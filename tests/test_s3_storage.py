"""Real S3 integration tests; set BIO2ZARR_MINIO_ENDPOINT to a local MinIO."""

import json
import os
import pathlib
import uuid

import boto3
import click.testing
import numpy.testing as nt
import pytest

from bio2zarr import bgen, cli, plink, storage, vcz, zarr_utils
from bio2zarr import vcf as vcf_mod
from tests.test_bgen import write_bgen
from tests.test_bgen_general import write_layout1
from tests.utils import assert_zarr_equal

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def minio(monkeypatch, tmp_path):
    endpoint = os.environ.get("BIO2ZARR_MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip("Set BIO2ZARR_MINIO_ENDPOINT to run real MinIO integration tests")
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.setenv("HTS_S3_HOST", endpoint.removeprefix("http://"))
    monkeypatch.setenv("HTS_S3_ADDRESS_STYLE", "path")
    client = boto3.client("s3", endpoint_url=endpoint)
    bucket = "bio2zarr-test-" + uuid.uuid4().hex
    client.create_bucket(Bucket=bucket)
    monkeypatch.chdir(tmp_path)
    copy_tree, move = storage.copy_tree, storage.move

    call_s3 = storage.path(f"s3://{bucket}/out").fs._call_s3

    async def without_copy(method, *args, **kwargs):
        if method in ("copy_object", "upload_part_copy"):
            pytest.fail("Remote conversion must not copy objects")
        return await call_s3(method, *args, **kwargs)

    monkeypatch.setattr(storage.path(f"s3://{bucket}/out").fs, "_call_s3", without_copy)

    def local_only(operation):
        def checked(src, dest):
            if storage.is_remote(src) or storage.is_remote(dest):
                pytest.fail("Remote conversion must not copy data or trees")
            return operation(src, dest)

        return checked

    monkeypatch.setattr(storage, "copy_tree", local_only(copy_tree))
    monkeypatch.setattr(storage, "move", local_only(move))
    try:
        yield client, bucket
    finally:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if keys:
                client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        client.delete_bucket(Bucket=bucket)


def upload(minio, category, filename):
    client, bucket = minio
    path = DATA / category / filename
    for suffix in ("", ".tbi", ".csi", ".metafile"):
        source = pathlib.Path(str(path) + suffix)
        if source.exists():
            client.upload_file(str(source), bucket, "input/" + filename + suffix)
    return f"s3://{bucket}/input/{filename}"


class TestMinioPartitionRetry:
    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_failed_retry_invalidates_completed_partition(
        self, minio, monkeypatch, zarr_format
    ):
        uri = upload(minio, "vcf", "sample.vcf.gz").replace("s3://", "s3+http://")
        icf = f"s3://{minio[1]}/icf"
        out = f"s3://{minio[1]}/out"
        vcf_mod.explode(icf, [uri])
        summary = vcf_mod.encode_init(
            icf, out, 3, variants_chunk_size=3, zarr_format=zarr_format
        )
        for partition in range(summary.num_partitions):
            vcf_mod.encode_partition(out, partition)
        marker = storage.path(out + "/wip/partitions/p0/.complete")
        assert marker.exists()
        encode = vcz.VcfZarrWriter.encode_partition_fields

        def fail_retry(writer, partition):
            encode(writer, partition)
            raise OSError("injected retry failure")

        with monkeypatch.context() as patch:
            patch.setattr(vcz.VcfZarrWriter, "encode_partition_fields", fail_retry)
            with pytest.raises(OSError, match="injected retry failure"):
                vcf_mod.encode_partition(out, 0)
        assert not marker.exists()
        with pytest.raises(FileNotFoundError, match="Partitions not encoded"):
            vcf_mod.encode_finalise(out)
        with pytest.raises(ValueError, match="conversion is incomplete"):
            zarr_utils.open_vcf_zarr(out)
        vcf_mod.encode_partition(out, 0)
        assert marker.exists()
        vcf_mod.encode_finalise(out)
        expected = vcf_mod.encode(icf, variants_chunk_size=3, zarr_format=zarr_format)
        assert_zarr_equal(expected, zarr_utils.open_vcf_zarr(out))


class TestMinioDirectLifecycle:
    @pytest.mark.parametrize("zarr_format", [2, 3])
    @pytest.mark.parametrize("filename", ["sample.vcf.gz", "sample.bcf"])
    def test_finalisation_retry_without_reopening_input(
        self, minio, monkeypatch, zarr_format, filename
    ):
        uri = upload(minio, "vcf", filename).replace("s3://", "s3+http://")
        out = f"s3://{minio[1]}/out"
        kwargs = dict(
            infer_shape=True,
            variants_chunk_size=3,
            samples_chunk_size=2,
            worker_processes=2,
            zarr_format=zarr_format,
        )
        expected = vcf_mod.convert([DATA / "vcf" / filename], **kwargs)

        def fail_index(indexer):
            raise OSError("injected direct index failure")

        with monkeypatch.context() as patch:
            patch.setattr(vcz.VcfZarrIndexer, "create_index", fail_index)
            with pytest.raises(OSError, match="injected direct index failure"):
                vcf_mod.convert([uri], out, **kwargs)
        with pytest.raises(ValueError, match="conversion is incomplete"):
            zarr_utils.open_vcf_zarr(out)
        metadata = json.loads(storage.path(out + "/wip/metadata.json").read_text())
        assert metadata["source_path"] == uri
        for partition in range(len(metadata["partitions"])):
            assert storage.path(
                out + f"/wip/partitions/p{partition}/.complete"
            ).exists()
        minio[0].delete_object(Bucket=minio[1], Key="input/" + filename)
        vcf_mod.encode_finalise(out)
        assert not storage.path(out + "/wip").exists()
        assert_zarr_equal(expected, zarr_utils.open_vcf_zarr(out))

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_failed_chunk_keeps_output_unreadable(
        self, minio, monkeypatch, zarr_format
    ):
        uri = upload(minio, "vcf", "sample.vcf.gz").replace("s3://", "s3+http://")
        out = f"s3://{minio[1]}/out"
        neighbour = storage.path(out + "-neighbour/keep")
        neighbour.write_text("keep")
        encode = vcz.VcfZarrWriter.encode_partition_fields

        def fail_chunk(writer, partition):
            encode(writer, partition)
            raise OSError("injected direct chunk failure")

        monkeypatch.setattr(vcz.VcfZarrWriter, "encode_partition_fields", fail_chunk)
        with pytest.raises(OSError, match="injected direct chunk failure"):
            vcf_mod.convert(
                [uri],
                out,
                infer_shape=True,
                variants_chunk_size=3,
                zarr_format=zarr_format,
            )
        assert not storage.path(out + "/wip/partitions/p0/.complete").exists()
        with pytest.raises(ValueError, match="conversion is incomplete"):
            zarr_utils.open_vcf_zarr(out)
        with pytest.raises(FileNotFoundError, match="Partitions not encoded"):
            vcf_mod.encode_finalise(out)
        assert neighbour.read_text() == "keep"
        assert vcf_mod.direct_vcf_writer.cache_info().currsize == 0


def upload_plink(minio, filename):
    for suffix in (".bed", ".bim", ".fam"):
        upload(minio, "plink", filename + suffix)
    return f"s3://{minio[1]}/input/{filename}"


class TestMinioConversions:
    @pytest.mark.parametrize("zarr_format", [2, 3])
    @pytest.mark.parametrize("filename", ["sample.vcf.gz", "sample.bcf"])
    @pytest.mark.parametrize("direct", [False, True])
    def test_minio_vcf_end_to_end(self, minio, tmp_path, filename, zarr_format, direct):
        uri = upload(minio, "vcf", filename).replace("s3://", "s3+http://")
        bucket = minio[1]
        out = f"s3://{bucket}/output/run"
        kwargs = dict(
            variants_chunk_size=3,
            samples_chunk_size=2,
            zarr_format=zarr_format,
            worker_processes=2,
        )
        expected = vcf_mod.convert(
            [DATA / "vcf" / filename], infer_shape=direct, **kwargs
        )
        if not direct:
            kwargs["temp_icf_dir"] = f"s3://{bucket}/tmp"
        actual = vcf_mod.convert([uri], out, infer_shape=direct, **kwargs)
        assert_zarr_equal(expected, actual)
        assert_zarr_equal(expected, zarr_utils.open_vcf_zarr(out))
        assert vcz.VcfZarr(out).summary_table()
        assert not storage.path(out + "/wip").exists()
        assert not storage.path(f"s3://{bucket}/tmp").exists()

    @pytest.mark.parametrize("zarr_format", [2, 3])
    @pytest.mark.parametrize(
        "filename", ["empty.vcf.gz", "mixed_ploidy.vcf.gz", "field_type_combos.vcf.gz"]
    )
    def test_minio_direct_write_edge_cases(self, minio, zarr_format, filename):
        uri = upload(minio, "vcf", filename).replace("s3://", "s3+http://")
        out = f"s3://{minio[1]}/out"
        kwargs = dict(
            zarr_format=zarr_format,
            variants_chunk_size=3,
            samples_chunk_size=2,
            worker_processes=2,
            infer_shape=True,
        )
        expected = vcf_mod.convert([DATA / "vcf" / filename], **kwargs)
        assert_zarr_equal(expected, vcf_mod.convert([uri], out, **kwargs))

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_minio_distributed_retry(self, minio, tmp_path, monkeypatch, zarr_format):
        bucket = minio[1]
        icf = f"s3://{bucket}/icf"
        uri = upload(minio, "vcf", "sample.vcf.gz").replace("s3://", "s3+http://")
        vcf_mod.explode(icf, [uri], worker_processes=2)
        out = f"s3://{bucket}/out"
        neighbour = storage.path(out + "-neighbour/keep")
        with neighbour.open("wb") as f:
            f.write(b"keep")
        summary = vcf_mod.encode_init(
            icf, out, 3, variants_chunk_size=3, zarr_format=zarr_format
        )
        original = vcz.VcfZarrWriter.encode_partition_fields

        def fail_partition(writer, index):
            original(writer, index)
            if index == 0:
                raise OSError("injected partition failure")

        with monkeypatch.context() as patch:
            patch.setattr(vcz.VcfZarrWriter, "encode_partition_fields", fail_partition)
            with pytest.raises(OSError, match="injected"):
                vcf_mod.encode_partition(out, 0)
        with pytest.raises(FileNotFoundError, match="Partitions not encoded"):
            vcf_mod.encode_finalise(out)
        assert not storage.path(out + "/wip/partitions/p0/.complete").exists()
        with pytest.raises(ValueError, match="conversion is incomplete"):
            zarr_utils.open_vcf_zarr(out)
        for j in range(summary.num_partitions):
            vcf_mod.encode_partition(out, j)
        vcf_mod.encode_partition(out, 0)
        assert not storage.path(out + "/wip/partitions/wip_p0").exists()

        def fail_index(self):
            raise OSError("injected index failure")

        with monkeypatch.context() as patch:
            patch.setattr(vcz.VcfZarrIndexer, "create_index", fail_index)
            with pytest.raises(OSError, match="injected"):
                vcf_mod.encode_finalise(out)
        assert storage.path(out + "/wip/partitions/p0/.complete").exists()
        vcf_mod.encode_finalise(out)
        vcf_mod.create_index(out)
        expected = vcf_mod.encode(icf, variants_chunk_size=3, zarr_format=zarr_format)
        assert_zarr_equal(expected, zarr_utils.open_vcf_zarr(out))
        assert neighbour.read_bytes() == b"keep"
        assert not storage.path(out + "/wip").exists()

    @pytest.mark.parametrize("zarr_format", [2, 3])
    @pytest.mark.parametrize("filename", ["haplotypes.bgen", "complex.23bits.bgen"])
    def test_minio_bgen(self, minio, tmp_path, filename, zarr_format):
        uri = upload(minio, "bgen", filename)
        out = f"s3://{minio[1]}/out"
        kwargs = dict(
            variants_chunk_size=3,
            samples_chunk_size=2,
            worker_processes=2,
            zarr_format=zarr_format,
        )
        expected = bgen.convert(DATA / "bgen" / filename, **kwargs)
        actual = bgen.convert(uri, out, **kwargs)
        assert_zarr_equal(expected, actual)
        assert_zarr_equal(expected, zarr_utils.open_vcf_zarr(out))
        assert not list(
            storage.path(f"s3://{minio[1]}/out").parent.glob(".bio2zarr-bgen-*")
        )

    @pytest.mark.parametrize("layout", [1, 2])
    @pytest.mark.parametrize("compression", [0, 1, 2])
    def test_minio_bgen_encodings(self, minio, tmp_path, layout, compression):
        if layout == 1 and compression == 2:
            pytest.skip("Layout 1 only supports raw or zlib encodings")
        local = tmp_path / "input.bgen"
        if layout == 1:
            write_layout1(local, compression)
        else:
            write_bgen(local, compression=compression, bits=32)
        minio[0].upload_file(str(local), minio[1], "input/synthetic.bgen")
        uri = f"s3://{minio[1]}/input/synthetic.bgen"
        kwargs = dict(variants_chunk_size=2, samples_chunk_size=2)
        assert_zarr_equal(bgen.convert(local, **kwargs), bgen.convert(uri, **kwargs))

    def test_minio_bgen_validation_leaves_output_absent(self, minio):
        uri = upload(minio, "bgen", "example.8bits.bgen")
        out = f"s3://{minio[1]}/out"
        with pytest.raises(ValueError, match="positions must be nondecreasing"):
            bgen.convert(uri, out)
        assert not storage.path(out).exists()

    def test_minio_bgen_partial_conversion_is_not_readable(self, minio, monkeypatch):
        uri = upload(minio, "bgen", "haplotypes.bgen")
        out = storage.path(f"s3://{minio[1]}/out")
        encode = bgen.BgenZarrWriter.encode_partition_fields

        def fail(writer, index):
            encode(writer, index)
            raise OSError("injected conversion failure")

        monkeypatch.setattr(bgen.BgenZarrWriter, "encode_partition_fields", fail)
        with pytest.raises(OSError, match="injected"):
            bgen.convert(uri, out)
        assert (out / "wip" / ".incomplete").exists()
        with pytest.raises(ValueError, match="conversion is incomplete"):
            zarr_utils.open_vcf_zarr(out)

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_minio_local_bed_remote_output(self, minio, zarr_format):
        source = DATA / "plink" / "example"
        out = f"s3://{minio[1]}/bed"
        kwargs = dict(
            zarr_format=zarr_format, variants_chunk_size=2, worker_processes=2
        )
        assert_zarr_equal(
            plink.convert(source, **kwargs), plink.convert(source, out, **kwargs)
        )

    @pytest.mark.parametrize("zarr_format", [2, 3])
    @pytest.mark.parametrize(
        "filename", ["example", "example_with_fam", "plink_sim_10s_100v_10pmiss"]
    )
    @pytest.mark.parametrize("remote_output", [False, True])
    def test_minio_remote_plink(
        self, minio, tmp_path, zarr_format, filename, remote_output
    ):
        uri = upload_plink(minio, filename)
        out = f"s3://{minio[1]}/out" if remote_output else tmp_path / "local.vcz"
        kwargs = dict(
            zarr_format=zarr_format,
            variants_chunk_size=3,
            samples_chunk_size=2,
            worker_processes=2,
        )
        expected = plink.convert(DATA / "plink" / filename, **kwargs)
        actual = plink.convert(uri, out, **kwargs)
        assert_zarr_equal(expected, actual)
        assert_zarr_equal(expected, zarr_utils.open_vcf_zarr(out))

    def test_minio_bed_range_reads(self, minio, monkeypatch):
        uri = upload_plink(minio, "plink_sim_10s_100v_10pmiss")
        bed = storage.path(uri + ".bed")
        calls = []
        call_s3 = bed.fs._call_s3

        async def record(method, *args, **kwargs):
            if method == "get_object":
                calls.append(kwargs.get("Range"))
            return await call_s3(method, *args, **kwargs)

        monkeypatch.setattr(bed.fs, "_call_s3", record)
        actual = plink.BedReader(bed, 100, 10).decode(90, 93)
        expected = plink.BedReader(
            DATA / "plink" / "plink_sim_10s_100v_10pmiss.bed", 100, 10
        ).decode(90, 93)
        nt.assert_array_equal(expected, actual)
        assert calls == ["bytes=0-2", "bytes=273-281"]

    @pytest.mark.parametrize("suffix", [".bed", ".bim", ".fam"])
    def test_minio_missing_plink_file(self, minio, suffix):
        uri = upload_plink(minio, "example")
        minio[0].delete_object(Bucket=minio[1], Key="input/example" + suffix)
        with pytest.raises(FileNotFoundError):
            plink.convert(uri)

    def test_minio_invalid_and_truncated_bed(self, minio):
        uri = upload_plink(minio, "example")
        key = "input/example.bed"
        minio[0].put_object(Bucket=minio[1], Key=key, Body=b"invalid")
        with pytest.raises(ValueError, match="Invalid BED file magic"):
            plink.convert(uri)
        minio[0].put_object(Bucket=minio[1], Key=key, Body=b"\x6c\x1b\x01")
        with pytest.raises(ValueError, match="Truncated BED"):
            plink.convert(uri)

    def test_minio_remote_plink_cli(self, minio):
        uri = upload_plink(minio, "example")
        out = f"s3://{minio[1]}/out"
        result = click.testing.CliRunner().invoke(
            cli.plink2zarr_main,
            ["convert", uri, out, "-Q", "--zarr-format", "3"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        expected = plink.convert(DATA / "plink" / "example", zarr_format=3)
        assert_zarr_equal(expected, zarr_utils.open_vcf_zarr(out))

    def test_minio_bgen_sidecar_and_samples(self, minio, tmp_path):
        import cbgen  # noqa: PLC0415

        filename = "haplotypes.bgen"
        uri = upload(minio, "bgen", filename)
        index = tmp_path / "index.metafile"
        with cbgen.BgenFile(str(DATA / "bgen" / filename)) as source:
            source.create_metafile(str(index))
        minio[0].upload_file(str(index), minio[1], f"input/{filename}.metafile")
        ids = tmp_path / "ids.sample"
        ids.write_text("ID_1 missing\n0 0\none 0\ntwo 0\nthree 0\nfour 0\n")
        minio[0].upload_file(str(ids), minio[1], "input/ids.sample")
        sample = f"s3://{minio[1]}/input/ids.sample"
        actual = bgen.convert(uri, sample_file=sample, worker_processes=2)
        expected = bgen.convert(DATA / "bgen" / filename, sample_file=ids)
        assert_zarr_equal(expected, actual)
        assert index.read_bytes() == storage.path(uri + ".metafile").read_bytes()

    def test_minio_cli(self, minio, tmp_path):
        uri = upload(minio, "vcf", "sample.bcf").replace("s3://", "s3+http://")
        out = f"s3://{minio[1]}/out"
        result = click.testing.CliRunner().invoke(
            cli.vcf2zarr_main,
            ["convert", uri, out, "--infer-shape", "-Q"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        for args in (["inspect", out], ["create-index", out]):
            result = click.testing.CliRunner().invoke(
                cli.vcf2zarr_main, args, catch_exceptions=False
            )
            assert result.exit_code == 0
        uri = upload(minio, "bgen", "haplotypes.bgen")
        result = click.testing.CliRunner().invoke(
            cli.bgen2zarr_main,
            ["convert", uri, f"s3://{minio[1]}/bgen", "-Q"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    def test_minio_cleanup_and_bucket_protection(self, minio):
        parent = storage.path(f"s3://{minio[1]}/tmp")
        neighbour = parent / "neighbour" / "keep"
        neighbour.write_text("keep")

        def fail():
            with storage.temporary_path(parent, prefix="run-") as tmp:
                (tmp / "data").write_text("data")
                raise RuntimeError("injected")

        with pytest.raises(RuntimeError, match="injected"):
            fail()
        assert list(parent.iterdir()) == [neighbour.parent]
        assert neighbour.read_text() == "keep"
        missing = "bio2zarr-absent-" + uuid.uuid4().hex
        with pytest.raises(FileNotFoundError, match="bucket does not exist"):
            storage.mkdir(storage.path(f"s3://{missing}/out"))
        assert missing not in [b["Name"] for b in minio[0].list_buckets()["Buckets"]]
        metadata = vcf_mod.discover_vcf_shape([DATA / "vcf" / "sample.vcf.gz"])[
            0
        ].asdict()
        assert os.environ["AWS_SECRET_ACCESS_KEY"] not in json.dumps(metadata)
