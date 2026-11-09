import gzip
import pathlib
import sys
from unittest import mock

import pytest

from bio2zarr import core, vcf, vcf_utils
from tests.utils import assert_zarr_equal

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="VCF support requires cyvcf2"
)


HEADER = (
    "##fileformat=VCFv4.3\n"
    "##contig=<ID=1,length=10000>\n"
    '##INFO=<ID=IV,Number=.,Type=Integer,Description="Variable integer">\n'
    '##INFO=<ID=FV,Number=.,Type=Float,Description="Variable float">\n'
    '##INFO=<ID=TXT,Number=.,Type=String,Description="Variable string">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depth">\n'
    '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Likelihood">\n'
    '##FORMAT=<ID=VF,Number=.,Type=Float,Description="Variable float">\n'
    '##FORMAT=<ID=VS,Number=.,Type=String,Description="Variable string">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
)
ROWS = [
    "1\t1\trs1\tA\tC\t.\tPASS\tIV=-120,0;FV=1.5,.;TXT=a,b\t"
    "GT:AD:PL:VF:VS\t0/1:10,20:0,1,2:0.5,.:x,y\t./.:.,.:.,.,.:.,.:.",
    "1\t2\t.\tG\tA,T\t12.5\tPASS\tIV=32768,.;FV=.,-2.25,3;TXT=c\t"
    "GT:AD:PL:VF:VS\t1|2:2,3,4:0,1,2,3,4,5:1.25,2.5,3.75:p,q,r\t"
    "0:8,0,0:0,1,2,3,4,5:.:z",
    "1\t3\t.\tT\tG\t.\tPASS\t.\tGT\t0/1\t.",
    "1\t4\tlast\tC\tA,G,T\t.\tPASS\tIV=-40000,5,6,7;FV=0.25;"
    "TXT=last,one,two,three\tGT:AD:PL:VF:VS\t"
    "0/1/2:1,2,3,4:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19:"
    ".,-3,4,5:u,v,w,x\t"
    "0/0:9,0,0,0:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19:"
    "0,1:.",
]


def write_plain_fixture(path, newline="\n", final_newline=True, rows=ROWS):
    text = HEADER + "\n".join(rows) + ("\n" if final_newline else "")
    path.write_bytes(text.replace("\n", newline).encode())


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [False, True])
def test_plain_byte_ranges_exact_boundaries(tmp_path, newline, final_newline):
    path = tmp_path / "input.vcf"
    write_plain_fixture(path, newline, final_newline)
    expected = [1, 2, 3, 4]
    data = path.read_bytes()
    row_boundary = data.index(b"\n1\t3\t") + 1
    for boundary in sorted({len(data) // 2, row_boundary}):
        actual = []
        for start, stop in [(0, boundary), (boundary, len(data))]:
            actual.extend(row.POS for row in vcf.iter_vcf_byte_range(path, start, stop))
        assert actual == expected, boundary


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [False, True])
def test_parallel_plain_discovery_and_conversion(
    tmp_path, workers, newline, final_newline
):
    path = tmp_path / "input.vcf"
    write_plain_fixture(path, newline, final_newline)
    expected, expected_header = vcf.discover_vcf_shape([path])
    actual, actual_header = vcf.discover_vcf_shape([path], worker_processes=workers)
    assert actual.asdict() == expected.asdict()
    assert actual_header == expected_header
    fields = {field.full_name: field for field in actual.fields}
    assert actual.num_records == len(ROWS)
    assert fields["FORMAT/GT"].summary.max_number == 4
    assert fields["INFO/IV"].summary.max_number == 4
    assert fields["INFO/IV"].smallest_dtype() == "i4"
    assert fields["FORMAT/AD"].summary.max_number == 4
    assert fields["FORMAT/PL"].summary.max_number == 20
    assert fields["FORMAT/VF"].summary.max_number == 4
    assert fields["FORMAT/VS"].summary.max_number == 4
    serial = vcf.convert([path], infer_shape=True, variants_chunk_size=2)
    parallel = vcf.convert(
        [path],
        infer_shape=True,
        worker_processes=workers,
        variants_chunk_size=2,
    )
    assert_zarr_equal(serial, parallel)


def test_plain_discovery_matches_icf(tmp_path):
    path = tmp_path / "input.vcf"
    write_plain_fixture(path)
    icf = vcf.explode(tmp_path / "icf", [path])
    expected = vcf.encode(icf.path, variants_chunk_size=2)
    actual = vcf.convert([path], infer_shape=True, variants_chunk_size=2)
    assert_zarr_equal(expected, actual)


def test_unindexed_gzip_parallel_request_warns_and_scans_serially(tmp_path, caplog):
    plain = tmp_path / "input.vcf"
    write_plain_fixture(plain)
    compressed = tmp_path / "input.vcf.gz"
    with gzip.open(compressed, "wb") as out:
        out.write(plain.read_bytes())
    expected, _ = vcf.discover_vcf_shape([compressed])
    actual, _ = vcf.discover_vcf_shape([compressed], worker_processes=2)
    assert actual.asdict() == expected.asdict()
    assert "shape discovery will be serial" in caplog.text
    assert "indexed BGZF" in caplog.text


@pytest.mark.parametrize(
    "filename", ["sample.vcf.gz", "sample.bcf", "field_type_combos.vcf.gz"]
)
def test_indexed_discovery_still_reduces_regions(filename):
    path = f"tests/data/vcf/{filename}"
    expected, header = vcf.discover_vcf_shape([path])
    actual, actual_header = vcf.discover_vcf_shape([path], worker_processes=2)
    assert actual.asdict() == expected.asdict()
    assert actual.num_records == expected.num_records
    assert actual_header == header


def test_empty_plain_parallel_discovery(tmp_path):
    path = tmp_path / "empty.vcf"
    path.write_text(HEADER)
    expected, _ = vcf.discover_vcf_shape([path])
    actual, _ = vcf.discover_vcf_shape([path], worker_processes=2)
    assert actual.asdict() == expected.asdict()


def test_plain_partition_end_tracks_last_input_record(tmp_path):
    path = tmp_path / "unsorted.vcf"
    rows = [
        "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0",
        "1\t4\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0",
        "1\t8\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0",
        "1\t2\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0",
    ]
    write_plain_fixture(path, rows=rows)
    expected, _ = vcf.discover_vcf_shape([path])
    actual, _ = vcf.discover_vcf_shape([path], worker_processes=2)
    assert expected.partitions[0].region.end == 2
    assert actual.partitions[0].region.end == expected.partitions[0].region.end


def test_plain_parallel_discovery_rejects_multiple_contigs(tmp_path):
    path = tmp_path / "multi.vcf"
    text = HEADER.replace(
        "##contig=<ID=1,length=10000>\n",
        "##contig=<ID=1,length=10000>\n##contig=<ID=2,length=10000>\n",
    )
    path.write_text(
        text
        + "1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\n"
        + "2\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\n"
    )
    with pytest.raises(ValueError, match="Multi-contig VCFs must be indexed"):
        vcf.discover_vcf_shape([path], worker_processes=2)


def test_plain_range_early_close_releases_feeder(tmp_path):
    path = tmp_path / "input.vcf"
    write_plain_fixture(path)
    with path.open("ab") as out:
        for i in range(10, 100):
            out.write(f"1\t{i}\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\n".encode())
    iterator = vcf.iter_vcf_byte_range(path, 0, path.stat().st_size)
    assert next(iterator).POS == 1
    iterator.close()


@pytest.mark.parametrize("kind", ["plain", "gzip", "indexed"])
def test_discovery_progress_uses_record_total(tmp_path, kind):
    path = tmp_path / "input.vcf"
    write_plain_fixture(path)
    if kind == "gzip":
        compressed = tmp_path / "input.vcf.gz"
        with gzip.open(compressed, "wb") as out:
            out.write(path.read_bytes())
        path = compressed
    elif kind == "indexed":
        path = pathlib.Path("tests/data/vcf/sample.vcf.gz")
    with mock.patch.object(core.tqdm, "tqdm", wraps=core.tqdm.tqdm) as progress:
        metadata, _ = vcf.discover_vcf_shape(
            [path], worker_processes=2, show_progress=True
        )
    assert progress.call_args.kwargs["unit"] == "records"
    assert progress.call_args.kwargs["total"] == (9 if kind == "indexed" else None)
    assert progress.call_args.kwargs["disable"] is False
    assert core.get_progress() == metadata.num_records


def test_discovery_progress_disabled(tmp_path, capsys):
    path = tmp_path / "input.vcf"
    write_plain_fixture(path)
    with mock.patch.object(core.tqdm, "tqdm", wraps=core.tqdm.tqdm) as progress:
        vcf.discover_vcf_shape([path], show_progress=False)
    assert progress.call_args.kwargs["disable"] is True
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    assert core.get_progress() == 0


def test_summary_progress_batches_record_updates(tmp_path):
    path = tmp_path / "input.vcf"
    rows = [f"1\t{i}\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0" for i in range(1, 1027)]
    write_plain_fixture(path, rows=rows)
    metadata, _ = vcf.scan_vcf(path, target_num_partitions=1)
    updates = []
    with (
        vcf_utils.VcfFile(path) as file,
        mock.patch.object(
            vcf, "variant_values", wraps=vcf.variant_values
        ) as extract_values,
        mock.patch.object(vcf.time, "monotonic", return_value=0),
        mock.patch.object(
            core,
            "update_progress",
            side_effect=lambda inc: updates.append((extract_values.call_count, inc)),
        ),
    ):
        summary = vcf.summarise_vcf_variants(
            file.variants(), metadata, ploidy=None, show_progress=True
        )
    assert summary.num_records == 1026
    assert updates == [(1024, 1024), (1026, 2)]
