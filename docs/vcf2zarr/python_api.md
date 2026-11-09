(sec-vcf2zarr-python-api)=
# Python API

Basic usage:
```python
import bio2zarr.vcf as v2z

v2z.convert([vcf_path], vcz_path)
```

To convert directly to an in-memory Zarr store (without writing to disk):
```python
root = v2z.convert([vcf_path])
```

To convert to a zip archive:
```python
root = v2z.convert([vcf_path], "output.vcz.zip")
```

To avoid intermediate storage, use a two-pass conversion:
```python
root = v2z.convert([vcf_path], infer_shape=True)
root = v2z.convert([vcf_path], infer_shape=True, ploidy=2)
```

The first pass discovers array dimensions and dtypes; the second reads input
sequentially and encodes bounded chunks. Input files must remain unchanged
between passes. Supplying `ploidy` skips genotype discovery, pads genotypes,
and rejects larger ploidies. Invalid allele indices and field overflows raise
`ValueError`. Oversized chunks require a smaller `variants_chunk_size` or the
default ICF conversion. `infer_shape` cannot be combined with `icf_path`.

For S3 input/output and temporary ICF placement in the default conversion:
```python
root = v2z.convert(
    ["s3://my-bucket/input/sample.vcf.gz"],
    "s3://my-bucket/output/sample.vcz",
    temp_icf_dir="s3://my-bucket/tmp",
)
```

See the overview for credentials and endpoint configuration. `temp_icf_dir`
and `icf_path` are mutually exclusive and cannot be used with `infer_shape`.

## API reference

```{eval-rst}

.. autofunction:: bio2zarr.vcf.convert

```
