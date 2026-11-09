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

The first pass discovers array dimensions and dtypes; the second reads owned
input regions in workers and feeds bounded batches through the shared VCF
adapter and Zarr writer. Input files must remain unchanged
between passes. Supplying `ploidy` skips genotype discovery, pads genotypes,
and rejects larger ploidies. Invalid allele indices and field overflows raise
`ValueError`. Batches can be smaller than output chunks; a single row or physical
codec chunk exceeding the fixed memory budget requires smaller output chunks or
the default ICF conversion. This budget accounts for batch and encoding buffers,
not a hard process RSS limit. `infer_shape` cannot be combined with `icf_path`.

## API reference

```{eval-rst}

.. autofunction:: bio2zarr.vcf.convert

```
