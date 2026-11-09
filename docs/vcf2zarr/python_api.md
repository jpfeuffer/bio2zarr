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
root = v2z.convert([vcf_path], infer_shape=True, max_alleles=4)
```

The first pass discovers array dimensions and dtypes; the second reads input
sequentially and encodes bounded chunks. Input files must remain unchanged
between passes. Supplying `ploidy` skips genotype discovery, pads genotypes,
and rejects larger ploidies. Invalid allele indices and field overflows raise
`ValueError`. Oversized chunks require a smaller `variants_chunk_size` or the
default ICF conversion. `infer_shape` cannot be combined with `icf_path`.

`max_alleles` is an optional bound on the total number of alleles, including
REF (so `2` means biallelic). It sizes the allele dimensions and, when GT is
present, the corresponding Number=G genotype dimension. Values below `2` are
invalid, and a record exceeding the bound raises `ValueError`. The option is
supported only with explicit `infer_shape=True`; it does not switch the default
ICF conversion mode and is not supported by custom-schema encoding. For inputs
without GT, Number=G retains its observed width rather than inferring a
theoretical width from `max_alleles`.

## API reference

```{eval-rst}

.. autofunction:: bio2zarr.vcf.convert

```
