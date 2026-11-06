(sec-plink2zarr)=
# plink2zarr

Convert plink data to the
[VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
reliably in parallel.

See {ref}`sec-plink2zarr-cli-ref` for detailed documentation on
command line options.

Conversion of the plink data model to VCF follows the semantics of plink1.9 as closely
as possible. That is, given a binary plink fileset with prefix "fileset" (i.e.,
fileset.bed, fileset.bim, fileset.fam), running
```
$ plink2zarr convert fileset out.vcz
```
should produce the same result in ``out.vcz`` as
```
$ plink1.9 --bfile fileset --keep-allele-order --recode vcf-iid --out tmp
$ vcf2zarr convert tmp.vcf out.vcz
```

## S3 input and output

Install `bio2zarr[s3]` to read a PLINK 1 BED/BIM/FAM fileset from S3, write
Zarr output to S3, or both:

```bash
plink2zarr convert s3://my-bucket/input/fileset s3://my-bucket/output/fileset.vcz
```

Use the common prefix without `.bed`, `.bim` or `.fam`. BED input is seeked
using S3 byte-range reads for each worker's assigned variants, without a
whole-file download or local cache. BIM/FAM metadata is streamed through UPath.
Missing files, invalid BED headers and truncated genotype blocks raise errors.
Local filesets and local/ZIP output remain supported; ZIP output is local-only.

Credentials use the standard `s3fs` provider chain; `AWS_ENDPOINT_URL_S3` or
`AWS_ENDPOINT_URL` selects an S3-compatible endpoint. Each worker needs the
same configuration. See the [S3 storage documentation](../vcf2zarr/overview.md#s3-input-and-output-storage)
for direct output, completion markers, retry semantics and MinIO testing.
PGEN input is a separate format and remains outside this layer.

:::{warning}
It is important to note that we follow the same conventions as plink 2.0
where the A1 allele in the [bim file](https://www.cog-genomics.org/plink/2.0/formats#bim)
is the VCF ALT and A2 is the REF.
:::

:::{note}
Currently we only convert the basic VCF-like data from plink, and don't include
phenotypes and pedigree information. These are planned as future enhancements.
Please comment on [this issue](https://github.com/sgkit-dev/bio2zarr/issues/392)
if you are interested in this functionality.
:::



