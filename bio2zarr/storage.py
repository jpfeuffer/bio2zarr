"""Local paths and S3 object prefixes used by conversion storage."""

import contextlib
import os
import pathlib
import shutil
import tempfile
import uuid

import zarr
from fsspec.utils import get_protocol
from upath import UPath
from upath.implementations.cloud import S3Path

S3_PROTOCOL = "s3"
HTSLIB_S3_PROTOCOLS = {"s3+http": "http", "s3+https": "https"}
S3_PROTOCOLS = frozenset({S3_PROTOCOL, *HTSLIB_S3_PROTOCOLS})


def protocol(value):
    """Read a UPath protocol or parse a path using fsspec's public API."""
    return value.protocol if isinstance(value, UPath) else get_protocol(str(value))


def path(value):
    scheme = protocol(value)
    if scheme == S3_PROTOCOL:
        result = (
            value
            if isinstance(value, S3Path)
            else UPath(
                value,
                endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3")
                or os.environ.get("AWS_ENDPOINT_URL"),
                use_listings_cache=False,
            )
        )
        if (
            not result.drive
            or not result.name
            or result.path.rstrip("/") == result.drive
        ):
            raise ValueError("S3 paths must include a bucket and an object key/prefix")
        return result
    if scheme not in ("", "file"):
        raise ValueError(f"Unsupported storage protocol: {scheme}")
    return pathlib.Path(value)


def mkdir(value, *, parents=False, exist_ok=False):
    if isinstance(value, S3Path):
        # Object prefixes need no mkdir; never create a bucket implicitly.
        if not value.fs.exists(value.drive):
            raise FileNotFoundError(f"S3 bucket does not exist: {value.drive}")
    else:
        value.mkdir(parents=parents, exist_ok=exist_ok)


def remove_tree(value):
    value = path(value)
    if isinstance(value, S3Path):
        # Match only descendants: a similarly named neighbouring prefix is not ours.
        objects = value.fs.find(value.path.rstrip("/") + "/")
        if objects:
            value.fs.rm(objects)
    else:
        shutil.rmtree(value)


def file_size(value):
    value = path(value)
    return value.stat().st_size


def zarr_store(value):
    if not isinstance(value, (str, os.PathLike, S3Path)):
        return value
    value = path(value)
    if isinstance(value, S3Path):
        return zarr.storage.FsspecStore.from_url(
            str(value), storage_options=dict(value.storage_options)
        )
    return value


def copy_tree(src, dest):
    """Copy a tree without deleting the source on a partial failure."""
    src, dest = path(src), path(dest)
    if isinstance(src, S3Path) and isinstance(dest, S3Path):
        prefix = src.path.rstrip("/") + "/"
        for key in src.fs.find(prefix):
            dest.fs.cp_file(key, dest.path.rstrip("/") + "/" + key[len(prefix) :])
    else:
        shutil.copytree(src, dest, dirs_exist_ok=True)


def move(src, dest):
    """Move locally; on S3 remove the source only after every copy succeeds."""
    src, dest = path(src), path(dest)
    if isinstance(src, S3Path):
        if src.is_file():
            src.fs.cp_file(src.path, dest.path)
            src.unlink()
        else:
            copy_tree(src, dest)
            remove_tree(src)
    else:
        os.rename(src, dest)


@contextlib.contextmanager
def temporary_path(parent, *, prefix):
    # An internal sibling parent may be the bucket root; only the unique child
    # is created or removed, never the parent.
    parent = parent if isinstance(parent, S3Path) else path(parent)
    if isinstance(parent, S3Path):
        result = path(parent / (prefix + uuid.uuid4().hex))
        mkdir(result)
        try:
            yield result
        finally:
            remove_tree(result)
    else:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as tmp:
            yield pathlib.Path(tmp)
