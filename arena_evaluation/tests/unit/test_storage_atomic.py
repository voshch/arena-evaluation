"""One atomic-write helper: a reader sees the old file or the new one, never a partial one."""

from __future__ import annotations

import pathlib


from arena_evaluation.storage.atomic import (
    atomic_write_bytes,
)


def test_atomic_write_bytes_replaces_the_document(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "index.parquet"
    path.write_bytes(b"old")

    atomic_write_bytes(path, b"new")

    assert path.read_bytes() == b"new"
