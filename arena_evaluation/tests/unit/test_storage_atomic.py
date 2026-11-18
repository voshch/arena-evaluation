"""One atomic-write helper: a reader sees the old file or the new one, never a partial one."""

from __future__ import annotations

import pathlib

import pytest

from arena_evaluation.storage.atomic import (
    atomic_write_bytes,
    atomic_write_text,
    staged_temp_path,
)


def test_atomic_write_text_replaces_the_document_and_keeps_its_mode(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "notes.yaml"
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o640)

    atomic_write_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert path.stat().st_mode & 0o777 == 0o640
    assert not list(tmp_path.glob(".notes.yaml.*.tmp"))


def test_atomic_write_bytes_replaces_the_document(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "index.parquet"
    path.write_bytes(b"old")

    atomic_write_bytes(path, b"new")

    assert path.read_bytes() == b"new"


def test_a_failed_staged_write_keeps_the_original_and_removes_the_temporary(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "notes.yaml"
    path.write_text("old\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="renderer failed"):
        with staged_temp_path(path) as temporary:
            temporary.write_text("new\n", encoding="utf-8")
            raise RuntimeError("renderer failed")

    assert path.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".notes.yaml.*.tmp"))
