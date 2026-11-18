"""Atomic replacement and advisory locking for files rewritten in place.

``fcntl.flock`` is POSIX-only: the Arena evaluation pipeline runs on Linux, so the
helper deliberately has no Windows fallback.

Every read-modify-write of one file takes :func:`file_lock` around the read and the
write, and :func:`staged_temp_path` replaces the file in one rename, so a reader never
sees a partial document and concurrent writers never lose an update.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pathlib
import stat
import tempfile
from collections.abc import Iterator

__all__ = ["atomic_write_bytes", "atomic_write_text", "file_lock", "staged_temp_path"]


@contextlib.contextmanager
def file_lock(path: pathlib.Path) -> Iterator[None]:
    """Hold the exclusive advisory lock of ``path``, the guard every read-modify-write of it takes."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def staged_temp_path(path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield a sibling temporary path, renaming it onto ``path`` on success.

    The caller writes the temporary file (text, bytes or Parquet); the context flushes it
    and replaces ``path`` in one ``os.replace``. A failed body removes the temporary file
    and leaves the original untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: pathlib.Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` in one rename."""
    with staged_temp_path(path) as temporary:
        temporary.write_bytes(payload)


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one rename, written as UTF-8 with Unix newlines."""
    with staged_temp_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
