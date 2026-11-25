from __future__ import annotations

import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _extra in (_ROOT / "scripts", _ROOT / "tests" / "unit"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


@pytest.fixture(autouse=True)
def _restore_arena_data_dir():
    """Restore ARENA_DATA_DIR after each test so os.environ edits stay isolated."""
    previous = os.environ.get("ARENA_DATA_DIR")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ARENA_DATA_DIR", None)
        else:
            os.environ["ARENA_DATA_DIR"] = previous
