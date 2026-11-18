from __future__ import annotations

import pathlib
import subprocess

import pytest

from arena_evaluation.benchmark.runner import _arena_git_provenance
from arena_evaluation.ingestion.metadata import IngestionMetadata


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def colcon_workspace(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    arena = tmp_path / "src" / "Arena"
    arena.mkdir(parents=True)
    _git(arena, "init")
    _git(arena, "config", "user.email", "fix9@example.invalid")
    _git(arena, "config", "user.name", "FIX9")
    (arena / "README.md").write_text("arena\n")
    _git(arena, "add", "README.md")
    _git(arena, "commit", "-m", "init")
    share = tmp_path / "install" / "share" / "arena_evaluation"
    share.mkdir(parents=True)
    return arena, share


def test_arena_git_provenance_reads_the_checkout_that_owns_the_installed_share(
    colcon_workspace: tuple[pathlib.Path, pathlib.Path],
) -> None:
    arena, share = colcon_workspace

    sha, dirty = _arena_git_provenance(share)

    assert sha == _git(arena, "rev-parse", "HEAD")
    assert dirty is False


def test_arena_git_provenance_reports_a_dirty_checkout(
    colcon_workspace: tuple[pathlib.Path, pathlib.Path],
) -> None:
    arena, share = colcon_workspace
    (arena / "README.md").write_text("changed\n")

    assert _arena_git_provenance(share)[1] is True


def test_arena_git_provenance_is_null_outside_a_git_checkout(
    tmp_path: pathlib.Path,
) -> None:
    share = tmp_path / "install" / "share" / "arena_evaluation"
    share.mkdir(parents=True)

    assert _arena_git_provenance(share) == (None, False)


def test_resolve_git_workspace_finds_the_colcon_src_checkout(
    tmp_path: pathlib.Path,
) -> None:
    arena = tmp_path / "src" / "Arena"
    arena.mkdir(parents=True)
    (arena / ".git").mkdir()

    assert IngestionMetadata.resolve_git_workspace(str(tmp_path / "install")) == str(
        arena
    )
