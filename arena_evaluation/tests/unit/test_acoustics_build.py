"""The acoustic solver build path and its ABI contract."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from arena_evaluation.processing.acoustics import impedance_grid
from arena_evaluation.processing.acoustics.constants import (
    ACOUSTIC_SOLVER_ABI,
    ACOUSTIC_SOLVER_ABI_SYMBOL,
    ACOUSTIC_SOLVER_SYMBOLS,
)
from arena_evaluation.storage.exceptions import MetricInputError

PACKAGE_ROOT = Path(impedance_grid.__file__).resolve().parents[3]
SOLVER_SOURCE = PACKAGE_ROOT / "arena_evaluation/processing/acoustics/solver.cpp"
ENV_SO_VAR = "ARENA_ACOUSTIC_SOLVER_SO"


def _compile(tmp_path: Path, name: str, body: str, definition: str = "") -> Path:
    """Compile a stand-in solver library from `body`, optionally with a PARETO_K override."""
    source = tmp_path / f"{name}.cpp"
    source.write_text(body, encoding="utf-8")
    library = tmp_path / f"{name}.so"
    command = ["g++", "-O3", "-fPIC", "-shared", "-std=c++17"]
    if definition:
        command.append(f"-D{definition}")
    subprocess.run([*command, str(source), "-o", str(library)], check=True)
    return library


def _solver_from(tmp_path: Path, name: str, definition: str = "") -> Path:
    """The real solver.cpp compiled with an optional override."""
    library = tmp_path / f"{name}.so"
    command = ["g++", "-O3", "-fPIC", "-shared", "-std=c++17"]
    if definition:
        command.append(f"-D{definition}")
    subprocess.run([*command, str(SOLVER_SOURCE), "-o", str(library)], check=True)
    return library


def _use_library(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Point the loader at `path` and forget the previously loaded library."""
    monkeypatch.setenv(ENV_SO_VAR, str(path))
    impedance_grid._solver_library.cache_clear()


def _attenuations(measure: float) -> np.ndarray:
    """Attenuations from one corner of a walled grid to a fan of targets behind the wall."""
    size = 64
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[30:, :] = 1
    grid[30:, 20:24] = 0
    targets_x = np.linspace(2.0, 60.0, 200, dtype=np.float32)
    targets_y = np.linspace(40.0, 62.0, 200, dtype=np.float32)
    return impedance_grid.compute_attenuations(
        grid,
        0.1,
        2.0,
        2.0,
        targets_x,
        targets_y,
        wall_tl=measure,
        mic_distance=1.0,
    )


def test_the_default_library_sits_next_to_the_module() -> None:
    """Without an override the loader looks where the build writes."""
    assert impedance_grid.solver_library_path() == (Path(impedance_grid.__file__).resolve().parent / "solver.so")


def test_the_loaded_library_exports_the_wrappers_abi() -> None:
    """The library in the package tree is the one this wrapper binds."""
    assert impedance_grid.solver_abi(impedance_grid._solver_library()) == ACOUSTIC_SOLVER_ABI


def test_the_solver_source_declares_the_wrappers_abi() -> None:
    """The C++ ABI constant and the Python one cannot drift apart."""
    source = SOLVER_SOURCE.read_text(encoding="utf-8")
    assert f"#define ARENA_ACOUSTIC_SOLVER_ABI {ACOUSTIC_SOLVER_ABI}" in source


def test_a_library_with_another_abi_version_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A library built from a different model does not publish numbers under this ABI."""
    exports = f'extern "C" int {ACOUSTIC_SOLVER_ABI_SYMBOL}(void) {{ return 999; }}\n' + "".join(f'extern "C" void {symbol}() {{}}\n' for symbol in ACOUSTIC_SOLVER_SYMBOLS)
    foreign = _compile(tmp_path, "foreign", exports)
    _use_library(monkeypatch, foreign)
    with pytest.raises(MetricInputError, match="ABI 999"):
        impedance_grid._solver_library()


def test_a_library_missing_a_solver_symbol_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An ABI match without the full symbol set is still not loadable."""
    exports = f'extern "C" int {ACOUSTIC_SOLVER_ABI_SYMBOL}(void) {{ return {ACOUSTIC_SOLVER_ABI}; }}\nextern "C" void solve_acoustic_field() {{}}\n'
    partial = _compile(tmp_path, "partial", exports)
    _use_library(monkeypatch, partial)
    with pytest.raises(MetricInputError, match="solve_acoustic_grid"):
        impedance_grid._solver_library()


def test_cmake_declares_a_target_for_both_solvers() -> None:
    """`arena build` has something to compile: one CMake target per solver library."""
    project = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "arena_evaluation/processing/acoustics/solver.cpp" in project
    assert "arena_evaluation/processing/path/solver.cpp" in project
    assert "OUTPUT_NAME solver" in project
    assert "-O3" in project
    assert "CXX_STANDARD 17" in project
    assert "ARENA_SOLVERS_ONLY" in project


def test_setup_py_builds_the_solvers_through_cmake() -> None:
    """colcon builds this package with `setup.py develop`, so build_ext must compile the solvers."""
    package = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"build_ext": SolverBuildExt' in package
    assert '"build": SolverBuild' in package
    assert "ARENA_SOLVERS_ONLY=ON" in package


def test_pareto_truncation_matches_a_wide_frontier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """K=2 truncates nothing observable: it matches a 32-label frontier."""
    _use_library(monkeypatch, _solver_from(tmp_path, "k2", "PARETO_K=2"))
    narrow = _attenuations(47.0)
    _use_library(monkeypatch, _solver_from(tmp_path, "k32", "PARETO_K=32"))
    wide = _attenuations(47.0)
    finite = np.isfinite(narrow) & np.isfinite(wide)
    assert np.any(finite)
    assert np.max(np.abs(narrow[finite] - wide[finite])) <= 1e-3
