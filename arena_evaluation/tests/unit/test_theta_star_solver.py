import math
import pathlib
import subprocess

import numpy as np
import pytest

from arena_evaluation.processing.path import theta_star
from arena_evaluation.processing.path.theta_star import (
    StaticGrid,
    _line_of_sight,
    plan_route,
)
from arena_evaluation.storage.exceptions import MetricInputError

PACKAGE_ROOT = pathlib.Path(theta_star.__file__).resolve().parents[3]
SOLVER_SOURCE = PACKAGE_ROOT / "arena_evaluation/processing/path/solver.cpp"
ENV_SO_VAR = "ARENA_THETA_STAR_SO"


def _compile(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """Compile a stand-in solver library from `body`."""
    source = tmp_path / f"{name}.cpp"
    source.write_text(body, encoding="utf-8")
    library = tmp_path / f"{name}.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-fPIC",
            "-shared",
            "-std=c++17",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


def _use_library(monkeypatch: pytest.MonkeyPatch, path: pathlib.Path) -> None:
    """Point the loader at `path` and forget the previously loaded library."""
    monkeypatch.setenv(ENV_SO_VAR, str(path))
    theta_star._solver_library.cache_clear()


def _open(
    size: int = 40, resolution: float = 0.1, origin: tuple[float, float] = (0.0, 0.0)
) -> StaticGrid:
    return StaticGrid(
        origin_xy=origin,
        resolution_m=resolution,
        lethal=np.zeros((size, size), dtype=bool),
    )


def test_line_of_sight_leaving_the_grid_is_blocked():
    grid = np.zeros((5, 5), dtype=bool)

    assert _line_of_sight(grid, 0, 0, 7, 0) is False


@pytest.mark.parametrize("blocked", [(0, 1), (1, 0)])
def test_line_of_sight_diagonal_step_refuses_to_cut_an_occupied_corner(
    blocked: tuple[int, int],
):
    grid = np.zeros((3, 3), dtype=bool)
    grid[blocked] = True

    assert _line_of_sight(grid, 0, 0, 1, 1) is False


@pytest.mark.parametrize(
    ("start", "end"),
    [((0, 0), (2, 2)), ((2, 2), (0, 0)), ((0, 0), (1, 3)), ((3, 1), (0, 0))],
)
def test_line_of_sight_through_free_cells_in_every_direction(
    start: tuple[int, int], end: tuple[int, int]
):
    grid = np.zeros((4, 4), dtype=bool)

    assert _line_of_sight(grid, *start, *end) is True


def test_line_of_sight_stops_at_an_obstacle_on_the_line():
    grid = np.zeros((5, 5), dtype=bool)
    grid[0, 2] = True

    assert _line_of_sight(grid, 0, 0, 4, 0) is False


def test_cell_of_floors_each_point_into_its_cell_and_rejects_points_outside():
    grid = _open(size=10, resolution=0.5, origin=(-1.0, 2.0))

    assert grid.cell_of(-1.0, 2.0) == (0, 0)
    assert grid.cell_of(0.2, 3.3) == (2, 2)
    assert grid.cell_of(4.0, 2.0) is None
    assert grid.cell_of(float("inf"), 2.0) is None


def test_cell_centre_lies_half_a_cell_into_its_cell():
    grid = _open(size=10, resolution=0.5, origin=(-1.0, 2.0))

    assert grid.cell_centre(2, 2) == pytest.approx((0.25, 3.25))


def test_open_grid_route_is_the_exact_straight_segment_with_its_hypot_length():
    grid = _open()
    start = (0.27502482987015087, 0.46768277259018687)
    goal = (3.3, 3.1)

    res = plan_route(grid, start, goal)

    assert res.null_reason is None
    assert res.points.tolist() == [list(start), list(goal)]
    assert res.length_m == math.hypot(goal[0] - start[0], goal[1] - start[1])


def test_route_around_a_wall_is_longer_than_the_line_and_keeps_exact_endpoints():
    grid = _open()
    grid.lethal[0:30, 20] = True

    res = plan_route(grid, (1.03, 1.0), (3.07, 1.0))

    assert res.null_reason is None
    assert res.points[0].tolist() == [1.03, 1.0]
    assert res.points[-1].tolist() == [3.07, 1.0]
    assert len(res.points) >= 3
    assert res.points[:, 1].max() >= 3.0
    assert res.length_m > 2 * math.hypot(1.0, 2.0) - 0.2


def test_goal_walled_in_has_no_route_and_no_straight_line():
    grid = _open(size=30)
    grid.lethal[16:25, 16] = True
    grid.lethal[16:25, 24] = True
    grid.lethal[16, 16:25] = True
    grid.lethal[24, 16:25] = True

    res = plan_route(grid, (0.5, 0.5), (2.0, 2.0))

    assert res.points is None
    assert res.length_m is None
    assert res.null_reason == "no_path"


@pytest.mark.parametrize(
    ("start", "goal", "reason"),
    [
        ((-0.5, 1.0), (3.0, 3.0), "start_outside_grid"),
        ((1.0, 1.0), (9.0, 1.0), "goal_outside_grid"),
        ((float("nan"), 1.0), (3.0, 3.0), "endpoint_not_finite"),
    ],
)
def test_endpoint_that_cannot_be_planned_from_gives_its_reason(
    start: tuple[float, float], goal: tuple[float, float], reason: str
):
    res = plan_route(_open(), start, goal)

    assert res.null_reason == reason
    assert res.points is None


def test_endpoint_inside_a_lethal_cell_is_refused_not_routed_around():
    grid = _open()
    grid.lethal[10:14, 10:14] = True

    assert (
        plan_route(grid, (1.05, 1.05), (3.0, 3.0)).null_reason == "start_in_lethal_cell"
    )
    assert (
        plan_route(grid, (3.0, 3.0), (1.05, 1.05)).null_reason == "goal_in_lethal_cell"
    )


def test_missing_search_library_raises_metric_input_error_with_the_path_and_the_build(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    missing_library = tmp_path / "solver.so"
    monkeypatch.setenv(ENV_SO_VAR, str(missing_library))
    grid = _open()
    grid.lethal[0:30, 20] = True
    theta_star._solver_library.cache_clear()
    try:
        with pytest.raises(MetricInputError) as raised:
            plan_route(grid, (1.0, 1.0), (3.0, 1.0))
    finally:
        theta_star._solver_library.cache_clear()
    assert str(missing_library) in str(raised.value)
    assert ENV_SO_VAR in str(raised.value)
    assert "arena build" in str(raised.value)
    assert "cmake" in str(raised.value)
    assert not missing_library.exists()


def test_solver_path_honours_the_environment_variable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    configured_library = tmp_path / "theta_star_solver.so"
    monkeypatch.setenv(ENV_SO_VAR, str(configured_library))

    assert theta_star._solver_path() == configured_library

    monkeypatch.delenv(ENV_SO_VAR)
    assert theta_star._solver_path() == theta_star._SOLVER_LIBRARY


def test_the_default_library_sits_next_to_the_module():
    """Without an override the loader looks where the CMake build writes."""
    assert theta_star._SOLVER_LIBRARY == (
        pathlib.Path(theta_star.__file__).resolve().parent / "solver.so"
    )


def test_the_loaded_library_exports_the_wrappers_abi():
    """The library the suite loads is the one this wrapper binds."""
    assert (
        theta_star.solver_abi(theta_star._solver_library())
        == theta_star.THETA_STAR_SOLVER_ABI
    )


def test_the_solver_source_declares_the_wrappers_abi():
    """The C++ ABI constant and the Python one cannot drift apart."""
    source = SOLVER_SOURCE.read_text(encoding="utf-8")
    assert (
        f"#define ARENA_THETA_STAR_SOLVER_ABI {theta_star.THETA_STAR_SOLVER_ABI}"
        in source
    )
    assert theta_star.THETA_STAR_SOLVER_ABI_SYMBOL in source


def test_a_library_without_the_abi_symbol_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    """A library from before the ABI (the old import-time g++ build) raises instead of binding."""
    stale = _compile(
        tmp_path,
        "stale",
        'extern "C" void solve_geometric_theta_star() {}\n',
    )
    _use_library(monkeypatch, stale)
    try:
        with pytest.raises(
            MetricInputError, match=theta_star.THETA_STAR_SOLVER_ABI_SYMBOL
        ):
            theta_star._solver_library()
    finally:
        theta_star._solver_library.cache_clear()


def test_a_library_with_another_abi_version_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    """A library built from a different search does not plan under this ABI."""
    foreign = _compile(
        tmp_path,
        "foreign",
        'extern "C" int arena_theta_star_solver_abi(void) { return 999; }\n'
        'extern "C" void solve_geometric_theta_star() {}\n',
    )
    _use_library(monkeypatch, foreign)
    try:
        with pytest.raises(MetricInputError, match="ABI 999"):
            theta_star._solver_library()
    finally:
        theta_star._solver_library.cache_clear()


def test_a_library_missing_the_search_symbol_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    """An ABI match without the search symbol is still not loadable."""
    partial = _compile(
        tmp_path,
        "partial",
        'extern "C" int arena_theta_star_solver_abi(void) { return 1; }\n',
    )
    _use_library(monkeypatch, partial)
    try:
        with pytest.raises(MetricInputError, match=theta_star.THETA_STAR_SOLVER_SYMBOL):
            theta_star._solver_library()
    finally:
        theta_star._solver_library.cache_clear()


def test_a_library_that_cannot_be_loaded_names_the_build_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    """A file that is not a shared library raises the hinted error instead of a bare OSError."""
    broken = tmp_path / "broken.so"
    broken.write_text("not a shared library\n", encoding="utf-8")
    _use_library(monkeypatch, broken)
    try:
        with pytest.raises(MetricInputError) as raised:
            theta_star._solver_library()
    finally:
        theta_star._solver_library.cache_clear()
    assert "arena build" in str(raised.value)
    assert ENV_SO_VAR in str(raised.value)


def test_search_that_exhausts_its_iteration_budget_reports_the_cap_not_no_path():
    lethal = np.zeros((40, 40), dtype=bool)
    lethal[:-1, 20] = True
    assert (
        theta_star._search(lethal, 0.1, (5, 5), (35, 5), max_iterations=1)
        == "iteration_cap"
    )
    assert (
        theta_star._search(lethal, 0.1, (5, 5), (35, 5), max_iterations=100000)
        != "iteration_cap"
    )


def test_search_reports_no_path_when_a_full_wall_separates_the_endpoints():
    lethal = np.zeros((40, 40), dtype=bool)
    lethal[:, 20] = True
    assert (
        theta_star._search(lethal, 0.1, (5, 5), (35, 5), max_iterations=100000)
        == "no_path"
    )
