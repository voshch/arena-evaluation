from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import yaml

from arena_evaluation.processing.metrics.social.encounter_costs import (
    EncounterCostsCalculator,
    max_orthogonal_deviation_m,
    mutual_accommodation_ratio,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
ROBOT_RADIUS_M = 0.3
PED_RADIUS_M = 0.3
NULL_REASON_SUFFIX = "_null_reason"
ENCOUNTER_OUTPUTS = (
    "detour_time_s",
    "ped_path_excess_m",
    "robot_path_excess_m",
    "robot_path_deviation_vs_planned_m",
    "mar_vs_planned_route",
    "pfi_vs_planned_route",
    "kcsc_wh_vs_planned_route",
)


def _write_world(run_dir: pathlib.Path, polygon: list[list[float]]) -> pathlib.Path:
    world_dir = run_dir / "worlds" / "test_map"
    (world_dir / "assets").mkdir(parents=True)
    world_spec = {
        "format": "arena-world/2",
        "levels": [{"id": "0", "zones": [{"name": "open", "polygon": polygon}]}],
    }
    (world_dir / "world.yaml").write_text(yaml.safe_dump(world_spec))
    return run_dir


@pytest.fixture
def open_run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_world(
        tmp_path, [[-5.0, -5.0], [15.0, -5.0], [15.0, 15.0], [-5.0, 15.0]]
    )


def _times_and_xy(
    samples: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(samples, dtype=np.float64)
    return array[:, 0], array[:, 1], array[:, 2]


def _robot_table(samples: list[tuple[float, float, float]]) -> pl.DataFrame:
    t, x, y = _times_and_xy(samples)
    return pl.DataFrame(
        {
            "t_ns": (t * SEC).astype(np.int64),
            "x": x,
            "y": y,
            "yaw": np.zeros_like(t),
            "vx": np.gradient(x, t),
            "vy": np.gradient(y, t),
        }
    )


def _ped_table(
    samples: list[tuple[float, float, float]], radius_m: float | None = PED_RADIUS_M
) -> pl.DataFrame:
    t, x, y = _times_and_xy(samples)
    count = len(t)
    return pl.DataFrame(
        {
            "t_ns": (t * SEC).astype(np.int64),
            "ped_id": ["p1"] * count,
            "ped_name": ["alice"] * count,
            "x": x,
            "y": y,
            "vx": np.gradient(x, t),
            "vy": np.gradient(y, t),
            "radius_m": pl.Series([radius_m] * count, dtype=pl.Float64),
            "desired_speed": [1.0] * count,
        }
    )


def _power_table(
    samples: list[tuple[float, float, float]], watts: float
) -> pl.DataFrame:
    t, _, _ = _times_and_xy(samples)
    return pl.DataFrame(
        {"t_ns": (t * SEC).astype(np.int64), "total_power_w": [watts] * len(t)}
    )


def _episode(
    robot: pl.DataFrame,
    ped: pl.DataFrame,
    power: pl.DataFrame | None = None,
    *,
    run_dir: pathlib.Path | None = None,
) -> AlignedEpisodeBundle:
    topics: dict[str, pl.DataFrame] = {"robot_tracks": robot, "pedestrian_tracks": ped}
    if power is not None:
        topics["robot_power"] = power
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.0, 0.0, 0.0],
        topics=topics,
        map="test_map",
        run_dir=run_dir,
    )


def _window(
    start_s: float, end_s: float, actor_kind: str = "pedestrian"
) -> SimpleNamespace:
    return SimpleNamespace(
        scope=MetricScope.ENCOUNTER,
        identity={"actor_2": "p1", "actor_2_kind": actor_kind},
        t_start_ns=int(start_s * SEC),
        t_end_ns=int(end_s * SEC),
    )


def _calculator() -> EncounterCostsCalculator:
    calculator = EncounterCostsCalculator(
        RobotParams(model="jackal", robot_radius=ROBOT_RADIUS_M)
    )
    calculator.metrics_config = {}
    return calculator


def _straight_pair(
    end_s: float = 4.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    times = np.arange(0.0, end_s + 1e-9, 0.5)
    robot = [(float(t), float(t) - 2.0, 0.0) for t in times]
    pedestrian = [(float(t), float(t) - 2.0, 4.0) for t in times]
    return robot, pedestrian


def test_agents_on_their_planned_routes_have_no_excess(
    open_run_dir: pathlib.Path,
) -> None:
    robot_samples, ped_samples = _straight_pair()
    times = [t for t, _, _ in robot_samples]
    power = _power_table([(t, 0.0, 0.0) for t in times], watts=5.0)
    episode = _episode(
        _robot_table(robot_samples),
        _ped_table(ped_samples),
        power,
        run_dir=open_run_dir,
    )

    outputs = _calculator().calculate_for_scope(episode, {}, _window(1.0, 3.0))

    assert outputs["robot_path_excess_m"] == pytest.approx(0.0, abs=1e-9)
    assert outputs["ped_path_excess_m"] == pytest.approx(0.0, abs=1e-9)
    assert outputs["robot_path_deviation_vs_planned_m"] == pytest.approx(0.0, abs=1e-9)
    assert outputs["detour_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert outputs["pfi_vs_planned_route"] == pytest.approx(0.0, abs=1e-9)
    assert outputs["kcsc_wh_vs_planned_route"] == pytest.approx(0.0, abs=1e-9)
    assert outputs["mar_vs_planned_route"] is None
    assert outputs[f"mar_vs_planned_route{NULL_REASON_SUFFIX}"] == "zero_denominator"


def test_robot_detour_gives_its_excess_deviation_and_mar_one(
    open_run_dir: pathlib.Path,
) -> None:
    robot_samples = [
        (0.0, -2.0, 0.0),
        (0.5, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.5, 0.5, 0.5),
        (2.0, 1.0, 1.0),
        (2.5, 1.5, 0.5),
        (3.0, 2.0, 0.0),
        (3.5, 3.0, 0.0),
        (4.0, 4.0, 0.0),
    ]
    ped_samples = [(t, t - 2.0, 4.0) for t, _, _ in robot_samples]
    episode = _episode(
        _robot_table(robot_samples), _ped_table(ped_samples), run_dir=open_run_dir
    )

    outputs = _calculator().calculate_for_scope(episode, {}, _window(1.0, 3.0))

    assert outputs["robot_path_excess_m"] == pytest.approx(
        2.0 * np.sqrt(2.0) - 2.0, abs=1e-6
    )
    assert outputs["robot_path_deviation_vs_planned_m"] == pytest.approx(1.0, abs=1e-6)
    assert outputs["ped_path_excess_m"] == pytest.approx(0.0, abs=1e-9)
    assert outputs["mar_vs_planned_route"] == pytest.approx(1.0, abs=1e-6)


def test_endpoint_in_a_lethal_cell_gives_reference_missing_for_the_robot_only(
    tmp_path: pathlib.Path,
) -> None:
    run_dir = _write_world(
        tmp_path,
        [
            [-5.0, -5.0],
            [-1.0, -5.0],
            [-1.0, 2.0],
            [15.0, 2.0],
            [15.0, 15.0],
            [-5.0, 15.0],
        ],
    )
    robot_samples, ped_samples = _straight_pair()
    episode = _episode(
        _robot_table(robot_samples), _ped_table(ped_samples), run_dir=run_dir
    )

    outputs = _calculator().calculate_for_scope(episode, {}, _window(1.0, 3.0))

    assert outputs["robot_path_excess_m"] is None
    assert outputs[f"robot_path_excess_m{NULL_REASON_SUFFIX}"] == "reference_missing"
    assert outputs["ped_path_excess_m"] == pytest.approx(0.0, abs=1e-9)


def test_pedestrian_without_a_radius_has_no_planned_route(
    open_run_dir: pathlib.Path,
) -> None:
    robot_samples, ped_samples = _straight_pair()
    episode = _episode(
        _robot_table(robot_samples),
        _ped_table(ped_samples, radius_m=0.0),
        run_dir=open_run_dir,
    )

    outputs = _calculator().calculate_for_scope(episode, {}, _window(1.0, 3.0))

    assert outputs["ped_path_excess_m"] is None
    assert outputs[f"ped_path_excess_m{NULL_REASON_SUFFIX}"] == "not_recorded"
    assert outputs["detour_time_s"] is None
    assert outputs["mar_vs_planned_route"] is None


def test_an_encounter_with_a_non_pedestrian_is_not_applicable(
    open_run_dir: pathlib.Path,
) -> None:
    robot_samples, ped_samples = _straight_pair()
    episode = _episode(
        _robot_table(robot_samples), _ped_table(ped_samples), run_dir=open_run_dir
    )

    outputs = _calculator().calculate_for_scope(
        episode, {}, _window(1.0, 3.0, actor_kind="robot")
    )

    assert all(outputs[name] is None for name in ENCOUNTER_OUTPUTS)
    assert all(
        outputs[f"{name}{NULL_REASON_SUFFIX}"] == "not_applicable"
        for name in ENCOUNTER_OUTPUTS
    )


def test_the_calculator_publishes_only_encounter_scope_outputs() -> None:
    assert {metric.name for metric in EncounterCostsCalculator.OUTPUTS} == set(
        ENCOUNTER_OUTPUTS
    )
    assert all(
        metric.scopes == (MetricScope.ENCOUNTER,)
        for metric in EncounterCostsCalculator.OUTPUTS
    )


@pytest.mark.parametrize(
    ("robot_excess_m", "pedestrian_excess_m", "expected"),
    [(0.828, 0.0, 1.0), (1.0, 1.0, 0.5), (-0.5, 2.0, 0.0), (0.0, 0.0, None)],
)
def test_mutual_accommodation_ratio_floors_each_excess_at_zero(
    robot_excess_m: float, pedestrian_excess_m: float, expected: float | None
) -> None:
    ratio = mutual_accommodation_ratio(robot_excess_m, pedestrian_excess_m)

    if expected is None:
        assert ratio is None
    else:
        assert ratio == pytest.approx(expected, abs=1e-9)


def test_orthogonal_deviation_is_perpendicular_inside_a_segment_and_clipped_at_its_ends() -> (
    None
):
    polyline = np.array([[0.0, 0.0], [2.0, 0.0]])

    assert max_orthogonal_deviation_m(
        np.array([[1.0, 1.0]]), polyline
    ) == pytest.approx(1.0)
    assert max_orthogonal_deviation_m(
        np.array([[3.0, 0.0]]), polyline
    ) == pytest.approx(1.0)
