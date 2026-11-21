from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import Direction
from arena_evaluation.processing.metrics.social.interaction_common import (
    Trajectory2D,
    extract_episode_ped_tracks,
    pedestrian_encounter_spans,
    resolve_robot_trajectory,
)
from arena_evaluation.processing.metrics.social.probst_responsibility import (
    ProbstPair,
    ProbstParts,
    ProbstResponsibilityCalculator,
    ProbstResult,
    compute_normalization,
    derive_probst_shares,
    distance_to_closest_encounter,
    evaluate_probst,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

# One pedestrian trajectory: canonical id, name, (N, 2) positions, (N, 2) velocities, collision radius in metres.
_PedSpec = tuple[str, str, np.ndarray, np.ndarray, float | None]
SHARE_KEYS = frozenset({"r_robot", "r_target", "r_time", "e_robot", "e_target", "e_time"})
PED_RADIUS_M = 0.4
METRE = 1_000_000_000

PEDS_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "ped_id": pl.String,
        "ped_name": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "radius_m": pl.Float64,
        "desired_speed": pl.Float64,
    }
)


def _make_pair(
    t: np.ndarray,
    pos_r: np.ndarray,
    vel_r: np.ndarray,
    pos_t: np.ndarray,
    vel_t: np.ndarray,
    r_robot: float = 0.5,
    r_target: float = 0.5,
) -> ProbstPair:
    """Construct a ProbstPair from planar position and velocity arrays."""
    t_ns = (t * 1e9).astype(np.int64)
    robot = Trajectory2D(t_ns, pos_r[:, 0], pos_r[:, 1], vel_r[:, 0], vel_r[:, 1], r_robot)
    target = Trajectory2D(t_ns, pos_t[:, 0], pos_t[:, 1], vel_t[:, 0], vel_t[:, 1], r_target)
    return ProbstPair(robot=robot, target=target)


def _make_zero_pair(times_s: np.ndarray, *, window_size_s: float = 12.0) -> ProbstPair:
    """ProbstPair at the origin with the given time grid, for normalization boundary tests."""
    t_ns = np.rint(times_s * 1e9).astype(np.int64)
    zeros = np.zeros(len(times_s), dtype=np.float64)
    trajectory = Trajectory2D(t_ns, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), 0.5)
    return ProbstPair(robot=trajectory, target=trajectory, window_size_s=window_size_s)


def _pedestrian_tracks(t_ns: np.ndarray, ped_tracks: Sequence[_PedSpec]) -> pl.DataFrame:
    """Flat pedestrian_tracks table, one row per pedestrian per frame, built from _PedSpec trajectories."""
    rows: list[dict[str, object]] = []
    for index, time_ns in enumerate(t_ns):
        for ped_id, ped_name, xy, vel, radius_m in ped_tracks:
            rows.append(
                {
                    "t_ns": int(time_ns),
                    "ped_id": ped_id,
                    "ped_name": ped_name,
                    "x": float(xy[index, 0]),
                    "y": float(xy[index, 1]),
                    "vx": float(vel[index, 0]),
                    "vy": float(vel[index, 1]),
                    "radius_m": None if radius_m is None else float(radius_m),
                    "desired_speed": None,
                }
            )
    return pl.DataFrame(rows, schema=PEDS_SCHEMA)


def _make_episode(
    times_s: np.ndarray,
    robot_xy: np.ndarray,
    robot_vel: np.ndarray,
    *,
    ped_tracks: Sequence[_PedSpec] | None = None,
) -> AlignedEpisodeBundle:
    """Build a flat AlignedEpisodeBundle from robot and pedestrian trajectories."""
    t_ns = (times_s * 1e9).astype(np.int64)
    num_samples = len(times_s)
    robot_tracks = pl.DataFrame(
        {
            "t_ns": t_ns,
            "x": robot_xy[:, 0],
            "y": robot_xy[:, 1],
            "yaw": np.zeros(num_samples, dtype=np.float64),
            "vx": robot_vel[:, 0],
            "vy": robot_vel[:, 1],
        }
    )
    topics: dict[str, pl.DataFrame] = {"robot_tracks": robot_tracks}
    if ped_tracks is not None:
        topics["pedestrian_arena_peds"] = _pedestrian_tracks(t_ns, ped_tracks)
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[float(robot_xy[0, 0]), float(robot_xy[0, 1]), 0.0],
        goal_pos=[float(robot_xy[-1, 0]), float(robot_xy[-1, 1]), 0.0],
        topics=topics,
    )


def _straight_robot(times_s: np.ndarray, speed_mps: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Straight +x robot motion at constant speed."""
    robot_xy = np.column_stack((speed_mps * times_s, np.zeros_like(times_s)))
    robot_vel = np.column_stack((np.full_like(times_s, speed_mps), np.zeros_like(times_s)))
    return robot_xy, robot_vel


def _stationary_pedestrian(x: float, y: float, num_samples: int, radius_m: float = PED_RADIUS_M) -> tuple[np.ndarray, np.ndarray]:
    """Stationary pedestrian positions and velocities."""
    positions = np.tile(np.array([x, y], dtype=np.float64), (num_samples, 1))
    return positions, np.zeros((num_samples, 2), dtype=np.float64)


@pytest.mark.parametrize(
    ("robot_pos", "robot_vel", "target_pos", "target_vel", "expected_m"),
    [
        ([0.0, 0.0], [1.0, 0.0], [10.0, 0.4], [-1.0, 0.0], 0.4),  # head-on
        ([0.0, 0.0], [-1.0, 0.0], [3.0, 4.0], [1.0, 0.0], 5.0),  # diverging
        ([0.0, 0.0], [1.0, 0.0], [10.0, 0.4], [0.0, 0.0], 0.4),  # static target
        ([0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [1.0, 0.0], 5.0),  # zero relative velocity
    ],
    ids=["head-on", "diverging", "static-target", "zero-relative-velocity"],
)
def test_distance_to_closest_encounter(
    robot_pos: Sequence[float],
    robot_vel: Sequence[float],
    target_pos: Sequence[float],
    target_vel: Sequence[float],
    expected_m: float,
) -> None:
    distance_m = distance_to_closest_encounter(
        (
            np.asarray(robot_pos, dtype=np.float64),
            np.asarray(robot_vel, dtype=np.float64),
        ),
        (
            np.asarray(target_pos, dtype=np.float64),
            np.asarray(target_vel, dtype=np.float64),
        ),
    )
    assert distance_m == pytest.approx(expected_m, abs=1e-9)


def _sum_of(outputs: dict[str, object], keys: Sequence[str]) -> float:
    """Sum of the published shares, after asserting each is defined."""
    values = [outputs[key] for key in keys]
    assert all(value is not None for value in values)
    return float(sum(value for value in values if value is not None))


def test_probst_calculator_encounter_and_pre_contact_scopes() -> None:
    dt = 0.1
    t = np.arange(0.0, 5.0, dt)
    vy_r = np.where((t >= 1.0) & (t <= 2.5), 0.4, 0.0)
    robot_xy = np.column_stack((1.0 * t, np.cumsum(vy_r * dt)))
    robot_vel = np.column_stack((np.ones_like(t), vy_r))
    ped_xy = np.column_stack((6.0 - 1.0 * t, np.zeros_like(t)))
    ped_vel = np.column_stack((-np.ones_like(t), np.zeros_like(t)))

    episode = _make_episode(
        t,
        robot_xy,
        robot_vel,
        ped_tracks=[("Pedestrian_1", "ped_1", ped_xy, ped_vel, PED_RADIUS_M)],
    )
    calculator = ProbstResponsibilityCalculator(make_robot_params(robot_radius=PED_RADIUS_M))

    out_enc = calculator.calculate(episode, {})
    assert set(out_enc) == set(calculator.output_keys())
    assert _sum_of(out_enc, ("probst_R_robot", "probst_R_actor2", "probst_R_time")) == pytest.approx(1.0, abs=1e-9)
    assert _sum_of(out_enc, ("probst_E_robot", "probst_E_actor2", "probst_E_time")) == pytest.approx(1.0, abs=1e-9)

    robot, _ = resolve_robot_trajectory(calculator, episode)
    tracks = extract_episode_ped_tracks(episode)
    spans = pedestrian_encounter_spans(episode, PED_RADIUS_M)
    assert len(spans) == 1
    window_result, _ = calculator.evaluate_encounter_track(robot, tracks["Pedestrian_1"], (spans[0][1], spans[0][2]))
    assert out_enc["probst_R_robot"] == pytest.approx(window_result.r_robot, abs=1e-12)

    out_pre = calculator.calculate_pre_contact(episode, int(3.0 * 1e9), "Pedestrian_1")
    assert set(out_pre) == set(calculator.output_keys())
    assert _sum_of(out_pre, ("probst_R_robot", "probst_R_actor2", "probst_R_time")) == pytest.approx(1.0, abs=1e-9)


def test_compute_normalization_non_uniform_grid() -> None:
    times_s = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0])
    pair = _make_zero_pair(times_s, window_size_s=1.0)
    norm = compute_normalization(pair, len(times_s) - 1)
    assert norm is not None
    np.testing.assert_allclose(norm, [0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    "parts",
    [
        None,
        ProbstParts(
            c_total=0.0,
            cc_minus_robot=0.0,
            cc_minus_target=0.0,
            cc_minus_time=0.0,
            cc_plus_robot=0.0,
            cc_plus_target=0.0,
            cc_plus_time=0.0,
        ),
    ],
    ids=["no-parts", "zero-total"],
)
def test_derive_probst_shares_returns_none_without_total(
    parts: ProbstParts | None,
) -> None:
    shares = derive_probst_shares(parts)
    assert set(shares) == SHARE_KEYS
    assert all(value is None for value in shares.values())


@pytest.mark.parametrize("num_samples", [0, 1])
def test_evaluate_probst_requires_two_samples(num_samples: int) -> None:
    t = np.arange(num_samples, dtype=np.float64) * 0.1
    empty = np.zeros((num_samples, 2), dtype=np.float64)
    assert evaluate_probst(_make_pair(t, empty, empty, empty, empty)) == ProbstResult()


def test_calculate_with_empty_pedestrian_tracks_returns_none_outputs() -> None:
    t = np.arange(0.0, 3.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    episode = _make_episode(t, robot_xy, robot_vel, ped_tracks=[])
    calculator = ProbstResponsibilityCalculator(make_robot_params(robot_radius=PED_RADIUS_M))

    out = calculator.calculate(episode, {})
    assert set(out) == set(calculator.output_keys())
    assert all(value is None for value in out.values())


@pytest.mark.parametrize("target_agent_id", [1, "1", "Pedestrian_1", "ped_1", "alpha"])
def test_calculate_resolves_target_identifiers(target_agent_id: int | str) -> None:
    t = np.arange(0.0, 5.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    ped_xy, ped_vel = _stationary_pedestrian(2.5, 0.1, len(t))
    episode = _make_episode(
        t,
        robot_xy,
        robot_vel,
        ped_tracks=[("Pedestrian_1", "alpha", ped_xy, ped_vel, PED_RADIUS_M)],
    )
    calculator = ProbstResponsibilityCalculator(make_robot_params(robot_radius=PED_RADIUS_M))
    calculator.metrics_config = {"target_agent_id": target_agent_id}

    out = calculator.calculate(episode, {})
    # Both bodies keep their velocities, so no velocity change resolves the conflict and time carries all of it.
    assert out["probst_R_robot"] == pytest.approx(0.0, abs=1e-9)
    assert out["probst_R_time"] == pytest.approx(1.0, abs=1e-9)
    assert _sum_of(out, ("probst_R_robot", "probst_R_actor2", "probst_R_time")) == pytest.approx(1.0, abs=1e-9)


def test_calculate_pre_contact_unknown_pedestrian_returns_none_outputs() -> None:
    t = np.arange(0.0, 5.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    ped_xy, ped_vel = _stationary_pedestrian(2.5, 0.1, len(t))
    episode = _make_episode(
        t,
        robot_xy,
        robot_vel,
        ped_tracks=[("Pedestrian_1", "alpha", ped_xy, ped_vel, PED_RADIUS_M)],
    )
    calculator = ProbstResponsibilityCalculator(make_robot_params(robot_radius=PED_RADIUS_M))

    out = calculator.calculate_pre_contact(episode, int(3.0 * 1e9), "Pedestrian_99")
    assert set(out) == set(calculator.output_keys())
    assert all(value is None for value in out.values())


def test_probst_requires_recorded_pedestrian_radius() -> None:
    t = np.arange(0.0, 5.0, 0.1)
    robot_xy, robot_vel = _straight_robot(t)
    ped_xy, ped_vel = _stationary_pedestrian(2.5, 0.1, len(t))
    episode = _make_episode(
        t,
        robot_xy,
        robot_vel,
        ped_tracks=[("Pedestrian_1", "alpha", ped_xy, ped_vel, None)],
    )
    calculator = ProbstResponsibilityCalculator(make_robot_params(robot_radius=PED_RADIUS_M))
    calculator.metrics_config = {"target_agent_id": "Pedestrian_1"}

    with pytest.raises(MetricInputError, match="radius_m"):
        calculator.calculate(episode, {})


def test_probst_directions_do_not_judge_responsibility() -> None:
    assert "probst_R_robot" not in ProbstResponsibilityCalculator.DIRECTIONS
    assert "probst_R_actor2" not in ProbstResponsibilityCalculator.DIRECTIONS
    assert "probst_R_time" not in ProbstResponsibilityCalculator.DIRECTIONS
    assert ProbstResponsibilityCalculator.DIRECTIONS["probst_E_robot"] is Direction.LOWER
    assert ProbstResponsibilityCalculator.DIRECTIONS["probst_E_actor2"] is Direction.LOWER
