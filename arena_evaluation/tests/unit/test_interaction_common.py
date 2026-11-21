"""Unit contracts for the shared flat-telemetry trajectory, encounter and visibility helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from arena_evaluation.processing.metrics.safety.contact_attribution import (
    ContactAttributionCalculator,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_PED_RADIUS_M,
    Trajectory2D,
    episode_encounter_windows,
    extract_episode_ped_tracks,
    pedestrian_encounter_spans,
)
from arena_evaluation.processing.windows import empty_encounter_windows
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

NS_PER_S = 1_000_000_000
ROBOT_RADIUS_M = 0.25

_PED_SCHEMA = pl.Schema(
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
_ROBOT_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
    }
)


def _episode(
    topics: dict[str, pl.DataFrame],
    fleet: dict[str, pl.DataFrame] | None = None,
) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0, 0.0],
        topics=topics,
        fleet=fleet or {},
    )


def _calculator() -> ContactAttributionCalculator:
    return ContactAttributionCalculator(make_robot_params(robot_radius=ROBOT_RADIUS_M))


def _robot_frame(
    t_s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    yaw: np.ndarray | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": (np.asarray(t_s, dtype=np.float64) * NS_PER_S).astype(np.int64),
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "yaw": np.zeros(len(t_s), dtype=np.float64)
            if yaw is None
            else np.asarray(yaw, dtype=np.float64),
            "vx": np.asarray(vx, dtype=np.float64),
            "vy": np.asarray(vy, dtype=np.float64),
        }
    )


def _ped_row(
    t_ns: int,
    ped_id: str,
    x: float,
    y: float,
    *,
    ped_name: str = "",
    vx: float = 0.0,
    vy: float = 0.0,
    radius_m: float | None = None,
    desired_speed: float | None = None,
) -> dict[str, object]:
    return {
        "t_ns": t_ns,
        "ped_id": ped_id,
        "ped_name": ped_name,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "radius_m": radius_m,
        "desired_speed": desired_speed,
    }


def _trajectory(
    x: list[float],
    t_s: list[float],
    *,
    vx: list[float] | None = None,
    radius_m: float = DEFAULT_PED_RADIUS_M,
) -> Trajectory2D:
    t_ns = (np.asarray(t_s, dtype=np.float64) * NS_PER_S).astype(np.int64)
    x_array = np.asarray(x, dtype=np.float64)
    zeros = np.zeros(len(x_array), dtype=np.float64)
    vx_array = zeros.copy() if vx is None else np.asarray(vx, dtype=np.float64)
    return Trajectory2D(
        t_ns=t_ns,
        x=x_array,
        y=zeros.copy(),
        vx=vx_array,
        vy=zeros.copy(),
        radius_m=radius_m,
    )


def test_extract_ped_tracks_derives_velocity_from_positions_but_leaves_the_first_sample_unmeasured() -> (
    None
):
    frame = pl.DataFrame(
        [
            _ped_row(0, "Pedestrian_3", 0.0, 0.0, vx=0.0, vy=0.0),
            _ped_row(NS_PER_S, "Pedestrian_3", 1.0, 0.0, vx=0.0, vy=0.0),
            _ped_row(2 * NS_PER_S, "Pedestrian_3", 2.0, 0.0, vx=0.0, vy=0.0),
        ],
        schema=_PED_SCHEMA,
    )

    track = extract_episode_ped_tracks(_episode({"pedestrian_arena_peds": frame}))[
        "Pedestrian_3"
    ]

    # The difference needs two poses, so the first sample is unmeasured (NaN), never a copy of the second.
    assert np.isnan(track.vx[0]) and np.isnan(track.vy[0])
    np.testing.assert_allclose(track.vx[1:], [1.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(track.vy[1:], [0.0, 0.0], atol=1e-9)


def _encounter_episode(
    *, with_pedestrians: bool = True, fleet: dict[str, pl.DataFrame] | None = None
) -> AlignedEpisodeBundle:
    """Robot at x = t and one pedestrian 2 m ahead on the robot's line."""
    robot = _robot_frame(
        t_s=np.arange(5.0),
        x=np.arange(5.0),
        y=np.zeros(5),
        vx=np.ones(5),
        vy=np.zeros(5),
    )
    topics = {"robot_tracks": robot}
    if with_pedestrians:
        topics["pedestrian_arena_peds"] = pl.DataFrame(
            [
                _ped_row(
                    t * NS_PER_S,
                    "Pedestrian_1",
                    2.0,
                    0.0,
                    ped_name="alice",
                    radius_m=0.3,
                )
                for t in range(5)
            ],
            schema=_PED_SCHEMA,
        )
    return _episode(topics, fleet=fleet)


def _fleet_poses(radius_m: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": np.arange(5) * NS_PER_S,
            "pos_x": np.full(5, 0.5, dtype=np.float64),
            "pos_y": np.zeros(5, dtype=np.float64),
            "yaw": np.zeros(5, dtype=np.float64),
            "radius_m": np.full(5, radius_m, dtype=np.float64),
        },
        schema={
            "t_ns": pl.Int64,
            "pos_x": pl.Float64,
            "pos_y": pl.Float64,
            "yaw": pl.Float64,
            "radius_m": pl.Float64,
        },
    )


def test_episode_encounters_exclude_robot_actors_from_pedestrian_metrics() -> None:
    episode = _encounter_episode(
        with_pedestrians=False, fleet={"env_0_robot_2": _fleet_poses(0.25)}
    )

    fact_windows = episode_encounter_windows(episode, ROBOT_RADIUS_M)

    assert fact_windows.height == 1
    assert_series_equal(
        fact_windows["actor_kind"], pl.Series("actor_kind", ["robot"], dtype=pl.String)
    )
    assert pedestrian_encounter_spans(episode, ROBOT_RADIUS_M) == []


def test_episode_encounter_windows_without_tracks_is_empty() -> None:
    windows = episode_encounter_windows(_episode({}), ROBOT_RADIUS_M)

    assert windows.height == 0
    assert windows.schema == empty_encounter_windows().schema
