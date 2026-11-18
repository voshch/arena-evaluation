"""Unit contracts for the shared flat-telemetry trajectory, encounter and visibility helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from arena_evaluation.processing.metrics.safety.contact_attribution import (
    ContactAttributionCalculator,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    DEFAULT_PED_RADIUS_M,
    Trajectory2D,
    _identity_aliases,
    contiguous_spans,
    episode_encounter_windows,
    extract_episode_ped_tracks,
    extract_ped_tracks,
    merge_intervals_ns,
    pedestrian_encounter_spans,
    resolve_ped_track,
    resolve_robot_trajectory,
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


def test_trajectory2d_exposes_xy_vel_and_seconds() -> None:
    trajectory = Trajectory2D(
        t_ns=np.array([0, NS_PER_S], dtype=np.int64),
        x=np.array([0.0, 1.0]),
        y=np.array([2.0, 3.0]),
        vx=np.array([0.5, 0.5]),
        vy=np.array([-1.0, -2.0]),
        radius_m=0.4,
    )

    np.testing.assert_allclose(trajectory.xy, [[0.0, 2.0], [1.0, 3.0]])
    np.testing.assert_allclose(trajectory.vel, [[0.5, -1.0], [0.5, -2.0]])
    np.testing.assert_allclose(trajectory.times_s, [0.0, 1.0])
    assert trajectory.radius_m == 0.4


@pytest.mark.parametrize(
    ("window_s", "expected_x"),
    [
        ((0.0, 2.0), [0.0, 1.0, 2.0]),
        ((1.0, 1.0), [1.0]),
        ((0.4, 1.6), [1.0]),
        ((10.0, 20.0), [2.0]),
    ],
)
def test_trajectory2d_slice_ns_selects_the_window_or_the_nearest_sample(
    window_s: tuple[float, float], expected_x: list[float]
) -> None:
    trajectory = _trajectory([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    sliced = trajectory.slice_ns(
        int(window_s[0] * NS_PER_S), int(window_s[1] * NS_PER_S)
    )

    np.testing.assert_allclose(sliced.x, expected_x)


def test_trajectory2d_resample_interpolates_positions_and_velocities() -> None:
    trajectory = Trajectory2D(
        t_ns=np.array([0, 2 * NS_PER_S], dtype=np.int64),
        x=np.array([0.0, 10.0]),
        y=np.array([0.0, -10.0]),
        vx=np.array([1.0, 3.0]),
        vy=np.array([0.0, 2.0]),
    )
    query = np.array([int(0.5 * NS_PER_S), int(1.5 * NS_PER_S)], dtype=np.int64)

    resampled = trajectory.resample_onto(query)

    np.testing.assert_array_equal(resampled.t_ns, query)
    np.testing.assert_allclose(resampled.x, [2.5, 7.5])
    np.testing.assert_allclose(resampled.y, [-2.5, -7.5])
    np.testing.assert_allclose(resampled.vx, [1.5, 2.5])
    np.testing.assert_allclose(resampled.vy, [0.5, 1.5])


def test_trajectory2d_resample_of_empty_trajectory_is_undefined_and_not_the_origin() -> (
    None
):
    """An empty trajectory has no position to resample: NaN, never a fabricated (0, 0)."""
    empty = Trajectory2D(
        t_ns=np.array([], dtype=np.int64),
        x=np.array([]),
        y=np.array([]),
        vx=np.array([]),
        vy=np.array([]),
        radius_m=0.3,
    )
    query = np.array([0, NS_PER_S], dtype=np.int64)

    resampled = empty.resample_onto(query)

    np.testing.assert_array_equal(resampled.t_ns, query)
    assert np.isnan(resampled.x).all()
    assert np.isnan(resampled.y).all()
    assert np.isnan(resampled.vx).all()
    assert resampled.radius_m == 0.3


def test_resolve_robot_trajectory_reads_the_flat_robot_tracks_table() -> None:
    frame = _robot_frame(
        t_s=np.array([0.0, 1.0, 2.0]),
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 0.5, 1.0]),
        vx=np.array([1.0, 1.0, 1.0]),
        vy=np.array([0.5, 0.5, 0.5]),
        yaw=np.array([0.0, 0.1, 0.2]),
    )

    trajectory, yaw = resolve_robot_trajectory(
        _calculator(), _episode({"robot_tracks": frame})
    )

    np.testing.assert_array_equal(trajectory.t_ns, [0, NS_PER_S, 2 * NS_PER_S])
    np.testing.assert_allclose(trajectory.x, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(trajectory.y, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(trajectory.vx, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(trajectory.vy, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(yaw, [0.0, 0.1, 0.2])
    assert trajectory.radius_m == ROBOT_RADIUS_M


@pytest.mark.parametrize(
    "frame",
    [
        None,
        pl.DataFrame(schema=_ROBOT_SCHEMA),
    ],
)
def test_resolve_robot_trajectory_without_rows_is_empty(
    frame: pl.DataFrame | None,
) -> None:
    topics = {} if frame is None else {"robot_tracks": frame}

    trajectory, yaw = resolve_robot_trajectory(_calculator(), _episode(topics))

    assert len(trajectory.t_ns) == 0
    assert len(trajectory.x) == 0
    assert len(trajectory.vx) == 0
    assert len(yaw) == 0
    assert trajectory.radius_m == ROBOT_RADIUS_M


def test_resolve_robot_trajectory_keeps_an_unmeasured_velocity_null() -> None:
    """A recorded null velocity stays no measurement (NaN), never a zero."""
    frame = _robot_frame(
        t_s=np.array([0.0, 1.0]),
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 0.0]),
        vx=np.array([1.0, 1.0]),
        vy=np.array([0.0, 0.0]),
    ).with_columns(pl.lit(None, dtype=pl.Float64).alias("vx"))

    trajectory, _yaw = resolve_robot_trajectory(
        _calculator(), _episode({"robot_tracks": frame})
    )

    assert np.isnan(trajectory.vx).all()
    assert np.isfinite(trajectory.vy).all()


def test_resolve_robot_trajectory_keeps_an_unmeasured_yaw_null() -> None:
    """Yaw is not a metric, but an unrecorded sample is still not the zero heading."""
    frame = _robot_frame(
        t_s=np.array([0.0, 1.0]),
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 0.0]),
        vx=np.array([1.0, 1.0]),
        vy=np.array([0.0, 0.0]),
    ).with_columns(pl.lit(None, dtype=pl.Float64).alias("yaw"))

    _trajectory, yaw = resolve_robot_trajectory(
        _calculator(), _episode({"robot_tracks": frame})
    )

    assert np.isnan(yaw).all()


def test_extract_episode_ped_tracks_groups_rows_by_ped_id() -> None:
    frame = pl.DataFrame(
        [
            _ped_row(
                0, "Pedestrian_1", 0.0, 0.0, ped_name="alice", vx=1.0, radius_m=0.3
            ),
            _ped_row(0, "Pedestrian_2", 5.0, 5.0, ped_name="bob"),
            _ped_row(
                NS_PER_S,
                "Pedestrian_1",
                1.0,
                0.0,
                ped_name="alice",
                vx=1.0,
                radius_m=0.3,
            ),
            _ped_row(NS_PER_S, "Pedestrian_2", 5.0, 5.0, ped_name="bob"),
            _ped_row(
                2 * NS_PER_S,
                "Pedestrian_1",
                2.0,
                0.0,
                ped_name="alice",
                vx=1.0,
                radius_m=0.3,
            ),
            _ped_row(2 * NS_PER_S, "Pedestrian_2", 5.0, 5.0, ped_name="bob"),
        ],
        schema=_PED_SCHEMA,
    )

    tracks = extract_episode_ped_tracks(_episode({"pedestrian_arena_peds": frame}))

    assert set(tracks) == {"Pedestrian_1", "Pedestrian_2"}
    alice = tracks["Pedestrian_1"]
    assert alice.ped_name == "alice"
    assert alice.radius_m == 0.3
    np.testing.assert_allclose(alice.x, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(alice.vx, [1.0, 1.0, 1.0])
    bob = tracks["Pedestrian_2"]
    assert bob.ped_name == "bob"
    assert bob.radius_m is None
    assert bob.trajectory.radius_m == DEFAULT_PED_RADIUS_M
    np.testing.assert_allclose(bob.x, [5.0, 5.0, 5.0])


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


def test_extract_ped_tracks_keeps_recorded_velocity_when_the_pedestrian_moves() -> None:
    frame = pl.DataFrame(
        [
            _ped_row(0, "Pedestrian_4", 0.0, 0.0, vx=2.0, vy=-1.0),
            _ped_row(NS_PER_S, "Pedestrian_4", 2.0, -1.0, vx=2.0, vy=-1.0),
        ],
        schema=_PED_SCHEMA,
    )

    track = extract_episode_ped_tracks(_episode({"pedestrian_arena_peds": frame}))[
        "Pedestrian_4"
    ]

    np.testing.assert_allclose(track.vx, [2.0, 2.0])
    np.testing.assert_allclose(track.vy, [-1.0, -1.0])


def test_extract_ped_tracks_keeps_an_unmeasured_velocity_null() -> None:
    """A static pedestrian with a null twist keeps the null (NaN): it is no measured speed."""
    frame = pl.DataFrame(
        [
            _ped_row(0, "Pedestrian_6", 1.0, 0.0),
            _ped_row(NS_PER_S, "Pedestrian_6", 1.0, 0.0),
        ],
        schema=_PED_SCHEMA,
    ).with_columns(pl.lit(None, dtype=pl.Float64).alias("vx"))

    track = extract_episode_ped_tracks(_episode({"pedestrian_arena_peds": frame}))[
        "Pedestrian_6"
    ]

    assert np.isnan(track.vx).all()


def test_extract_ped_tracks_without_table_is_empty() -> None:
    assert extract_episode_ped_tracks(_episode({})) == {}


def test_extract_ped_tracks_empty_frame_is_empty() -> None:
    assert (
        extract_episode_ped_tracks(
            _episode({"pedestrian_arena_peds": pl.DataFrame(schema=_PED_SCHEMA)})
        )
        == {}
    )


def test_extract_ped_tracks_drops_non_finite_positions() -> None:
    frame = pl.DataFrame(
        [
            _ped_row(0, "Pedestrian_5", float("nan"), 0.0),
            _ped_row(NS_PER_S, "Pedestrian_5", 1.0, 0.0),
        ],
        schema=_PED_SCHEMA,
    )

    track = extract_episode_ped_tracks(_episode({"pedestrian_arena_peds": frame}))[
        "Pedestrian_5"
    ]

    np.testing.assert_allclose(track.x, [1.0])


@pytest.mark.parametrize(
    ("mask", "times_s", "merge_gap_s", "expected"),
    [
        (
            [True, True, False, True, True],
            [0.0, 1.0, 2.0, 4.0, 5.0],
            2.0,
            [(0, 1), (3, 4)],
        ),
        ([True, True, False, True, True], [0.0, 1.0, 2.0, 4.0, 5.0], 4.0, [(0, 4)]),
        ([False, False, False], [0.0, 1.0, 2.0], 1.0, []),
        ([True, True, True], [0.0, 1.0, 2.0], 1.0, [(0, 2)]),
        ([True, True, True], [0.0, 1.0, 3.0], 1.0, [(0, 1), (2, 2)]),
    ],
)
def test_contiguous_spans_bridges_gaps_within_the_merge_window(
    mask: list[bool],
    times_s: list[float],
    merge_gap_s: float,
    expected: list[tuple[int, int]],
) -> None:
    t_ns = (np.asarray(times_s, dtype=np.float64) * NS_PER_S).astype(np.int64)

    assert (
        contiguous_spans(
            np.asarray(mask, dtype=bool), t_ns, int(merge_gap_s * NS_PER_S)
        )
        == expected
    )


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([], []),
        ([(0, 10)], [(0, 10)]),
        ([(0, 5), (5, 10)], [(0, 10)]),
        ([(0, 6), (4, 9)], [(0, 9)]),
        ([(0, 6), (7, 9)], [(0, 6), (7, 9)]),
        ([(5, 10), (0, 3)], [(0, 3), (5, 10)]),
        ([(3, 2)], []),
    ],
)
def test_merge_intervals_ns_unions_overlapping_intervals(
    intervals: list[tuple[int, int]], expected: list[tuple[int, int]]
) -> None:
    assert merge_intervals_ns(intervals) == expected


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


def test_episode_and_fact_windows_share_one_membership_rule() -> None:
    episode = _encounter_episode()

    fact_windows = episode_encounter_windows(episode, ROBOT_RADIUS_M)
    spans = pedestrian_encounter_spans(episode, ROBOT_RADIUS_M)

    assert fact_windows.height == 1
    assert_series_equal(
        fact_windows["actor_id"],
        pl.Series("actor_id", ["Pedestrian_1"], dtype=pl.String),
    )
    assert spans == [("Pedestrian_1", 0, 4 * NS_PER_S)]


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


def test_resolve_ped_track_never_aliases_a_robot_to_a_pedestrian() -> None:
    tracks = extract_ped_tracks(
        pl.DataFrame(
            [_ped_row(0, "Pedestrian_2", 1.0, 0.0, ped_name="bob")], schema=_PED_SCHEMA
        )
    )

    assert _identity_aliases("robot_2") == {"robot_2"}
    assert resolve_ped_track(tracks, "robot_2") is None
    assert resolve_ped_track(tracks, "ped_2") is not None
    assert resolve_ped_track(tracks, 2) is not None
