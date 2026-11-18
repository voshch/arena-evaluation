"""Scoped, vectorized proxemics_extended: zone dwell, events, closing time and typed missingness."""

from __future__ import annotations

import polars as pl
import pytest
from flat_peds import flat_peds
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.social.proxemics import ProxemicsCalculator
from arena_evaluation.processing.metrics.social.proxemics_extended import (
    SERIES_SCHEMA,
    ZONE_EVENT_MERGE_GAP_NS,
    ProxemicsExtendedCalculator,
    zone_entry_events,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
RADIUS = 0.25
PED_RADIUS_M = 0.3
D_COMBINED = RADIUS + PED_RADIUS_M
ROBOT_TRACK_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "pose_source": pl.String,
    }
)


def _at_clearance(d_eff: float) -> list[float]:
    """One pedestrian straight ahead at the given edge-to-edge clearance."""
    return [d_eff + D_COMBINED, 0.0]


def _episode(
    times_ns: list[int],
    peds_positions: list[list[float]],
    robot_xy: list[tuple[float, float]] | None = None,
    twists: list[list[float]] | None = None,
    robot_times_ns: list[int] | None = None,
) -> AlignedEpisodeBundle:
    """Episode of flat pedestrian frames; ``robot_times_ns`` may end before the pedestrian axis to leave it unresolved."""
    n = len(times_ns)
    robot_xy = robot_xy or [(0.0, 0.0)] * n
    frames = []
    for i, (t, flat) in enumerate(zip(times_ns, peds_positions, strict=True)):
        pedestrians = []
        for j in range(len(flat) // 2):
            ped = {"x": flat[2 * j], "y": flat[2 * j + 1]}
            if twists is not None:
                ped["vx"], ped["vy"] = twists[i][3 * j], twists[i][3 * j + 1]
            pedestrians.append(ped)
        frames.append((t, pedestrians))
    peds = flat_peds(frames).with_columns(
        radius_m=pl.lit(PED_RADIUS_M, dtype=pl.Float64)
    )
    robot_times = times_ns if robot_times_ns is None else robot_times_ns
    robot = pl.DataFrame(
        {
            "t_ns": robot_times,
            "x": [p[0] for p in robot_xy[: len(robot_times)]],
            "y": [p[1] for p in robot_xy[: len(robot_times)]],
            "yaw": [0.0] * len(robot_times),
            "pose_source": ["tf_gt"] * len(robot_times),
        },
        schema=ROBOT_TRACK_SCHEMA,
    )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(
            {
                "t_ns": times_ns,
                "x": [p[0] for p in robot_xy],
                "y": [p[1] for p in robot_xy],
                "yaw": [0.0] * n,
            }
        ),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics={"robot_tracks": robot, "pedestrian_arena_peds": peds},
        peds=peds,
    )


@pytest.fixture
def calc() -> ProxemicsExtendedCalculator:
    return ProxemicsExtendedCalculator(make_robot_params(robot_radius=RADIUS))


def test_output_keys_do_not_collide_with_legacy_proxemics():
    """Both calculators land in the same results dict, so keys must be disjoint."""
    overlap = set(ProxemicsCalculator.output_keys()) & set(
        ProxemicsExtendedCalculator.output_keys()
    )
    assert overlap == set()


def test_no_nested_series_column_is_an_output_key():
    """The per-frame clearance series is a 1NF table, never a list-valued metric."""
    assert not any(
        key.startswith("timeseries_")
        for key in ProxemicsExtendedCalculator.output_keys()
    )
    assert set(SERIES_SCHEMA) == {
        "run_id",
        "episode_id",
        "robot_id",
        "t_ns",
        "time_s",
        "min_ped_clearance_m",
    }


def test_no_topics_yields_all_none(calc: ProxemicsExtendedCalculator):
    episode = AlignedEpisodeBundle(
        episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0
    )
    results = calc.calculate(episode, {})
    assert set(results) == set(calc.output_keys())
    assert all(value is None for value in results.values())


@pytest.mark.parametrize("radius_m", [None, 0.0, -0.3])
def test_a_pedestrian_without_a_positive_recorded_radius_is_refused(
    calc: ProxemicsExtendedCalculator, radius_m: float | None
):
    """The calculator never invents a body size; a missing or non-positive recorded radius raises."""
    episode = _episode([0, SEC], [[1.0, 0.0], [1.0, 0.0]])
    episode.topics["pedestrian_arena_peds"] = episode.topics[
        "pedestrian_arena_peds"
    ].with_columns(radius_m=pl.lit(radius_m, dtype=pl.Float64))
    with pytest.raises(MetricInputError):
        calc.calculate(episode, {})


def test_time_is_attributed_to_the_hall_zone_of_the_edge_distance(
    calc: ProxemicsExtendedCalculator,
):
    """One second in each zone, classified on d_eff rather than center distance."""
    times = [0, SEC, 2 * SEC, 3 * SEC]
    peds = [
        _at_clearance(0.2),  # intimate
        _at_clearance(0.8),  # personal
        _at_clearance(2.0),  # social
        _at_clearance(5.0),  # public, final frame carries a negligible dt
    ]
    results = calc.calculate(_episode(times, peds), {})

    assert results["time_in_intimate_zone"] == pytest.approx(1.0)
    assert results["time_in_personal_zone"] == pytest.approx(1.0)
    assert results["time_in_social_zone"] == pytest.approx(1.0)
    assert results["time_in_public_zone"] == pytest.approx(0.0, abs=1e-3)


def test_zone_boundaries_use_edge_to_edge_not_center_distance(
    calc: ProxemicsExtendedCalculator,
):
    """A ped 1.0 m from the robot center is intimate once radii are removed."""
    center_distance = 1.0
    episode = _episode([0, SEC], [[center_distance, 0.0]] * 2)
    results = calc.calculate(episode, {})
    assert results["time_in_intimate_zone"] > 0.0
    assert results["time_in_personal_zone"] == 0.0


def test_series_table_reports_the_frame_clearance(
    calc: ProxemicsExtendedCalculator,
):
    """The 1NF series carries one row per frame with the edge clearance of its closest pedestrian."""
    center_distance = 1.0
    episode = _episode([0, SEC], [[center_distance, 0.0]] * 2)
    table = calc.series_table(episode, run_id="run-1", robot_id="jackal_0")
    expected = pl.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "episode_id": [1, 1],
            "robot_id": ["jackal_0", "jackal_0"],
            "t_ns": [0, SEC],
            "time_s": [0.0, 1.0],
            "min_ped_clearance_m": [
                center_distance - D_COMBINED,
                center_distance - D_COMBINED,
            ],
        },
        schema=SERIES_SCHEMA,
    )
    assert_frame_equal(table, expected)


def test_a_pedestrian_frame_without_a_robot_sample_is_null_not_zero(
    calc: ProxemicsExtendedCalculator,
):
    """A frame further than the asof tolerance from the robot track has no clearance, never a fabricated zero."""
    episode = _episode([0, SEC], [[1.0, 0.0], [1.0, 0.0]], robot_times_ns=[0])
    table = calc.series_table(episode)
    clearances = table.get_column("min_ped_clearance_m").to_list()
    assert clearances[0] == pytest.approx(1.0 - D_COMBINED)
    assert clearances[1] is None


def test_frames_without_pedestrians_report_no_clearance(
    calc: ProxemicsExtendedCalculator,
):
    episode = _episode([0, SEC], [[], []])
    results = calc.calculate(episode, {})
    assert results["movement_towards_peds_ratio"] is None
    assert results["time_in_intimate_zone"] is None
    assert calc.series_table(episode).schema == SERIES_SCHEMA
    assert calc.series_table(episode).height == 0


def test_movement_towards_peds_ratio_counts_approaching_frames(
    calc: ProxemicsExtendedCalculator,
):
    """Robot drives +x with the ped ahead, so every moving frame is approaching."""
    times = [0, SEC, 2 * SEC]
    robot = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    peds = [[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]
    results = calc.calculate(_episode(times, peds, robot_xy=robot), {})
    assert results["movement_towards_peds_ratio"] == pytest.approx(2 / 3)


def test_tti_is_none_when_nothing_is_closing(calc: ProxemicsExtendedCalculator):
    times = [0, SEC]
    peds = [_at_clearance(2.0), _at_clearance(2.0)]
    results = calc.calculate(_episode(times, peds), {})
    assert results["tti_min"] is None
    assert results["tti_mean"] is None


def test_tti_uses_relative_velocity_of_an_approaching_pedestrian(
    calc: ProxemicsExtendedCalculator,
):
    """Ped 2 m clear closing at 1 m/s on a stationary robot gives TTI = 2 s."""
    times = [0, SEC]
    peds = [_at_clearance(2.0), _at_clearance(2.0)]
    twists = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    results = calc.calculate(_episode(times, peds, twists=twists), {})
    assert results["tti_min"] == pytest.approx(2.0, rel=1e-6)


def test_max_speed_is_recorded_per_zone(calc: ProxemicsExtendedCalculator):
    """Speed is only credited to the zone the robot was in at that frame."""
    times = [0, SEC, 2 * SEC]
    robot = [(0.0, 0.0), (0.0, 0.0), (2.0, 0.0)]
    peds = [
        [_at_clearance(0.2)[0], 0.0],
        [_at_clearance(0.2)[0], 0.0],
        [2.0 + _at_clearance(3.0)[0], 0.0],
    ]
    results = calc.calculate(_episode(times, peds, robot_xy=robot), {})
    assert results["max_speed_intimate_zone"] == pytest.approx(0.0)
    assert results["max_speed_social_zone"] == pytest.approx(2.0)
    assert results["max_speed_public_zone"] is None


def test_psi_events_merge_within_the_gap_and_split_beyond_it(
    calc: ProxemicsExtendedCalculator,
):
    """Intrusions closer than the 2 s gap are one event, further apart are two."""
    near, far = _at_clearance(0.2), _at_clearance(3.0)

    close_times = [0, SEC // 2, SEC]
    close = calc.calculate(_episode(close_times, [near, far, near]), {})
    assert close["psi_intimate_events"] == 1

    spread_times = [0, 5 * SEC // 2, 5 * SEC]
    spread = calc.calculate(_episode(spread_times, [near, far, near]), {})
    assert spread["psi_intimate_events"] == 2


def test_psi_events_are_zero_without_intrusions(calc: ProxemicsExtendedCalculator):
    results = calc.calculate(_episode([0, SEC], [_at_clearance(3.0)] * 2), {})
    assert results["psi_intimate_events"] == 0
    assert results["psi_personal_events"] == 0


def test_zone_entry_events_bridge_stays_within_the_merge_gap():
    """Two stays of one zone that resume inside the gap count as one event."""
    merged = pl.DataFrame(
        {
            "t_ns": [0, SEC // 2, 3 * SEC // 2, 2 * SEC],
            "zone": ["intimate", "personal", "intimate", "intimate"],
            "dt_s": [0.5, 1.0, 0.5, 1e-6],
        }
    )
    assert zone_entry_events(merged, "intimate") == 1
    spread = pl.DataFrame(
        {
            "t_ns": [
                0,
                SEC // 2,
                ZONE_EVENT_MERGE_GAP_NS,
                ZONE_EVENT_MERGE_GAP_NS + SEC,
            ],
            "zone": ["intimate", "personal", "intimate", "intimate"],
            "dt_s": [0.5, 1.0, 1.0, 1e-6],
        }
    )
    assert zone_entry_events(spread, "intimate") == 2


def _slice(scope: MetricScope, bundle: AlignedEpisodeBundle) -> ScopeSlice:
    """Scope slice over one bundle, used to prove which population a scope measures."""
    return ScopeSlice(
        scope=scope,
        slice_id="slice",
        t_start_ns=0,
        t_end_ns=SEC,
        identity={},
        window_bundle=bundle,
        full_episode=bundle,
    )


def test_episode_scope_recomputes_and_unserved_scopes_do_not_masquerade(
    calc: ProxemicsExtendedCalculator,
):
    """The EPISODE scope is the calculator's own population; a scope it does not serve yields typed nulls."""
    episode = _episode([0, SEC], [_at_clearance(0.8), _at_clearance(0.8)])
    assert calc.calculate_for_scope(
        episode, {}, _slice(MetricScope.EPISODE, episode)
    ) == calc.calculate(episode, {})
    pedestrian_slice = _slice(MetricScope.PEDESTRIAN, episode)
    assert all(
        value is None
        for value in calc.calculate_for_scope(episode, {}, pedestrian_slice).values()
    )


def test_encounter_scope_recomputes_the_window_telemetry(
    calc: ProxemicsExtendedCalculator,
):
    """An ENCOUNTER slice measures its own window bundle, never the episode bundle passed alongside it."""
    episode = _episode([0, SEC, 2 * SEC], [_at_clearance(0.8)] * 3)
    window = _episode([0, SEC], [_at_clearance(0.8)] * 2)
    window_metrics = calc.calculate_for_scope(
        episode, {}, _slice(MetricScope.ENCOUNTER, window)
    )
    assert window_metrics == calc.calculate(window, {})
    assert window_metrics["time_in_personal_zone"] == pytest.approx(1.0, abs=1e-5)
    assert window_metrics["psi_personal_events"] == 1
