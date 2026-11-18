"""Holistic metrics: AEPS integration from the published 1NF series, typed missingness and e_cot."""

from __future__ import annotations

import math

import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.holistic.holistic_metrics import (
    HolisticMetricsCalculator,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

S = 1_000_000_000


def _episode(
    times_s: tuple[float, ...],
    *,
    levels: pl.DataFrame | None,
    clearances: pl.DataFrame | None,
) -> AlignedEpisodeBundle:
    series: dict[str, pl.DataFrame] = {}
    if levels is not None:
        series["acoustic_exposure"] = levels
    if clearances is not None:
        series["proxemics_extended"] = clearances
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [int(seconds * S) for seconds in times_s]}),
        start_pos=[],
        goal_pos=[],
        series=series,
    )


def _levels(pairs: list[tuple[float, list[float]]]) -> pl.DataFrame:
    """Acoustic exposure rows: one per (t_ns, ped_id, exposure_dba)."""
    rows = [
        (int(seconds * S), f"ped_{index}", value)
        for seconds, values in pairs
        for index, value in enumerate(values)
    ]
    return pl.DataFrame(
        {
            "t_ns": [row[0] for row in rows],
            "ped_id": [row[1] for row in rows],
            "exposure_dba": [row[2] for row in rows],
        },
        schema={"t_ns": pl.Int64, "ped_id": pl.String, "exposure_dba": pl.Float64},
    )


def _clearances(pairs: list[tuple[float, float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [int(seconds * S) for seconds, _value in pairs],
            "min_ped_clearance_m": [value for _seconds, value in pairs],
        },
        schema={"t_ns": pl.Int64, "min_ped_clearance_m": pl.Float64},
    )


def _aeps(
    times_s: tuple[float, ...],
    levels: pl.DataFrame | None,
    clearances: pl.DataFrame | None,
) -> float | None:
    return HolisticMetricsCalculator(make_robot_params()).calculate(
        _episode(times_s, levels=levels, clearances=clearances), {}
    )["aeps_linear_s"]


def test_output_keys() -> None:
    assert HolisticMetricsCalculator.output_keys() == ["aeps_linear_s", "e_cot"]


def test_declares_the_published_series_dependencies() -> None:
    """AEPS reads the 1NF tables, not the retired nested list outputs."""
    assert HolisticMetricsCalculator.SERIES_DEPENDS_ON == (
        "acoustic_exposure",
        "proxemics_extended",
    )


def test_no_series_yields_none() -> None:
    assert _aeps((0, 1), None, None) is None


def test_aeps_integrates_the_series_over_intruded_frames() -> None:
    levels = _levels([(0, [60.0]), (1, [70.0]), (3, [80.0])])
    clearances = _clearances([(0, 2.0), (1, 1.0), (3, 0.5), (4, None)])
    assert _aeps((0, 1, 3, 4), levels, clearances) == pytest.approx(
        1e7 * 1.0 + 1e8 * 2.0, rel=1e-12
    )


def test_aeps_takes_the_loudest_pedestrian_per_frame() -> None:
    levels = _levels([(0, [50.0, 70.0]), (2, [60.0])])
    clearances = _clearances([(0, 1.0), (1, 1.0), (2, 1.0)])
    assert _aeps((0, 1, 2), levels, clearances) == pytest.approx(1e6, rel=1e-12)


def test_aeps_is_zero_without_intrusions() -> None:
    levels = _levels([(0, [90.0]), (1, [90.0]), (2, [90.0])])
    clearances = _clearances([(0, 1.2), (1, 3.0), (2, None)])
    assert _aeps((0, 1, 2), levels, clearances) == 0.0


def test_aeps_is_none_without_the_acoustic_series() -> None:
    assert _aeps((0, 1), None, _clearances([(0, 0.5), (1, 0.5)])) is None


def test_aeps_is_none_without_the_clearance_series() -> None:
    assert _aeps((0, 1), _levels([(0, [90.0]), (1, [90.0])]), None) is None


def test_aeps_is_none_for_empty_series_frames() -> None:
    empty_levels = pl.DataFrame(
        schema={"t_ns": pl.Int64, "ped_id": pl.String, "exposure_dba": pl.Float64}
    )
    empty_clearance = pl.DataFrame(
        schema={"t_ns": pl.Int64, "min_ped_clearance_m": pl.Float64}
    )
    assert _aeps((0, 1), empty_levels, empty_clearance) is None


def test_aeps_is_none_without_an_aligned_time_axis() -> None:
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[],
        goal_pos=[],
        series={
            "acoustic_exposure": _levels([(0, [90.0])]),
            "proxemics_extended": _clearances([(0, 0.5)]),
        },
    )
    assert (
        HolisticMetricsCalculator(make_robot_params()).calculate(episode, {})[
            "aeps_linear_s"
        ]
        is None
    )


def test_aeps_clamps_backwards_timestamps_to_zero_dt() -> None:
    levels = _levels([(0, [60.0]), (2, [60.0]), (1, [60.0])])
    clearances = _clearances([(0, 0.5), (1, 0.5), (2, 0.5)])
    assert _aeps((0, 2, 1), levels, clearances) == pytest.approx(2e6, rel=1e-12)


def test_registry_reports_the_missing_series_and_never_returns_a_value() -> None:
    """A declared dependency that is absent is a typed null, not an invented zero."""
    registry = MetricRegistry(make_robot_params())
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [0, S]}),
        start_pos=[],
        goal_pos=[],
        series={"proxemics_extended": _clearances([(0, 0.5), (1, 0.5)])},
    )
    robot_slice = ScopeSlice(
        scope=MetricScope.ROBOT,
        slice_id="robot",
        t_start_ns=0,
        t_end_ns=S,
        identity={},
        window_bundle=episode,
        full_episode=episode,
    )
    outcome = registry.run_scope(
        MetricScope.ROBOT,
        episode,
        available_topics={"robot_tracks", "pedestrian_arena_peds"},
        scope_slice=robot_slice,
    )
    assert outcome["aeps_linear_s"] is None
    assert (
        outcome.provenance.get("aeps_linear_s")
        == "aeps_linear_s:not_recorded:acoustic_exposure"
    )


def test_e_cot_penalizes_collisions_and_intrusion_rate() -> None:
    prior = {
        "specific_cost_of_transport": 1.5,
        "collision_amount": 2,
        "personal_space_intrusion_integral": 4.0,
        "time_to_goal": 10.0,
    }
    assert HolisticMetricsCalculator(make_robot_params())._effective_cost_of_transport(
        prior
    ) == pytest.approx(1.5 * (1.0 + 4.0 + 0.2), abs=1e-9)


@pytest.mark.parametrize("cot", [None, 0.0, -1.0])
def test_e_cot_is_none_without_positive_cost_of_transport(cot: float | None) -> None:
    assert (
        HolisticMetricsCalculator(make_robot_params())._effective_cost_of_transport(
            {"specific_cost_of_transport": cot}
        )
        is None
    )


@pytest.mark.parametrize(
    "missing",
    ["collision_amount", "personal_space_intrusion_integral", "time_to_goal"],
)
def test_e_cot_is_null_when_a_penalty_input_is_not_measured(missing: str) -> None:
    prior = {
        "specific_cost_of_transport": 3.0,
        "collision_amount": 0,
        "personal_space_intrusion_integral": 0.0,
        "time_to_goal": 10.0,
    }
    del prior[missing]
    assert (
        HolisticMetricsCalculator(make_robot_params())._effective_cost_of_transport(
            prior
        )
        is None
    )


def test_e_cot_rounds_to_six_decimals() -> None:
    prior = {
        "specific_cost_of_transport": 1.0 / 3.0,
        "collision_amount": 0,
        "personal_space_intrusion_integral": 0.0,
        "time_to_goal": 10.0,
    }
    value = HolisticMetricsCalculator(make_robot_params())._effective_cost_of_transport(
        prior
    )
    assert value == 0.333333


def test_e_cot_is_finite_for_a_positive_cost() -> None:
    prior = {
        "specific_cost_of_transport": 2.0,
        "collision_amount": 0,
        "personal_space_intrusion_integral": 0.0,
        "time_to_goal": 10.0,
    }
    value = HolisticMetricsCalculator(make_robot_params())._effective_cost_of_transport(
        prior
    )
    assert value is not None and math.isfinite(value)
