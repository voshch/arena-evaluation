"""Holistic metrics: AEPS integration from the published 1NF series, typed missingness and e_cot."""

from __future__ import annotations


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


def test_declares_the_published_series_dependencies() -> None:
    """AEPS reads the 1NF tables, not the retired nested list outputs."""
    assert HolisticMetricsCalculator.SERIES_DEPENDS_ON == (
        "acoustic_exposure",
        "proxemics_extended",
    )


_LEVELS = _levels([(0, [90.0]), (1, [90.0])])
_CLEARANCES = _clearances([(0, 0.5), (1, 0.5)])
_EMPTY_LEVELS = pl.DataFrame(
    schema={"t_ns": pl.Int64, "ped_id": pl.String, "exposure_dba": pl.Float64}
)
_EMPTY_CLEARANCES = pl.DataFrame(
    schema={"t_ns": pl.Int64, "min_ped_clearance_m": pl.Float64}
)


@pytest.mark.parametrize(
    ("levels", "clearances"),
    [
        (None, None),
        (None, _CLEARANCES),
        (_LEVELS, None),
        (_EMPTY_LEVELS, _EMPTY_CLEARANCES),
    ],
    ids=["no-series", "no-acoustic-series", "no-clearance-series", "empty-frames"],
)
def test_aeps_is_none_without_its_inputs(levels, clearances) -> None:
    assert _aeps((0, 1), levels, clearances) is None


def test_aeps_is_zero_without_intrusions() -> None:
    levels = _levels([(0, [90.0]), (1, [90.0]), (2, [90.0])])
    clearances = _clearances([(0, 1.2), (1, 3.0), (2, None)])
    assert _aeps((0, 1, 2), levels, clearances) == 0.0


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


def _robot_slice(
    window: AlignedEpisodeBundle, full: AlignedEpisodeBundle
) -> ScopeSlice:
    return ScopeSlice(
        scope=MetricScope.ROBOT,
        slice_id="robot_1",
        t_start_ns=0,
        t_end_ns=2 * S,
        identity={"robot_id": "robot_1"},
        window_bundle=window,
        full_episode=full,
    )


def test_the_robot_scope_gives_the_episode_values_for_its_own_window() -> None:
    """Holistic metrics are declared at robot scope: a robot slice carries the values its window gives."""
    bundle = _episode(
        (0.0, 1.0, 2.0),
        levels=_levels([(0.0, [60.0]), (1.0, [60.0]), (2.0, [60.0])]),
        clearances=_clearances([(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]),
    )
    prior = {
        "specific_cost_of_transport": 2.0,
        "collision_amount": 0,
        "personal_space_intrusion_integral": 1.5,
        "time_to_goal": 4.0,
    }
    calculator = HolisticMetricsCalculator(make_robot_params())
    expected = calculator.calculate(bundle, prior)

    robot = calculator.calculate_for_scope(bundle, prior, _robot_slice(bundle, bundle))

    assert robot["aeps_linear_s"] == pytest.approx(expected["aeps_linear_s"])
    assert robot["e_cot"] == pytest.approx(expected["e_cot"])


def test_the_robot_scope_measures_its_own_window_not_the_full_episode() -> None:
    window = _episode(
        (0.0, 1.0),
        levels=_levels([(0.0, [60.0]), (1.0, [60.0])]),
        clearances=_clearances([(0.0, 2.0), (1.0, 2.0)]),
    )
    full = _episode(
        (0.0, 1.0),
        levels=_levels([(0.0, [60.0]), (1.0, [60.0])]),
        clearances=_clearances([(0.0, 1.0), (1.0, 1.0)]),
    )
    calculator = HolisticMetricsCalculator(make_robot_params())

    robot = calculator.calculate_for_scope(window, {}, _robot_slice(window, full))

    assert robot["aeps_linear_s"] == 0.0


def test_apa_weights_linear_power_by_clamped_inverse_square_distance() -> None:
    bundle = _episode(
        (0.0, 1.0, 2.0),
        levels=_levels([(0.0, [50.0]), (1.0, [50.0]), (2.0, [50.0])]),
        clearances=_clearances([(0.0, 0.1), (1.0, 1.0), (2.0, 2.0)]),
    )
    out = HolisticMetricsCalculator(make_robot_params()).calculate(bundle, {})
    # t=0 has dt=0; t=1 has d=1.0 -> 1e5/1.0*1.0; t=2 has d=2.0 -> 1e5/4.0*1.0
    assert out["acoustic_proxemic_annoyance"] == pytest.approx(1e5 + 1e5 / 4.0)
    assert out["relative_social_compliance_cost"] is None
    assert out["relative_social_compliance_cost_null_reason"] == "reference_missing"


def test_pedestrian_summary_integrates_poi_psychoacoustic_exposure() -> None:
    from arena_evaluation.processing.metrics.ecological.acoustic_exposure import (
        AcousticExposureCalculator,
    )

    summary = AcousticExposureCalculator.pedestrian_summary(
        [(0, 60.0), (1_000_000_000, 60.0), (2_000_000_000, 60.0)]
    )
    assert summary["poi_psychoacoustic_exposure"] == pytest.approx(3.0 * 1e6)
