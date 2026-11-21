"""The standard report labels local-planner plots with the local planner and splits aware from blind worlds."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml
from star_benchmark import read_summary, write_benchmark

MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "configs" / "reports" / "standard" / "report.yaml"


def _manifest() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _channel_values(plot: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for value in (plot.get("channels") or {}).values():
        values.update([value] if isinstance(value, str) else list(value or []))
    return values


def test_summary_local_planner_is_the_local_name_not_the_contestant(
    tmp_path: pathlib.Path,
) -> None:
    rows = [
        {
            "episode_id": 0,
            "planner": "dwb-navigate_w_replanning_time",
            "local_planner": "dwb",
            "inter_planner": "navigate_w_replanning_time",
            "world": "concourse_headon_aware",
        }
    ]

    summary = read_summary(write_benchmark(tmp_path / "bench", rows))

    assert summary["local_planner"].to_list() == ["dwb"]


def test_summary_pairs_each_local_planner_with_its_world(
    tmp_path: pathlib.Path,
) -> None:
    rows = [
        {
            "episode_id": 0,
            "planner": "dwb-navigate_w_replanning_time",
            "local_planner": "dwb",
            "inter_planner": "navigate_w_replanning_time",
            "world": "concourse_headon_aware",
        }
    ]

    summary = read_summary(write_benchmark(tmp_path / "bench", rows))

    assert summary["planner_world"].to_list() == ["dwb · concourse_headon_aware"]


def test_standard_report_never_reads_the_composite_planner_column() -> None:
    offenders = [plot["id"] for plot in _manifest()["plots"] if "planner" in _channel_values(plot)]

    assert offenders == []


def test_standard_report_summarises_by_local_planner_and_world() -> None:
    assert _manifest()["summary_group_by"] == ["local_planner", "world"]


def test_local_planner_comparisons_split_aware_from_blind_worlds() -> None:
    comparisons = [plot for plot in _manifest()["plots"] if plot.get("type") in {"box", "violin", "bar"} and "by Local Planner" in str(plot.get("title"))]

    assert comparisons
    unsplit = [plot["id"] for plot in comparisons if (plot.get("channels") or {}).get("color") != "world"]
    assert unsplit == []


def test_timeseries_plots_draw_one_curve_per_planner_and_world() -> None:
    series = [plot for plot in _manifest()["plots"] if plot.get("type") == "timeseries"]

    assert series
    assert [plot["channels"]["color"] for plot in series] == ["planner_world"] * len(series)


def test_summary_relative_throughput_uses_reference_time_to_goal(
    tmp_path: pathlib.Path,
) -> None:
    rows = [
        {
            "episode_id": 0,
            "planner": "dwb-nav",
            "is_reference": True,
            "success": True,
            "time_to_goal": 10.0,
        },
        {
            "episode_id": 1,
            "planner": "dwb-nav",
            "is_reference": False,
            "success": True,
            "time_to_goal": 20.0,
        },
        {
            "episode_id": 2,
            "planner": "dwb-nav",
            "is_reference": False,
            "success": False,
            "time_to_goal": 5.0,
        },
        {
            "episode_id": 3,
            "planner": "mppi-nav",
            "is_reference": False,
            "success": True,
            "time_to_goal": 15.0,
        },
    ]

    summary = read_summary(write_benchmark(tmp_path / "bench", rows)).sort("episode_id")

    assert summary["relative_throughput"].to_list() == [None, 0.5, None, None]
    assert summary["relative_throughput_null_reason"].to_list() == [
        "is_reference",
        None,
        "not_successful",
        "reference_missing",
    ]
