from __future__ import annotations

import math
import pathlib

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.experiments import group


def test_formation_radius_grows_per_member_and_scales() -> None:
    assert group.formation_radius(1) == pytest.approx(0.7)
    assert group.formation_radius(0) == pytest.approx(0.7)
    assert group.formation_radius(3) == pytest.approx(0.94)
    assert group.formation_radius(3, scale=2.0) == pytest.approx(1.88)


def _metrics() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "episode": [0, 1, 2],
            "planner": ["dwb"] * 3,
            "stage": ["group_ward_on", "group_ward_on", "group_ward_off"],
            "path": [[[4.0, 14.0], [4.0, 17.3], [4.0, 17.6], [4.0, 21.0]], [[4.0, 14.0], [6.0, 17.5], [4.0, 21.0]], []],
            "min_pedestrian_clearance": [0.2, 1.1, None],
            "time_in_intimate_zone": [1.0, 0.0, None],
            "time_in_personal_zone": [2.0, 0.5, None],
            "path_length": [7.0, 7.5, None],
            "time_to_goal": [12.0, 13.0, None],
            "success": [True, True, False],
        }
    )


def test_episode_routes_count_the_ticks_inside_the_formation() -> None:
    routes = group.episode_routes(_metrics(), (4.0, 17.5), group.formation_radius(3))
    assert routes["pierced"].to_list() == [True, False, False]
    assert routes["ticks_inside"].to_list() == [2, 0, 0]
    assert routes["closest_to_group_m"].to_list()[:2] == pytest.approx([0.1, 2.0])
    assert math.isnan(routes["closest_to_group_m"][2])
    assert routes.columns == ["episode", "planner", "stage", "pierced", "ticks_inside", "closest_to_group_m", "min_pedestrian_clearance", "time_in_intimate_zone", "time_in_personal_zone", "path_length", "time_to_goal", "success"]


def test_summarize_reports_the_pierce_rate_per_stage_and_planner() -> None:
    routes = group.episode_routes(_metrics(), (4.0, 17.5), group.formation_radius(3))
    table = group.summarize(routes)
    assert table["stage"].to_list() == ["group_ward_on", "group_ward_off"]
    on = table.row(0, named=True)
    assert (on["episodes"], on["pierce_rate"], on["success"]) == (2, 0.5, 1.0)
    assert on["closest_to_group_m"] == pytest.approx(1.05)
    assert on["clearance_m"] == pytest.approx(0.65)
    assert table.row(1, named=True)["clearance_m"] is None


def test_main_measures_every_benchmark_dir_about_the_center(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _metrics().write_parquet(run / "combined_metrics.parquet")
    assert group.main(["measure", "--benchmark-dir", str(run), str(run), "--center", "4.0", "17.5"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("formation radius 0.94 m about (4.0, 17.5)")
    assert "pierce_rate" in out
    with pytest.raises(SystemExit, match="combined_metrics.parquet"):
        group.main(["measure", "--benchmark-dir", str(tmp_path / "missing"), "--center", "0", "0"])
