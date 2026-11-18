from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.experiments import group
from star_benchmark import read_summary, write_benchmark


def test_formation_radius_grows_per_member_and_scales() -> None:
    assert group.formation_radius(1) == pytest.approx(0.7)
    assert group.formation_radius(0) == pytest.approx(0.7)
    assert group.formation_radius(3) == pytest.approx(0.94)
    assert group.formation_radius(3, scale=2.0) == pytest.approx(1.88)


def _summary() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "episode_id": [0, 1, 2],
            "planner": ["dwb"] * 3,
            "stage": ["group_ward_on", "group_ward_on", "group_ward_off"],
            "min_ped_clearance_m": [0.2, 1.1, None],
            "time_in_intimate_zone": [3.0, 5.0, None],
            "time_in_personal_zone": [8.0, 4.0, None],
            "path_length_m": [7.0, 7.5, None],
            "time_to_goal": [12.0, 13.0, None],
            "success": [True, True, False],
        },
        schema={
            "episode_id": pl.Int64,
            "planner": pl.String,
            "stage": pl.String,
            "min_ped_clearance_m": pl.Float64,
            "time_in_intimate_zone": pl.Float64,
            "time_in_personal_zone": pl.Float64,
            "path_length_m": pl.Float64,
            "time_to_goal": pl.Float64,
            "success": pl.Boolean,
        },
    )


def _tracks() -> pl.DataFrame:
    rows = []
    for episode_id, points in enumerate(
        [
            [(4.0, 14.0), (4.0, 17.3), (4.0, 17.6), (4.0, 21.0)],
            [(4.0, 14.0), (6.0, 17.5), (4.0, 21.0)],
            [],
        ]
    ):
        rows.extend({"episode_id": episode_id, "x": x, "y": y} for x, y in points)
    return pl.DataFrame(
        rows, schema={"episode_id": pl.Int64, "x": pl.Float64, "y": pl.Float64}
    )


def test_episode_routes_count_the_ticks_inside_the_formation() -> None:
    routes = group.episode_routes(
        _summary(), _tracks(), (4.0, 17.5), group.formation_radius(3)
    )
    assert routes["pierced"].to_list() == [True, False, False]
    assert routes["ticks_inside"].to_list() == [2, 0, 0]
    assert routes["closest_to_group_m"].to_list()[:2] == pytest.approx([0.1, 2.0])
    assert routes["closest_to_group_m"][2] is None
    assert routes.columns == [
        "episode_id",
        "planner",
        "stage",
        "pierced",
        "ticks_inside",
        "closest_to_group_m",
        "min_ped_clearance_m",
        "time_in_intimate_zone",
        "time_in_personal_zone",
        "path_length_m",
        "time_to_goal",
        "success",
    ]


def test_summarize_reports_the_pierce_rate_per_stage_and_planner() -> None:
    routes = group.episode_routes(
        _summary(), _tracks(), (4.0, 17.5), group.formation_radius(3)
    )
    table = group.summarize(routes)
    assert table["stage"].to_list() == ["group_ward_on", "group_ward_off"]
    on = table.row(0, named=True)
    assert (on["episodes"], on["pierce_rate"], on["success"]) == (2, 0.5, 1.0)
    assert on["closest_to_group_m"] == pytest.approx(1.05)
    assert on["clearance_m"] == pytest.approx(0.65)
    # The two proxemics time metrics are carried through and reported as medians (L3-071).
    assert on["intimate_s"] == pytest.approx(4.0)
    assert on["personal_s"] == pytest.approx(6.0)
    assert table.row(1, named=True)["clearance_m"] is None
    assert table.row(1, named=True)["intimate_s"] is None


def test_main_measures_every_benchmark_dir_about_the_center(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    write_benchmark(run, _summary().to_dicts(), robot_tracks=_tracks().to_dicts())
    assert (
        group.main(
            [
                "measure",
                "--benchmark-dir",
                str(run),
                str(run),
                "--center",
                "4.0",
                "17.5",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("formation radius 0.94 m about (4.0, 17.5)")
    assert "pierce_rate" in out
    with pytest.raises(SystemExit, match="Star Schema"):
        group.main(
            [
                "measure",
                "--benchmark-dir",
                str(tmp_path / "missing"),
                "--center",
                "0",
                "0",
            ]
        )


def test_episode_summary_is_read_from_the_catalog(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "run"
    write_benchmark(run, _summary().to_dicts(), robot_tracks=_tracks().to_dicts())
    assert sorted(read_summary(run)["episode_id"].to_list()) == [0, 1, 2]
