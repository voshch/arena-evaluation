from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.experiments import group
from star_benchmark import write_benchmark


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
