from __future__ import annotations

import math
import pathlib

import polars as pl
import pytest
import yaml

from arena_evaluation.experiments import social
from star_benchmark import write_benchmark


def test_contact_partners_straddle_the_center_facing_each_other() -> None:
    a, b = social.contact_positions((4.0, 17.5), 0.30)
    assert a == pytest.approx((3.85, 17.5, 0.0))
    assert b == pytest.approx((4.15, 17.5, 3.14159))


def test_group_members_stand_on_the_formation_circle_facing_its_center() -> None:
    poses = social.group_positions((4.0, 17.5))
    assert len(poses) == 3
    for x, y, yaw in poses:
        assert math.hypot(x - 4.0, y - 17.5) == pytest.approx(social.GROUP_RADIUS_M)
        assert math.atan2(math.sin(yaw), math.cos(yaw)) == pytest.approx(
            math.atan2(17.5 - y, 4.0 - x)
        )
    assert poses[0] == pytest.approx((4.94, 17.5, math.pi))


def test_write_scenario_emits_one_agent_file_per_person_with_rounded_poses(
    tmp_path: pathlib.Path,
) -> None:
    layout = social.LAYOUTS["ward"]
    social.write_scenario(
        tmp_path / "contact_hug_ward",
        "HUG",
        layout,
        [(3.85, 17.5, 0.0), (4.15, 17.5, 3.14159)],
    )
    scenario = yaml.safe_load(
        (tmp_path / "contact_hug_ward" / "scenario.yaml").read_text()
    )
    assert scenario["robots"] == [
        {"start": [4.0, 12.0, 1.5708], "phases": [{"goto": [4.0, 23.0, 1.5708]}]}
    ]
    assert [d["name"] for d in scenario["dynamic"]] == ["person_0", "person_1"]
    assert [d["model"] for d in scenario["dynamic"]] == list(social.MODELS[:2])
    assert [d["pose"] for d in scenario["dynamic"]] == [
        [3.85, 17.5, 0.0],
        [4.15, 17.5, 3.142],
    ]
    assert scenario["dynamic"][1]["agent"] == {"agent_type": "./person_1.yaml"}
    assert scenario["dynamic"][1]["waypoints"] == [[4.15, 17.5]]
    agent = yaml.safe_load(
        (tmp_path / "contact_hug_ward" / "person_1.yaml").read_text()
    )
    steps = agent["sequences"]["seq"]["steps"]
    assert (agent["name"], agent["initial_sequence"], list(steps)) == (
        "person_1",
        "seq",
        ["settle", "interact"],
    )
    assert steps["interact"]["interaction"] == "HUG"
    assert steps["interact"]["attention"] == {"gaze": "partner"}


def test_stage_carries_the_arm_as_the_gesture_mode() -> None:
    stage = social._stage("group_ward", "off", "disabled", 76003, 3)
    assert (stage["name"], stage["seed"]) == ("group_ward_off", 76003)
    assert stage["config"] == {
        "scenario": {
            "file": "group_ward",
            "gesture_mode": "disabled",
            "contact_mode": "enabled",
        }
    }
    assert (stage["map"], stage["robot"], stage["episodes"], stage["timeout"]) == (
        social.WORLD,
        social.ROBOT,
        3,
        "90s",
    )


def _rows(
    *,
    arm: str,
    cases: tuple[str, ...] = ("contact_hug_ward", "group_ward"),
    planners: tuple[str, ...] = ("dwb",),
    episodes: int = 2,
    through: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    episode = 0
    for planner in planners:
        for case in cases:
            for seed in range(episodes):
                rows.append(
                    {
                        "stage": f"{case}_{arm}",
                        "planner": planner,
                        "episode_id": episode,
                        "result": "GOAL_REACHED",
                        "success": True,
                        "min_ped_clearance_m": 0.5
                        + 0.1 * seed
                        + (0.0 if through else 1.0),
                        "time_in_intimate_zone": 2.0 + 0.5 * seed,
                        "time_in_personal_zone": 5.0 + 0.5 * seed,
                        "path_length_m": 6.0 + seed,
                        "time_to_goal": 10.0 + seed,
                    }
                )
                episode += 1
    rows.append({**rows[-1], "episode_id": episode, "result": "UNRESOLVED"})
    return rows


def _tracks(rows: list[dict[str, object]], *, through: bool) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for row in rows:
        cx, cy = social.CENTERS[str(row["stage"]).rsplit("_", 1)[0]]
        offset = 0.0 if through else 2.0
        samples.extend(
            {
                "episode_id": row["episode_id"],
                "x": x,
                "y": y,
            }
            for x, y in ((cx, cy - 3.0), (cx + offset, cy), (cx, cy + 3.0))
        )
    return samples


def _run(tmp_path: pathlib.Path, arm: str, **kwargs: object) -> pathlib.Path:
    rows = _rows(arm=arm, **kwargs)
    directory = tmp_path / arm
    return write_benchmark(
        directory,
        rows,
        robot_tracks=_tracks(rows, through=bool(kwargs.get("through", False))),
    )


def test_analyze_pairs_the_arms_by_seed_within_each_case(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    on = _run(tmp_path, "on")
    off = _run(tmp_path, "off", through=True)
    tables = social.analyze(on, off, n_boot=100, out_dir=tmp_path / "out")
    assert set(tables) == {"descriptive", "matched", "paired"}
    assert tables["descriptive"].select("case", "planner", "arm").rows() == [
        ("contact_hug_ward", "dwb", "off"),
        ("contact_hug_ward", "dwb", "on"),
        ("group_ward", "dwb", "off"),
        ("group_ward", "dwb", "on"),
    ]
    paired = tables["paired"]
    assert set(paired["metric"]) == set(social.METRICS)
    pierced = paired.filter(pl.col("metric") == "pierced").row(0, named=True)
    assert (pierced["off_minus_on"], pierced["n_cases"], pierced["n_pairs"]) == (
        1.0,
        2,
        4,
    )
    clearance = paired.filter(pl.col("metric") == "min_ped_clearance_m").row(
        0, named=True
    )
    assert clearance["off_minus_on"] == pytest.approx(-1.0)
    # The two proxemics time metrics are paired like every other metric (L3-071/B-016).
    intimate = paired.filter(pl.col("metric") == "time_in_intimate_zone").row(
        0, named=True
    )
    assert intimate["off_minus_on"] == pytest.approx(0.0)
    assert tables["descriptive"].select("intimate_s", "personal_s").columns == [
        "intimate_s",
        "personal_s",
    ]
    assert tables["matched"].select("intimate_s", "personal_s").columns == [
        "intimate_s",
        "personal_s",
    ]
    assert paired.filter(pl.col("metric") == "success")["p"][0] == 1.0
    assert "p_holm" in paired.columns
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "descriptive.csv",
        "matched.csv",
        "paired.csv",
    }
    out = capsys.readouterr().out
    assert "== descriptive" in out and "== matched" in out and "== paired" in out


def test_main_analyze_returns_zero(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    on = _run(tmp_path, "on")
    off = _run(tmp_path, "off", through=True)
    assert (
        social.main(["analyze", "--on", str(on), "--off", str(off), "--boot", "50"])
        == 0
    )
    assert "== paired" in capsys.readouterr().out
