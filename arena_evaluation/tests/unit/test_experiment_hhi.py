from __future__ import annotations

import math
import pathlib

import numpy as np
import polars as pl
import pytest
import yaml

from arena_evaluation.experiments import hhi


def test_geometry_is_admissible_and_meets_near_the_path() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = hhi.sample_geometry(rng)
        for p in (*g.ped_start, *g.ped_exit, g.meet):
            assert hhi._inside(p)
        s, e = np.array(g.start[:2]), np.array(g.goal[:2])
        m = np.array(g.meet)
        d, v = e - s, m - s
        off = abs(d[0] * v[1] - d[1] * v[0]) / np.linalg.norm(d)
        assert off <= 1.0 + 1e-9
        gap = np.linalg.norm(np.subtract(*g.ped_meet))
        assert gap == pytest.approx(2 * hhi.MEET_OFFSET)  # both arms approach to the same points
        assert 4.0 <= g.hold_s <= 8.0


def test_generate_writes_paired_stages_sharing_seeds(tmp_path: pathlib.Path) -> None:
    suite = hhi.generate(3, seed=1, scenarios_dir=tmp_path / "scenarios", suite_path=tmp_path / "suite.yaml")
    stages = suite["stages"]
    assert len(stages) == 2 * 3 * 2
    by_name = {s["name"]: s for s in stages}
    for kind in hhi.KINDS:
        for k in range(3):
            contact, loco = by_name[hhi.stage_name(kind, k, "contact")], by_name[hhi.stage_name(kind, k, "loco")]
            assert contact["seed"] == loco["seed"]
            assert contact["config"]["scenario"]["file"] == loco["config"]["scenario"]["file"]
            assert (contact["config"]["scenario"]["contact_mode"], loco["config"]["scenario"]["contact_mode"]) == ("enabled", "locomotion_only")
    assert len({s["seed"] for s in stages}) == 2 * 3  # distinct across scenarios
    reloaded = yaml.safe_load((tmp_path / "suite.yaml").read_text())
    assert reloaded == suite

    scenario = yaml.safe_load((tmp_path / "scenarios" / "hhi_shake_01" / "scenario.yaml").read_text())
    assert [d["agent"]["agent_type"] for d in scenario["dynamic"]] == ["./partner_0.yaml", "./partner_1.yaml"]
    partner = yaml.safe_load((tmp_path / "scenarios" / "hhi_shake_01" / "partner_1.yaml").read_text())
    steps = partner["sequences"]["contact_seq"]["steps"]
    assert list(steps) == ["wait", "approach", "contact", "separate", "rest"]
    assert steps["contact"]["interaction"] == "SHAKE_HAND"


def test_generated_agent_types_load_in_humansim(tmp_path: pathlib.Path) -> None:
    loader = pytest.importorskip("arena_humansim.core.agents.loader")
    hhi.generate(1, scenarios_dir=tmp_path, suite_path=tmp_path / "suite.yaml")
    agent = loader.load_agent_type_from_file(tmp_path / "hhi_hug_00" / "partner_0.yaml")
    assert agent is not None


def _metrics() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    episode = 0
    for planner in ("dwb", "mppi"):
        for kind in ("hug", "shake"):
            for k in range(8):
                for arm in ("contact", "loco"):
                    for _ in range(3):
                        bump = 2 if (planner == "dwb" and arm == "contact") else 0
                        stalls = int(rng.poisson(1)) + bump
                        post = min(stalls, bump + int(rng.random() < 0.2))
                        rows.append(
                            {
                                "stage": hhi.stage_name(kind, k, arm),
                                "planner": planner,
                                "episode": episode,
                                "status": "evaluated",
                                "is_reference": False,
                                "success": rng.random() < 0.9,
                                "time_to_goal": 40.0 + rng.normal(0, 2) + 3 * bump,
                                "hhi_stall_count": stalls,
                                "hhi_passes_between": 0 if arm == "contact" else int(rng.random() < 0.3),
                                "hhi_stalls_post": post,
                                "hhi_share_post": 0.1,
                                "hhi_episode_s": 50.0,
                                "hhi_release_s": 20.0,
                                "hhi_hold_onset_s": 14.0,
                                "hhi_pair_gap_hold": 0.3 if arm == "contact" else 1.2,
                            }
                        )
                        episode += 1
    rows.append({"stage": "some_other_stage", "planner": "dwb", "episode": episode, "status": "evaluated"})
    return pl.DataFrame(rows, infer_schema_length=None)


def test_load_episodes_tags_arm_scenario_and_seed() -> None:
    df = hhi.load_episodes(_metrics())
    assert set(df["arm"]) == {"contact", "loco"} and set(df["interaction"]) == {"hug", "shake_hand"}
    assert sorted(set(df["seed"])) == [0, 1, 2]
    assert df.filter(pl.col("stage") == "some_other_stage").height == 0
    assert df.group_by(["planner", "stage"]).agg(pl.col("seed").n_unique())["seed"].to_list() == [3] * df["stage"].n_unique() * 2


def test_family_and_window_tables(tmp_path: pathlib.Path) -> None:
    metrics = _metrics()
    (tmp_path / "run").mkdir()
    metrics.write_parquet(tmp_path / "run" / "combined_metrics.parquet")
    tables = hhi.analyze(tmp_path / "run", n_boot=300)
    fam = tables["family_a"]
    assert fam.height == 2 * 2 * 4  # planner x interaction x metric
    dwb_stalls = fam.filter((pl.col("planner") == "dwb") & (pl.col("metric") == "stalls"))
    assert dwb_stalls["estimate"].to_list() == pytest.approx([2.0, 2.0], abs=0.6)
    assert dwb_stalls["excludes_zero"].to_list() == [True, True]
    mppi_stalls = fam.filter((pl.col("planner") == "mppi") & (pl.col("metric") == "stalls"))
    assert not any(mppi_stalls["excludes_zero"].to_list())
    windows = tables["windows"].sort("planner")
    assert windows["time_share_post"].to_list() == pytest.approx([0.1, 0.1])
    assert windows["excess"][0] > windows["excess"][1]
    check = tables["manipulation_check"]
    assert check.filter(pl.col("arm") == "contact")["pair_gap_median"].to_list() == [0.3, 0.3]
    assert (tmp_path / "run" / "hhi" / "family_a.csv").exists()


def test_paper_outcomes_fail_collisions_and_drop_failed_times() -> None:
    df = pl.DataFrame({"success": [True, True, False], "time_to_goal": [30.0, 40.0, 120.0], "collision_amount": [0, 1, 0]})
    out = hhi.paper_outcomes(df)
    assert out["success"].to_list() == [True, False, False]
    assert out["time_to_goal"].to_list() == [30.0, None, None]


def test_stage_regex_round_trips() -> None:
    for kind in hhi.KINDS:
        m = hhi.STAGE_RE.match(hhi.stage_name(kind, 7, "loco"))
        assert m is not None and (m["kind"], int(m["scenario"]), m["arm"]) == (kind, 7, "loco")
    assert math.isclose(hhi.MEET_OFFSET * 2, hhi.STANDING_DISTANCE + 0.2)
