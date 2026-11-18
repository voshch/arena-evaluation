from __future__ import annotations

import math
import pathlib

import numpy as np
import polars as pl
import pytest
import yaml

from arena_evaluation.experiments import hhi
from arena_evaluation.experiments.source import contacts
from star_benchmark import read_summary, write_benchmark


@pytest.mark.parametrize("seed", [0, 186])
def test_geometry_is_admissible_and_meets_near_the_path(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(100):
        g = hhi.sample_geometry(rng)
        for p in (*g.ped_start, *g.ped_exit, g.meet):
            assert hhi._inside(p)
        s, e = np.array(g.start[:2]), np.array(g.goal[:2])
        m = np.array(g.meet)
        d, v = e - s, m - s
        off = abs(d[0] * v[1] - d[1] * v[0]) / np.linalg.norm(d)
        assert off <= 1.0 + 1e-9
        gap = np.linalg.norm(np.subtract(*g.ped_meet))
        assert gap == pytest.approx(2 * hhi.MEET_OFFSET)
        assert 4.0 <= g.hold_s <= 8.0


def test_generate_writes_paired_stages_sharing_seeds(tmp_path: pathlib.Path) -> None:
    suite = hhi.generate(
        3,
        seed=1,
        scenarios_dir=tmp_path / "scenarios",
        suite_path=tmp_path / "suite.yaml",
    )
    stages = suite["stages"]
    assert len(stages) == 2 * 3 * 2
    by_name = {s["name"]: s for s in stages}
    for kind in hhi.KINDS:
        for k in range(3):
            contact, loco = (
                by_name[hhi.stage_name(kind, k, "contact")],
                by_name[hhi.stage_name(kind, k, "loco")],
            )
            assert contact["seed"] == loco["seed"]
            assert (
                contact["config"]["scenario"]["file"]
                == loco["config"]["scenario"]["file"]
            )
            assert (
                contact["config"]["scenario"]["contact_mode"],
                loco["config"]["scenario"]["contact_mode"],
            ) == ("enabled", "locomotion_only")
    assert len({s["seed"] for s in stages}) == 2 * 3
    reloaded = yaml.safe_load((tmp_path / "suite.yaml").read_text())
    assert reloaded == suite

    scenario = yaml.safe_load(
        (tmp_path / "scenarios" / "hhi_shake_01" / "scenario.yaml").read_text()
    )
    assert [d["agent"]["agent_type"] for d in scenario["dynamic"]] == [
        "./partner_0.yaml",
        "./partner_1.yaml",
    ]
    partner = yaml.safe_load(
        (tmp_path / "scenarios" / "hhi_shake_01" / "partner_1.yaml").read_text()
    )
    steps = partner["sequences"]["contact_seq"]["steps"]
    assert list(steps) == ["wait", "approach", "contact", "separate", "rest"]
    assert steps["contact"]["interaction"] == "SHAKE_HAND"


def test_generated_agent_types_load_in_humansim(tmp_path: pathlib.Path) -> None:
    loader = pytest.importorskip("arena_humansim.core.agents.loader")
    hhi.generate(1, scenarios_dir=tmp_path, suite_path=tmp_path / "suite.yaml")
    agent = loader.load_agent_type_from_file(tmp_path / "hhi_hug_00" / "partner_0.yaml")
    assert agent is not None


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    episode = 0
    for planner in ("dwb", "mppi"):
        for kind in ("hug", "shake"):
            for k in range(8):
                for arm in ("contact", "loco"):
                    for _ in range(3):
                        rows.append(
                            {
                                "stage": hhi.stage_name(kind, k, arm),
                                "planner": planner,
                                "episode_id": episode,
                                "is_reference": False,
                                "success": True,
                                "time_to_goal": 40.0
                                + (
                                    10.0
                                    if planner == "dwb" and arm == "contact"
                                    else 0.0
                                ),
                                "collision_amount": 0,
                                "collision_amount_pedestrian": 0,
                                "collision_amount_wall": 0,
                                "collision_amount_static": 0,
                                "hhi_interaction": "hug"
                                if kind == "hug"
                                else "shake_hand",
                                "hhi_hold_onset_s": 2.0,
                                "hhi_release_s": 6.0 if arm == "contact" else None,
                                "hhi_episode_s": 10.0,
                                "hhi_stall_count": 4 if arm == "contact" else 1,
                                "hhi_stall_time_s": 8.0 if arm == "contact" else 1.0,
                                "hhi_stalls_approach": 0,
                                "hhi_stalls_hold": 0,
                                "hhi_stalls_post": 3 if arm == "contact" else 1,
                                "hhi_stalls_rest": 1 if arm == "contact" else 0,
                                "hhi_share_approach": 0.2,
                                "hhi_share_hold": 0.4,
                                "hhi_share_post": 0.25,
                                "hhi_share_rest": 0.15,
                                "hhi_passes_between": 2 if arm == "contact" else 0,
                                "hhi_min_pair_clearance": 0.5,
                                "hhi_pair_gap_hold": 0.6 if arm == "contact" else 1.8,
                            }
                        )
                        episode += 1
    rows.append(
        {
            "stage": "some_other_stage",
            "planner": "dwb",
            "episode_id": episode,
            "is_reference": False,
            "success": True,
        }
    )
    return rows


def _run(tmp_path: pathlib.Path) -> pathlib.Path:
    return write_benchmark(tmp_path / "run", _rows())


def test_load_episodes_tags_arm_scenario_and_seed(tmp_path: pathlib.Path) -> None:
    df = hhi.load_episodes(read_summary(_run(tmp_path)))
    assert set(df["arm"]) == {"contact", "loco"} and set(df["interaction"]) == {
        "hug",
        "shake_hand",
    }
    assert sorted(set(df["seed"])) == [0, 1, 2]
    assert df.filter(pl.col("stage") == "some_other_stage").height == 0
    assert (
        df.group_by(["planner", "stage"])
        .agg(pl.col("seed").n_unique())["seed"]
        .to_list()
        == [3] * df["stage"].n_unique() * 2
    )


def test_family_table_compares_the_contact_arm_against_locomotion(
    tmp_path: pathlib.Path,
) -> None:
    tables = hhi.analyze(_run(tmp_path), n_boot=1000)
    fam = tables["family_a"]
    assert fam.height == 2 * 2 * 4
    dwb_ttg = fam.filter((pl.col("planner") == "dwb") & (pl.col("metric") == "ttg"))
    assert dwb_ttg["estimate"].to_list() == pytest.approx([10.0, 10.0])
    assert dwb_ttg["excludes_zero"].to_list() == [True, True]
    mppi_ttg = fam.filter((pl.col("planner") == "mppi") & (pl.col("metric") == "ttg"))
    assert not any(mppi_ttg["excludes_zero"].to_list())
    assert (tmp_path / "run" / "hhi" / "family_a.csv").exists()


def test_family_table_carries_the_two_hhi_metrics(tmp_path: pathlib.Path) -> None:
    """The published study keeps all four family metrics, not only the two episode outcomes (L3-070)."""
    family = hhi.analyze(_run(tmp_path), n_boot=50)["family_a"]

    assert set(family["metric"].to_list()) == {
        "success",
        "ttg",
        "stalls",
        "passes_between",
    }
    assert family.height == 2 * 2 * 4
    dwb_stalls = family.filter(
        (pl.col("planner") == "dwb") & (pl.col("metric") == "stalls")
    )
    # Contact arm 4 stalls, locomotion arm 1, over the paired scenarios.
    assert dwb_stalls["estimate"].to_list() == pytest.approx([3.0, 3.0])
    dwb_passes = family.filter(
        (pl.col("planner") == "dwb") & (pl.col("metric") == "passes_between")
    )
    assert dwb_passes["estimate"].to_list() == pytest.approx([2.0, 2.0])


def test_paper_outcomes_fail_collisions_and_drop_failed_times() -> None:
    df = pl.DataFrame(
        {
            "success": [True, True, False],
            "time_to_goal": [30.0, 40.0, 120.0],
            "collision_amount": [0, 1, 0],
            "collision_amount_pedestrian": [0, 1, 0],
            "collision_amount_wall": [0, 0, 0],
            "collision_amount_static": [0, 0, 0],
        },
        schema={
            "success": pl.Boolean,
            "time_to_goal": pl.Float64,
            "collision_amount": pl.Int64,
            "collision_amount_pedestrian": pl.Int64,
            "collision_amount_wall": pl.Int64,
            "collision_amount_static": pl.Int64,
        },
    )
    out = hhi.paper_outcomes(df)
    assert out["success"].to_list() == [True, False, False]
    assert out["time_to_goal"].to_list() == [30.0, None, None]


def test_window_table_and_manipulation_check_use_the_episode_rows_without_contacts(
    tmp_path: pathlib.Path,
) -> None:
    """A real HHI run has no robot collision: the tables must come from the episode rows alone."""
    run = _run(tmp_path)
    summary = read_summary(run)
    assert {
        "hhi_stalls_post",
        "hhi_stall_count",
        "hhi_share_post",
        "hhi_episode_s",
        "hhi_release_s",
    } <= set(summary.columns)
    assert contacts(run).is_empty()

    tables = hhi.analyze(run, n_boot=50)
    assert not tables["windows"].is_empty()
    assert not tables["manipulation_check"].is_empty()
    assert set(tables["family_a"]["metric"].to_list()) == {
        "success",
        "ttg",
        "stalls",
        "passes_between",
    }
    dwb_win = tables["windows"].filter(pl.col("planner") == "dwb").row(0, named=True)
    assert dwb_win["stall_share_post"] == pytest.approx(0.75)
    assert dwb_win["time_share_post"] == pytest.approx(0.25)
    assert dwb_win["excess"] == pytest.approx(0.5)
    contact_hug = (
        tables["manipulation_check"]
        .filter((pl.col("interaction") == "hug") & (pl.col("arm") == "contact"))
        .row(0, named=True)
    )
    assert contact_hug["pair_gap_median"] == pytest.approx(0.6)


def test_stage_regex_round_trips() -> None:
    for kind in hhi.KINDS:
        m = hhi.STAGE_RE.match(hhi.stage_name(kind, 7, "loco"))
        assert m is not None and (m["kind"], int(m["scenario"]), m["arm"]) == (
            kind,
            7,
            "loco",
        )
    assert math.isclose(hhi.MEET_OFFSET * 2, hhi.STANDING_DISTANCE + 0.2)


def test_sampling_gives_up_after_its_tries() -> None:
    with pytest.raises(RuntimeError, match="no admissible geometry"):
        hhi.sample_geometry(np.random.default_rng(0), max_tries=0)


def test_default_paths_sit_beside_the_package() -> None:
    suite = hhi._default_suite_path()
    assert suite.parts[-4:] == ("configs", "suites", "hhi_contact", "suite.yaml")
    assert (suite.parents[3] / "arena_evaluation" / "experiments" / "hhi.py").is_file()
    try:
        scenarios = hhi._default_scenarios_dir()
    except FileNotFoundError:
        pytest.skip(f"world {hhi.MAP} is not in the source tree")
    assert scenarios.parts[-3:] == ("worlds", hhi.MAP, "scenarios")


def test_main_generates_then_analyzes(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        hhi.main(
            [
                "generate",
                "--scenarios",
                "1",
                "--episodes",
                "2",
                "--scenarios-dir",
                str(tmp_path / "scenarios"),
                "--suite",
                str(tmp_path / "suite.yaml"),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "4 stages x 2 episodes per contestant"
    run = _run(tmp_path)
    assert (
        hhi.main(
            [
                "analyze",
                "--benchmark-dir",
                str(run),
                "--boot",
                "50",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "== family_a" in out
    assert (tmp_path / "out" / "family_a.csv").is_file()


def test_window_table_and_manipulation_check_raise_on_missing_columns() -> None:
    empty_missing = pl.DataFrame({"episode_id": pl.Series([], dtype=pl.Int64)})
    with pytest.raises(ValueError, match="window_table missing required columns"):
        hhi.window_table(empty_missing)
    with pytest.raises(ValueError, match="manipulation_check missing required columns"):
        hhi.manipulation_check(empty_missing)


def test_paper_outcomes_preserves_null_collisions_and_success() -> None:
    df = pl.DataFrame(
        {
            "success": pl.Series([True, True, True, False, None], dtype=pl.Boolean),
            "time_to_goal": [10.0, 12.0, 14.0, 15.0, 16.0],
            "collision_amount": pl.Series([0, None, 1, None, 0], dtype=pl.Int64),
        }
    )
    out = hhi.paper_outcomes(df)
    assert out["success"].to_list() == [True, None, False, False, None]
    assert out["time_to_goal"].to_list() == [10.0, None, None, None, None]
