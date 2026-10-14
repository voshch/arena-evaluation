from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

pl = pytest.importorskip("polars")

from arena_evaluation.experiments import beckon


def test_robot_loops_the_route_until_the_call_is_seen_or_the_route_passes_the_caller() -> None:
    loop, serve = beckon.robot_phases()
    assert [step["goto"] for step in loop["loop"]] == [list(p) for p in beckon.ROUTE]
    assert {step["tolerance_radius"] for step in loop["loop"]} == {beckon.CHECKPOINT_RADIUS_M}
    assert loop["until"] == "robot sees caller within 12 and caller plays beckon for 0.5 or robot within 2.5 of caller"
    assert serve == {"goto": "caller", "tolerance_radius": 2.5, "hold_time": 0.0}


def test_caller_stands_then_calls_with_its_gaze_on_the_robot() -> None:
    agent = beckon.caller_agent_yaml("burger")
    seq = agent["sequences"]["call_seq"]
    assert (agent["name"], agent["initial_sequence"], seq["then"]) == ("caller", "call_seq", "call_seq")
    assert list(seq["steps"]) == ["wait", "call"]
    assert seq["steps"]["wait"] == {"attention": {"posture": "standing"}, "duration": {"mean": 5.0, "std": 0.0}}
    assert seq["steps"]["call"]["attention"] == {"clip": "beckon", "gaze": ["robot:burger", "robot:robot_0", "robot"]}
    assert seq["steps"]["call"]["duration"]["mean"] > beckon.TIMEOUT_S


def test_scenario_places_the_caller_at_its_pose_with_a_single_waypoint() -> None:
    scenario = beckon.scenario_yaml((9.0, 16.0, 0.0))
    assert scenario["robots"] == [{"start": [10.0, 2.0, 1.5708], "phases": beckon.robot_phases()}]
    (caller,) = scenario["dynamic"]
    assert (caller["name"], caller["model"], caller["pose"]) == ("caller", beckon.MODEL, [9.0, 16.0, 0.0])
    assert caller["agent"] == {"agent_type": "./caller.yaml"}
    assert caller["waypoints"] == [[9.0, 16.0]]


def test_route_samples_walk_the_patrol_line_at_half_meter_spacing() -> None:
    samples = beckon._route_samples()
    assert samples[0] == (10.0, 2.0)
    assert samples[-1] == (10.0, 2.0)
    assert (10.0, 18.0) in samples
    assert len(samples) == 2 * 33
    assert {x for x, _ in samples} == {10.0}
    assert np.diff([y for _, y in samples[:33]]).tolist() == pytest.approx([0.5] * 32)


def _synthetic_world(tmp_path: pathlib.Path) -> pathlib.Path:
    from PIL import Image

    world = tmp_path / "world"
    (world / "0").mkdir(parents=True)
    grid = np.full((100, 100), 255, dtype=np.uint8)
    grid[:61, 50] = 0
    Image.fromarray(grid, mode="L").save(world / "0" / "map.png")
    (world / "0" / "map.yaml").write_text(yaml.safe_dump({"image": "map.png", "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}))
    return world


def test_map_rejects_walls_and_borders_within_the_clearance(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("PIL")
    world_map = beckon._Map(_synthetic_world(tmp_path))
    assert world_map.free(2.0, 2.0)
    assert world_map.free(5.3, 2.0)
    assert not world_map.free(5.3, 8.0)
    assert world_map.free(5.3, 8.0, clearance=0.2)
    assert not world_map.free(0.3, 2.0)


def test_line_of_sight_is_cut_by_the_wall_and_the_map_edge(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("PIL")
    world_map = beckon._Map(_synthetic_world(tmp_path))
    assert world_map.line_of_sight((2.0, 2.0), (8.0, 2.0))
    assert not world_map.line_of_sight((2.0, 8.0), (8.0, 8.0))
    assert not world_map.line_of_sight((2.0, 2.0), (2.0, 12.0))


def test_route_window_names_the_visible_stretch_and_the_closest_approach(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("PIL")
    world_map = beckon._Map(_synthetic_world(tmp_path))
    samples = [(2.0, float(y)) for y in range(1, 10)]
    assert world_map.route_window(samples, (8.0, 2.0), 7.0) == ("y=1.0..5.0", 6.0)
    assert world_map.route_window(samples, (8.0, 2.0), 5.0) == ("nowhere", 6.0)


def test_generate_writes_the_scenarios_and_the_paired_suites(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("PIL")
    try:
        beckon._world_dir()
    except FileNotFoundError:
        pytest.skip(f"world {beckon.WORLD} is not in the source tree")
    (tmp_path / "suites").mkdir()
    suite = beckon.generate(scenarios_dir=tmp_path / "scenarios", suite_path=tmp_path / "suites" / "beckon_service.yaml", episodes=1)

    stages = suite["stages"]
    assert [s["name"] for s in stages] == [beckon.stage_name(case, arm) for case in beckon.CALLERS for arm in beckon.ARMS]
    by_name = {s["name"]: s for s in stages}
    for index, (case, (pose, _)) in enumerate(beckon.CALLERS.items()):
        on, off = by_name[beckon.stage_name(case, "on")], by_name[beckon.stage_name(case, "off")]
        assert on["seed"] == off["seed"] == beckon.SEED_BASE + index
        assert (on["config"]["scenario"]["gesture_mode"], off["config"]["scenario"]["gesture_mode"]) == ("enabled", "disabled")
        assert on["config"]["scenario"]["file"] == off["config"]["scenario"]["file"] == f"beckon_{case}"
        assert (on["map"], on["robot"], on["episodes"], on["timeout"]) == (beckon.WORLD, beckon.ROBOT, 1, "90s")
        directory = tmp_path / "scenarios" / f"beckon_{case}"
        assert yaml.safe_load((directory / "scenario.yaml").read_text()) == beckon.scenario_yaml(pose)
        assert yaml.safe_load((directory / "caller.yaml").read_text()) == beckon.caller_agent_yaml()
    assert yaml.safe_load((tmp_path / "suites" / "beckon_service.yaml").read_text()) == suite
    for arm in beckon.ARMS:
        arm_suite = yaml.safe_load((tmp_path / "suites" / f"beckon_{arm}.yaml").read_text())
        assert [s["name"] for s in arm_suite["stages"]] == [beckon.stage_name(case, arm) for case in beckon.CALLERS]
    out = capsys.readouterr().out
    assert "beckon_hall_near" in out and "closest approach" in out
    assert "6 stages x 1 episodes per contestant" in out


def _metrics(planners: tuple[str, ...] = ("dwb",), cases: tuple[str, ...] = ("hall_near", "ward_door"), episodes: int = 2) -> pl.DataFrame:
    rows = []
    episode = 0
    for planner in planners:
        for case in cases:
            for arm in ("on", "off"):
                for seed in range(episodes):
                    served = arm == "on" or case == "hall_near"
                    rows.append(
                        {
                            "stage": beckon.stage_name(case, arm),
                            "planner": planner,
                            "episode": episode,
                            "status_reason": None if served else "timeout",
                            "success": 1 if served else 0,
                            "time_to_goal": (20.0 if arm == "on" else 40.0) + seed if served else None,
                        }
                    )
                    episode += 1
    rows.append({"stage": "other_stage_on", "planner": "dwb", "episode": episode, "status_reason": None, "success": 1, "time_to_goal": 1.0})
    return pl.DataFrame(rows, infer_schema_length=None)


def test_load_episodes_tags_case_arm_and_seed_and_drops_other_stages() -> None:
    df = beckon.load_episodes(_metrics())
    assert set(df["case"]) == {"hall_near", "ward_door"} and set(df["arm"]) == {"on", "off"}
    assert df.filter(pl.col("stage") == "other_stage_on").height == 0
    assert df.group_by(["planner", "stage"]).agg(pl.col("seed").sort())["seed"].to_list() == [[0, 1]] * 4


def test_paper_outcomes_censor_unserved_calls_at_the_longest_episode() -> None:
    df = pl.DataFrame({"stage": ["a", "a", "a"], "status_reason": [None, "timeout", None], "success": [1, 0, 0], "time_to_goal": [25.0, None, 95.0]}, infer_schema_length=None)
    out = beckon.paper_outcomes(df)
    assert out["served"].to_list() == [True, False, False]
    assert out["censored"].to_list() == [False, True, True]
    assert out["censor_at_s"].to_list() == [95.0] * 3
    assert out["wait_s"].to_list() == [20.0, 90.0, 90.0]


def test_paper_outcomes_censor_at_the_stage_timeout_when_known() -> None:
    df = pl.DataFrame({"stage": ["cp", "cp", "hall"], "status_reason": [None, "timeout", "timeout"], "success": [1, 0, 0], "time_to_goal": [150.0, None, None]}, infer_schema_length=None)
    out = beckon.paper_outcomes(df, {"cp": 180.0})
    assert out["censor_at_s"].to_list() == [180.0, 180.0, 150.0]
    assert out["wait_s"].to_list() == [145.0, 175.0, 145.0]


def test_stage_timeouts_come_from_the_run_manifest(tmp_path: pathlib.Path) -> None:
    assert beckon.stage_timeouts(tmp_path) == {}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({"suite": {"stages": [{"name": "beckon_checkpoint_on", "timeout": "180s"}, {"name": "quick", "timeout": "2m"}, {"name": "untimed"}]}}))
    assert beckon.stage_timeouts(tmp_path) == {"beckon_checkpoint_on": 180.0, "quick": 120.0}


def test_case_table_pairs_the_arms_by_seed_and_adjusts_within_the_family() -> None:
    episodes = beckon.paper_outcomes(beckon.load_episodes(_metrics(episodes=4)))
    table = beckon.case_table(episodes, n_boot=200).sort("case")
    assert table["case"].to_list() == ["hall_near", "ward_door"]
    assert table["n_pairs"].to_list() == [4, 4]
    assert table["wait_on"].to_list() == pytest.approx([16.5, 16.5])
    assert table["wait_off"].to_list() == pytest.approx([36.5, 85.0])
    assert table["served_off"].to_list() == [1.0, 0.0]
    assert table["diff"].to_list() == pytest.approx([-20.0, -68.5])
    assert table["p_holm"].to_list() == [0.0, 0.0]


def test_case_table_leaves_an_unpaired_case_without_a_difference() -> None:
    episodes = beckon.paper_outcomes(beckon.load_episodes(_metrics()))
    one_armed = episodes.filter(~((pl.col("case") == "ward_door") & (pl.col("arm") == "off")))
    ward = beckon.case_table(one_armed, n_boot=50).filter(pl.col("case") == "ward_door").row(0, named=True)
    assert (ward["n_pairs"], ward["diff"], ward["p"], ward["p_holm"], ward["wait_off"]) == (0, None, None, None, None)
    table = beckon.case_table(episodes.filter(pl.col("arm") == "on"), n_boot=50)
    assert "p_holm" not in table.columns and table["n_pairs"].to_list() == [0, 0]


def test_planner_table_pools_the_paired_differences_over_cases() -> None:
    episodes = beckon.paper_outcomes(beckon.load_episodes(_metrics(planners=("dwb", "mppi"), episodes=4)))
    table = beckon.planner_table(episodes, n_boot=200).sort("planner")
    assert table["planner"].to_list() == ["dwb", "mppi"]
    assert table["n_cases"].to_list() == [2, 2]
    assert table["diff"].to_list() == pytest.approx([-44.25, -44.25])
    assert table["served_on"].to_list() == [1.0, 1.0] and table["served_off"].to_list() == [0.5, 0.5]
    assert all(row["ci_high"] < 0 for row in table.iter_rows(named=True))


def test_analyze_reads_the_processed_run_and_writes_both_tables(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="combined_metrics.parquet"):
        beckon.analyze(run)
    _metrics().write_parquet(run / "combined_metrics.parquet")
    tables = beckon.analyze(run, n_boot=50, out_dir=tmp_path / "out")
    assert set(tables) == {"cases", "planners"}
    assert pl.read_csv(tmp_path / "out" / "cases.csv")["case"].sort().to_list() == ["hall_near", "ward_door"]
    assert pl.read_csv(tmp_path / "out" / "planners.csv")["planner"].to_list() == ["dwb"]
    out = capsys.readouterr().out
    assert "== cases" in out and "== planners" in out


def test_main_analyze_returns_zero_after_printing_the_tables(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _metrics().write_parquet(run / "combined_metrics.parquet")
    assert beckon.main(["analyze", "--benchmark-dir", str(run), "--boot", "50", "--out", str(tmp_path / "out")]) == 0
    assert "== planners" in capsys.readouterr().out
    assert (tmp_path / "out" / "planners.csv").is_file()
