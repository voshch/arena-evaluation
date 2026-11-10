from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_assets.importers import IMPORTERS

from arena_evaluation.adapters import sidewalkbench
from arena_evaluation.adapters.sidewalkbench import BEHAVIORS, SidewalkBench

BINDING = {"modes": {"robots": "scenario", "obstacles": "scenario"}, "params": {"robots": {"file": "route"}, "obstacles": {"file": "route"}}}
SCENARIOS = [("straight_0", 30.0), ("straight_1", 31.0), ("curve_0", 40.0), ("crosswalk_0", 35.0), ("long_0", 120.0), ("long_1", 200.0)]
BLOCK_ROUTE = {"robots": [{"start": [0.0, 0.0, 0.0], "phases": [{"goto": [150.0, 0.0, 0.0]}]}]}


def _stages() -> list[dict]:
    stages = []
    for name, layout, _, _ in sidewalkbench.generated_specs():
        timeout = sidewalkbench.route_budget_s(sidewalkbench.route_length(BLOCK_ROUTE)) if layout == "block" else sidewalkbench.route_budget_s(30.0)
        stages.append(sidewalkbench.stage(name, name, BINDING, timeout))
    return stages + sidewalkbench.scene_stages("ucla_crosswalk", SCENARIOS)


def _importer() -> SidewalkBench:
    return SidewalkBench(cache=Path("/nonexistent"), provider=object())


def test_registry_loads_this_class():
    assert IMPORTERS.load("suite", "sidewalkbench") is SidewalkBench
    assert SidewalkBench.meta.kind == "suite"
    assert list(_importer().items()) == ["sidewalkbench"]


def test_generated_specs_cover_layouts_behaviors_and_seeds():
    specs = sidewalkbench.generated_specs()
    assert len(specs) == 60
    assert len({name for name, *_ in specs}) == 60
    assert {seed for *_, seed in specs} == {42, 43, 44, 45, 46}
    assert {(layout, scenario) for _, layout, scenario, _ in specs} == {
        ("straight", "empty"),
        ("corner", "empty"),
        ("crosswalk", "empty"),
        ("block", "default"),
        *((("crosswalk" if b == "crossing" else "straight"), b) for b in BEHAVIORS),
    }
    assert ("sidewalk_straight_42", "straight", "empty", 42) in specs
    assert ("sidewalk_crosswalk_crossing_46", "crosswalk", "crossing", 46) in specs
    assert ("sidewalk_block_44", "block", "default", 44) in specs


def test_generator_configurations_use_the_declared_fields():
    assert sidewalkbench.generated_config("straight", "empty") == {"layout": "straight", "scenario": "empty", "length": 25.0}
    assert sidewalkbench.generated_config("straight", "frontal") == {"layout": "straight", "scenario": "frontal", "length": 30.0}
    assert sidewalkbench.generated_config("block", "default") == {"layout": "block", "scenario": "default", "blocks": 2, "pedestrians": 10}


def test_long_budget_derives_from_route_length_at_the_nominal_speed():
    assert sidewalkbench.route_budget_s(100.0) == 260
    assert sidewalkbench.route_budget_s(120.0) == 310
    assert sidewalkbench.route_budget_s(30.0) == 90
    assert sidewalkbench.route_budget_s(0.0) == 60


def test_route_length_follows_start_and_pose_gotos_only():
    scenario = {"robots": [{"start": [0.0, 0.0, 0.0], "phases": [{"goto": [3.0, 4.0, 0.0]}, {"goto": "kitchen"}, {"goto": [3.0, 14.0, 0.0]}]}]}
    assert sidewalkbench.route_length(scenario) == pytest.approx(15.0)


def test_route_length_reads_mapping_poses():
    scenario = {"robots": [{"start": {"x": 0.0, "y": 0.0, "z": 0.5, "yaw": 0.0}, "phases": [{"goto": {"x": 3.0, "y": 4.0}}, {"goto": [3.0, 14.0, 0.0]}]}]}
    assert sidewalkbench.route_length(scenario) == pytest.approx(15.0)


def test_scene_stages_give_every_route_a_budget_from_its_length():
    stages = {s["name"]: s for s in sidewalkbench.scene_stages("ucla_crosswalk", SCENARIOS)}
    assert stages["ucla_crosswalk_straight_0"] == {
        "name": "ucla_crosswalk_straight_0",
        "map": "sidewalkbench_ucla_crosswalk",
        "robot": "jackal",
        "episodes": 1,
        "modes": {"robots": "scenario", "obstacles": "scenario"},
        "config": {"scenario": {"file": "straight_0"}},
        "timeout": "90s",
    }
    assert stages["ucla_crosswalk_crosswalk_0"]["timeout"] == "100s"
    assert stages["ucla_crosswalk_long_0"]["timeout"] == "310s"
    assert stages["ucla_crosswalk_long_1"]["timeout"] == "510s"


def test_stage_merges_both_axes_into_one_scenario_block():
    built = sidewalkbench.stage("a", "b", BINDING, 60)
    assert built["config"] == {"scenario": {"file": "route"}}
    assert built["modes"] == {"robots": "scenario", "obstacles": "scenario"}


def test_suite_validates_with_the_runner_parser():
    from arena_evaluation.mcp.authoring import check_suite

    verdict = check_suite({"metrics": {"max_collisions": 1}, "stages": _stages()})
    assert verdict["problems"] == []
    assert verdict["valid"] is True
    assert verdict["n_stages"] == 60 + len(SCENARIOS)
    assert [w for w in verdict["warnings"] if "does not resolve locally" not in w] == []


def test_contest_validates_and_pins_the_collision_rule():
    from arena_runtime.env.constants.runtime import EPISODE_PARAMS

    from arena_evaluation.mcp.authoring import check_contest

    contest = [yaml.safe_load(yaml.safe_dump(sidewalkbench.CONTESTANT))]
    verdict = check_contest(contest)
    assert verdict["problems"] == []
    assert verdict["valid"] is True
    assert contest[0]["task.episode.fail_on_collision"] is True
    assert {key for key in contest[0] if key.startswith("task.")} <= set(EPISODE_PARAMS)


def test_classify_reads_layout_and_behavior_from_the_stage_name():
    assert sidewalkbench.classify("sidewalk_straight_42") == ("straight", None)
    assert sidewalkbench.classify("sidewalk_corner_43") == ("corner", None)
    assert sidewalkbench.classify("sidewalk_straight_obstructing_42") == ("straight", "obstructing")
    assert sidewalkbench.classify("sidewalk_crosswalk_crossing_42") == ("crosswalk", "crossing")
    assert sidewalkbench.classify("sidewalk_block_42") == ("long", None)
    assert sidewalkbench.classify("ucla_sidewalk_backward_curve_3") == ("corner", None)
    assert sidewalkbench.classify("ucla_crosswalk_long_0") == ("long", None)


def test_manifest_validates_and_selects_the_right_stages():
    from arena_evaluation.mcp.authoring import check_manifest
    from arena_evaluation.processing.metrics.performance.sidewalk_metrics import SidewalkMetricsCalculator

    manifest = sidewalkbench.manifest([s["name"] for s in _stages()])
    verdict = check_manifest(manifest)
    assert verdict["problems"] == []
    assert verdict["valid"] is True
    plots = {plot["id"]: plot for plot in manifest["plots"]}
    layouts = {f"sidewalkbench_layout_{g}" for g in ("straight", "corner", "crosswalk", "long")}
    assert set(plots) == layouts | {f"sidewalkbench_behavior_{b}" for b in BEHAVIORS} | {"sidewalkbench_long_horizon"}
    straight = plots["sidewalkbench_layout_straight"]["filter"]["stage"]
    assert "sidewalk_straight_42" in straight
    assert "ucla_crosswalk_straight_1" in straight
    assert not any("obstructing" in n or "curve" in n for n in straight)
    assert plots["sidewalkbench_behavior_crossing"]["filter"]["stage"] == [f"sidewalk_crosswalk_crossing_{s}" for s in range(42, 47)]
    assert plots["sidewalkbench_layout_corner"]["filter"]["stage"] == [f"sidewalk_corner_{s}" for s in range(42, 47)] + ["ucla_crosswalk_curve_0"]
    assert set(plots["sidewalkbench_long_horizon"]["filter"]["stage"]) == {f"sidewalk_block_{s}" for s in range(42, 47)} | {"ucla_crosswalk_long_0", "ucla_crosswalk_long_1"}
    assert {plot["options"]["group_by"][0] for plot in manifest["plots"]} == {"planner"}
    used = {column["metric"] for plot in manifest["plots"] for column in plot["options"]["columns"]} | {column["metric"] for column in manifest["summary"]}
    assert used <= set(SidewalkMetricsCalculator.output_keys())


def test_attribution_names_the_paper_and_both_licenses():
    attribution = _importer().attribution("sidewalkbench")
    assert attribution.url == "https://github.com/VAIL-UCLA/SidewalkBench"
    assert "Apache 2.0" in attribution.license
    assert "MIT" in attribution.license
    assert attribution.authors.count(",") == 6
    assert "2606.16953" in attribution.citation


def test_unknown_item_is_rejected(tmp_path: Path):
    with pytest.raises(KeyError):
        SidewalkBench(cache=tmp_path, provider=object()).convert("barn", tmp_path, tmp_path / "out")


def test_convert_bundles_the_generated_worlds_and_validates_the_bundle(tmp_path: Path):
    from arena_simulation_setup.generative.world import WorldGeneratorType

    if "SIDEWALK" not in WorldGeneratorType.__members__:
        pytest.skip("sidewalk world generator not available")
    out = tmp_path / "out"
    assert SidewalkBench(cache=tmp_path / "cache", provider=object(), scenes=[]).convert("sidewalkbench", tmp_path, out) == "suites/sidewalkbench"
    suite = yaml.safe_load((out / "suites" / "sidewalkbench" / "suite.yaml").read_text())
    assert suite["launch"] == {"human": "arena"}
    assert [stage["name"] for stage in suite["stages"]] == [name for name, *_ in sidewalkbench.generated_specs()]
    assert sorted(path.name for path in (out / "suites" / "sidewalkbench" / "worlds").iterdir()) == sorted(stage["map"] for stage in suite["stages"])
    budgets = {stage["name"]: int(stage["timeout"].removesuffix("s")) for stage in suite["stages"]}
    assert all(budgets[sidewalkbench.world_name(layout, seed)] == 80 for layout in sidewalkbench.UNIT_LAYOUTS if layout != "corner" for seed in sidewalkbench.SEEDS)
    assert all(budgets[sidewalkbench.world_name("straight", seed, "frontal")] == 90 for seed in sidewalkbench.SEEDS)
    assert all(budgets[sidewalkbench.world_name("block", seed)] > 260 for seed in sidewalkbench.SEEDS)
    assert yaml.safe_load((out / "contests" / "sidewalkbench" / "contest.yaml").read_text()) == [sidewalkbench.CONTESTANT]
