"""Tests for the knowledge tools: get_config_template, inspect_map,
describe_task_mode, describe_metric, planner catalog."""

import pathlib

import pytest

pytest.importorskip("polars")
pytest.importorskip("mcp")
pytest.importorskip("arena_evaluation")
pytest.importorskip("arena_evaluation_mcp")


@pytest.fixture(scope="module")
def bridge():
    from arena_evaluation_mcp.eval_bridge import EvalBridge

    return EvalBridge()


@pytest.fixture(scope="module")
def metrics(bridge):
    try:
        found = bridge.discover_available_metrics()
    except Exception as exc:
        pytest.skip(f"metric registry unavailable: {exc}")
    if not found:
        pytest.skip("no metrics registered")
    return found


class TestConfigTemplates:
    def test_suite_template(self, bridge):
        doc = bridge.read_config_template("suite", "basic")
        if doc is None:
            pytest.skip("bundled suite configs not found")
        assert doc["kind"] == "suite"
        assert "stages" in doc["content"]

    def test_contest_template(self, bridge):
        doc = bridge.read_config_template("contest", "basic")
        if doc is None:
            pytest.skip("bundled contest configs not found")
        assert doc["kind"] == "contest"

    def test_manifest_template(self, bridge):
        doc = bridge.read_config_template("manifest", "standard")
        if doc is None:
            pytest.skip("bundled manifest configs not found")
        assert doc["kind"] == "manifest"
        assert "plots" in doc["content"]

    def test_manifest_with_yaml_suffix(self, bridge):
        doc = bridge.read_config_template("manifest", "standard.yaml")
        if doc is None:
            pytest.skip("bundled manifest configs not found")
        assert doc["name"] == "standard"

    def test_unknown_name_returns_none(self, bridge):
        assert bridge.read_config_template("suite", "zz_no_such_suite") is None

    def test_traversal_name_rejected(self, bridge):
        with pytest.raises(ValueError):
            bridge.read_config_template("manifest", "../secrets")

    def test_list_suites_and_contests(self, bridge):
        suites = bridge.list_suite_stems()
        contests = bridge.list_contest_stems()
        if not suites and not contests:
            pytest.skip("no bundled benchmark configs found")
        assert len(suites) >= 3
        assert len(contests) >= 3

    def test_suites_contests_deduplicated(self, bridge):
        suites = bridge.list_suite_stems()
        contests = bridge.list_contest_stems()
        assert len(suites) == len(set(suites)), "duplicate suite stems!"
        assert len(contests) == len(set(contests)), "duplicate contest stems!"

    def test_write_targets_are_absolute_in_share_dir(self, bridge):
        """create_suite/contest must write where listing + runner read:
        the install share dir - never a relative path."""
        for kind, method in (("suite", bridge.suite_path), ("contest", bridge.contest_path)):
            p = method("zz_never_exists")
            assert p.is_absolute(), f"{kind} write target must be absolute"
            assert "install" in str(p) or "src" in str(p), f"{kind} write target should resolve in install/source tree"
            assert "configs/benchmark" in str(p)

    def test_write_target_traversal_rejected(self, bridge):
        with pytest.raises(ValueError):
            bridge.suite_path("../../evil")
        with pytest.raises(ValueError):
            bridge.contest_path("/etc/passwd")


@pytest.mark.skipif(
    not pathlib.Path("/opt/arena_ws/source").exists(),
    reason="arena workspace not installed",
)
class TestCliWrapper:
    def test_cli_wrapper_finds_arena_function(self, bridge):
        """arena is a shell function, not an executable - the wrapper must
        source the environment so the CLI is reachable."""
        r = bridge.run_cli("--help", timeout=60)
        assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr[:200]}"
        assert "Usage: arena evaluation" in (r.stdout or "")


class TestRunBenchmark:
    def test_default_launch_config(self):
        from arena_evaluation_mcp.tools import _benchmark_cmd_args

        args = _benchmark_cmd_args({"suite": "s", "contest": "c"})
        assert "sim:=gazebo" in args
        assert "headless:=true" in args
        assert "env.n:=2" in args
        assert "optim.obstacles:=bbox" in args

    def test_launch_config_overridable(self):
        from arena_evaluation_mcp.tools import _benchmark_cmd_args

        args = _benchmark_cmd_args(
            {
                "suite": "s",
                "contest": "c",
                "sim": "isaac",
                "headless": False,
                "env_n": 4,
                "optim_obstacles": "full",
            }
        )
        assert "sim:=isaac" in args
        assert "headless:=false" in args
        assert "env.n:=4" in args
        assert "optim.obstacles:=full" in args

    def test_extra_passthrough(self):
        from arena_evaluation_mcp.tools import _benchmark_cmd_args

        args = _benchmark_cmd_args(
            {
                "suite": "s",
                "contest": "c",
                "extra_passthrough": {"task.fail_on_collision": True, "complexity": 2},
            }
        )
        assert "task.fail_on_collision:=True" in args
        assert "complexity:=2" in args

    def test_command_shape(self):
        from arena_evaluation_mcp.tools import _benchmark_cmd_args

        args = _benchmark_cmd_args(
            {
                "suite": "my_suite",
                "contest": "my_contest",
                "scale_episodes": 2.0,
                "run_id": "run123",
            }
        )
        assert args[0] == "benchmark"
        assert "--suite" in args and args[args.index("--suite") + 1] == "my_suite"
        assert "--contest" in args
        assert "--scale-episodes" in args
        assert "--run-id" in args
        assert "headless:=true" in args


class TestInspectMap:
    def test_known_map_returns_zones_and_scenarios(self, bridge):
        info = bridge.inspect_map("hospital_1")
        if "error" in info:
            pytest.skip(f"hospital_1 unavailable: {info['error']}")
        assert len(info.get("zones", [])) >= 5
        assert "map_bounds" in info
        assert len(info.get("scenarios", [])) >= 1
        # Scenario should carry a robot route with coordinates
        routes = [s for s in info["scenarios"] if s.get("robot")]
        assert routes, "expected at least one scenario with a robot route"

    def test_map_metadata_present(self, bridge):
        info = bridge.inspect_map("map_empty")
        if "error" in info:
            pytest.skip(f"map_empty unavailable: {info['error']}")
        assert info.get("map_metadata", {}).get("resolution_m_per_px") is not None

    def test_unknown_map_returns_error(self, bridge):
        info = bridge.inspect_map("zz_no_such_map")
        assert "error" in info


class TestDescribeTaskMode:
    def test_full_catalog(self, bridge):
        ref = bridge.describe_task_mode()
        assert "tm_robots" in ref["catalog"]
        assert "tm_obstacles" in ref["catalog"]
        assert "pedestrian_models" in ref
        assert ref["pedestrian_models"]["bundled"]  # non-empty

    def test_guidance_present(self, bridge):
        ref = bridge.describe_task_mode()
        assert "guidance" in ref
        assert "same_path_across_stages" in ref["guidance"]
        assert "seed" in ref["guidance"]["same_path_across_stages"]
        assert "construction" in ref["guidance"]
        assert "NOT infer" in ref["guidance"]["construction"]

    def test_random_mode_example_has_seed(self, bridge):
        ref = bridge.describe_task_mode("random")
        assert "random" in ref
        assert "example" in ref["random"]
        assert "seed: 42" in ref["random"]["example"]

    def test_scenario_mode_documented(self, bridge):
        ref = bridge.describe_task_mode("scenario")
        assert "scenario" in ref
        assert "file" in ref["scenario"]["config_keys"]
        assert "purpose" in ref["scenario"]
        assert "example" in ref["scenario"]

    def test_unknown_mode_returns_error(self, bridge):
        ref = bridge.describe_task_mode("zz_mode")
        assert "error" in ref


class TestDescribeMetric:
    def test_known_metric(self, bridge, metrics):
        from arena_evaluation_mcp.tools import _describe_metric

        result = _describe_metric({"metric_name": "energy"}, bridge)
        assert "error" not in result
        assert result["category"] == "ecological"
        assert "outputs" in result
        assert "lower_is_better" in result
        assert result["lower_is_better"] is True  # energy_total_wh is lower-better

    def test_unknown_metric(self, bridge, metrics):
        from arena_evaluation_mcp.tools import _describe_metric

        result = _describe_metric({"metric_name": "zz_metric"}, bridge)
        assert "error" in result
        assert "available" in result

    def test_directions_declared_by_metric(self, bridge, metrics):
        """Directions must come from the calculator declarations - e.g.
        success is 'higher', time_to_goal 'lower'."""
        from arena_evaluation_mcp.tools import _describe_metric

        coll = _describe_metric({"metric_name": "collision_metrics"}, bridge)
        assert coll["outputs"]["success"]["lower_is_better"] is False
        assert coll["outputs"]["collision_amount"]["lower_is_better"] is True

        eff = _describe_metric({"metric_name": "path_efficiency"}, bridge)
        assert eff["outputs"]["path_efficiency"]["lower_is_better"] is False

    def test_default_compare_metrics_from_declarations(self, bridge, metrics):
        """Default comparison metrics = union of PRIMARY_OUTPUTS declared by
        each calculator - no hardcoded list."""
        from arena_evaluation_mcp.tools import _default_compare_metrics

        defaults = _default_compare_metrics(bridge)
        assert "success" in defaults
        assert "time_to_goal" in defaults
        assert "energy_total_wh" in defaults
        # timeseries-only calculators contribute nothing
        assert not any("timeseries" in m for m in defaults)


class TestPlannerCatalog:
    def test_catalog_shapes(self, bridge):
        cat = bridge.planner_catalog()
        assert cat["drivers"]  # non-empty
        assert "nav2" in cat["drivers"]
        assert "mobile" in cat["cap_keys"]
        if not cat["local_planners"]:
            pytest.skip("arena_robots nav2 catalog not installed")
        assert cat["inter_planners"]  # non-empty


class TestBenchmarkFilters:
    def test_filters_subset_of_unfiltered(self, bridge):
        all_runs = bridge.list_benchmarks()
        if len(all_runs) < 2:
            pytest.skip("need >=2 runs to test filters")
        by_suite = bridge.list_benchmarks(suite=all_runs[0]["suite"])
        assert all(r["suite"] == all_runs[0]["suite"] for r in by_suite)
        by_query = bridge.list_benchmarks(query=all_runs[0]["run_id"][:8])
        assert all(all_runs[0]["run_id"][:8] in r["run_id"] for r in by_query)
        assert len(by_query) >= 1

    def test_status_filter_consistent(self, bridge):
        all_runs = bridge.list_benchmarks()
        if not all_runs:
            pytest.skip("no runs to test")
        completed = bridge.list_benchmarks(status="completed")
        assert all(r["status"] == "completed" for r in completed)
        # A completed run in the unfiltered list must appear in the filtered one
        done = [r for r in all_runs if r["status"] == "completed"]
        if done:
            assert done[0]["run_id"] in {r["run_id"] for r in completed}


class TestModelWarnings:
    def test_unknown_pedestrian_model_warns(self, bridge):
        from arena_evaluation_mcp.tools import _warn_unknown_models

        warnings = _warn_unknown_models(
            """
stages:
  - name: s1
    map: hospital_1
    robot: jackal
    tm_robots: random
    tm_obstacles: random
    episodes: 1
    config:
      random:
        dynamic: {min: 1, max: 2, models: [arenien]}
""",
            bridge,
        )
        assert len(warnings) == 1
        assert "arenien" in warnings[0]
        assert "arenian" in warnings[0]  # mentions the fallback

    def test_known_models_no_warnings(self, bridge):
        from arena_evaluation_mcp.tools import _warn_unknown_models

        warnings = _warn_unknown_models(
            """
stages:
  - name: s1
    map: hospital_1
    robot: jackal
    tm_robots: random
    tm_obstacles: random
    episodes: 1
    config:
      random:
        dynamic: {min: 1, max: 2, models: [arenian]}
""",
            bridge,
        )
        assert warnings == []

    def test_validate_suite_includes_model_warnings(self, bridge):
        from arena_evaluation_mcp.tools import _dispatch

        result = _dispatch(
            "validate_suite",
            {
                "yaml_content": """
stages:
  - name: s1
    map: hospital_1
    robot: jackal
    tm_robots: random
    tm_obstacles: random
    episodes: 1
    config:
      random:
        dynamic: {min: 1, max: 2, models: [bogus_ped]}
""",
            },
            bridge,
        )
        if not result.get("valid"):
            pytest.skip(f"suite schema validation unavailable: {result.get('error')}")
        assert result["model_warnings"], "expected model warnings"
        assert any("bogus_ped" in w for w in result["model_warnings"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
