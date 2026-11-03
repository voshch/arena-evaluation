from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


# ---------------------------------------------------------------------------
# Suite.parse
# ---------------------------------------------------------------------------


def _make_stage_dict(**overrides: object) -> dict:
    base: dict = {
        "name": "stage_one",
        "episodes": 5,
        "robot": "turtlebot3_burger",
        "map": "map1",
        "tm_robots": "RANDOM",
        "tm_obstacles": "RANDOM",
        "config": {},
    }
    base.update(overrides)
    return base


def test_suite_parse_happy_path():
    from arena_evaluation.benchmark.config import Suite

    obj = {"stages": [_make_stage_dict()]}
    result = Suite.parse("test_suite", obj)

    assert isinstance(result, Suite)
    assert result.name == "test_suite"
    assert len(result.stages) == 1
    assert isinstance(result.stages[0], Suite.Stage)


def test_suite_parse_multiple_stages():
    from arena_evaluation.benchmark.config import Suite

    obj = {"stages": [_make_stage_dict(name="s1"), _make_stage_dict(name="s2")]}
    result = Suite.parse("multi", obj)

    assert len(result.stages) == 2
    assert result.stages[0].name == "s1"
    assert result.stages[1].name == "s2"


def test_suite_parse_missing_stages_raises():
    from arena_evaluation.benchmark.config import Suite

    with pytest.raises(KeyError):
        Suite.parse("s", {})


def test_suite_parse_launch_args_defaults_empty():
    from arena_evaluation.benchmark.config import Suite

    obj = {"stages": [_make_stage_dict()]}
    result = Suite.parse("test_suite", obj)

    assert result.launch_args == {}


def test_suite_parse_launch_args_from_launch_mapping():
    from arena_evaluation.benchmark.config import Suite

    obj = {"stages": [_make_stage_dict()], "launch": {"sim.isaac.physics": "newton", "headless": True}}
    result = Suite.parse("test_suite", obj)

    assert result.launch_args == {"sim.isaac.physics": "newton", "headless": "True"}


# ---------------------------------------------------------------------------
# Suite.Stage.parse
# ---------------------------------------------------------------------------


def test_stage_parse_happy_path():
    from arena_evaluation.benchmark.config import Suite

    obj = _make_stage_dict(seed=42, timeout="60")
    result = Suite.Stage.parse(obj)

    assert isinstance(result, Suite.Stage)
    assert result.name == "stage_one"
    assert result.episodes == 5
    assert result.robot == "turtlebot3_burger"
    assert result.map == "map1"
    assert result.modes == {"robots": "random", "obstacles": "random"}
    assert result.seed == 42
    assert result.timeout == 60.0


def test_stage_parse_missing_timeout_defaults_to_60():
    from arena_evaluation.benchmark.config import Suite

    obj = _make_stage_dict()
    obj.pop("timeout", None)
    result = Suite.Stage.parse(obj)

    assert result.timeout == 60.0


def test_stage_parse_seed_defaults_to_hash():
    from arena_evaluation.benchmark.config import Suite

    obj = _make_stage_dict()
    obj.pop("seed", None)
    result = Suite.Stage.parse(obj)

    assert isinstance(result.seed, int)


def test_stage_parse_takes_any_case_of_a_vocabulary_name_on_every_axis():
    from arena_runtime.env.task_layer import registered_modes, task_layer

    from arena_evaluation.benchmark.config import Suite

    for axis, names in registered_modes(task_layer()).items():
        for name in names:
            result = Suite.Stage.parse(_make_stage_dict(**{f"tm_{axis}": name.upper()}))
            assert result.modes[axis] == name


def test_stage_parse_modes_mapping_and_legacy_keys_parse_the_same():
    from arena_evaluation.benchmark.config import Suite

    legacy = _make_stage_dict(tm_robots="RANDOM", tm_obstacles="Scenario")
    mapping = {key: value for key, value in legacy.items() if not key.startswith("tm_")}
    mapping["modes"] = {"robots": "RANDOM", "obstacles": "Scenario"}
    assert Suite.Stage.parse(mapping) == Suite.Stage.parse(legacy)


def test_stage_parse_emits_modes_and_no_legacy_keys():
    from arena_evaluation.benchmark.config import Suite

    legacy = _make_stage_dict(tm_robots="RANDOM", tm_obstacles="Scenario")
    mapping = {key: value for key, value in legacy.items() if not key.startswith("tm_")}
    mapping["modes"] = {"robots": "RANDOM", "obstacles": "Scenario"}
    from_legacy = Suite.Stage.parse(legacy)
    from_mapping = Suite.Stage.parse(mapping)

    assert from_legacy.modes == from_mapping.modes == {"robots": "random", "obstacles": "scenario"}
    assert "modes" in from_legacy._asdict()
    assert not [key for key in from_legacy._asdict() if key.startswith("tm_")]


def test_stage_parse_names_an_unknown_axis():
    from arena_evaluation.benchmark.config import Suite

    with pytest.raises(ValueError, match="no_such_axis"):
        Suite.Stage.parse(_make_stage_dict(modes={"no_such_axis": "random"}))


def test_stage_parse_rejects_an_axis_set_twice():
    from arena_evaluation.benchmark.config import Suite

    with pytest.raises(ValueError, match="twice"):
        Suite.Stage.parse(_make_stage_dict(modes={"robots": "random"}))


def test_stage_parse_modules_axis_takes_a_comma_list():
    from arena_runtime.env.task_layer import registered_modes, task_layer

    from arena_evaluation.benchmark.config import Suite

    names = registered_modes(task_layer())["modules"][:2]
    result = Suite.Stage.parse(_make_stage_dict(modes={"modules": ",".join(name.upper() for name in names)}))
    assert result.modes["modules"] == ",".join(names)


def test_stage_seed_hashes_the_legacy_keys_of_the_raw_stage():
    import hashlib
    import json

    from arena_evaluation.benchmark.config import Suite

    raw = {"name": "s", "episodes": 2, "robot": "jackal", "map": "map1", "tm_obstacles": "random", "tm_robots": "scenario", "config": {"scenario": {"file": "x"}}}
    expected = 0x7FFFFFFF & int.from_bytes(hashlib.sha1(json.dumps({k: v for k, v in raw.items() if k != "config"}).encode()).digest()[-4:], byteorder="big")
    assert Suite.Stage.parse(raw).seed == expected


def test_every_shipped_suite_keeps_its_stage_seeds():
    import hashlib
    import json
    import pathlib

    import yaml

    from arena_evaluation.benchmark.config import Suite

    def old_seed(raw: dict) -> int:
        legacy = {key: value for key, value in raw.items() if key not in ("timeout", "timeout_peds", "optim")}
        for key in ("tm_robots", "tm_obstacles"):
            if key in legacy:
                legacy[key] = legacy[key].lower()
        return 0x7FFFFFFF & int.from_bytes(hashlib.sha1(json.dumps({k: v for k, v in legacy.items() if k != "config"}).encode()).digest()[-4:], byteorder="big")

    suites = sorted((pathlib.Path(__file__).parents[1] / "configs" / "benchmark" / "suites").rglob("*.yaml"))
    checked = 0
    for path in suites:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict) or "stages" not in data or "task_layer" in data:
            continue
        for raw, stage in zip(data["stages"], Suite.parse(path.stem, data).stages, strict=True):
            if "seed" not in raw and "modes" not in raw:
                assert stage.seed == old_seed(raw), (path.name, raw["name"])
                checked += 1
    assert checked


def test_stage_parse_invalid_mode_name_raises():
    from arena_evaluation.benchmark.config import Suite

    for key in ("tm_robots", "tm_obstacles"):
        with pytest.raises(ValueError, match="(?i)NOT_A_VALID_MODE"):
            Suite.Stage.parse(_make_stage_dict(**{key: "NOT_A_VALID_MODE"}))


def test_stage_parse_passes_on_the_message_of_the_runtime_validator():
    from arena_runtime.env.task_layer import task_layer

    from arena_evaluation.benchmark.config import Suite

    for modes in ({"robots": ""}, {"obstacles": ""}, {"robots": "random,random"}, {"robots": "NOT_A_VALID_MODE"}):
        stage = _make_stage_dict(modes=modes)
        stage.pop("tm_robots")
        stage.pop("tm_obstacles")
        with pytest.raises(ValueError) as parsed:
            Suite.Stage.parse(stage)
        axis, value = next(iter(modes.items()))
        with pytest.raises(ValueError) as validated:
            task_layer().validate_modes({axis: [name for name in value.split(",") if name]})
        assert str(parsed.value) == str(validated.value)


# ---------------------------------------------------------------------------
# Contest.parse - list form
# ---------------------------------------------------------------------------


def test_contest_parse_list_happy_path():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "planner_a", "mobile.local_planner": "dwa"}]
    result = Contest.parse("my_contest", obj)

    assert isinstance(result, Contest)
    assert result.name == "my_contest"
    assert len(result.contestants) == 1
    c = result.contestants[0]
    assert c.name == "planner_a"
    assert c.args["mobile.local_planner"] == "dwa"


def test_contest_parse_list_multiple_contestants():
    from arena_evaluation.benchmark.config import Contest

    obj = [
        {"name": "p1", "mobile.local_planner": "dwa"},
        {"name": "p2", "mobile.local_planner": "teb"},
    ]
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 2
    assert result.contestants[0].name == "p1"
    assert result.contestants[1].name == "p2"


def test_contest_parse_list_description_is_none():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "p1", "mobile.local_planner": "dwa"}]
    result = Contest.parse("c", obj)

    assert result.description is None


def test_contest_parse_list_missing_name_raises():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"mobile.local_planner": "dwa"}]
    with pytest.raises(ValueError, match="requires 'name'"):
        Contest.parse("c", obj)


def test_contest_parse_list_duplicate_names_raises():
    from arena_evaluation.benchmark.config import Contest

    obj = [
        {"name": "same", "mobile.local_planner": "dwa"},
        {"name": "same", "mobile.local_planner": "teb"},
    ]
    with pytest.raises(ValueError, match="duplicate contestant name"):
        Contest.parse("c", obj)


def test_contest_parse_list_all_keys_except_name_go_to_args():
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "p", "mobile.local_planner": "teb", "mobile.inter_planner": "foo", "mobile.agent": "bar"}]
    result = Contest.parse("c", obj)

    c = result.contestants[0]
    assert c.args == {"mobile.local_planner": "teb", "mobile.inter_planner": "foo", "mobile.agent": "bar"}


# ---------------------------------------------------------------------------
# Contest.parse - list form (dict-cap)
# ---------------------------------------------------------------------------


def test_contest_parse_list_dict_cap():
    """Dict-valued arg stores the cap dict as-is."""
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "teb", "mobile": {"driver": "nav2", "local_planner": "teb"}}]
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 1
    c = result.contestants[0]
    assert c.name == "teb"
    assert c.args["mobile"] == {"driver": "nav2", "local_planner": "teb"}


def test_contest_parse_list_dict_cap_multiple():
    from arena_evaluation.benchmark.config import Contest

    obj = [
        {"name": "a", "mobile": {"driver": "nav2", "local_planner": "teb"}},
        {"name": "b", "mobile": {"driver": "nav2", "local_planner": "dwa"}},
    ]
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 2
    assert result.contestants[0].args["mobile"]["local_planner"] == "teb"
    assert result.contestants[1].args["mobile"]["local_planner"] == "dwa"


def test_contest_parse_list_dict_cap_no_driver():
    """Dict-cap without driver: only sub-keys are emitted."""
    from arena_evaluation.benchmark.config import Contest

    obj = [{"name": "x", "mobile": {"local_planner": "teb"}}]
    result = Contest.parse("c", obj)

    cap = result.contestants[0].args["mobile"]
    assert "driver" not in cap
    assert cap["local_planner"] == "teb"


# ---------------------------------------------------------------------------
# Contest.parse - dict-wrapped list form (name + contestants)
# ---------------------------------------------------------------------------


def test_contest_parse_dict_with_contestants_list():
    from arena_evaluation.benchmark.config import Contest

    obj = {
        "name": "social_planners_core",
        "contestants": [
            {
                "name": "dwb-replan-time",
                "mobile": {
                    "driver": "nav2",
                    "local_planner": "dwb",
                    "inter_planner": "navigate_w_replanning_time",
                },
            },
            {
                "name": "teb-replan-time",
                "mobile": {
                    "driver": "nav2",
                    "local_planner": "teb",
                    "inter_planner": "navigate_w_replanning_time",
                },
            },
        ],
    }
    result = Contest.parse("social_planners_core", obj)

    assert isinstance(result, Contest)
    assert result.name == "social_planners_core"
    assert len(result.contestants) == 2
    assert result.contestants[0].name == "dwb-replan-time"
    assert result.contestants[0].args["mobile"]["local_planner"] == "dwb"
    assert result.contestants[1].name == "teb-replan-time"
    assert result.contestants[1].args["mobile"]["local_planner"] == "teb"


def test_contest_parse_dict_with_contestants_list_and_description():
    from arena_evaluation.benchmark.config import Contest

    obj = {
        "name": "my_contest",
        "description": "A test contest",
        "contestants": [
            {"name": "p1", "mobile": {"driver": "nav2", "local_planner": "dwb"}},
        ],
    }
    result = Contest.parse("fallback_name", obj)

    assert result.name == "my_contest"
    assert result.description == "A test contest"
    assert len(result.contestants) == 1
    assert result.contestants[0].name == "p1"


# ---------------------------------------------------------------------------
# Contest.parse - sweep form
# ---------------------------------------------------------------------------


def test_contest_parse_sweep_one_axis():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "teb"], "mobile.inter_planner": "foo"}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 2
    names = [c.name for c in result.contestants]
    assert names == ["dwa", "teb"]
    assert result.contestants[0].args["mobile.local_planner"] == "dwa"
    assert result.contestants[0].args["mobile.inter_planner"] == "foo"
    assert result.contestants[1].args["mobile.local_planner"] == "teb"
    assert result.contestants[1].args["mobile.inter_planner"] == "foo"


def test_contest_parse_sweep_two_axes():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "teb"], "mobile.global_planner": ["navfn", "smac"]}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 4
    names = [c.name for c in result.contestants]
    assert names == ["dwa-navfn", "dwa-smac", "teb-navfn", "teb-smac"]


def test_contest_parse_sweep_prefix():
    from arena_evaluation.benchmark.config import Contest

    obj = {"name": "foo", "mobile.local_planner": ["dwa", "teb"]}
    result = Contest.parse("c", obj)

    names = [c.name for c in result.contestants]
    assert names == ["foo-dwa", "foo-teb"]


def test_contest_parse_sweep_description_stored_not_in_args():
    from arena_evaluation.benchmark.config import Contest

    obj = {"description": "my desc", "mobile.local_planner": ["dwa", "teb"]}
    result = Contest.parse("c", obj)

    assert result.description == "my desc"
    for c in result.contestants:
        assert "description" not in c.args


def test_contest_parse_sweep_name_not_in_args():
    from arena_evaluation.benchmark.config import Contest

    obj = {"name": "prefix", "mobile.local_planner": ["dwa", "teb"]}
    result = Contest.parse("c", obj)

    for c in result.contestants:
        assert "name" not in c.args


def test_contest_parse_sweep_no_axes_single_contestant():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": "dwa", "mobile.inter_planner": "foo"}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 1
    assert result.contestants[0].name == "single"
    assert result.contestants[0].args["mobile.local_planner"] == "dwa"


def test_contest_parse_sweep_duplicate_names_raises():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "dwa"]}
    with pytest.raises(ValueError, match="duplicate contestant name"):
        Contest.parse("c", obj)


def test_contest_parse_sweep_constants_in_all_args():
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile.local_planner": ["dwa", "teb"], "mobile.inter_planner": "bypass"}
    result = Contest.parse("c", obj)

    for c in result.contestants:
        assert c.args["mobile.inter_planner"] == "bypass"


def test_contest_parse_invalid_type_raises():
    from arena_evaluation.benchmark.config import Contest

    with pytest.raises(ValueError, match="contest must be list or dict"):
        Contest.parse("c", "not_a_valid_type")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contest.parse - sweep form (dict-cap)
# ---------------------------------------------------------------------------


def test_contest_parse_sweep_dict_cap_one_inner_axis():
    """Dict-form cap with one sweep axis inside."""
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile": {"driver": "nav2", "local_planner": ["teb", "dwa"], "inter_planner": "bypass"}}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 2
    names = [c.name for c in result.contestants]
    assert names == ["teb", "dwa"]
    for c in result.contestants:
        cap = c.args["mobile"]
        assert cap["driver"] == "nav2"
        assert cap["inter_planner"] == "bypass"
    assert result.contestants[0].args["mobile"]["local_planner"] == "teb"
    assert result.contestants[1].args["mobile"]["local_planner"] == "dwa"


def test_contest_parse_sweep_dict_cap_two_inner_axes():
    """Dict-form cap with two sweep axes -> cartesian product."""
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile": {"driver": "nav2", "local_planner": ["teb", "dwa"], "inter_planner": ["polite", "aggressive"]}}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 4
    names = [c.name for c in result.contestants]
    assert names == ["teb-polite", "teb-aggressive", "dwa-polite", "dwa-aggressive"]


def test_contest_parse_sweep_dict_cap_no_axes():
    """Dict-form cap with no list values -> single contestant."""
    from arena_evaluation.benchmark.config import Contest

    obj = {"mobile": {"driver": "nav2", "local_planner": "teb"}}
    result = Contest.parse("c", obj)

    assert len(result.contestants) == 1
    assert result.contestants[0].name == "single"
    assert result.contestants[0].args["mobile"] == {"driver": "nav2", "local_planner": "teb"}


def test_contest_parse_sweep_dict_cap_with_prefix():
    from arena_evaluation.benchmark.config import Contest

    obj = {"name": "basic", "mobile": {"driver": "nav2", "inter_planner": "bypass", "local_planner": ["teb", "dwa"]}}
    result = Contest.parse("c", obj)

    names = [c.name for c in result.contestants]
    assert names == ["basic-teb", "basic-dwa"]


def test_shipped_landmarks_suite_and_contest_parse():
    import pathlib

    import yaml

    from arena_evaluation.benchmark.config import Contest, Suite

    configs = pathlib.Path(__file__).parents[1] / "configs" / "benchmark"
    suite = Suite.parse("landmarks", yaml.safe_load((configs / "suites" / "landmarks.yaml").read_text()))
    contest = Contest.parse("landmarks", yaml.safe_load((configs / "contests" / "landmarks.yaml").read_text()))

    assert {stage.map for stage in suite.stages} == {"hospital_1"}
    assert all(stage.config["scenario"]["file"].startswith("landmark_") and stage.modes["robots"] == "scenario" for stage in suite.stages)
    assert len({stage.name for stage in suite.stages}) == len(suite.stages) == 5
    assert {stage.robot for stage in suite.stages} == {"jackal"}
    assert suite.launch_args["lockstep"] == "True" and suite.launch_args["lockstep.paused"] == "False"
    wordings = {c.name: c.args["mobile"].get("wording") for c in contest.contestants}
    assert wordings["nav2"] is None and wordings["omnivla-edge-pose"] is None
    assert {name.rsplit("-", 1)[1] for name, wording in wordings.items() if wording} == {"goal", "route"}
    assert all(name.endswith(wording) for name, wording in wordings.items() if wording)
    cameras = {c.args["mobile"]["planner"]: c.parts["camera"] for c in contest.contestants if c.name != "nav2"}
    assert set(cameras) == {"cma", "navid", "navila", "streamvln", "omnivla", "omnivla-edge", "pi05", "navdp", "iplanner", "viplanner", "socialnav", "citywalker", "xmobility"}
    assert cameras["cma"] == "vln_rgbd" and cameras["navila"] == "vln_navila"
    assert next(c for c in contest.contestants if c.name == "nav2").parts == {}
    assert {c.parts["camera"] for c in contest.contestants if c.args["mobile"].get("config.variant") == "uninavid"} == {"vln_uninavid"}


@pytest.mark.parametrize(("contest", "planners"), [("edge_case_differential", {"cadrl", "her-drl", "sicnav"}), ("edge_case_omnidirectional", {"rlrvo", "kdma", "navistar", "cd-sarl"})])
def test_shipped_edge_case_contests_parse(contest, planners):
    import pathlib

    import yaml

    from arena_evaluation.benchmark.config import Contest

    path = pathlib.Path(__file__).parents[1] / "configs" / "benchmark" / "contests" / f"{contest}.yaml"
    parsed = Contest.parse(contest, yaml.safe_load(path.read_text()))
    assert parsed.contestants[0].name == "nav2" and parsed.contestants[0].args["mobile"]["driver"] == "nav2"
    assert {c.args["mobile"]["planner"] for c in parsed.contestants[1:]} == planners
    assert all(c.args["mobile"]["driver"] == "drl" and c.args["mobile"]["rate"] == 10 for c in parsed.contestants[1:])


def test_a_generated_edge_case_suite_parses():
    from arena_simulation_setup.generative.scenario.suite import build

    from arena_evaluation.benchmark.config import Suite

    suite = Suite.parse("edge", build({"hospital_1": ["normal__u_001", "normal__u_002"]}, robot="jackal", episodes=3, seed=7, timeout="2m"))
    assert [stage.name for stage in suite.stages] == ["hospital_1.normal__u_001", "hospital_1.normal__u_002"]
    assert all(stage.map == "hospital_1" and stage.robot == "jackal" and stage.episodes == 3 and stage.seed == 7 for stage in suite.stages)
    assert {(stage.modes["robots"], stage.modes["obstacles"]) for stage in suite.stages} == {("scenario", "scenario")}
    assert [stage.config["scenario"]["file"] for stage in suite.stages] == ["normal__u_001", "normal__u_002"]
    assert suite.stages[0].timeout == 120.0 and suite.launch_args == {"human": "arena"}


def test_contest_parse_list_parts_stay_out_of_args():
    from arena_evaluation.benchmark.config import Contest

    result = Contest.parse("c", [{"name": "cma", "parts": {"camera": "vln_rgbd"}, "mobile": {"driver": "vla", "planner": "cma"}}])

    assert result.contestants[0].parts == {"camera": "vln_rgbd"}
    assert result.contestants[0].args == {"mobile": {"driver": "vla", "planner": "cma"}}


def test_contest_parse_sweep_shares_parts():
    from arena_evaluation.benchmark.config import Contest

    result = Contest.parse("c", {"name": "navid", "parts": {"camera": "vln_navid"}, "mobile": {"driver": "vla", "planner": "navid", "wording": ["goal", "route"]}})

    assert [c.name for c in result.contestants] == ["navid-goal", "navid-route"]
    assert all(c.parts == {"camera": "vln_navid"} and "parts" not in c.args for c in result.contestants)


def test_contest_parse_parts_rejects_lists():
    from arena_evaluation.benchmark.config import Contest

    with pytest.raises(ValueError, match="parts must map"):
        Contest.parse("c", [{"name": "cma", "parts": {"camera": ["vln_rgbd", "vln_navid"]}}])


def test_robot_with_parts_replaces_items_of_the_same_key():
    from arena_evaluation.benchmark.config import robot_with_parts

    assert robot_with_parts("jackal", {}) == "jackal"
    assert robot_with_parts("jackal", {"camera": "vln_rgbd"}) == "jackal[camera=vln_rgbd]"
    assert robot_with_parts("jackal[camera=vln_navid]", {"camera": "vln_rgbd"}) == "jackal[camera=vln_rgbd]"
    assert robot_with_parts("jackal[2,camera=[a,b],lidar=x],burger", {"camera": "vln_rgbd"}) == "jackal[2,lidar=x,camera=vln_rgbd],burger[camera=vln_rgbd]"


# ---------------------------------------------------------------------------
# _parse_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("60", 60.0), ("60.0", 60.0), ("500ms", 0.5), ("5s", 5.0), ("5m", 300.0), ("1h", 3600.0), ("1h30m", 5400.0)],
)
def test_parse_duration_reads_plain_numbers_and_units(text, seconds):
    from arena_evaluation.benchmark.config import _parse_duration

    assert _parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["not_a_duration", "abc", ""])
def test_parse_duration_rejects_text_without_a_duration(text):
    from arena_evaluation.benchmark.config import _parse_duration

    with pytest.raises(ValueError):
        _parse_duration(text)


def test_suite_parse_metrics_defaults_empty():
    from arena_evaluation.benchmark.config import Suite

    suite = Suite.parse("test_suite", {"stages": [_make_stage_dict()]})
    assert suite.metrics == {}


def test_suite_parse_metrics_max_collisions():
    from arena_evaluation.benchmark.config import Suite

    suite = Suite.parse("test_suite", {"stages": [_make_stage_dict()], "metrics": {"max_collisions": 1}})
    assert suite.metrics == {"max_collisions": 1}


def test_suite_parse_metrics_unknown_key_raises():
    from arena_evaluation.benchmark.config import Suite

    with pytest.raises(ValueError, match="unknown keys"):
        Suite.parse("test_suite", {"stages": [_make_stage_dict()], "metrics": {"max_colisions": 1}})


def test_suite_parse_metrics_nonpositive_raises():
    from arena_evaluation.benchmark.config import Suite

    with pytest.raises(ValueError, match="max_collisions"):
        Suite.parse("test_suite", {"stages": [_make_stage_dict()], "metrics": {"max_collisions": 0}})


def test_suite_parse_metrics_barn_reference():
    from arena_evaluation.benchmark.config import Suite

    barn = {"start": [2.25, 3], "goal": [2.25, 13], "path_length": {"map1": 11.5}}
    suite = Suite.parse("test_suite", {"stages": [_make_stage_dict()], "metrics": {"barn": barn}})
    assert suite.metrics == {"barn": {"start": [2.25, 3.0], "goal": [2.25, 13.0], "path_length": {"map1": 11.5}}}


def test_suite_parse_metrics_barn_reference_junk_raises():
    from arena_evaluation.benchmark.config import Suite

    with pytest.raises(ValueError, match="suite metrics: barn: path_length"):
        Suite.parse("test_suite", {"stages": [_make_stage_dict()], "metrics": {"barn": {"start": [0, 0], "goal": [0, 1], "path_length": {"map1": -1}}}})


def test_stage_fields_are_plain_yaml_once_task_modes_are_values():
    import yaml

    from arena_evaluation.benchmark.config import Suite

    stage = Suite.Stage.parse(_make_stage_dict(timeout="90s"))
    record = stage._asdict()
    assert set(record) == set(Suite.Stage._fields)
    assert yaml.safe_load(yaml.dump({"stage": record}, sort_keys=False))["stage"] == record
    assert record["timeout"] == 90.0
    assert record["map"] == "map1"
