from __future__ import annotations

import os
import pathlib
import subprocess
import time

import pytest

from arena_evaluation.benchmark.config import Contest, Suite
from arena_evaluation.benchmark.runner import (
    _all_steps,
    _all_steps_grid,
    _default_run_id,
    _is_inline_contest,
    _is_inline_suite,
    _is_substantive_log_line,
    _load_suite_contest,
    _preflight_contest,
    _proc_starttime,
    _proc_tree,
    _resolve_contest_source,
    _resolve_suite_source,
    _shared_lane,
    _suite_bundle_dir,
    _walk_dict,
    _warn_config_drift,
    build_launch_args,
    build_pending,
    group_pending,
    resolve_planner_identity,
    robot_has_depth_camera,
)
from arena_evaluation.benchmark.state import Manifest, RunDir, compute_config_hash
from arena_evaluation.benchmark.step import Step, StepErrorKind, StepResult
from rcl_interfaces.msg import ParameterType
from task_generator.constants import Constants

RANDOM_ROBOTS = Constants.TaskMode.TM_Robots.RANDOM
RANDOM_OBSTACLES = Constants.TaskMode.TM_Obstacles.RANDOM


def _stage(name: str = "s1", *, episodes: int = 4, robot: str = "jackal", world: str = "map1", optim: dict | None = None) -> Suite.Stage:
    return Suite.Stage(name=name, episodes=episodes, robot=robot, map=world, tm_robots=RANDOM_ROBOTS, tm_obstacles=RANDOM_OBSTACLES, config={}, seed=7, timeout=60.0, optim=optim)


def _suite(*stages: Suite.Stage, references: bool = False) -> Suite:
    return Suite(name="suite", stages=list(stages), references=references)


def _contest(*names: str, args: dict | None = None) -> Contest:
    return Contest(name="contest", description=None, contestants=[Contest.Contestant(name=n, args=args or {}) for n in names])


def _step(contestant: str = "dwb", stage: Suite.Stage | None = None, args: dict | None = None, **kwargs: object) -> Step:
    return Step(contestant=Contest.Contestant(name=contestant, args=args or {}), stage=stage or _stage(), **kwargs)


def _result(key: str, status: str, kind: StepErrorKind | None = None) -> StepResult:
    return StepResult(key, status, 0, 0.0, 1.0, kind, None)


def test_launch_args_carry_stage_optim_overrides():
    args = build_launch_args(_step(stage=_stage(optim={"lr": 0.1, "beta": 2})), None)

    assert "optim.lr:=0.1" in args
    assert "optim.beta:=2" in args
    assert not any(a.startswith("sim:=") for a in args)


def test_launch_args_drop_stage_owned_keys_from_cap_dicts(capfd: pytest.CaptureFixture[str]):
    step = _step(args={"robot": {"driver": "other"}, "task": {"robots": "scenario", "speed": 2, "unset": ""}, "mobile": {"driver": "nav2", "local_planner": "dwb"}})

    args = build_launch_args(step, "gazebo")

    assert "robot:=other" not in args
    assert "task.robots:=scenario" not in args
    assert "task.speed:=2" in args
    assert "task.unset:=" not in args
    assert "robot.mobile:=nav2" in args
    assert "robot.mobile.local_planner:=dwb" in args
    assert capfd.readouterr().err.count("controlled by stage") == 2


def test_launch_args_forward_passthrough_except_runner_owned_and_stage_keys():
    args = build_launch_args(_step(), "gazebo", passthrough={"headless": "true", "env.n": "4", "env_n": "4", "world": "other", "viz.view": "robot"})

    assert "viz.view:=robot" in args
    assert "world:=map1" in args
    assert not any(a.startswith(("headless:=", "env.n:=", "env_n:=", "world:=other")) for a in args)


def test_reference_grid_adds_an_unobstructed_run_per_cell_and_one_ped_run_per_stage(tmp_path: pathlib.Path):
    suite = _suite(_stage("s1"), _stage("s2", episodes=3), references=True)

    steps = _all_steps_grid(suite, _contest("a", "b"), 2.0, tmp_path)

    assert [s.key for s in steps] == [
        "a/s1",
        "a_unobstructed_robot/s1",
        "a/s2",
        "a_unobstructed_robot/s2",
        "b/s1",
        "b_unobstructed_robot/s1",
        "b/s2",
        "b_unobstructed_robot/s2",
        "unhindered_peds/s1",
        "unhindered_peds/s2",
    ]
    assert [s.episodes for s in steps[:4]] == [8, 8, 6, 6]
    assert {s.record_dir for s in steps} == {tmp_path / "episodes"}


@pytest.mark.parametrize("references", [False, True])
def test_grid_refuses_duplicate_stage_names(references: bool):
    suite = _suite(_stage("s1"), _stage("s1"), references=references)

    with pytest.raises(ValueError, match="duplicate step key"):
        _all_steps_grid(suite, _contest("a"), 1.0)


def test_grid_refuses_a_contestant_that_collides_with_the_reference_name():
    suite = _suite(_stage("s1"), references=True)

    with pytest.raises(ValueError, match="duplicate step key: 'a_unobstructed_robot/s1'"):
        _all_steps_grid(suite, _contest("a", "a_unobstructed_robot"), 1.0)


def test_grid_refuses_a_contestant_named_like_the_ped_reference():
    suite = _suite(_stage("s1"), references=True)

    with pytest.raises(ValueError, match="duplicate step key: 'unhindered_peds/s1'"):
        _all_steps_grid(suite, _contest("unhindered_peds"), 1.0)


def test_all_steps_has_no_record_dir():
    steps = _all_steps(_contest("a"), _suite(_stage("s1", episodes=3)), 1.0)

    assert [(s.key, s.episodes, s.record_dir) for s in steps] == [("a/s1", 3, None)]


def test_pending_skips_ok_and_plain_failures_but_retries_systemic_ones(tmp_path: pathlib.Path):
    suite = _suite(*(_stage(f"s{i}") for i in range(5)))
    states = {
        "a/s0": _result("a/s0", "ok"),
        "a/s1": _result("a/s1", "failed", StepErrorKind.INTERNAL),
        "a/s2": _result("a/s2", "failed", StepErrorKind.SIM_DEAD),
        "a/s3": _result("a/s3", "partial"),
    }
    RunDir.create(tmp_path, "r", _manifest("0" * 40)).state.write(states)
    run_dir = RunDir.open(tmp_path, "r")

    pending = build_pending(suite, _contest("a"), 1.0, run_dir, retry_failed=False, record_root=tmp_path / "r")
    retried = build_pending(suite, _contest("a"), 1.0, run_dir, retry_failed=True, record_root=tmp_path / "r")

    assert [s.key for s in pending] == ["a/s2", "a/s3", "a/s4"]
    assert [s.key for s in retried] == ["a/s1", "a/s2", "a/s3", "a/s4"]


def test_planner_identity_prefers_the_mobile_cap_planners():
    assert resolve_planner_identity(Contest.Contestant(name="x", args={"mobile": {"local_planner": "dwb", "inter_planner": "spline"}})) == ("dwb", "spline")
    assert resolve_planner_identity(Contest.Contestant(name="x", args={"mobile": {"local_planner": "mppi"}})) == ("mppi", "none")


def test_planner_identity_falls_back_to_the_contestant_name():
    from arena_evaluation.storage.planner_names import split_planner_name

    contestant = Contest.Contestant(name="teb-spline", args={"mobile": {"driver": "nav2"}})

    assert resolve_planner_identity(contestant) == split_planner_name("teb-spline")
    assert resolve_planner_identity(Contest.Contestant(name="teb-spline", args={})) == split_planner_name("teb-spline")


def test_preflight_names_missing_nav2_planners_and_skips_other_drivers():
    pytest.importorskip("arena_robots")
    contest = Contest(
        name="c",
        description=None,
        contestants=[
            Contest.Contestant(name="good", args={"mobile": {"driver": "nav2", "local_planner": "dwb", "inter_planner": "default"}}),
            Contest.Contestant(name="no_inter", args={"mobile": {"driver": "nav2", "local_planner": "dwb", "inter_planner": "none"}}),
            Contest.Contestant(name="bad", args={"mobile": {"driver": "nav2", "local_planner": "nope_planner", "inter_planner": "nope_inter"}}),
            Contest.Contestant(name="drl", args={"mobile": {"driver": "rosnav", "local_planner": "nope_planner"}}),
            Contest.Contestant(name="flat", args={"mobile.local_planner": "nope_planner"}),
            Contest.Contestant(name="bare_nav2", args={"mobile": {"driver": "nav2"}}),
        ],
    )

    problems = _preflight_contest(contest)

    assert len(problems) == 2
    assert problems[0].startswith("bad: local_planner 'nope_planner' has no ")
    assert "'dwb'" in problems[0]
    assert problems[1].startswith("bad: inter_planner 'nope_inter' has no ")
    assert "'default'" in problems[1]


def test_unhindered_peds_steps_join_the_block_of_their_robot_and_world():
    jackal = _step("a", _stage("s1", robot="jackal", world="w1"))
    burger = _step("b", _stage("s2", robot="burger", world="w1"))
    peds_jackal = _step("unhindered_peds", _stage("s1", robot="jackal", world="w1"), is_reference=True, reference_type="unhindered_peds")
    peds_orphan = _step("unhindered_peds", _stage("s3", robot="jackal", world="w9"), is_reference=True, reference_type="unhindered_peds")

    groups = group_pending([jackal, burger, peds_jackal, peds_orphan], "gazebo")

    assert [[s.key for s in g] for g in groups] == [["a/s1", "unhindered_peds/s1"], ["b/s2"], ["unhindered_peds/s3"]]


@pytest.mark.parametrize(
    ("value", "kind", "field", "expected"),
    [
        (True, ParameterType.PARAMETER_BOOL, "bool_value", True),
        (3, ParameterType.PARAMETER_INTEGER, "integer_value", 3),
        (0.5, ParameterType.PARAMETER_DOUBLE, "double_value", 0.5),
        ("x", ParameterType.PARAMETER_STRING, "string_value", "x"),
        ([], ParameterType.PARAMETER_STRING_ARRAY, "string_array_value", []),
        ([True, False], ParameterType.PARAMETER_BOOL_ARRAY, "bool_array_value", [True, False]),
        ([1, 2], ParameterType.PARAMETER_INTEGER_ARRAY, "integer_array_value", [1, 2]),
        ([0.5, 1.5], ParameterType.PARAMETER_DOUBLE_ARRAY, "double_array_value", [0.5, 1.5]),
        (["a", "b"], ParameterType.PARAMETER_STRING_ARRAY, "string_array_value", ["a", "b"]),
    ],
)
def test_walk_dict_types_each_leaf(value: object, kind: int, field: str, expected: object):
    (param,) = _walk_dict({"leaf": value})

    assert param.name == "leaf"
    assert param.value.type == kind
    actual = getattr(param.value, field)
    assert (list(actual) if isinstance(expected, list) else actual) == expected


def test_walk_dict_turns_float_ranges_into_double_pairs_and_keeps_sibling_keys():
    params = _walk_dict({"dynamic": {"min": 1, "max": 2.5, "models": ["a"]}})

    assert [p.name for p in params] == ["dynamic.n", "dynamic.models"]
    assert params[0].value.type == ParameterType.PARAMETER_DOUBLE_ARRAY
    assert list(params[0].value.double_array_value) == [1.0, 2.5]


@pytest.mark.parametrize(("value", "message"), [([1, "a"], "unsupported mixed list type for 'leaf'"), (None, "unsupported param type for 'leaf': NoneType")])
def test_walk_dict_rejects_values_without_a_parameter_type(value: object, message: str):
    with pytest.raises(TypeError, match=message):
        _walk_dict({"leaf": value})


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", False),
        ("   ", False),
        ("[env_0] waiting on env_0 to come up", False),
        ("still loading spawn_env for a/s1", False),
        ("[gz] Loaded world map_empty", True),
    ],
)
def test_spawn_chatter_is_not_progress(line: str, expected: bool):
    assert _is_substantive_log_line(line) is expected


def test_proc_starttime_of_a_live_and_a_vanished_process():
    proc = subprocess.Popen(["sleep", "30"])
    try:
        assert isinstance(_proc_starttime(proc.pid), int)
    finally:
        proc.kill()
        proc.wait()
    assert _proc_starttime(proc.pid) is None


def test_proc_tree_lists_grandchildren_with_their_start_ticks():
    proc = subprocess.Popen(["sh", "-c", "sleep 30 & sleep 30 & wait"])
    try:
        deadline = time.monotonic() + 10.0
        tree = _proc_tree(proc.pid)
        while len(tree) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
            tree = _proc_tree(proc.pid)
        assert len(tree) == 2
        assert all(_proc_starttime(pid) == start for pid, start in tree.items())
    finally:
        for pid in _proc_tree(proc.pid):
            os.kill(pid, 9)
        proc.kill()
        proc.wait()


def test_robot_without_an_assembly_is_judged_by_its_model_sensors():
    pytest.importorskip("arena_robots")

    assert robot_has_depth_camera("ridgeback_plus") is False


@pytest.mark.parametrize(("raw", "expected"), [("[{name: a}]", True), ("  {stages: []}", True), ("basic", False), ("basic.yaml", False)])
def test_inline_configs_are_recognized_by_their_first_character(raw: str, expected: bool):
    assert _is_inline_contest(raw) is expected
    assert _is_inline_suite(raw) is expected


def test_default_run_id_names_lane_and_config_stems():
    run_id = _default_run_id("dir/basic.yaml", "[{name: a}]", "p1")

    assert run_id.endswith("-p1-basic-inline")
    assert len(run_id.split("-")[0]) == 8


def test_default_run_id_without_lane():
    assert _default_run_id("{stages: []}", "allplanners", None).endswith("-inline-allplanners")


def test_named_suite_and_contest_resolve_to_shipped_files():
    suite_path, bundle_dir, provenance = _resolve_suite_source("basic.yaml")
    contest_path, contest_provenance = _resolve_contest_source("basic")

    assert suite_path.name == "basic.yaml" and suite_path.parent.name == "suites"
    assert bundle_dir is None
    assert provenance["path"] == str(suite_path)
    assert contest_path.parent.name == "contests"
    assert contest_provenance["path"] == str(contest_path)
    assert _suite_bundle_dir("basic") is None


def test_inline_suite_has_no_bundle_dir():
    assert _suite_bundle_dir("{stages: []}") is None


def test_load_suite_contest_parses_inline_and_named_configs():
    suite, contest, suite_dict, contest_dict, bundle_dir, suite_prov, contest_prov = _load_suite_contest("{stages: [{name: s, episodes: 1, robot: jackal, map: m, tm_robots: random, tm_obstacles: random}]}", "basic.yaml")

    assert (suite.name, [s.name for s in suite.stages]) == ("inline", ["s"])
    assert suite_dict["stages"][0]["map"] == "m"
    assert (bundle_dir, suite_prov) == (None, None)
    assert contest.name == "basic"
    assert [c.name for c in contest.contestants] == [c["name"] for c in contest_dict]
    assert contest_prov["path"].endswith("basic.yaml")


def test_load_suite_contest_parses_a_named_suite_and_an_inline_contest():
    suite, contest, *_ = _load_suite_contest("basic", "[{name: solo}]")

    assert suite.name == "basic"
    assert len(suite.stages) > 0
    assert (contest.name, [c.name for c in contest.contestants]) == ("inline", ["solo"])


def _manifest(config_hash: str) -> Manifest:
    return Manifest(
        run_id="r",
        created_at="2026-10-13T00:00:00+00:00",
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=[],
        env_n=1,
        headless=True,
        config_hash=config_hash,
        simulator=None,
        scale_episodes=1.0,
        suite_name="basic",
        contest_name="basic",
        suite={},
        contest=[],
        steps=[],
    )


def test_config_drift_warns_only_when_the_shipped_config_changed(capfd: pytest.CaptureFixture[str]):
    _, _, suite_dict, contest_dict, *_ = _load_suite_contest("basic", "basic")

    _warn_config_drift(_manifest(compute_config_hash(suite_dict, contest_dict)))
    assert "differs from this run's stored config" not in capfd.readouterr().err
    _warn_config_drift(_manifest("0" * 40))

    assert "differs from this run's stored config" in capfd.readouterr().err


def test_shared_lane_is_off_without_a_token():
    assert _shared_lane(None, None, None) == (None, None)


def test_shared_token_must_be_a_plain_file_name():
    lane, error = _shared_lane("../escape", "r", None)

    assert lane is None
    assert "must be a plain file name" in error


@pytest.mark.skipif("ARENA_LANE" in os.environ, reason="needs a runner without a lane")
def test_shared_run_needs_a_lane_name():
    assert _shared_lane("tok", "r", None) == (None, "--shared needs ARENA_LANE set to this runner's lane name (e.g. ARENA_LANE=p1)")


@pytest.mark.skipif(not os.environ.get("ARENA_LANE"), reason="needs a runner inside a lane")
def test_shared_run_needs_one_explicit_run_for_every_lane():
    lane = os.environ["ARENA_LANE"]

    assert _shared_lane("tok", None, "__auto__") == (None, "--shared refuses bare --resume, pass --resume <run_id> so every lane opens the same run")
    assert _shared_lane("tok", None, None) == (None, "--shared needs --run-id <id> for a fresh run or --resume <run_id>, shared by every lane")
    assert _shared_lane("tok", "r", None) == (lane, None)
    assert _shared_lane("tok", None, "r") == (lane, None)
