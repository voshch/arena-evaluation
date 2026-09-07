from __future__ import annotations

import asyncio
import csv
import datetime
import json
import logging
import pathlib
import re
import time
import types

import pytest
from arena_evaluation.benchmark.config import Contest, Suite, _parse_duration
from arena_evaluation.benchmark.runner import (
    _MAX_SIM_DEATHS,
    _RetryBudget,
    _SYSTEMIC,
    _default_run_id,
    _orphaned_env_ids,
    _requeue_front,
    _resolve_resume_config,
    build_launch_args,
    build_pending,
    cell_verdict,
    closed_fraction,
)
from arena_evaluation.benchmark.state import (
    Manifest,
    ProgressLog,
    StateFile,
    compute_config_hash,
)
from arena_evaluation.benchmark.step import Step, StepErrorKind, StepResult
from task_generator.constants import Constants

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_stage(name: str = "s1") -> Suite.Stage:
    return Suite.Stage(
        name=name,
        episodes=10,
        robot="turtlebot3_burger",
        map="map1",
        tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
        tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
        config={},
        seed=0,
        timeout=120.0,
    )


def _make_contestant(name: str = "planner_a", args: dict | None = None) -> Contest.Contestant:
    return Contest.Contestant(
        name=name,
        args=args if args is not None else {"mobile.local_planner": "dwa"},
    )


def _make_episode_record(
    *,
    episode_id: int = 1,
    world: str = "map1",
    seed: int = 42,
    tm_robots: str = "random",
    tm_obstacles: str = "random",
    tm_modules: list[str] | None = None,
    robots: list[str] | None = None,
    outcome_state: int = 1,
    outcome_info: str = "",
    robots_params: list | None = None,
    obstacles_params: list | None = None,
    goal_dist_start: float = 0.0,
    goal_dist_min: float = 0.0,
    path_length: float = 0.0,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        episode_id=episode_id,
        world=world,
        seed=seed,
        tm_robots=tm_robots,
        tm_obstacles=tm_obstacles,
        tm_modules=tm_modules or [],
        robots=robots or ["turtlebot3_burger"],
        outcome_state=outcome_state,
        outcome_info=outcome_info,
        robots_params=robots_params or [],
        obstacles_params=obstacles_params or [],
        goal_dist_start=goal_dist_start,
        goal_dist_min=goal_dist_min,
        path_length=path_length,
    )


def _make_env_record(env_id: int, ready: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(env_id=env_id, ready=ready)


# ---------------------------------------------------------------------------
# Step.key
# ---------------------------------------------------------------------------


def test_step_key():
    step = Step(
        contestant=_make_contestant("planner_a"),
        stage=_make_stage("stage_one"),
        episodes=10,
        record_dir=None,
    )
    assert step.key == "planner_a/stage_one"


def test_step_key_uses_names():
    step = Step(
        contestant=_make_contestant("teb"),
        stage=_make_stage("indoor_10"),
        episodes=5,
        record_dir=None,
    )
    assert step.key == "teb/indoor_10"


# ---------------------------------------------------------------------------
# StateFile roundtrip
# ---------------------------------------------------------------------------


def test_state_file_roundtrip(tmp_path: pathlib.Path):
    steps: dict[str, StepResult] = {
        "p1/s1": StepResult(
            key="p1/s1",
            status="ok",
            env_id=1,
            started_at=1.0,
            ended_at=2.0,
            error_kind=None,
            error_detail=None,
        ),
        "p1/s2": StepResult(
            key="p1/s2",
            status="failed",
            env_id=None,
            started_at=3.0,
            ended_at=4.0,
            error_kind=StepErrorKind.ENV_SETUP,
            error_detail="oops",
        ),
    }
    sf = StateFile.open(tmp_path)
    sf.write(steps)

    sf2 = StateFile.open(tmp_path)
    assert set(sf2.steps.keys()) == {"p1/s1", "p1/s2"}
    r1 = sf2.steps["p1/s1"]
    assert r1.status == "ok"
    assert r1.env_id == 1
    assert r1.started_at == 1.0
    assert r1.ended_at == 2.0
    assert r1.error_kind is None
    assert r1.error_detail is None

    r2 = sf2.steps["p1/s2"]
    assert r2.status == "failed"
    assert r2.env_id is None
    assert r2.error_kind == StepErrorKind.ENV_SETUP
    assert r2.error_detail == "oops"


def test_state_file_empty_when_no_file(tmp_path: pathlib.Path):
    sf = StateFile.open(tmp_path)
    assert sf.steps == {}


def test_state_file_overwrite(tmp_path: pathlib.Path):
    sf = StateFile.open(tmp_path)
    sf.write({"k/s": StepResult("k/s", "in_progress", None, 0.0, None, None, None)})
    sf.write({"k/s": StepResult("k/s", "ok", 2, 0.0, 1.0, None, None)})
    sf2 = StateFile.open(tmp_path)
    assert sf2.steps["k/s"].status == "ok"


def test_state_file_roundtrip_episodes_fields(tmp_path: pathlib.Path):
    steps = {
        "p/s": StepResult(
            key="p/s",
            status="partial",
            env_id=0,
            started_at=0.0,
            ended_at=5.0,
            error_kind=None,
            error_detail=None,
            episodes_run=8,
            episodes_failed=2,
        )
    }
    sf = StateFile.open(tmp_path)
    sf.write(steps)
    sf2 = StateFile.open(tmp_path)
    r = sf2.steps["p/s"]
    assert r.episodes_run == 8
    assert r.episodes_failed == 2


def test_state_file_backward_compat_error_field(tmp_path: pathlib.Path):
    """Old state files with a single 'error' field are loaded into error_detail."""
    import json

    state_data = {
        "steps": {
            "p/s": {
                "status": "failed",
                "env_id": None,
                "started_at": 0.0,
                "ended_at": 1.0,
                "error": "old error message",
                "episodes_run": 0,
                "episodes_failed": 0,
            }
        }
    }
    state_path = tmp_path / ".benchmark_state.json"
    state_path.write_text(json.dumps(state_data))
    sf = StateFile.open(tmp_path)
    r = sf.steps["p/s"]
    assert r.error_detail == "old error message"
    assert r.error_kind is None


# ---------------------------------------------------------------------------
# ProgressLog schema and append
# ---------------------------------------------------------------------------

_EXPECTED_HEADERS = [
    "ts_iso",
    "run_id",
    "step_key",
    "contestant",
    "stage",
    "env_id",
    "episode_id",
    "parent_episode_id",
    "is_reference",
    "reference_type",
    "world",
    "seed",
    "tm_robots",
    "tm_obstacles",
    "tm_modules",
    "robots",
    "outcome_state",
    "outcome_info",
    "started_at",
    "ended_at",
    "runtime_s",
    "robots_params_json",
    "obstacles_params_json",
    "error_kind",
    "error_detail",
    "lockstep_stalls",
    "lockstep_max_stall_s",
    "lockstep_rtf",
    "lockstep_beats",
    "goal_dist_start",
    "goal_dist_min",
    "path_length",
]


def test_progress_log_headers(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    log.close()
    with (tmp_path / "progress.csv").open(newline="") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
    assert headers == _EXPECTED_HEADERS


def test_progress_log_header_column_count(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    log.close()
    with (tmp_path / "progress.csv").open(newline="") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
    assert len(headers) == 32


def test_progress_log_append(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    t0 = time.time()

    rec1 = _make_episode_record(
        episode_id=1,
        world="map1",
        seed=42,
        tm_robots="random",
        tm_obstacles="random",
        tm_modules=["benchmark"],
        robots=["burger"],
        outcome_state=1,
        outcome_info="",
    )
    rec2 = _make_episode_record(
        episode_id=2,
        world="map1",
        seed=43,
        outcome_state=2,
        outcome_info="collision",
    )

    ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
    log.append(
        ts_iso=ts,
        run_id="run-abc",
        step_key="pa/s1",
        contestant="pa",
        stage="s1",
        env_id=0,
        episode_id=1,
        episode_record=rec1,
        started_at=t0,
        ended_at=t0 + 5.0,
    )
    log.append(
        ts_iso=ts,
        run_id="run-abc",
        step_key="pa/s1",
        contestant="pa",
        stage="s1",
        env_id=0,
        episode_id=2,
        episode_record=rec2,
        started_at=t0 + 5.0,
        ended_at=t0 + 8.0,
        error_kind=StepErrorKind.EPISODE_TIMEOUT,
        error_detail="stage.timeout exceeded",
    )
    log.close()

    with (tmp_path / "progress.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 2

    r0 = rows[0]
    assert r0["step_key"] == "pa/s1"
    assert r0["contestant"] == "pa"
    assert r0["stage"] == "s1"
    assert r0["env_id"] == "0"
    assert r0["episode_id"] == "1"
    assert r0["world"] == "map1"
    assert r0["seed"] == "42"
    assert r0["tm_robots"] == "random"
    assert r0["tm_obstacles"] == "random"
    assert r0["tm_modules"] == "benchmark"
    assert r0["robots"] == "burger"
    assert r0["outcome_state"] == "1"
    assert r0["outcome_info"] == ""
    assert json.loads(r0["robots_params_json"]) == []
    assert json.loads(r0["obstacles_params_json"]) == []
    assert r0["error_kind"] == ""
    assert r0["error_detail"] == ""

    r1 = rows[1]
    assert r1["episode_id"] == "2"
    assert r1["outcome_state"] == "2"
    assert r1["outcome_info"] == "collision"
    assert r1["error_kind"] == "episode_timeout"
    assert r1["error_detail"] == "stage.timeout exceeded"


def test_progress_log_append_to_existing(tmp_path: pathlib.Path):
    ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
    t0 = time.time()
    rec = _make_episode_record(episode_id=1)

    log1 = ProgressLog(tmp_path / "progress.csv")
    log1.append(
        ts_iso=ts,
        run_id="r",
        step_key="p/s",
        contestant="p",
        stage="s",
        env_id=0,
        episode_id=1,
        episode_record=rec,
        started_at=t0,
        ended_at=t0 + 1.0,
    )
    log1.close()

    log2 = ProgressLog(tmp_path / "progress.csv")
    log2.append(
        ts_iso=ts,
        run_id="r",
        step_key="p/s",
        contestant="p",
        stage="s",
        env_id=0,
        episode_id=2,
        episode_record=rec,
        started_at=t0 + 1.0,
        ended_at=t0 + 2.0,
    )
    log2.close()

    with (tmp_path / "progress.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# compute_config_hash determinism
# ---------------------------------------------------------------------------


def test_compute_config_hash_deterministic():
    suite = {"stages": [{"name": "s1"}]}
    contest = [{"name": "c1"}]
    assert compute_config_hash(suite, contest) == compute_config_hash(suite, contest)


def test_compute_config_hash_differs_on_suite():
    contest = [{"name": "c1"}]
    h1 = compute_config_hash({"stages": [{"name": "a"}]}, contest)
    h2 = compute_config_hash({"stages": [{"name": "b"}]}, contest)
    assert h1 != h2


def test_compute_config_hash_differs_on_contest():
    suite = {"stages": [{"name": "s"}]}
    h1 = compute_config_hash(suite, [{"name": "a"}])
    h2 = compute_config_hash(suite, [{"name": "b"}])
    assert h1 != h2


def test_compute_config_hash_is_string():
    h = compute_config_hash({"a": 1}, [{"b": 2}])
    assert isinstance(h, str)
    assert len(h) > 0


# ---------------------------------------------------------------------------
# resume config resolution (regression: resume must use the manifest, not argv)
# ---------------------------------------------------------------------------


def _make_resume_manifest() -> Manifest:
    suite_dict = {
        "stages": [
            {
                "name": "ladder_01",
                "map": "ladder_01",
                "robot": "jackal",
                "tm_robots": "scenario",
                "tm_obstacles": "random",
                "episodes": 3,
                "config": {"scenario": {"file": "ladder.json"}},
            },
            {
                "name": "ladder_05",
                "map": "ladder_05",
                "robot": "jackal",
                "tm_robots": "scenario",
                "tm_obstacles": "random",
                "episodes": 3,
                "config": {"scenario": {"file": "ladder.json"}},
            },
        ]
    }
    contest_dict = [{"name": "dwb", "mobile": {"driver": "nav2", "local_planner": "dwb"}}]
    return Manifest(
        run_id="20260623-202509-ladder-basic",
        created_at="2026-06-23T20:25:09+00:00",
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=["--suite", "ladder", "--contest", "basic"],
        env_n=1,
        headless=False,
        config_hash=compute_config_hash(suite_dict, contest_dict),
        simulator="gazebo",
        scale_episodes=2.0,
        suite_name="ladder",
        contest_name="basic",
        suite=suite_dict,
        contest=contest_dict,
        steps=[],
    )


def test_resume_reconstructs_config_from_manifest():
    man = _make_resume_manifest()
    # Round-trip through YAML exactly as RunDir.open reads it back from disk.
    man = Manifest.from_yaml(man.to_yaml())

    suite, contest, scale_episodes, simulator = _resolve_resume_config(man)

    # The reported bug: resume fell back to the argparse default suite ("basic").
    assert suite.name == "ladder"
    assert [s.name for s in suite.stages] == ["ladder_01", "ladder_05"]
    assert [c.name for c in contest.contestants] == ["dwb"]
    # scale_episodes and simulator also come from the run, not from CLI defaults.
    assert scale_episodes == 2.0
    assert simulator == "gazebo"


def test_resume_replays_stored_config_hash():
    man = Manifest.from_yaml(_make_resume_manifest().to_yaml())
    # Resuming hashes the stored dicts, which must reproduce the run's stored hash.
    assert compute_config_hash(man.suite, man.contest) == man.config_hash


def test_manifest_round_trips_launch_args():
    man = _make_resume_manifest()
    man.launch_args = {"sim.isaac.physics": "newton", "headless": "true"}
    assert Manifest.from_yaml(man.to_yaml()).launch_args == {"sim.isaac.physics": "newton", "headless": "true"}


def test_manifest_without_launch_args_defaults_empty():
    import yaml

    data = yaml.safe_load(_make_resume_manifest().to_yaml())
    del data["launch_args"]
    assert Manifest.from_yaml(yaml.dump(data)).launch_args == {}


# ---------------------------------------------------------------------------
# _parse_duration
# ---------------------------------------------------------------------------


def test_parse_duration_plain_int():
    assert _parse_duration("60") == 60.0


def test_parse_duration_plain_float():
    assert _parse_duration("60.0") == 60.0


def test_parse_duration_ms():
    assert _parse_duration("500ms") == pytest.approx(0.5)


def test_parse_duration_seconds_suffix():
    assert _parse_duration("5s") == pytest.approx(5.0)


def test_parse_duration_minutes():
    assert _parse_duration("5m") == pytest.approx(300.0)


def test_parse_duration_hours():
    assert _parse_duration("1h") == pytest.approx(3600.0)


def test_parse_duration_compound():
    assert _parse_duration("1h30m") == pytest.approx(5400.0)


def test_parse_duration_garbage_raises():
    with pytest.raises(ValueError):
        _parse_duration("not_a_duration")


def test_parse_duration_empty_raises():
    with pytest.raises(ValueError):
        _parse_duration("abc")


# ---------------------------------------------------------------------------
# build_launch_args
# ---------------------------------------------------------------------------


def _make_cell(
    contestant_name: str = "planner_a",
    stage_name: str = "s1",
    record_dir: pathlib.Path | None = None,
    episodes: int = 10,
    stage_config: dict | None = None,
    contestant_args: dict | None = None,
) -> Step:
    stage = Suite.Stage(
        name=stage_name,
        episodes=episodes,
        robot="turtlebot3_burger",
        map="map1",
        tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
        tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
        config=stage_config or {},
        seed=42,
        timeout=120.0,
    )
    args = contestant_args if contestant_args is not None else {"mobile.local_planner": "dwa"}
    contestant = Contest.Contestant(name=contestant_name, args=args)
    return Step(contestant=contestant, stage=stage, episodes=episodes, record_dir=record_dir)


def test_build_launch_args_required_fields():
    cell = _make_cell()
    args = build_launch_args(cell, "gazebo")
    assert "sim:=gazebo" in args
    assert "robot:=turtlebot3_burger" in args
    assert "world:=map1" in args
    assert f"task.robots:={Constants.TaskMode.TM_Robots.RANDOM.value}" in args
    assert f"task.obstacles:={Constants.TaskMode.TM_Obstacles.RANDOM.value}" in args
    assert not any(a.startswith("task.episodes:=") for a in args)
    assert "run_seed:=42" in args
    assert "task.auto_reset:=false" in args
    assert "task.modules:=" in args


def test_build_launch_args_no_record_dir_by_default():
    cell = _make_cell(record_dir=None)
    args = build_launch_args(cell, "gazebo")
    assert not any(a.startswith("record.dir:=") for a in args)


def test_build_launch_args_record_dir_appended(tmp_path: pathlib.Path):
    cell = _make_cell(record_dir=tmp_path / "out")
    args = build_launch_args(cell, "gazebo")
    assert any(a.startswith("record.dir:=") for a in args)


def test_build_launch_args_simulator_propagated():
    cell = _make_cell()
    assert "sim:=dummy" in build_launch_args(cell, "dummy")
    assert "sim:=isaac" in build_launch_args(cell, "isaac")


def test_build_launch_args_scenario_not_in_launch():
    """scenario_file is set via QueueEpisode, not as a launch arg."""
    cell = _make_cell(stage_config={"scenario": {"file": "/some/path/my_scenario.yaml"}})
    args = build_launch_args(cell, "gazebo")
    assert not any(a.startswith("task.scenario:=") for a in args)


def test_build_launch_args_cap_scoped_planner_forwarded():
    cell = _make_cell(contestant_args={"mobile.local_planner": "teb"})
    args = build_launch_args(cell, "gazebo")
    assert "robot.mobile.local_planner:=teb" in args


def test_build_launch_args_unknown_contestant_arg_forwarded():
    """Non-stage-owned keys pass through verbatim; launch layer is the gate."""
    cell = _make_cell(contestant_args={"mobile.local_planner": "dwa", "secret_knob": "x"})
    args = build_launch_args(cell, "gazebo")
    assert "secret_knob:=x" in args


def test_build_launch_args_stage_owned_key_dropped():
    """A contestant key colliding with stage-owned args is logged and dropped."""
    cell = _make_cell(contestant_args={"mobile.local_planner": "teb", "sim": "isaac"})
    logger = logging.getLogger("arena_evaluation.benchmark.runner")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        args = build_launch_args(cell, "gazebo")
    finally:
        logger.removeHandler(handler)
    assert "sim:=gazebo" in args
    assert "sim:=isaac" not in args
    msgs = [r.getMessage() for r in records if r.levelname == "WARNING"]
    assert any("'sim'" in m and "ignored" in m for m in msgs), msgs


def test_build_launch_args_stage_owned_robot_dropped(caplog: pytest.LogCaptureFixture):
    cell = _make_cell(contestant_args={"robot": "jackal", "world": "map2"})
    with caplog.at_level("WARNING"):
        args = build_launch_args(cell, "gazebo")
    assert "robot:=turtlebot3_burger" in args
    assert "robot:=jackal" not in args
    assert "world:=map1" in args
    assert "world:=map2" not in args


def test_build_launch_args_empty_value_skipped():
    """Empty-string values are skipped without forwarding or warning."""
    cell = _make_cell(contestant_args={"mobile.local_planner": ""})
    args = build_launch_args(cell, "gazebo")
    assert not any(a.startswith("robot.mobile.local_planner") for a in args)


def test_build_launch_args_arm_cap_forwarded():
    cell = _make_cell(contestant_args={"arm": "moveit", "arm.controller": "ompl"})
    args = build_launch_args(cell, "gazebo")
    assert "robot.arm:=moveit" in args
    assert "robot.arm.controller:=ompl" in args


def test_build_launch_args_passthrough_cli_args_forwarded():
    passthrough = {"optim.obstacles": "bbox", "headless": "true", "env.n": "1"}
    cell = _make_cell(contestant_args=passthrough)
    args = build_launch_args(cell, "gazebo")
    assert "optim.obstacles:=bbox" in args
    assert "headless:=true" in args
    assert "env.n:=1" in args


def test_build_launch_args_multiple_cap_keys_all_pass():
    cell = _make_cell(
        contestant_args={
            "mobile.local_planner": "teb",
            "mobile.inter_planner": "bypass",
            "mobile.global_planner": "smac",
        }
    )
    args = build_launch_args(cell, "gazebo")
    assert "robot.mobile.local_planner:=teb" in args
    assert "robot.mobile.inter_planner:=bypass" in args
    assert "robot.mobile.global_planner:=smac" in args


def test_build_launch_args_mobile_adapter_forwarded():
    cell = _make_cell(contestant_args={"mobile": "rosnav_rl", "mobile.agent": "best"})
    args = build_launch_args(cell, "gazebo")
    assert "robot.mobile:=rosnav_rl" in args
    assert "robot.mobile.agent:=best" in args


def test_build_launch_args_no_mobile_when_absent():
    cell = _make_cell(contestant_args={"mobile.local_planner": "dwa"})
    args = build_launch_args(cell, "gazebo")
    assert not any(a == "robot.mobile:=" or a.startswith("robot.mobile:=") for a in args)


def test_build_launch_args_no_sim_when_simulator_none():
    cell = _make_cell()
    args = build_launch_args(cell, None)
    assert not any(a.startswith("sim:=") for a in args)
    assert "robot:=turtlebot3_burger" in args
    assert "world:=map1" in args


def test_build_launch_args_dict_cap_driver_and_kwargs():
    """Dict-form cap emits driver as top-level and kwargs as dot-joined args."""
    cell = _make_cell(contestant_args={"mobile": {"driver": "nav2", "local_planner": "teb", "inter_planner": "bypass"}})
    args = build_launch_args(cell, "gazebo")
    assert "robot.mobile:=nav2" in args
    assert "robot.mobile.local_planner:=teb" in args
    assert "robot.mobile.inter_planner:=bypass" in args


def test_build_launch_args_dict_cap_no_driver():
    """Dict-form cap without driver: only sub-keys are emitted."""
    cell = _make_cell(contestant_args={"mobile": {"local_planner": "dwa"}})
    args = build_launch_args(cell, "gazebo")
    assert not any(a == "robot.mobile:=" or a == "robot.mobile:=None" for a in args)
    assert "robot.mobile.local_planner:=dwa" in args


def test_build_launch_args_dict_cap_stage_collision_dropped():
    """Dict-form cap sub-key colliding with stage-owned key is dropped."""
    cell = _make_cell(contestant_args={"mobile": {"driver": "nav2", "local_planner": "teb"}, "sim": "isaac"})
    args = build_launch_args(cell, "gazebo")
    assert "sim:=gazebo" in args
    assert "sim:=isaac" not in args
    assert "robot.mobile:=nav2" in args
    assert "robot.mobile.local_planner:=teb" in args


# ---------------------------------------------------------------------------
# build_pending
# ---------------------------------------------------------------------------


def _fake_run_dir(state_steps: dict[str, StepResult]) -> types.SimpleNamespace:
    state = types.SimpleNamespace(steps=state_steps)
    return types.SimpleNamespace(state=state)


def _make_suite(*stage_names: str) -> Suite:
    stages = [
        Suite.Stage(
            name=n,
            episodes=5,
            robot="turtlebot3_burger",
            map="map1",
            tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
            tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
            config={},
            seed=0,
            timeout=120.0,
        )
        for n in stage_names
    ]
    return Suite(name="test_suite", stages=stages)


def _make_contest(*contestant_names: str) -> Contest:
    from arena_evaluation.benchmark.config import Contestant

    return Contest(
        name="test_contest",
        description=None,
        contestants=[Contestant(name=n, args={}) for n in contestant_names],
    )


def test_build_pending_empty_state_all_steps_pending(tmp_path: pathlib.Path):
    suite = _make_suite("s1", "s2")
    suite = suite._replace(references=False)
    contest = _make_contest("pa", "pb")
    run_dir = _fake_run_dir({})
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    keys = {c.key for c in steps}
    assert keys == {
        "pa/s1",
        "pa/s2",
        "pb/s1",
        "pb/s2",
    }


def test_build_pending_ok_steps_skipped(tmp_path: pathlib.Path):
    suite = _make_suite("s1")
    contest = _make_contest("pa")
    state_steps = {
        "pa/s1": StepResult("pa/s1", "ok", None, 0.0, 1.0, None, None),
        "pa_unobstructed_robot/s1": StepResult("pa_unobstructed_robot/s1", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s1": StepResult("unhindered_peds/s1", "ok", None, 0.0, 1.0, None, None),
    }
    run_dir = _fake_run_dir(state_steps)
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    assert steps == []


def test_build_pending_failed_without_retry_skipped(tmp_path: pathlib.Path):
    suite = _make_suite("s1")
    contest = _make_contest("pa")
    state_steps = {
        "pa/s1": StepResult("pa/s1", "failed", None, 0.0, 1.0, StepErrorKind.ENV_SETUP, "error"),
        "pa_unobstructed_robot/s1": StepResult("pa_unobstructed_robot/s1", "failed", None, 0.0, 1.0, StepErrorKind.ENV_SETUP, "error"),
        "unhindered_peds/s1": StepResult("unhindered_peds/s1", "failed", None, 0.0, 1.0, StepErrorKind.ENV_SETUP, "error"),
    }
    run_dir = _fake_run_dir(state_steps)
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    assert steps == []


def test_build_pending_failed_with_retry_included(tmp_path: pathlib.Path):
    suite = _make_suite("s1")
    contest = _make_contest("pa")
    state_steps = {
        "pa/s1": StepResult("pa/s1", "failed", None, 0.0, 1.0, StepErrorKind.ENV_SETUP, "error"),
        "pa_unobstructed_robot/s1": StepResult("pa_unobstructed_robot/s1", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s1": StepResult("unhindered_peds/s1", "ok", None, 0.0, 1.0, None, None),
    }
    run_dir = _fake_run_dir(state_steps)
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=True, record_root=tmp_path)
    assert len(steps) == 1
    assert steps[0].key == "pa/s1"


def test_build_pending_partial_always_retried(tmp_path: pathlib.Path):
    # partial = definitionally incomplete; retry regardless of --retry-failed.
    suite = _make_suite("s1")
    contest = _make_contest("pa")
    state_steps = {
        "pa/s1": StepResult(
            "pa/s1",
            "partial",
            None,
            0.0,
            1.0,
            None,
            None,
            episodes_run=3,
            episodes_failed=2,
        ),
        "pa_unobstructed_robot/s1": StepResult("pa_unobstructed_robot/s1", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s1": StepResult("unhindered_peds/s1", "ok", None, 0.0, 1.0, None, None),
    }
    run_dir = _fake_run_dir(state_steps)
    steps_no_flag = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    steps_with_flag = build_pending(suite, contest, 1.0, run_dir, retry_failed=True, record_root=tmp_path)
    assert len(steps_no_flag) == 1
    assert len(steps_with_flag) == 1


def test_build_pending_skipped_always_retried(tmp_path: pathlib.Path):
    suite = _make_suite("s1")
    contest = _make_contest("pa")
    state_steps = {
        "pa/s1": StepResult("pa/s1", "skipped", None, 0.0, 1.0, StepErrorKind.CANCELLED, "cancelled"),
        "pa_unobstructed_robot/s1": StepResult("pa_unobstructed_robot/s1", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s1": StepResult("unhindered_peds/s1", "ok", None, 0.0, 1.0, None, None),
    }
    run_dir = _fake_run_dir(state_steps)
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    assert len(steps) == 1


def test_build_pending_in_progress_always_retried(tmp_path: pathlib.Path):
    suite = _make_suite("s1")
    contest = _make_contest("pa")
    state_steps = {
        "pa/s1": StepResult("pa/s1", "in_progress", None, 0.0, None, None, None),
        "pa_unobstructed_robot/s1": StepResult("pa_unobstructed_robot/s1", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s1": StepResult("unhindered_peds/s1", "ok", None, 0.0, 1.0, None, None),
    }
    run_dir = _fake_run_dir(state_steps)
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    assert len(steps) == 1


def test_build_pending_mixed_states(tmp_path: pathlib.Path):
    suite = _make_suite("s1", "s2", "s3", "s4")
    contest = _make_contest("pa")
    state_steps = {
        "pa/s1": StepResult("pa/s1", "ok", None, 0.0, 1.0, None, None),
        "pa_unobstructed_robot/s1": StepResult("pa_unobstructed_robot/s1", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s1": StepResult("unhindered_peds/s1", "ok", None, 0.0, 1.0, None, None),
        "pa/s2": StepResult("pa/s2", "failed", None, 0.0, 1.0, StepErrorKind.INTERNAL, "err"),
        "pa_unobstructed_robot/s2": StepResult("pa_unobstructed_robot/s2", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s2": StepResult("unhindered_peds/s2", "ok", None, 0.0, 1.0, None, None),
        "pa/s3": StepResult("pa/s3", "partial", None, 0.0, 1.0, None, None),
        "pa_unobstructed_robot/s3": StepResult("pa_unobstructed_robot/s3", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s3": StepResult("unhindered_peds/s3", "ok", None, 0.0, 1.0, None, None),
        "pa_unobstructed_robot/s4": StepResult("pa_unobstructed_robot/s4", "ok", None, 0.0, 1.0, None, None),
        "unhindered_peds/s4": StepResult("unhindered_peds/s4", "ok", None, 0.0, 1.0, None, None),
        # pa/s4 not in state -> pending
    }
    run_dir = _fake_run_dir(state_steps)
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    keys = {c.key for c in steps}
    # ok=skipped, failed+no-retry=skipped, partial=retried, missing=pending
    assert keys == {"pa/s3", "pa/s4"}


def test_build_pending_record_dir_set_from_record_root(tmp_path: pathlib.Path):
    suite = _make_suite("s1")
    suite = suite._replace(references=False)
    contest = _make_contest("pa")
    run_dir = _fake_run_dir({})
    steps = build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)
    # All steps share the flat episodes/ directory under record_root.
    assert steps[0].record_dir == tmp_path / "episodes"


def test_build_pending_duplicate_key_raises(tmp_path: pathlib.Path):
    # Two contestants with the same name produce duplicate step keys.
    from arena_evaluation.benchmark.config import Contestant

    suite = _make_suite("s1")
    # Bypass Contest._reject_duplicate_names by constructing directly.
    contest = Contest(
        name="dup",
        description=None,
        contestants=[Contestant(name="pa", args={}), Contestant(name="pa", args={})],
    )
    run_dir = _fake_run_dir({})
    with pytest.raises(ValueError, match="duplicate step key"):
        build_pending(suite, contest, 1.0, run_dir, retry_failed=False, record_root=tmp_path)


def test_build_pending_scale_episodes(tmp_path: pathlib.Path):
    suite = _make_suite("s1")  # stage has episodes=5
    contest = _make_contest("pa")
    run_dir = _fake_run_dir({})
    steps = build_pending(suite, contest, 2.0, run_dir, retry_failed=False, record_root=tmp_path)
    assert steps[0].episodes == 10


# ---------------------------------------------------------------------------
# ProgressLog.dedupe_in_place
# ---------------------------------------------------------------------------


def test_dedupe_in_place_no_duplicates(tmp_path: pathlib.Path):
    log = ProgressLog(tmp_path / "progress.csv")
    t0 = time.time()
    ts1 = "2024-01-01T00:00:01+00:00"
    ts2 = "2024-01-01T00:00:02+00:00"
    rec = _make_episode_record(episode_id=1)
    log.append(ts_iso=ts1, run_id="r", step_key="p/s", contestant="p", stage="s", env_id=0, episode_id=1, episode_record=rec, started_at=t0, ended_at=t0 + 1.0)
    rec2 = _make_episode_record(episode_id=2)
    log.append(ts_iso=ts2, run_id="r", step_key="p/s", contestant="p", stage="s", env_id=0, episode_id=2, episode_record=rec2, started_at=t0 + 1.0, ended_at=t0 + 2.0)
    log.dedupe_in_place()
    log.close()

    with (tmp_path / "progress.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2


def test_dedupe_in_place_keeps_latest_ts(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    log = ProgressLog(path)
    t0 = time.time()
    ts_old = "2024-01-01T00:00:01+00:00"
    ts_new = "2024-01-01T00:00:05+00:00"
    rec = _make_episode_record(episode_id=1, outcome_info="old")
    rec_new = _make_episode_record(episode_id=1, outcome_info="new")
    log.append(ts_iso=ts_old, run_id="r", step_key="p/s", contestant="p", stage="s", env_id=0, episode_id=1, episode_record=rec, started_at=t0, ended_at=t0 + 1.0)
    log.append(ts_iso=ts_new, run_id="r", step_key="p/s", contestant="p", stage="s", env_id=0, episode_id=1, episode_record=rec_new, started_at=t0 + 1.0, ended_at=t0 + 2.0)
    log.dedupe_in_place()
    log.close()

    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["outcome_info"] == "new"
    assert rows[0]["ts_iso"] == ts_new


def test_dedupe_in_place_result_sorted_by_ts(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    log = ProgressLog(path)
    t0 = time.time()
    ts_a = "2024-01-01T00:00:03+00:00"
    ts_b = "2024-01-01T00:00:01+00:00"
    ts_c = "2024-01-01T00:00:02+00:00"
    for ts, eid in [(ts_a, 3), (ts_b, 1), (ts_c, 2)]:
        r = _make_episode_record(episode_id=eid)
        log.append(ts_iso=ts, run_id="r", step_key="p/s", contestant="p", stage="s", env_id=0, episode_id=eid, episode_record=r, started_at=t0, ended_at=t0 + 1.0)
    log.dedupe_in_place()
    log.close()

    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["ts_iso"] for r in rows] == [ts_b, ts_c, ts_a]


def test_dedupe_in_place_discards_comment_lines(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    log = ProgressLog(path)
    log.write_comment("resumed at 2024-01-01T00:00:00+00:00")
    t0 = time.time()
    rec = _make_episode_record(episode_id=1)
    log.append(ts_iso="2024-01-01T00:00:01+00:00", run_id="r", step_key="p/s", contestant="p", stage="s", env_id=0, episode_id=1, episode_record=rec, started_at=t0, ended_at=t0 + 1.0)
    log.dedupe_in_place()
    log.close()

    raw = path.read_text()
    assert "# resumed" not in raw
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1


def test_dedupe_in_place_preserves_header(tmp_path: pathlib.Path):
    path = tmp_path / "progress.csv"
    log = ProgressLog(path)
    t0 = time.time()
    rec = _make_episode_record(episode_id=1)
    log.append(ts_iso="2024-01-01T00:00:01+00:00", run_id="r", step_key="p/s", contestant="p", stage="s", env_id=0, episode_id=1, episode_record=rec, started_at=t0, ended_at=t0 + 1.0)
    log.dedupe_in_place()
    log.close()

    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
    assert headers == _EXPECTED_HEADERS


# ---------------------------------------------------------------------------
# _default_run_id format
# ---------------------------------------------------------------------------

_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[\w]+-[\w]+$")


def test_default_run_id_format():
    run_id = _default_run_id("basic", "basic")
    assert _RUN_ID_RE.match(run_id), f"run_id {run_id!r} did not match expected pattern"
    assert run_id.endswith("-basic-basic")


def test_default_run_id_inline():
    run_id = _default_run_id("basic", "[{name: teb, mobile.local_planner: teb}]")
    assert _RUN_ID_RE.match(run_id), f"run_id {run_id!r} did not match expected pattern"
    assert run_id.endswith("-basic-inline")


def test_default_run_id_strips_yaml_suffix():
    run_id = _default_run_id("basic.yaml", "planners.yaml")
    assert run_id.endswith("-basic-planners")


def test_default_run_id_lex_sort_is_chronological():
    run_id_a = _default_run_id("basic", "basic")
    run_id_b = _default_run_id("basic", "basic")
    assert run_id_a <= run_id_b


# test_data_root_uses_env_var: skipped, verifying that the resolution function
# reads ARENA_DATA_DIR from the environment requires monkeypatching os.environ,
# which is prohibited by the project no-mock rule. Follow-up: refactor
# _resolve_data_root to accept an optional env-dict argument so it can be
# tested with a plain dict.


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def _make_step_for(contestant_name: str, stage_name: str, robot: str = "jackal", map: str = "map1") -> Step:
    stage = Suite.Stage(
        name=stage_name,
        episodes=5,
        robot=robot,
        map=map,
        tm_robots=Constants.TaskMode.TM_Robots.RANDOM,
        tm_obstacles=Constants.TaskMode.TM_Obstacles.RANDOM,
        config={},
        seed=0,
        timeout=120.0,
    )
    contestant = Contest.Contestant(name=contestant_name, args={})
    return Step(contestant=contestant, stage=stage, episodes=5, record_dir=None)


def test_group_pending_single_contestant_same_robot():
    from arena_evaluation.benchmark.runner import group_pending

    steps = [_make_step_for("alpha", f"s{i}") for i in range(4)]
    groups = group_pending(steps, "gazebo")
    assert len(groups) == 1
    assert len(groups[0]) == 4


def test_group_pending_splits_on_contestant_change():
    from arena_evaluation.benchmark.runner import group_pending

    steps = [_make_step_for("alpha", f"s{i}") for i in range(3)] + [_make_step_for("beta", f"s{i}") for i in range(2)]
    groups = group_pending(steps, "gazebo")
    assert len(groups) == 2
    assert len(groups[0]) == 3
    assert len(groups[1]) == 2


def test_group_pending_splits_on_robot_change():
    from arena_evaluation.benchmark.runner import group_pending

    steps = [
        _make_step_for("alpha", "s0", robot="jackal"),
        _make_step_for("alpha", "s1", robot="jackal"),
        _make_step_for("alpha", "s2", robot="turtlebot3_burger"),
        _make_step_for("alpha", "s3", robot="turtlebot3_burger"),
    ]
    groups = group_pending(steps, "gazebo")
    assert len(groups) == 2
    assert all(s.stage.robot == "jackal" for s in groups[0])
    assert all(s.stage.robot == "turtlebot3_burger" for s in groups[1])


def test_group_pending_preserves_suite_order():
    from arena_evaluation.benchmark.runner import group_pending

    steps = [_make_step_for("alpha", f"s{i}") for i in range(3)]
    groups = group_pending(steps, None)
    assert len(groups) == 1
    assert [s.stage.name for s in groups[0]] == ["s0", "s1", "s2"]


def test_group_pending_empty():
    from arena_evaluation.benchmark.runner import group_pending

    assert group_pending([], "gazebo") == []


def test_env_key_components():
    from arena_evaluation.benchmark.runner import env_key

    step = _make_step_for("planner_a", "indoor", robot="jackal")
    key = env_key(step, "gazebo")
    assert key == ("planner_a", "jackal", "map1", "gazebo")


def test_env_key_simulator_none():
    from arena_evaluation.benchmark.runner import env_key

    step = _make_step_for("planner_a", "indoor", robot="jackal")
    key = env_key(step, None)
    assert key == ("planner_a", "jackal", "map1", None)


# ---------------------------------------------------------------------------
# _flatten_per_mode_params
# ---------------------------------------------------------------------------


def test_flatten_scenario_file_strips_suffix():
    from arena_evaluation.benchmark.runner import _flatten_per_mode_params
    from rcl_interfaces.msg import ParameterType

    obs, rob = _flatten_per_mode_params({"scenario": {"file": "4.json"}}, tm_obstacles="scenario", tm_robots="scenario")
    by_name = {p.name: p for p in obs}
    assert "file" in by_name
    p = by_name["file"]
    assert p.value.type == ParameterType.PARAMETER_STRING
    assert p.value.string_value == "4"
    assert {p.name for p in rob} == {"file"}


def test_flatten_random_nested_counts():
    from arena_evaluation.benchmark.runner import _flatten_per_mode_params
    from rcl_interfaces.msg import ParameterType

    obs, _rob = _flatten_per_mode_params(
        {"random": {"dynamic": {"min": 2, "max": 5}}},
        tm_obstacles="random",
        tm_robots="random",
    )
    by_name = {p.name: p for p in obs}
    assert "dynamic.n" in by_name
    assert by_name["dynamic.n"].value.type == ParameterType.PARAMETER_INTEGER_ARRAY
    assert list(by_name["dynamic.n"].value.integer_array_value) == [2, 5]


def test_flatten_empty_config_yields_empty():
    from arena_evaluation.benchmark.runner import _flatten_per_mode_params

    assert _flatten_per_mode_params({}, tm_obstacles="random", tm_robots="random") == ([], [])


def test_flatten_routes_per_active_mode():
    from arena_evaluation.benchmark.runner import _flatten_per_mode_params

    obs, rob = _flatten_per_mode_params(
        {"scenario": {"file": "x"}, "random": {"n": 3}},
        tm_obstacles="random",
        tm_robots="scenario",
    )
    assert {p.name for p in obs} == {"n"}
    assert {p.name for p in rob} == {"file"}


def test_flatten_drops_inactive_modes():
    from arena_evaluation.benchmark.runner import _flatten_per_mode_params

    obs, rob = _flatten_per_mode_params({"unrelated": {"key": "value"}}, tm_obstacles="random", tm_robots="random")
    assert obs == [] and rob == []


def test_flatten_skips_non_dict_top_level():
    from arena_evaluation.benchmark.runner import _flatten_per_mode_params

    obs, _rob = _flatten_per_mode_params(
        {"scenario": "not_a_dict", "random": {"n": 3}},
        tm_obstacles="random",
        tm_robots="random",
    )
    names = [p.name for p in obs]
    assert names == ["n"]


def test_flatten_typed_values():
    from arena_evaluation.benchmark.runner import _flatten_per_mode_params
    from rcl_interfaces.msg import ParameterType

    obs, _rob = _flatten_per_mode_params(
        {
            "random": {
                "a_int": 7,
                "a_str": "hello",
                "a_bool": True,
                "a_float": 3.14,
            }
        },
        tm_obstacles="random",
        tm_robots="random",
    )
    by_name = {p.name: p for p in obs}
    assert by_name["a_int"].value.type == ParameterType.PARAMETER_INTEGER
    assert by_name["a_int"].value.integer_value == 7
    assert by_name["a_str"].value.type == ParameterType.PARAMETER_STRING
    assert by_name["a_str"].value.string_value == "hello"
    assert by_name["a_bool"].value.type == ParameterType.PARAMETER_BOOL
    assert by_name["a_bool"].value.bool_value is True
    assert by_name["a_float"].value.type == ParameterType.PARAMETER_DOUBLE
    assert by_name["a_float"].value.double_value == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# BenchmarkProgressDisplay tests
# ---------------------------------------------------------------------------


def test_progress_display_slot_lifecycle():
    from arena_evaluation.benchmark.progress_display import BenchmarkProgressDisplay

    p = BenchmarkProgressDisplay(
        title="Test Benchmark",
        total_steps=10,
        env_n=2,
        run_id="run-123",
    )
    assert p.total_steps == 10
    assert p.completed_steps == 0

    # update slot
    p.update_slot(
        slot_index=0,
        env_id=0,
        contestant="dwb",
        stage="s01",
        step_key="dwb/s01",
        ep_idx=0,
        ep_total=3,
        state="RUNNING",
    )
    assert 0 in p.active_slots
    assert p.active_slots[0]["contestant"] == "dwb"
    assert p.active_slots[0]["state"] == "RUNNING"

    # update state
    p.update_slot_state(0, "CLEANING_UP")
    assert p.active_slots[0]["state"] == "CLEANING_UP"

    # render
    renderable = p._render()
    assert renderable is not None

    # step completed
    p.log_step_completed(
        step_key="dwb/s01",
        status="ok",
        contestant="dwb",
        stage="s01",
        episodes_run=3,
        episodes_total=3,
        episodes_failed=0,
        elapsed_sec=5.2,
    )
    assert p.completed_steps == 1
    assert p.ok_steps == 1

    # clear slot
    p.clear_slot(0)
    assert 0 not in p.active_slots


def test_progress_display_render_empty_and_with_slots():
    from arena_evaluation.benchmark.progress_display import BenchmarkProgressDisplay

    p = BenchmarkProgressDisplay(
        title="Test Benchmark",
        total_steps=5,
        env_n=1,
        run_id="run-456",
    )
    # render with no slots
    tbl_empty = p._render()
    assert tbl_empty is not None

    # add slot and render
    p.update_slot(0, 0, "teb", "s02", "teb/s02", 1, 5, "RUNNING")
    tbl_slots = p._render()
    assert tbl_slots is not None


def test_runner_start_and_restart_arena_defined():
    from arena_evaluation.benchmark.runner import BenchmarkRunner

    assert hasattr(BenchmarkRunner, "_start_arena")
    assert hasattr(BenchmarkRunner, "_restart_arena")


def test_world_batch_ordering():
    from arena_evaluation.benchmark.runner import group_pending

    steps = [
        _make_step_for("p1", "s1", map="hospital_1"),
        _make_step_for("p2", "s2", map="hospital_1"),
        _make_step_for("p1", "s3", map="hospital_2"),
        _make_step_for("p1", "s4", map="office_1"),
    ]
    world_maps = list(dict.fromkeys(s.stage.map for s in steps))
    assert world_maps == ["hospital_1", "hospital_2", "office_1"]

    h1_steps = [s for s in steps if s.stage.map == "hospital_1"]
    blocks = group_pending(h1_steps, "gazebo")
    assert len(blocks) == 2


# ---------------------------------------------------------------------------
# retry bookkeeping
# ---------------------------------------------------------------------------


def test_sim_dead_is_systemic():
    assert StepErrorKind.SIM_DEAD in _SYSTEMIC
    assert StepErrorKind.EPISODE_TIMEOUT not in _SYSTEMIC


def test_retry_budget_allows_one_retry_per_key_by_default():
    budget = _RetryBudget()
    assert budget.record("teb/s1", sim_death_generation=None) == (True, False)
    assert budget.record("teb/s1", sim_death_generation=None) == (False, False)


def test_retry_budget_keys_are_independent():
    budget = _RetryBudget()
    assert budget.record("teb/s1", sim_death_generation=None) == (True, False)
    assert budget.record("dwb/s1", sim_death_generation=None) == (True, False)


def test_retry_budget_zero_never_retries():
    budget = _RetryBudget(retries=0)
    assert budget.record("teb/s1", sim_death_generation=None) == (False, False)
    assert budget.limit == "0"


def test_retry_budget_unbounded_always_retries():
    budget = _RetryBudget(retries=-1)
    for _ in range(5):
        assert budget.record("teb/s1", sim_death_generation=None) == (True, False)
    assert budget.limit == "inf"


def test_retry_budget_counts_one_sim_death_per_generation():
    budget = _RetryBudget()
    budget.record("teb/s1", sim_death_generation=0)
    budget.record("dwb/s1", sim_death_generation=0)
    assert budget.sim_deaths == 1


def test_retry_budget_max_sim_deaths_aborts_the_run():
    budget = _RetryBudget()
    for generation in range(_MAX_SIM_DEATHS - 1):
        may_retry, run_abort = budget.record(f"p{generation}/s1", sim_death_generation=generation)
        assert (may_retry, run_abort) == (True, False)
    assert budget.record("last/s1", sim_death_generation=_MAX_SIM_DEATHS - 1) == (False, True)
    assert budget.sim_deaths == _MAX_SIM_DEATHS


def test_retry_budget_abort_wins_over_retry():
    budget = _RetryBudget(retries=-1, max_sim_deaths=1)
    assert budget.record("fresh/s1", sim_death_generation=0) == (False, True)


def test_retry_budget_spawn_phase_death_counts_the_generation_only():
    budget = _RetryBudget()
    assert budget.record(None, sim_death_generation=0) == (False, False)
    assert budget.sim_deaths == 1
    assert not budget.attempts
    assert budget.record("teb/s1", sim_death_generation=1) == (True, False)


def test_requeue_front_puts_the_step_back_at_the_head():
    q: asyncio.Queue = asyncio.Queue()
    first = _make_step_for("p1", "s1")
    second = _make_step_for("p1", "s2")
    third = _make_step_for("p1", "s3")
    for step in (second, third):
        q.put_nowait(step)

    _requeue_front(q, first)

    assert [q.get_nowait().key for _ in range(3)] == [first.key, second.key, third.key]


# ---------------------------------------------------------------------------
# orphan despawn on spawn timeout
# ---------------------------------------------------------------------------


def test_orphaned_env_ids_picks_new_not_ready_envs():
    known = {1}
    records = {
        1: _make_env_record(1, ready=True),
        2: _make_env_record(2, ready=False),
        3: _make_env_record(3, ready=True),
    }
    assert _orphaned_env_ids(known, records, []) == [2]


def test_orphaned_env_ids_ignores_new_ready_envs():
    known = {1}
    records = {1: _make_env_record(1, ready=True), 2: _make_env_record(2, ready=True)}
    assert _orphaned_env_ids(known, records, []) == []


def test_orphaned_env_ids_dedupes_with_registered():
    known = set()
    records = {2: _make_env_record(2, ready=False)}
    assert _orphaned_env_ids(known, records, [2]) == [2]


def test_orphaned_env_ids_keeps_registered_first():
    known = set()
    records = {2: _make_env_record(2, ready=False), 5: _make_env_record(5, ready=False)}
    assert _orphaned_env_ids(known, records, [5]) == [5, 2]


def test_orphaned_env_ids_empty_when_nothing_new():
    known = {1, 2}
    records = {1: _make_env_record(1, ready=True), 2: _make_env_record(2, ready=False)}
    assert _orphaned_env_ids(known, records, []) == []


# ---------------------------------------------------------------------------
# closed_fraction
# ---------------------------------------------------------------------------


def test_closed_fraction_zero_start_is_zero():
    assert closed_fraction(0.0, 0.0) == 0.0
    assert closed_fraction(0.0, 5.0) == 0.0


def test_closed_fraction_full_close():
    assert closed_fraction(10.0, 0.0) == 1.0


def test_closed_fraction_no_progress():
    assert closed_fraction(10.0, 10.0) == 0.0


def test_closed_fraction_partial():
    assert closed_fraction(10.0, 4.0) == pytest.approx(0.6)


def test_closed_fraction_clamps_above_start():
    assert closed_fraction(10.0, 15.0) == 0.0


def test_closed_fraction_clamps_below_zero_min():
    assert closed_fraction(10.0, -5.0) == 1.0


# ---------------------------------------------------------------------------
# cell_verdict
# ---------------------------------------------------------------------------


def test_cell_verdict_wedged_on_error_kind():
    r = StepResult(
        "p/s",
        "failed",
        0,
        0.0,
        1.0,
        StepErrorKind.ENV_SETUP,
        "boom",
        episodes_run=0,
        episodes_total=5,
    )
    assert cell_verdict(r) == "wedged"


def test_cell_verdict_wedged_on_incomplete_episodes():
    r = StepResult(
        "p/s",
        "partial",
        0,
        0.0,
        1.0,
        None,
        None,
        episodes_run=3,
        episodes_total=5,
    )
    assert cell_verdict(r) == "wedged"


def test_cell_verdict_weak_when_episodes_weak():
    r = StepResult(
        "p/s",
        "ok",
        0,
        0.0,
        1.0,
        None,
        None,
        episodes_run=5,
        episodes_total=5,
        episodes_weak=2,
    )
    assert cell_verdict(r) == "weak"


def test_cell_verdict_ok():
    r = StepResult(
        "p/s",
        "ok",
        0,
        0.0,
        1.0,
        None,
        None,
        episodes_run=5,
        episodes_total=5,
        episodes_weak=0,
    )
    assert cell_verdict(r) == "ok"


def test_requeue_front_into_an_empty_queue():
    q: asyncio.Queue = asyncio.Queue()
    only = _make_step_for("p1", "s1")
    _requeue_front(q, only)
    assert q.qsize() == 1
    assert q.get_nowait().key == only.key
