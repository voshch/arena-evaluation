from __future__ import annotations

import logging
import os
import pathlib

import pytest

from arena_evaluation.benchmark import runner
from arena_evaluation.benchmark.runner import cli_main
from arena_evaluation.benchmark.state import Manifest, RunDir

SUITE = "{stages: [{name: s1, episodes: 1, robot: jackal, map: map_empty, tm_robots: random, tm_obstacles: random}]}"


@pytest.fixture(autouse=True)
def restored_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    runner_level, runner_propagate = runner._log.level, runner._log.propagate
    yield
    root.handlers = handlers
    root.setLevel(level)
    runner._log.setLevel(runner_level)
    runner._log.propagate = runner_propagate


def _manifest(run_id: str, contest: list, simulator: str | None = "gazebo") -> Manifest:
    return Manifest(
        run_id=run_id,
        created_at="2026-10-13T00:00:00+00:00",
        arena_git_sha=None,
        arena_git_dirty=False,
        cli_args=[],
        env_n=1,
        headless=True,
        config_hash="0" * 40,
        simulator=simulator,
        scale_episodes=1.0,
        suite_name="basic",
        contest_name="basic",
        suite={"stages": [{"name": "s1", "episodes": 1, "robot": "jackal", "map": "map_empty", "tm_robots": "random", "tm_obstacles": "random"}]},
        contest=contest,
        steps=[],
    )


def test_unknown_extra_argument_is_an_argparse_error(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--data-root", str(tmp_path), "--bogus"])

    assert exc.value.code == 2
    assert "unrecognized argument: '--bogus'" in capsys.readouterr().err


def test_bad_shared_token_is_refused_before_anything_runs(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    assert cli_main(["--data-root", str(tmp_path), "--shared", "a/b", "--run-id", "r"]) == 2
    assert "--shared token 'a/b' must be a plain file name" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_empty_grid_is_a_config_error_and_env_n_is_renamed(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    assert cli_main(["--data-root", str(tmp_path), "--suite", SUITE, "--contest", "[]", "env_n:=2", "viz.view:=robot"]) == 2

    err = capsys.readouterr().err
    assert "env_n:= is deprecated, use env.n:=" in err
    assert f"data_root from --data-root: {tmp_path}" in err
    assert "empty grid (suite='inline' contest='inline' produced no steps)" in err
    assert list(tmp_path.iterdir()) == []


def test_data_root_defaults_to_the_arena_data_dir_or_the_package_share(capsys: pytest.CaptureFixture[str]):
    assert cli_main(["--suite", SUITE, "--contest", "[]"]) == 2

    err = capsys.readouterr().err
    expected = "data_root from ARENA_DATA_DIR" if os.environ.get("ARENA_DATA_DIR") else "data_root from default"
    assert expected in err


def test_preflight_rejects_a_nav2_contestant_with_an_unknown_planner(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    pytest.importorskip("arena_robots")

    assert cli_main(["--data-root", str(tmp_path), "--suite", SUITE, "--contest", "[{name: bad, mobile: {driver: nav2, local_planner: nope_planner}}]"]) == 2
    assert "benchmark: bad: local_planner 'nope_planner' has no " in capsys.readouterr().err


def test_duplicate_stage_names_are_reported_as_a_config_error(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    suite = "{stages: [{name: s1, episodes: 1, robot: jackal, map: m, tm_robots: random, tm_obstacles: random}, {name: s1, episodes: 1, robot: jackal, map: m, tm_robots: random, tm_obstacles: random}]}"

    assert cli_main(["--data-root", str(tmp_path), "--suite", suite, "--contest", "[{name: a}]"]) == 2
    assert "benchmark: duplicate step key: 'a/s1'" in capsys.readouterr().err


def test_malformed_inline_suite_is_reported_as_a_config_error(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    assert cli_main(["--data-root", str(tmp_path), "--suite", "{stages: [{name: s1}]}", "--contest", "[{name: a}]"]) == 2
    assert capsys.readouterr().err.splitlines()[-1].startswith("benchmark: ")


def test_bare_resume_without_runs_is_a_config_error(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    assert cli_main(["--data-root", str(tmp_path), "--resume"]) == 2
    assert f"no resumable runs in {tmp_path}" in capsys.readouterr().err


def test_resuming_an_unknown_run_reports_the_missing_manifest(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    assert cli_main(["--data-root", str(tmp_path), "--resume", "nope"]) == 2
    assert "benchmark: config file not found:" in capsys.readouterr().err


@pytest.mark.parametrize(("explicit", "simulator"), [(True, "gazebo"), (False, None)])
def test_resume_replays_the_stored_config_and_marks_the_progress_log(tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str], explicit: bool, simulator: str | None):
    RunDir.create(tmp_path, "20261013-000000-basic-basic", _manifest("20261013-000000-basic-basic", [], simulator)).progress.close()
    resume = ["--resume", "20261013-000000-basic-basic"] if explicit else ["--resume"]

    assert cli_main(["--data-root", str(tmp_path), *resume, "--suite", SUITE, "--contest", "[{name: ignored}]"]) == 2

    err = capfd.readouterr().err
    assert "empty grid (suite='basic' contest='basic' produced no steps)" in err
    assert "differs from this run's stored config, resuming with the stored config" in err
    assert ("auto-resume picked run_id=20261013-000000-basic-basic" in err) is not explicit
    assert "# resumed at " in (tmp_path / "20261013-000000-basic-basic" / "progress.csv").read_text()


@pytest.mark.skipif(not os.environ.get("ARENA_LANE"), reason="needs a runner inside a lane")
def test_shared_resume_opens_the_lane_files_of_the_run(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    RunDir.create(tmp_path, "shared", _manifest("shared", [])).progress.close()

    assert cli_main(["--data-root", str(tmp_path), "--shared", "tok", "--resume", "shared"]) == 2

    assert "empty grid" in capsys.readouterr().err
    assert (tmp_path / "shared" / f"progress.{os.environ['ARENA_LANE']}.csv").exists()


def test_fresh_run_refuses_to_reuse_an_existing_run_directory(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / "taken").mkdir()

    assert cli_main(["--data-root", str(tmp_path), "--suite", SUITE, "--contest", "[{name: a}]", "--run-id", "taken"]) == 2
    assert "File exists" in capsys.readouterr().err
    assert list((tmp_path / "taken").iterdir()) == []
