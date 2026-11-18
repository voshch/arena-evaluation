from __future__ import annotations

import pathlib

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


@pytest.mark.parametrize(
    ("argv", "flags"),
    [
        (["report", "--help"], ("--export", "--generate-gifs")),
        (["process", "--help"], ("--force-process", "--force-extract")),
        (["migrate", "--help"], ("--dry-run", "--force", "--report")),
    ],
)
def test_evaluation_cli_lists_the_layer_flags(
    argv: list[str], flags: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    from arena_evaluation.cli import _main_impl

    with pytest.raises(SystemExit) as raised:
        _main_impl(argv)
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for flag in flags:
        assert flag in help_text


def test_migrate_run_root_with_an_empty_episode_reports_one_line(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena_evaluation.cli import _main_impl

    run = tmp_path / "run"
    (run / "episodes" / "episode_000").mkdir(parents=True)
    (run / "manifest.yaml").write_text("run_id: run\n")

    assert _main_impl(["migrate", "--dry-run", str(run)]) == 1

    stderr = capsys.readouterr().err
    lines = [line for line in stderr.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("Error: ")
    assert "Traceback" not in stderr


def test_validate_suite_accepts_a_bundle(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena_evaluation.cli import _main_impl

    bundle = tmp_path / "configs" / "suites" / "basic"
    bundle.mkdir(parents=True)
    (bundle / "suite.yaml").write_text(
        "stages:\n"
        "  - name: preflight\n"
        "    episodes: 1\n"
        "    robot: jackal\n"
        "    map: map_empty\n"
        "    tm_robots: random\n"
        "    tm_obstacles: random\n"
    )

    assert _main_impl(["validate", "--suite", str(bundle)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "valid suite: basic"
    assert "1 stage(s) without an explicit seed: preflight" in captured.err


def test_validate_suite_rejects_a_non_mapping_root(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena_evaluation.cli import _main_impl

    config = tmp_path / "bad" / "suite.yaml"
    config.parent.mkdir()
    config.write_text("- not\n- a suite\n")

    assert _main_impl(["validate", "--suite", str(config)]) == 1
    assert "a suite needs a mapping with a stages list" in capsys.readouterr().err


def test_validate_contest_rejects_a_duplicate_contestant(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena_evaluation.cli import _main_impl

    config = tmp_path / "dupe" / "contest.yaml"
    config.parent.mkdir()
    config.write_text(
        "- name: same\n  mobile.local_planner: dwa\n- name: same\n  mobile.local_planner: teb\n"
    )

    assert _main_impl(["validate", "--contest", str(config)]) == 1
    assert capsys.readouterr().err.startswith("Error: ")


