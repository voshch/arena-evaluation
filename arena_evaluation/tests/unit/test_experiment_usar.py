from __future__ import annotations

import csv
import pathlib

import pytest
import yaml

pl = pytest.importorskip("polars")

from arena_evaluation.experiments import usar

# Two worlds with two stages each, the shape of a generate_usar bundle.
STAGES = {"w0__a": "w0", "w0__b": "w0", "w1__a": "w1", "w1__b": "w1"}
PROGRESS = ["ts_iso", "step_key", "contestant", "stage", "env_id", "episode_id", "world", "seed", "outcome_state", "outcome_info", "runtime_s"]


def _row(contestant: str, stage: str, seed: int, found_after: float | None, *, state: int | None = None, env: int = 0, episode_id: int | None = None, ts: str = "2026-11-19T10:00:00+00:00") -> dict:
    """A progress row; found_after None is a search that timed out."""
    state = state if state is not None else (2 if found_after is not None else 3)
    return {
        "ts_iso": ts,
        "step_key": f"{contestant}/{stage}",
        "contestant": contestant,
        "stage": stage,
        "env_id": env,
        "episode_id": seed if episode_id is None else episode_id,
        "world": STAGES[stage],
        "seed": seed,
        "outcome_state": state,
        "outcome_info": f"finished after {found_after}s" if found_after is not None else "timeout",
        "runtime_s": found_after if found_after is not None else 120.4,
    }


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROGRESS)
        writer.writeheader()
        writer.writerows(rows)


def _run(root: pathlib.Path, name: str, rows: list[dict], *, suite: str = "usar_tune", contest: list | None = None, timeout: str | None = "2m") -> pathlib.Path:
    """A benchmark run directory with the manifest and progress.csv the runner writes."""
    run = root / name
    run.mkdir(parents=True)
    stages = [{"name": s, "map": w, "robot": "turtlebot", "modes": {"robots": "search", "obstacles": "scenario"}, "episodes": 1, **({"timeout": timeout} if timeout else {})} for s, w in STAGES.items()]
    manifest = {"suite_name": suite, "suite": {"stages": stages}, "contest_name": name, "contest": contest or [], "simulator": "flatsim"}
    (run / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    _write_csv(run / "progress.csv", rows)
    return run


def _recording(run: pathlib.Path, number: int, contestant: str, stage: str, env: int, episode_id: int, *, state: int = 2, started: str = "2026-11-19T10:00:00+00:00") -> None:
    directory = run / "episodes" / f"episode_{number:03d}"
    directory.mkdir(parents=True)
    meta = {"episode_id": number, "planner": contestant, "stage": stage, "task_generator_episode_id": episode_id, "outcome_state": state, "recording_started_at": started, "env_ns_root": f"/arena/env_{env}/arena_env_node"}
    (directory / f"episode_{number:03d}.yaml").write_text(yaml.safe_dump(meta))


def _paired_runs(root: pathlib.Path, *, baseline_s: float, contestant_s: float) -> list[pathlib.Path]:
    """A baseline run and a contestant run on every stage with seeds 1 and 2, found after the given seconds."""
    base = _run(root, "base_run", [_row("baseline", s, seed, baseline_s) for s in STAGES for seed in (1, 2)])
    other = _run(root, "sweep_run", [_row("listen", s, seed, contestant_s) for s in STAGES for seed in (1, 2)])
    return [base, other]


def test_load_run_keeps_finished_episodes_and_censors_a_miss_at_the_stage_timeout(tmp_path: pathlib.Path) -> None:
    run = _run(tmp_path, "r", [_row("a", "w0__a", 1, 40.0), _row("a", "w0__a", 2, None), _row("a", "w0__b", 3, None, state=5)])
    episodes = usar.load_run(run).sort("seed")
    assert episodes["seed"].to_list() == [1, 2]
    assert episodes["found"].to_list() == [True, False]
    assert episodes["ttf_s"].to_list() == [40.0, 120.0]
    assert episodes["world"].to_list() == ["w0", "w0"]


def test_load_run_censors_a_miss_at_the_runners_default_timeout_when_the_stage_names_none(tmp_path: pathlib.Path) -> None:
    run = _run(tmp_path, "r", [_row("a", "w0__a", 1, None)], timeout=None)
    assert usar.load_run(run)["ttf_s"].to_list() == [60.0]


def test_load_run_takes_the_latest_attempt_and_fills_gaps_from_superseded_only_when_asked(tmp_path: pathlib.Path) -> None:
    run = _run(tmp_path, "r", [_row("a", "w0__a", 1, 90.0, episode_id=1), _row("a", "w0__a", 1, 50.0, episode_id=2, ts="2026-11-19T11:00:00+00:00")])
    _write_csv(run / "superseded_progress.csv", [_row("a", "w0__a", 1, 10.0), _row("a", "w0__b", 2, 70.0)])
    assert usar.load_run(run)["ttf_s"].to_list() == [50.0]
    filled = usar.load_run(run, superseded=True).sort("seed")
    assert filled["ttf_s"].to_list() == [50.0, 70.0]


def test_load_runs_rejects_a_missing_baseline_and_a_contestant_in_two_runs(tmp_path: pathlib.Path) -> None:
    sweep = _run(tmp_path, "sweep", [_row("listen", "w0__a", 1, 60.0)])
    with pytest.raises(ValueError, match="no contestant baseline"):
        usar.load_runs([sweep])
    base = _run(tmp_path, "base", [_row("baseline", "w0__a", 1, 60.0), _row("listen", "w0__a", 1, 60.0)])
    with pytest.raises(ValueError, match="contestant listen is in runs"):
        usar.load_runs([base, sweep])


def test_load_runs_warns_when_the_runs_drove_different_suites(tmp_path: pathlib.Path) -> None:
    base = _run(tmp_path, "base", [_row("baseline", "w0__a", 1, 60.0)], suite="usar_tune")
    sweep = _run(tmp_path, "sweep", [_row("listen", "w0__a", 1, 60.0)], suite="usar_test")
    with pytest.warns(UserWarning, match="different suites"):
        usar.load_runs([base, sweep])


def test_attach_measures_joins_each_episode_to_the_latest_finished_recording_on_its_env(tmp_path: pathlib.Path) -> None:
    # The step ran on env 0, then again on env 1; both attempts count their episodes from 1.
    run = _run(tmp_path, "r", [_row("baseline", "w0__a", 1, 60.0, env=1, episode_id=1), _row("baseline", "w0__a", 2, 70.0, env=1, episode_id=2)])
    _recording(run, 0, "baseline", "w0__a", 0, 1, started="2026-11-19T09:00:00+00:00")
    _recording(run, 1, "baseline", "w0__a", 1, 1, state=4, started="2026-11-19T12:00:00+00:00")
    _recording(run, 2, "baseline", "w0__a", 1, 1, started="2026-11-19T11:00:00+00:00")
    _recording(run, 3, "baseline", "w0__a", 1, 2, started="2026-11-19T11:05:00+00:00")
    _recording(run, 4, "baseline", "w0__a", 1, 1, started="2026-11-19T10:00:00+00:00")
    metrics = {"episode": [0, 1, 2, 3, 4], "is_reference": [False] * 5, "path_length": [1.0, 2.0, 3.0, 4.0, 5.0], "search_stops": [9.0, 8.0, 5.0, 0.0, 7.0]}
    pl.DataFrame(metrics).write_parquet(run / "combined_metrics.parquet")
    episodes = usar.attach_measures(usar.load_run(run), [run]).sort("seed")
    assert episodes.height == 2
    assert episodes["path_length"].to_list() == [3.0, 4.0]
    assert episodes["search_stops"].to_list() == [5.0, 0.0]
    assert episodes["search_listened_s"].to_list() == [None, None]


def test_attach_measures_leaves_an_unprocessed_run_empty_with_a_warning(tmp_path: pathlib.Path) -> None:
    run = _run(tmp_path, "r", [_row("baseline", "w0__a", 1, 60.0)])
    with pytest.warns(UserWarning, match="arena evaluation run --benchmark-dir r"):
        episodes = usar.attach_measures(usar.load_run(run), [run])
    assert episodes.select(usar.MEASURES).row(0) == (None, None, None)


def test_contestant_table_pairs_on_stage_and_seed_and_puts_the_baseline_first() -> None:
    rows = [("baseline", s, 1, True, 100.0, 0.0) for s in STAGES] + [("listen", s, 1, True, 80.0, 5.0) for s in STAGES] + [("listen", "w0__a", 9, False, 120.0, 5.0)]
    episodes = pl.DataFrame(rows, schema=["contestant", "stage", "seed", "found", "ttf_s", "search_listened_s"], orient="row").with_columns(pl.col("stage").replace_strict(STAGES).alias("world"), pl.lit(10.0).alias("path_length"), pl.lit(2.0).alias("search_stops"))
    table = usar.contestant_table(episodes, n_boot=200)
    assert table["contestant"].to_list() == ["baseline", "listen"]
    base, listen = table.iter_rows(named=True)
    assert (base["pairs"], base["found"], base["ttf_s"], base["diff_s"]) == (4, 4, 100.0, None)
    assert (listen["pairs"], listen["found"], listen["found_baseline"], listen["clusters"]) == (4, 4, 4, 2)
    assert listen["diff_s"] == pytest.approx(-20.0)
    assert listen["saved_per_stop_s"] == pytest.approx(((100.0 - 0.0) - (80.0 - 5.0)) / 5.0)
    assert listen["listened_s"] == pytest.approx(5.0)


def test_contestant_table_needs_episodes_shared_with_the_baseline() -> None:
    episodes = pl.DataFrame({"contestant": ["baseline", "listen"], "world": ["w0", "w0"], "stage": ["w0__a", "w0__b"], "seed": [1, 1], "found": [True, True], "ttf_s": [1.0, 1.0]})
    episodes = episodes.with_columns(pl.lit(None, pl.Float64).alias(m) for m in usar.MEASURES)
    with pytest.raises(ValueError, match="same suite"):
        usar.contestant_table(episodes)


def test_verdict_names_a_winner_only_when_its_difference_excludes_zero() -> None:
    def table(excludes: bool) -> pl.DataFrame:
        return pl.DataFrame({"contestant": ["baseline", "fast", "slow"], "diff_s": [None, -20.0, 5.0], "ci_low": [None, -30.0, -1.0], "ci_high": [None, -10.0, 9.0], "p_holm": [None, 0.01, 0.4], "excludes_zero": [None, excludes, False]})

    winner, result = usar.verdict(table(True))
    assert winner == "fast"
    assert result.startswith("fast finds the person faster than baseline: -20.0 s [-30.0, -10.0]")
    winner, result = usar.verdict(table(False))
    assert winner is None
    assert result.startswith("No contestant finds the person faster than baseline. Closest: fast")


def test_table_markdown_flags_too_few_clusters_for_a_test() -> None:
    row = {c: None for c in usar.TABLE_COLUMNS} | {"contestant": "listen", "pairs": 4, "found": 3, "found_baseline": 4, "diff_s": 12.0, "ci_low": 2.0, "ci_high": 20.0, "p_holm": 0.0, "clusters": 2}
    text = usar.table_markdown(pl.DataFrame([row]), suite="usar_tune", result="null")
    assert "| listen | 4 | 3/4 | 4/4 |  |  | 12.0 | [2, 20] | 0.000 |" in text
    assert "**Only 2 worlds.**" in text
    assert text.rstrip().endswith("null")


def test_main_writes_results_table_and_plot_and_reports_the_null(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    base, sweep = _paired_runs(tmp_path, baseline_s=60.0, contestant_s=70.0)
    with pytest.warns(UserWarning, match="combined_metrics.parquet not found"):
        assert usar.main(["analyze", "--benchmark-dir", str(base), str(sweep), "--boot", "200"]) == 0
    out = base / "usar"
    assert sorted(p.name for p in out.iterdir()) == ["paired_ttf.png", "results.csv", "table.md"]
    assert pl.read_csv(out / "results.csv")["contestant"].to_list() == ["baseline", "listen"]
    assert "No contestant finds the person faster than baseline. Closest: listen, +10.0 s" in capsys.readouterr().out


def test_main_reports_a_missing_baseline_as_a_usage_error(tmp_path: pathlib.Path) -> None:
    _, sweep = _paired_runs(tmp_path, baseline_s=60.0, contestant_s=70.0)
    with pytest.raises(SystemExit) as exit_info:
        usar.main(["analyze", "--benchmark-dir", str(sweep)])
    assert exit_info.value.code == 2


def test_launch_line_is_the_runners_line_for_the_winner_on_the_bundle_world_path(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("arena_evaluation.benchmark.runner")
    run = _run(tmp_path / "suites" / "usar_tune", "run", [], contest=[{"name": "listen", "acoustics": "arena", "robot.hearing.driver.stop_rule": "on_call"}])
    (run.parent / "worlds").mkdir()
    manifest = yaml.safe_load((run / "manifest.yaml").read_text()) | {"suite_provenance": {"path": str(run.parent / "suite.yaml")}}
    (run / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    line = usar.launch_line(run, "listen")
    assert line.startswith(f"ARENA_WORLD_PATH={run.parent / 'worlds'} arena launch sim:=flatsim ")
    for arg in ("robot:=turtlebot", "world:=w0", "acoustics:=arena", "robot.hearing.driver.stop_rule:=on_call"):
        assert arg in line.split()
