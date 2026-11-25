from __future__ import annotations

import pathlib

import polars as pl
import pytest
from star_benchmark import write_benchmark

from arena_evaluation.experiments import usar

STAGES = {"w0__a": "w0", "w0__b": "w0", "w1__a": "w1", "w1__b": "w1"}


def _row(contestant: str, stage: str, seed: int, found_after: float | None, *, episode_id: int | None = None, timeout_s: float = 120.0) -> dict:
    """A summary row of a search episode; found_after None is a search that ran out of time."""
    return {
        "episode_id": episode_id,
        "planner": contestant,
        "stage": stage,
        "world": STAGES[stage],
        "seed": seed,
        "result": "GOAL_REACHED",
        "is_reference": False,
        "search_found": found_after is not None,
        "search_time_to_find_s": timeout_s if found_after is None else found_after,
        "path_length_m": 10.0,
        "search_stops": 2,
        "search_listened_s": 4.0,
    }


def _run(root: pathlib.Path, name: str, rows: list[dict]) -> pathlib.Path:
    numbered = [{**row, "episode_id": index if row["episode_id"] is None else row["episode_id"]} for index, row in enumerate(rows)]
    return write_benchmark(root / name, numbered)


def _paired_runs(root: pathlib.Path, *, baseline_s: float, contestant_s: float) -> list[pathlib.Path]:
    """A baseline run and a contestant run on every stage with seeds 1 and 2, found after the given seconds."""
    base = _run(root, "base_run", [_row("baseline", s, seed, baseline_s) for s in STAGES for seed in (1, 2)])
    other = _run(root, "sweep_run", [_row("listen", s, seed, contestant_s) for s in STAGES for seed in (1, 2)])
    return [base, other]


def test_load_run_keeps_scored_search_episodes_and_the_last_recording_of_a_rerun(tmp_path: pathlib.Path) -> None:
    rows = [
        _row("a", "w0__a", 1, 40.0, episode_id=0),
        _row("a", "w0__a", 1, 30.0, episode_id=3),
        _row("a", "w0__a", 2, None, episode_id=1),
        {**_row("a", "w0__b", 3, 10.0, episode_id=2), "search_found": None},
        {**_row("a", "w1__a", 4, 20.0, episode_id=4), "is_reference": True},
    ]
    episodes = usar.load_run(_run(tmp_path, "r", rows)).sort("seed")
    assert episodes["seed"].to_list() == [1, 2]
    assert episodes["found"].to_list() == [True, False]
    assert episodes["ttf_s"].to_list() == [30.0, 120.0]
    assert episodes["world"].to_list() == ["w0", "w0"]
    assert episodes["contestant"].to_list() == ["a", "a"]


def test_load_run_refuses_an_unprocessed_run_and_a_run_without_seeds(tmp_path: pathlib.Path) -> None:
    (tmp_path / "bare").mkdir()
    with pytest.raises(FileNotFoundError, match="no Star Schema tables"):
        usar.load_run(tmp_path / "bare")
    null_seed = _run(tmp_path, "null", [{**_row("a", "w0__a", 1, 40.0), "seed": None}])
    with pytest.raises(ValueError, match="without a seed"):
        usar.load_run(null_seed)


def test_load_runs_rejects_a_missing_baseline_and_a_contestant_in_two_runs(tmp_path: pathlib.Path) -> None:
    sweep = _run(tmp_path, "sweep", [_row("listen", "w0__a", 1, 60.0)])
    with pytest.raises(ValueError, match="no contestant baseline"):
        usar.load_runs([sweep])
    base = _run(tmp_path, "base", [_row("baseline", "w0__a", 1, 60.0), _row("listen", "w0__a", 1, 60.0)])
    with pytest.raises(ValueError, match="contestant listen is in runs"):
        usar.load_runs([base, sweep])


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


def test_verdict_names_a_winner_only_when_its_difference_excludes_zero_and_flags_few_worlds() -> None:
    def table(excludes: bool, clusters: int) -> pl.DataFrame:
        return pl.DataFrame({"contestant": ["baseline", "fast", "slow"], "diff_s": [None, -20.0, 5.0], "ci_low": [None, -30.0, -1.0], "ci_high": [None, -10.0, 9.0], "p_holm": [None, 0.01, 0.4], "excludes_zero": [None, excludes, False], "clusters": [None, clusters, clusters]})

    winner, result = usar.verdict(table(True, 8))
    assert winner == "fast"
    assert result == "fast finds the person faster than baseline: -20.0 s [-30.0, -10.0], Holm p 0.010."
    winner, result = usar.verdict(table(False, 2))
    assert winner is None
    assert result.startswith("No contestant finds the person faster than baseline. Closest: fast")
    assert result.endswith("Only 2 worlds: the interval describes these runs and does not test them.")


def test_main_writes_results_and_reports_the_null(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    base, sweep = _paired_runs(tmp_path, baseline_s=60.0, contestant_s=70.0)
    assert usar.main(["analyze", "--benchmark-dir", str(base), str(sweep), "--boot", "200"]) == 0
    assert sorted(p.name for p in (base / "usar").iterdir()) == ["results.csv"]
    assert pl.read_csv(base / "usar" / "results.csv")["contestant"].to_list() == ["baseline", "listen"]
    assert "No contestant finds the person faster than baseline. Closest: listen, +10.0 s" in capsys.readouterr().out


def test_main_reports_a_missing_baseline_as_a_usage_error(tmp_path: pathlib.Path) -> None:
    _, sweep = _paired_runs(tmp_path, baseline_s=60.0, contestant_s=70.0)
    with pytest.raises(SystemExit) as exit_info:
        usar.main(["analyze", "--benchmark-dir", str(sweep)])
    assert exit_info.value.code == 2
