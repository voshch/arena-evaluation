"""USAR study - search driver configurations that stop to listen, paired against a baseline that only explores.

Every contestant drives all stages of one generate_usar suite bundle with the same seeds. The baseline contestant
(goal_rule nearest, stop_rule never) is the row every other row is read against.

analyze: per contestant, time to find minus the baseline's on the same (stage, seed), a missed person counted at the
  stage timeout; cluster bootstrap over worlds, Holm across the contestants. Also share found, stops, listening
  seconds, path length and driving seconds saved per stopped second.

    ros2 run arena_simulation_setup generate_usar configs/usar/tune.yaml --write --suite usar
    arena evaluation benchmark --suite usar_tune --contest usar_baseline
    arena evaluation benchmark --suite usar_tune --contest usar_sweep
    arena evaluation run --benchmark-dir <run_id>    # optional, adds stops, listening seconds and path length
    python -m arena_evaluation.experiments.usar analyze --benchmark-dir <baseline_run_id> <sweep_run_id>
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
import sys
import warnings

import polars as pl
import yaml

from arena_evaluation.benchmark.config import _DEFAULT_TIMEOUT_S
from arena_evaluation.benchmark.state import SUPERSEDED_PROGRESS, progress_rows
from arena_evaluation.experiments.beckon import stage_timeouts
from arena_evaluation.experiments.stats import DEFAULT_BOOT, summarize_family

BASELINE = "baseline"
FOUND = 2  # arena_runtime_msgs/msg/EpisodeRecord.SUCCESS
FINISHED = (2, 3)  # SUCCESS and FAILED, the search ran to its end; SKIPPED and FATAL say nothing about the search
FINISHED_AFTER = re.compile(r"finished after (?P<seconds>[\d.]+)s")
MEASURES = ("path_length", "search_stops", "search_listened_s")
FEW_CLUSTERS = 5
PLOT_COLOR, PLOT_MUTED, PLOT_GRID = "#2a78d6", "#898781", "#e1e0d9"
TABLE_COLUMNS = ["contestant", "pairs", "found", "found_baseline", "ttf_s", "ttf_baseline_s", "diff_s", "ci_low", "ci_high", "p", "p_holm", "excludes_zero", "clusters", "stops", "listened_s", "path_m", "saved_per_stop_s"]

EPISODE_SCHEMA = {
    "run": pl.String,
    "contestant": pl.String,
    "world": pl.String,
    "stage": pl.String,
    "seed": pl.Int64,
    "env": pl.Int64,
    "episode_id": pl.Int64,
    "found": pl.Boolean,
    "ttf_s": pl.Float64,
}


# -- loading -----------------------------------------------------------------------------------


def _manifest(run_dir: pathlib.Path) -> dict:
    path = run_dir / "manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found, pass a benchmark run directory <data_root>/<run_id>")
    return yaml.safe_load(path.read_text())


def _time_to_find(row: dict[str, str]) -> float:
    """The task's own 'finished after X s', the runner's runtime when the text lacks it."""
    match = FINISHED_AFTER.search(row["outcome_info"])
    return float(match["seconds"]) if match else float(row["runtime_s"])


def load_run(run_dir: pathlib.Path, *, superseded: bool = False) -> pl.DataFrame:
    """One row per finished episode, the latest per (contestant, stage, seed), with time to find censored at the stage timeout."""
    timeouts = stage_timeouts(run_dir)
    sources = [progress_rows(run_dir)]
    if superseded and (run_dir / SUPERSEDED_PROGRESS).is_file():
        with (run_dir / SUPERSEDED_PROGRESS).open(newline="") as fh:
            sources.append(list(csv.DictReader(fh)))
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for rows in sources:
        for row in sorted(rows, key=lambda r: r["ts_iso"], reverse=True):
            key = (row["contestant"], row["stage"], row["seed"])
            if row["outcome_state"].isdigit() and int(row["outcome_state"]) in FINISHED and key not in latest:
                latest[key] = row
    episodes = []
    for (contestant, stage, seed), row in latest.items():
        found = int(row["outcome_state"]) == FOUND
        episodes.append(
            {
                "run": run_dir.name,
                "contestant": contestant,
                "world": row["world"],
                "stage": stage,
                "seed": int(seed),
                "env": int(row["env_id"]),
                "episode_id": int(row["episode_id"]),
                "found": found,
                "ttf_s": _time_to_find(row) if found else timeouts.get(stage, _DEFAULT_TIMEOUT_S),
            }
        )
    return pl.DataFrame(episodes, schema=EPISODE_SCHEMA)


def load_runs(run_dirs: list[pathlib.Path], *, baseline: str = BASELINE, superseded: bool = False) -> pl.DataFrame:
    """Episodes of all runs, which must not share a contestant, with the baseline among them and a warning when the suites differ."""
    suites = {run_dir.name: _manifest(run_dir).get("suite_name") for run_dir in run_dirs}
    if len(set(suites.values())) > 1:
        warnings.warn(f"runs drove different suites {suites}, only episodes on the same (stage, seed) are paired", stacklevel=2)
    frames = [load_run(run_dir, superseded=superseded) for run_dir in run_dirs]
    owners: dict[str, str] = {}
    for frame in frames:
        for contestant, run in frame.select("contestant", "run").unique().iter_rows():
            if contestant in owners:
                raise ValueError(f"contestant {contestant} is in runs {owners[contestant]} and {run}, pass only one of them")
            owners[contestant] = run
    if baseline not in owners:
        raise ValueError(f"no contestant {baseline} in the runs (found {sorted(owners)}), add its run or name it with --baseline")
    return pl.concat(frames)


def _recorded_episodes(run_dir: pathlib.Path) -> pl.DataFrame:
    """Recording number of every finished episode keyed like its progress row, the latest recording where an episode was retried."""
    rows = []
    for path in sorted((run_dir / "episodes").glob("episode_*/episode_*.yaml")):
        meta = yaml.safe_load(path.read_text()) or {}
        env = re.search(r"env_(\d+)", meta.get("env_ns_root") or "")
        if env is None or meta.get("outcome_state") not in FINISHED:
            continue
        rows.append(
            {
                "episode": int(meta["episode_id"]),
                "contestant": meta["planner"],
                "stage": meta["stage"],
                "env": int(env[1]),
                "episode_id": int(meta["task_generator_episode_id"]),
                "recorded_at": meta.get("recording_started_at") or "",
            }
        )
    schema = {"episode": pl.Int64, "contestant": pl.String, "stage": pl.String, "env": pl.Int64, "episode_id": pl.Int64, "recorded_at": pl.String}
    return pl.DataFrame(rows, schema=schema).sort("recorded_at").unique(subset=["contestant", "stage", "env", "episode_id"], keep="last").drop("recorded_at")


def attach_measures(episodes: pl.DataFrame, run_dirs: list[pathlib.Path]) -> pl.DataFrame:
    """Episodes with path length, stops and listening seconds from each run's combined_metrics.parquet, null where a run is not processed."""
    parts = []
    for run_dir in run_dirs:
        path = run_dir / "combined_metrics.parquet"
        if not path.is_file():
            warnings.warn(f"{path} not found, its stops, listening seconds and path length stay empty; run `arena evaluation run --benchmark-dir {run_dir.name}` first", stacklevel=2)
            continue
        metrics = pl.scan_parquet(path)
        columns = metrics.collect_schema().names()
        if "is_reference" in columns:
            metrics = metrics.filter(~pl.col("is_reference").fill_null(False))
        metrics = metrics.select("episode", *(pl.col(m).cast(pl.Float64) if m in columns else pl.lit(None, pl.Float64).alias(m) for m in MEASURES)).collect()
        parts.append(_recorded_episodes(run_dir).join(metrics, on="episode").drop("episode").with_columns(pl.lit(run_dir.name).alias("run")))
    if not parts:
        return episodes.with_columns(pl.lit(None, pl.Float64).alias(m) for m in MEASURES)
    return episodes.join(pl.concat(parts), on=["run", "contestant", "stage", "env", "episode_id"], how="left")


# -- analysis ----------------------------------------------------------------------------------


def contestant_table(episodes: pl.DataFrame, *, baseline: str = BASELINE, cluster: str = "world", n_boot: int = DEFAULT_BOOT, seed: int = 0) -> pl.DataFrame:
    """One row per contestant, baseline first, then by paired difference in time to find with its cluster bootstrap interval and Holm p across the contestants."""
    base = episodes.filter(pl.col("contestant") == baseline)
    pairs = episodes.filter(pl.col("contestant") != baseline).join(base.select("stage", "seed", "found", "ttf_s", "search_listened_s"), on=["stage", "seed"], suffix="_base")
    if pairs.is_empty():
        raise ValueError(f"no episode shares a (stage, seed) with {baseline}, the runs must drive the same suite")
    family = summarize_family(pairs.with_columns((pl.col("ttf_s") - pl.col("ttf_s_base")).alias("diff")), cell=["contestant"], scenario=cluster, n_boot=n_boot, seed=seed)
    known = pl.col("search_listened_s").is_not_null() & pl.col("search_listened_s_base").is_not_null()
    listened = pl.col("search_listened_s").filter(known).sum()
    driving_saved = ((pl.col("ttf_s_base") - pl.col("search_listened_s_base")) - (pl.col("ttf_s") - pl.col("search_listened_s"))).filter(known).sum()
    paired = pairs.group_by("contestant").agg(
        pl.len().alias("pairs"),
        pl.col("found").sum(),
        pl.col("found_base").sum().alias("found_baseline"),
        pl.col("ttf_s").mean(),
        pl.col("ttf_s_base").mean().alias("ttf_baseline_s"),
        pl.when(listened > 0).then(driving_saved / listened).alias("saved_per_stop_s"),
    )
    paired = paired.join(family.rename({"estimate": "diff_s", "n_scenarios": "clusters"}).drop("n_pairs"), on="contestant").sort("diff_s")
    own = base.group_by("contestant").agg(pl.len().alias("pairs"), pl.col("found").sum(), pl.col("ttf_s").mean())
    measures = episodes.group_by("contestant").agg(pl.col("search_stops").mean().alias("stops"), pl.col("search_listened_s").mean().alias("listened_s"), pl.col("path_length").mean().alias("path_m"))
    return pl.concat([own, paired], how="diagonal_relaxed").join(measures, on="contestant", how="left", maintain_order="left").select(TABLE_COLUMNS)


def verdict(table: pl.DataFrame, baseline: str = BASELINE) -> tuple[str | None, str]:
    """The contestant ahead of the baseline by the most among those whose difference excludes zero, else None and the closest contestant as the null result."""
    rivals = table.filter(pl.col("contestant") != baseline)
    winners = rivals.filter(pl.col("excludes_zero") & (pl.col("diff_s") < 0))
    best = (winners if winners.height else rivals).row(0, named=True)
    result = f"{best['diff_s']:+.1f} s [{best['ci_low']:+.1f}, {best['ci_high']:+.1f}], Holm p {best['p_holm']:.3f}"
    if winners.height:
        return best["contestant"], f"{best['contestant']} finds the person faster than {baseline}: {result}."
    return None, f"No contestant finds the person faster than {baseline}. Closest: {best['contestant']}, {result}."


# -- report ------------------------------------------------------------------------------------


def launch_line(run_dir: pathlib.Path, contestant: str) -> str:
    """The runner's arena launch line for the contestant on the suite's first stage, on the bundle's world path when it has one."""
    from arena_evaluation.benchmark.config import Contest, Suite
    from arena_evaluation.benchmark.runner import build_launch_args  # imports rclpy, the rest of the study runs without ROS
    from arena_evaluation.benchmark.step import Step

    manifest = _manifest(run_dir)
    chosen = next(c for c in Contest.parse(manifest["contest_name"], manifest["contest"]).contestants if c.name == contestant)
    stage = Suite.parse(manifest["suite_name"], manifest["suite"]).stages[0]
    line = " ".join(["arena launch", *build_launch_args(Step(contestant=chosen, stage=stage), manifest.get("simulator"))])
    suite_path = (manifest.get("suite_provenance") or {}).get("path")
    worlds = pathlib.Path(suite_path).parent / "worlds" if suite_path else None
    return f"ARENA_WORLD_PATH={worlds} {line}" if worlds is not None and worlds.is_dir() else line


def _number(value: float | None, digits: int = 1) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def table_markdown(table: pl.DataFrame, *, suite: str, baseline: str = BASELINE, cluster: str = "world", n_boot: int = DEFAULT_BOOT, result: str = "", launch: str | None = None) -> str:
    """Configuration against measures, what each column means, the result and the winner's launch line."""
    header = ["contestant", "pairs", "found", "found_baseline", "ttf_s", "ttf_baseline_s", "diff_s", f"ci_{cluster}", "p_holm", "stops", "listened_s", "path_m", "saved_per_stop_s"]
    lines = [f"# {suite}: configuration against measures", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in table.iter_rows(named=True):
        cells = [
            r["contestant"],
            str(r["pairs"]),
            f"{r['found']}/{r['pairs']}",
            "" if r["found_baseline"] is None else f"{r['found_baseline']}/{r['pairs']}",
            _number(r["ttf_s"]),
            _number(r["ttf_baseline_s"]),
            _number(r["diff_s"]),
            "" if r["ci_low"] is None else f"[{_number(r['ci_low'], 0)}, {_number(r['ci_high'], 0)}]",
            _number(r["p_holm"], 3),
            _number(r["stops"]),
            _number(r["listened_s"]),
            _number(r["path_m"]),
            _number(r["saved_per_stop_s"], 2),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Columns",
        "",
        f"- `pairs`: episodes of the contestant and of {baseline} on the same (stage, seed). In the {baseline} row: all its finished episodes.",
        "- `found`, `found_baseline`, `ttf_s`, `ttf_baseline_s`: share found and mean time to find over those pairs, a missed person counted at the stage timeout.",
        f"- `diff_s`: mean paired difference of time to find, contestant minus {baseline}. Negative = faster.",
        f"- `ci_{cluster}`, `p_holm`: 95 % percentile cluster bootstrap of `diff_s` resampling whole {cluster}s, {n_boot} draws, and its two-sided p after Holm across the contestants (`experiments/stats.py`).",
        "- `stops`, `listened_s`, `path_m`: mean over all finished episodes of the contestant, from `combined_metrics.parquet`, empty for a run not processed with `arena evaluation run`.",
        "- `saved_per_stop_s`: sum over pairs of ((T_base - L_base) - (T - L)) divided by the sum of L, T = time to find, L = listening seconds. Negative = the contestant lost more driving time than it spent listening.",
    ]
    clusters = table.select(pl.col("clusters").min()).item()
    if clusters is not None and clusters < FEW_CLUSTERS:
        lines.append(f"- **Only {clusters} {cluster}s.** A cluster bootstrap over so few clusters has a handful of distinct resamples, so `ci_{cluster}` and `p_holm` describe the runs but do not test them. More worlds in the suite make them a test.")
    lines += ["", "## Result", "", result]
    if launch is not None:
        lines += ["", "```sh", launch, "```"]
    return "\n".join(lines) + "\n"


def plot_paired(table: pl.DataFrame, path: pathlib.Path, *, baseline: str = BASELINE, cluster: str = "world", n_boot: int = DEFAULT_BOOT) -> None:
    """Paired time to find against the baseline per contestant, the mean difference with its 95 % cluster bootstrap interval."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table.filter(pl.col("contestant") != baseline).reverse()
    y = list(range(rows.height))
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * rows.height + 1.6))
    ax.hlines(y, rows["ci_low"], rows["ci_high"], color=PLOT_COLOR, linewidth=2, label=f"95 % interval, cluster bootstrap over {cluster}s ({n_boot} draws)")
    ax.plot(rows["diff_s"], y, "o", color=PLOT_COLOR, markersize=8, label=f"mean paired difference, contestant - {baseline}")
    ax.axvline(0.0, color=PLOT_MUTED, linewidth=1, label="no difference")
    ax.set_yticks(y, [f"{c} ({n} pairs)" for c, n in rows.select("contestant", "pairs").iter_rows()])
    ax.set_xlabel(f"difference in time to find [s], negative = faster than {baseline}\na missed person counts at the stage timeout")
    ax.set_title(f"Paired time to find against {baseline}", loc="left")
    ax.grid(axis="x", color=PLOT_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_extra_artists=[legend], bbox_inches="tight")
    plt.close(fig)


def write_report(table: pl.DataFrame, episodes: pl.DataFrame, run_dirs: list[pathlib.Path], out_dir: pathlib.Path, *, baseline: str = BASELINE, cluster: str = "world", n_boot: int = DEFAULT_BOOT) -> str:
    """results.csv, table.md and paired_ttf.png in out_dir, returns the result with the winner's launch line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_dir / "results.csv")
    winner, result = verdict(table, baseline)
    launch = None
    if winner is not None:
        run = episodes.filter(pl.col("contestant") == winner)["run"][0]
        launch = launch_line(next(d for d in run_dirs if d.name == run), winner)
    suite = _manifest(run_dirs[0]).get("suite_name") or "usar"
    (out_dir / "table.md").write_text(table_markdown(table, suite=suite, baseline=baseline, cluster=cluster, n_boot=n_boot, result=result, launch=launch))
    plot_paired(table, out_dir / "paired_ttf.png", baseline=baseline, cluster=cluster, n_boot=n_boot)
    return result if launch is None else f"{result}\n{launch}"


def analyze(run_dirs: list[pathlib.Path], *, baseline: str = BASELINE, cluster: str = "world", superseded: bool = False, n_boot: int = DEFAULT_BOOT, out_dir: pathlib.Path | None = None) -> tuple[pl.DataFrame, str]:
    """Pair the runs' contestants against the baseline and write the report, by default to <first run>/usar."""
    episodes = attach_measures(load_runs(run_dirs, baseline=baseline, superseded=superseded), run_dirs)
    table = contestant_table(episodes, baseline=baseline, cluster=cluster, n_boot=n_boot)
    return table, write_report(table, episodes, run_dirs, out_dir or run_dirs[0] / "usar", baseline=baseline, cluster=cluster, n_boot=n_boot)


def main(argv: list[str] | None = None) -> int:
    from arena_evaluation.cli import resolve_paths

    p = argparse.ArgumentParser(prog="arena_evaluation.experiments.usar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="paired analysis of benchmark runs on one usar suite bundle")
    a.add_argument("--benchmark-dir", type=pathlib.Path, nargs="+", required=True, help="run directories or run ids, together holding the baseline and the other contestants")
    a.add_argument("--baseline", default=BASELINE, help="contestant every other one is paired against")
    a.add_argument("--cluster", choices=["world", "stage"], default="world", help="unit the bootstrap resamples")
    a.add_argument("--superseded", action="store_true", help="fill episodes missing from progress.csv with superseded attempts")
    a.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    a.add_argument("--out", type=pathlib.Path, default=None, help="output directory, default <first run>/usar")
    args = resolve_paths(p.parse_args(argv))
    try:
        table, result = analyze(args.benchmark_dir, baseline=args.baseline, cluster=args.cluster, superseded=args.superseded, n_boot=args.boot, out_dir=args.out)
    except (FileNotFoundError, ValueError) as error:
        p.error(str(error))
    summary = table.select("contestant", "pairs", "found", pl.col("ttf_s", "diff_s", "ci_low", "ci_high").round(1), pl.col("p_holm").round(3))
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=160, fmt_str_lengths=40):
        print(summary)
    print(f"\n{result}\nwritten to {args.out or args.benchmark_dir[0] / 'usar'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
