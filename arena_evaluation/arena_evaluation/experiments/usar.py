"""USAR study - search driver configurations that stop to listen, paired against a baseline that only explores.

Every contestant drives all stages of one generate_usar suite bundle with the same seeds. The baseline contestant
(goal_rule nearest, stop_rule never) is the row every other row is read against.

analyze: per contestant, time to find minus the baseline's on the same (stage, seed), a missed person counted at the
  stage timeout; cluster bootstrap over worlds, Holm across the contestants. Also share found, stops, listening
  seconds, path length and driving seconds saved per listened second.

    ros2 run arena_simulation_setup generate_usar configs/usar/tune.yaml --write --suite usar
    arena evaluation benchmark --suite usar_tune --contest usar_baseline
    arena evaluation benchmark --suite usar_tune --contest usar_sweep
    arena evaluation process --benchmark-dir <run_id>
    python -m arena_evaluation.experiments.usar analyze --benchmark-dir <baseline_run_id> <sweep_run_id>
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import polars as pl

from arena_evaluation.experiments.beckon import EVALUATED_STATUSES
from arena_evaluation.experiments.stats import DEFAULT_BOOT, summarize_family

BASELINE = "baseline"
MEASURES = ("path_length", "search_stops", "search_listened_s")
REQUIRED = ("episode", "planner", "stage", "map", "seed", "status", "search_found", "search_time_to_find_s", *MEASURES)
FEW_CLUSTERS = 5
TABLE_COLUMNS = ["contestant", "pairs", "found", "found_baseline", "ttf_s", "ttf_baseline_s", "diff_s", "ci_low", "ci_high", "p", "p_holm", "excludes_zero", "clusters", "stops", "listened_s", "path_m", "saved_per_stop_s"]


# -- loading -----------------------------------------------------------------------------------


def _metrics_path(run_dir: pathlib.Path) -> pathlib.Path:
    path = run_dir / "combined_metrics.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"no combined_metrics.parquet in {run_dir}, run `arena evaluation process --benchmark-dir {run_dir.name}` first")
    return path


def load_run(run_dir: pathlib.Path) -> pl.DataFrame:
    """One row per evaluated search episode of the run, the last recording where a (contestant, stage, seed) ran twice."""
    path = _metrics_path(run_dir)
    metrics = pl.read_parquet(path)
    missing = [column for column in REQUIRED if column not in metrics.columns]
    if missing:
        raise ValueError(f"{path} has no column {', '.join(missing)}, process the run again with this arena_evaluation")
    if "is_reference" in metrics.columns:
        metrics = metrics.filter(~pl.col("is_reference").fill_null(False))
    episodes = metrics.filter(pl.col("status").is_in(EVALUATED_STATUSES) & pl.col("search_found").is_not_null())
    if episodes["seed"].null_count():
        raise ValueError(f"{path} has search episodes without a seed, process the run again with this arena_evaluation")
    return (
        episodes.sort("episode")
        .unique(subset=["planner", "stage", "seed"], keep="last", maintain_order=True)
        .select(
            pl.lit(run_dir.name).alias("run"),
            pl.col("planner").alias("contestant"),
            pl.col("map").alias("world"),
            "stage",
            "seed",
            pl.col("search_found").alias("found"),
            pl.col("search_time_to_find_s").alias("ttf_s"),
            *(pl.col(m).cast(pl.Float64) for m in MEASURES),
        )
    )


def load_runs(run_dirs: list[pathlib.Path], *, baseline: str = BASELINE) -> pl.DataFrame:
    """Episodes of all runs, which must not share a contestant, with the baseline among them."""
    frames = [load_run(run_dir) for run_dir in run_dirs]
    owners: dict[str, str] = {}
    for frame in frames:
        for contestant, run in frame.select("contestant", "run").unique().iter_rows():
            if contestant in owners:
                raise ValueError(f"contestant {contestant} is in runs {owners[contestant]} and {run}, pass only one of them")
            owners[contestant] = run
    if baseline not in owners:
        raise ValueError(f"no contestant {baseline} in the runs (found {sorted(owners)}), add its run or name it with --baseline")
    return pl.concat(frames)


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


def verdict(table: pl.DataFrame, *, baseline: str = BASELINE, cluster: str = "world") -> tuple[str | None, str]:
    """The contestant ahead of the baseline by the most among those whose difference excludes zero, else None and the closest contestant as the null result."""
    rivals = table.filter(pl.col("contestant") != baseline)
    winners = rivals.filter(pl.col("excludes_zero") & (pl.col("diff_s") < 0))
    best = (winners if winners.height else rivals).row(0, named=True)
    result = f"{best['diff_s']:+.1f} s [{best['ci_low']:+.1f}, {best['ci_high']:+.1f}], Holm p {best['p_holm']:.3f}"
    text = f"{best['contestant']} finds the person faster than {baseline}: {result}." if winners.height else f"No contestant finds the person faster than {baseline}. Closest: {best['contestant']}, {result}."
    clusters = table.select(pl.col("clusters").min()).item()
    if clusters is not None and clusters < FEW_CLUSTERS:
        text += f" Only {clusters} {cluster}s: the interval describes these runs and does not test them."
    return (best["contestant"] if winners.height else None), text


def analyze(run_dirs: list[pathlib.Path], *, baseline: str = BASELINE, cluster: str = "world", n_boot: int = DEFAULT_BOOT, out_dir: pathlib.Path | None = None) -> tuple[pl.DataFrame, str]:
    """Pair the runs' contestants against the baseline and write results.csv, by default to <first run>/usar."""
    table = contestant_table(load_runs(run_dirs, baseline=baseline), baseline=baseline, cluster=cluster, n_boot=n_boot)
    out_dir = out_dir or run_dirs[0] / "usar"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_dir / "results.csv")
    return table, verdict(table, baseline=baseline, cluster=cluster)[1]


def main(argv: list[str] | None = None) -> int:
    from arena_evaluation.cli import resolve_paths

    p = argparse.ArgumentParser(prog="arena_evaluation.experiments.usar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="paired analysis of processed benchmark runs on one usar suite bundle")
    a.add_argument("--benchmark-dir", type=pathlib.Path, nargs="+", required=True, help="run directories or run ids, together holding the baseline and the other contestants")
    a.add_argument("--baseline", default=BASELINE, help="contestant every other one is paired against")
    a.add_argument("--cluster", choices=["world", "stage"], default="world", help="unit the bootstrap resamples")
    a.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    a.add_argument("--out", type=pathlib.Path, default=None, help="output directory, default <first run>/usar")
    args = resolve_paths(p.parse_args(argv))
    try:
        table, result = analyze(args.benchmark_dir, baseline=args.baseline, cluster=args.cluster, n_boot=args.boot, out_dir=args.out)
    except (FileNotFoundError, ValueError) as error:
        p.error(str(error))
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=160, fmt_str_lengths=40):
        print(table.select("contestant", "pairs", "found", pl.col("ttf_s", "diff_s", "ci_low", "ci_high").round(1), pl.col("p_holm").round(3), pl.col("stops", "listened_s").round(1)))
    print(f"\n{result}\nwritten to {args.out or args.benchmark_dir[0] / 'usar'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
