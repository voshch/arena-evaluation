"""Study A - two-person contact interactions, hug and handshake with the robot as bystander (EVALUATION.md §A).

generate: L scenarios per contact kind on map_empty, each run in two arms that share scenario and seeds:
  contact     - the pair walks to its meeting points, closes to contact, the clip plays, holds, releases
  locomotion  - same approach and meeting points, holds at the standing distance, no clip (motion-matched control)
  In both arms the pair then separates to its exit points and stands.

analyze: paired contact - locomotion differences per (planner, interaction, metric), cluster bootstrap over
  scenarios, Holm within the family; plus the release-time window check (post-release stall share vs time share).

    python -m arena_evaluation.experiments.hhi generate --scenarios 15
    python -m arena_evaluation.experiments.hhi analyze --benchmark-dir <data_root>/<run_id>
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
import typing

import numpy as np
import polars as pl
import yaml

from arena_evaluation.experiments.stats import DEFAULT_BOOT, cluster_bootstrap, holm, paired_differences, summarize_family

KINDS = {"hug": "HUG", "shake": "SHAKE_HAND"}
ARMS = {"contact": "enabled", "loco": "locomotion_only"}
STAGE_RE = re.compile(r"^hhi_(?P<kind>hug|shake)_(?P<scenario>\d+)_(?P<arm>contact|loco)$")

MAP = "map_empty"
BOUNDS = (1.5, 1.5, 28.5, 21.5)  # map_empty interior, walls at 0 / 30 x 0 / 23
STANDING_DISTANCE = 1.2  # locomotion-only hold separation
MEET_OFFSET = STANDING_DISTANCE / 2 + 0.1  # meeting point off the midpoint, identical in both arms
PED_SPEED = 1.2  # m/s, fixed so the timing is the scenario's, not the sample's
ROBOT_SPEED = 0.6  # m/s, nominal for timing the pair's arrival against the robot's
ROBOT_LATENCY = 2.0  # s before the robot moves
TIMEOUT = "180s"  # MPPI needs ~130 s on these ~26 m paths (pilot 2026-09-11)
EPISODES = 3  # seeds per scenario
SEED_BASE = 71_000

_ROOT = pathlib.Path(__file__).resolve()


def _default_scenarios_dir() -> pathlib.Path:
    for parent in _ROOT.parents:
        candidate = parent / "arena_simulation_setup" / "worlds" / MAP / "scenarios"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"arena_simulation_setup/worlds/{MAP}/scenarios not found above {_ROOT}")


def _default_suite_path() -> pathlib.Path:
    return _ROOT.parents[1] / "configs" / "benchmark" / "suites" / "hhi_contact.yaml"


# -- generation --------------------------------------------------------------------------------


class Geometry(typing.NamedTuple):
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    meet: tuple[float, float]  # pair midpoint
    axis: float  # pair axis heading, rad
    ped_start: tuple[tuple[float, float], tuple[float, float]]
    ped_meet: tuple[tuple[float, float], tuple[float, float]]
    ped_exit: tuple[tuple[float, float], tuple[float, float]]
    hold_s: float
    lead_s: float  # how long before the robot's nominal arrival the pair starts holding
    wait_s: float  # the pair stands at its start points this long, then walks


def _inside(p: tuple[float, float] | np.ndarray, margin: float = 0.0) -> bool:
    x0, y0, x1, y1 = BOUNDS
    return x0 + margin <= p[0] <= x1 - margin and y0 + margin <= p[1] <= y1 - margin


def sample_geometry(rng: np.random.Generator, max_tries: int = 1000) -> Geometry:
    """One scenario: the pair meets near the robot's straight path and holds around its arrival."""
    for _ in range(max_tries):
        sy, gy = rng.uniform(4.0, 19.0), rng.uniform(4.0, 19.0)
        if abs(sy - gy) > 6.0:
            continue
        s, g = np.array([2.5, sy]), np.array([27.5, gy])
        if rng.random() < 0.5:
            s, g = g, s
        path = g - s
        length = float(np.linalg.norm(path))
        along, normal = path / length, np.array([-path[1], path[0]]) / length
        m = s + rng.uniform(0.35, 0.6) * path + rng.uniform(-1.0, 1.0) * normal
        heading = math.atan2(along[1], along[0])
        axis = heading + math.radians(rng.uniform(15.0, 165.0))
        u = np.array([math.cos(axis), math.sin(axis)])
        lead = rng.uniform(1.0, 6.0)
        t_hold = (np.linalg.norm(m - s) / ROBOT_SPEED + ROBOT_LATENCY) - lead
        approach = rng.uniform(3.0, 8.0)
        wait = t_hold - approach / PED_SPEED
        if wait < 0.0:
            continue
        meet = (m - MEET_OFFSET * u, m + MEET_OFFSET * u)
        starts = (meet[0] - approach * u, meet[1] + approach * u)
        exits = []
        for side, p in zip((-1.0, 1.0), meet, strict=True):
            turn = math.radians(rng.uniform(-60.0, 60.0))
            d = side * np.array([math.cos(axis + turn), math.sin(axis + turn)])
            exits.append(p + rng.uniform(4.0, 7.0) * d)
        points = [*starts, *exits, m]
        if not all(_inside(p, 0.5) for p in points):
            continue
        if min(np.linalg.norm(p - q) for p in starts for q in (s, g)) < 2.0:
            continue
        return Geometry(
            start=(float(s[0]), float(s[1]), heading),
            goal=(float(g[0]), float(g[1]), heading),
            meet=(float(m[0]), float(m[1])),
            axis=axis,
            ped_start=tuple((float(p[0]), float(p[1])) for p in starts),  # type: ignore[arg-type]
            ped_meet=tuple((float(p[0]), float(p[1])) for p in meet),  # type: ignore[arg-type]
            ped_exit=tuple((float(p[0]), float(p[1])) for p in exits),  # type: ignore[arg-type]
            hold_s=float(rng.uniform(4.0, 8.0)),
            lead_s=float(lead),
            wait_s=float(wait),
        )
    raise RuntimeError("no admissible geometry, loosen the sampling ranges")


def _r(v: float) -> float:
    return round(float(v), 3)


def agent_yaml(name: str, interaction: str, meet: tuple[float, float], exit_: tuple[float, float], hold_s: float, wait_s: float) -> dict:
    return {
        "name": name,
        "mode": "behavior_tree",
        "extends": "adult",
        "initial_sequence": "contact_seq",
        "sequences": {
            "contact_seq": {
                "steps": {
                    "wait": {"duration": {"mean": _r(wait_s), "std": 0.0}},
                    "approach": {"kind": "go_to", "target_pose": {"x": _r(meet[0]), "y": _r(meet[1])}},
                    "contact": {"interaction": interaction, "attention": {"gaze": "partner"}, "duration": {"mean": _r(hold_s), "std": 0.0}},
                    "separate": {"kind": "go_to", "target_pose": {"x": _r(exit_[0]), "y": _r(exit_[1])}},
                    "rest": {"duration": {"mean": 10_000.0, "std": 0.0}},
                },
            },
        },
    }


def scenario_yaml(geo: Geometry) -> dict:
    models = ("female_adult_business_02", "male_adult_construction_01")
    return {
        "robots": [{"start": [_r(v) for v in geo.start], "goal": [_r(v) for v in geo.goal]}],
        "dynamic": [
            {
                "name": f"partner_{i}",
                "model": models[i],
                "pose": [_r(geo.ped_start[i][0]), _r(geo.ped_start[i][1]), _r(geo.axis + (0.0 if i == 0 else math.pi))],
                "agent": {"agent_type": f"./partner_{i}.yaml", "desired_velocity": PED_SPEED},
            }
            for i in range(2)
        ],
    }


def stage_name(kind: str, k: int, arm: str) -> str:
    return f"hhi_{kind}_{k:02d}_{arm}"


def generate(n_scenarios: int, *, seed: int = 0, scenarios_dir: pathlib.Path | None = None, suite_path: pathlib.Path | None = None, episodes: int = EPISODES) -> dict:
    """Write L scenarios per kind under map_empty/scenarios and the paired suite; returns the suite dict."""
    scenarios_dir = scenarios_dir or _default_scenarios_dir()
    suite_path = suite_path or _default_suite_path()
    rng = np.random.default_rng(seed)
    stages = []
    manifest = []
    for kind, interaction in KINDS.items():
        for k in range(n_scenarios):
            geo = sample_geometry(rng)
            name = f"hhi_{kind}_{k:02d}"
            out = scenarios_dir / name
            out.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (out / f"partner_{i}.yaml").write_text(yaml.safe_dump(agent_yaml(f"{name}_partner_{i}", interaction, geo.ped_meet[i], geo.ped_exit[i], geo.hold_s, geo.wait_s), sort_keys=False))
            (out / "scenario.yaml").write_text(yaml.safe_dump(scenario_yaml(geo), sort_keys=False))
            scenario_seed = SEED_BASE + 100 * len(manifest)
            manifest.append({"scenario": name, "seed": scenario_seed, "hold_s": _r(geo.hold_s), "lead_s": _r(geo.lead_s), "wait_s": _r(geo.wait_s), "axis_deg": _r(math.degrees(geo.axis) % 360.0)})
            for arm, mode in ARMS.items():
                stages.append(
                    {
                        "name": stage_name(kind, k, arm),
                        "map": MAP,
                        "robot": "jackal",
                        "tm_robots": "scenario",
                        "tm_obstacles": "scenario",
                        "episodes": episodes,
                        "seed": scenario_seed,  # shared by both arms: episode i runs seed + i in each
                        "timeout": TIMEOUT,
                        "config": {"scenario": {"file": name, "contact_mode": mode, "standing_distance": STANDING_DISTANCE}},
                    }
                )
    suite = {"stages": stages}
    header = (
        f"# Study A, two-person contact interactions (EVALUATION.md §A), generated by arena_evaluation.experiments.hhi generate --scenarios {n_scenarios} --seed {seed}.\n"
        "# Every scenario runs twice with the same seeds: _contact (contact_mode=enabled) and _loco (locomotion_only).\n"
        "# Run:    ros2 run arena_evaluation benchmark --suite hhi_contact --contest paper_planners headless:=true\n"
        "# Analyze: python -m arena_evaluation.experiments.hhi analyze --benchmark-dir <data_root>/<run_id>\n"
    )
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(header + yaml.safe_dump(suite, sort_keys=False))
    (suite_path.parent / f"{suite_path.stem}.scenarios.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return suite


# -- analysis ----------------------------------------------------------------------------------

FAMILY_METRICS = {
    "success": "success",
    "ttg": "time_to_goal",
    "stalls": "hhi_stall_count",
    "passes_between": "hhi_passes_between",
}


def load_episodes(metrics: pl.DataFrame) -> pl.DataFrame:
    """Tag every HHI episode row with interaction / scenario / arm / seed index, dropping other stages."""
    parsed = metrics.with_columns(pl.col("stage").str.extract_groups(STAGE_RE.pattern).alias("_m")).unnest("_m")
    parsed = parsed.filter(pl.col("kind").is_not_null())
    if "is_reference" in parsed.columns:
        parsed = parsed.filter(~pl.col("is_reference").fill_null(False))
    if "status" in parsed.columns:
        parsed = parsed.filter(pl.col("status") == "evaluated")
    # the runner hands episode i of a stage seed + i, in order, so the rank inside (planner, stage) is the seed
    parsed = parsed.sort("episode").with_columns((pl.col("episode").rank("ordinal").over(["planner", "stage"]) - 1).cast(pl.Int64).alias("seed"))
    return parsed.with_columns(pl.col("scenario").cast(pl.Int64), pl.col("kind").replace({"shake": "shake_hand"}).alias("interaction"))


def paper_outcomes(df: pl.DataFrame) -> pl.DataFrame:
    """Manuscript §4.1: success reaches the goal before the timeout without a collision; time to goal only over successes.

    The pipeline's own `success` tolerates collisions below its MAX_COLLISIONS, so a collision count is applied on top.
    A paired time difference then only exists where both arms of a (scenario, seed) pair succeeded.
    """
    success = pl.col("success").cast(pl.Float64).fill_null(0.0) > 0.5
    if "collision_amount" in df.columns:
        success = success & (pl.col("collision_amount").fill_null(0) == 0)
    return df.with_columns(success.alias("success"), pl.when(success).then(pl.col("time_to_goal")).otherwise(None).alias("time_to_goal"))


def family_table(df: pl.DataFrame, *, n_boot: int = DEFAULT_BOOT, seed: int = 0) -> pl.DataFrame:
    """Study A hypotheses: contact - locomotion per (planner, interaction, metric), Holm-corrected across all of them."""
    parts = []
    for label, column in FAMILY_METRICS.items():
        if column not in df.columns:
            continue
        diffs = paired_differences(df.with_columns(pl.col(column).cast(pl.Float64)), cell=["planner", "interaction"], metric=column, arm="arm", treatment="contact", control="loco")
        if diffs.height:
            parts.append(diffs.with_columns(pl.lit(label).alias("metric")))
    if not parts:
        return pl.DataFrame()
    return summarize_family(pl.concat(parts), cell=["planner", "interaction", "metric"], n_boot=n_boot, seed=seed)


def window_table(df: pl.DataFrame, *, n_boot: int = DEFAULT_BOOT, seed: int = 0) -> pl.DataFrame:
    """Registered direction 2, contact arm: post-release share of stall onsets minus post-release share of time, per planner.

    Both shares are pooled ratios over the drawn episodes (sum of post stalls / sum of stalls, sum of post time /
    sum of episode time), resampled by scenario.
    """
    need = {"hhi_stalls_post", "hhi_stall_count", "hhi_share_post", "hhi_episode_s"}
    if not need <= set(df.columns):
        return pl.DataFrame()
    contact = df.filter((pl.col("arm") == "contact") & pl.col("hhi_release_s").is_not_null())
    rows = []
    rng = np.random.default_rng(seed)

    def excess(drawn: list[np.ndarray]) -> float:
        m = np.concatenate(drawn)
        stalls = m[:, 1].sum()
        return float(m[:, 0].sum() / stalls - m[:, 2].sum() / m[:, 3].sum()) if stalls > 0 else float("nan")

    for (planner,), group in contact.group_by(["planner"], maintain_order=True):
        clusters = []
        for _, g in group.group_by(["interaction", "scenario"], maintain_order=True):
            post_time = (g["hhi_share_post"] * g["hhi_episode_s"]).to_numpy()
            clusters.append(np.column_stack([g["hhi_stalls_post"].to_numpy(), g["hhi_stall_count"].to_numpy(), post_time, g["hhi_episode_s"].to_numpy()]).astype(float))
        res = cluster_bootstrap(clusters, excess, n_boot=n_boot, rng=rng)
        m = np.concatenate(clusters) if clusters else np.zeros((0, 4))
        rows.append(
            {
                "planner": planner,
                "stall_share_post": float(m[:, 0].sum() / m[:, 1].sum()) if m[:, 1].sum() > 0 else None,
                "time_share_post": float(m[:, 2].sum() / m[:, 3].sum()) if m[:, 3].sum() > 0 else None,
                "excess": res.estimate,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p": res.p,
                "n_scenarios": res.n_scenarios,
                "n_episodes": res.n_pairs,
            }
        )
    out = pl.DataFrame(rows)
    return out.with_columns(pl.Series("p_holm", holm(out["p"].to_list()))) if out.height else out


def manipulation_check(df: pl.DataFrame) -> pl.DataFrame:
    """Median rendered pair gap during the hold per arm and interaction: contact must sit near contact distance."""
    if "hhi_pair_gap_hold" not in df.columns:
        return pl.DataFrame()
    return (
        df.group_by(["interaction", "arm"])
        .agg(
            pl.col("hhi_pair_gap_hold").median().alias("pair_gap_median"),
            pl.col("hhi_hold_onset_s").is_not_null().mean().alias("held_share"),
            pl.col("hhi_release_s").is_not_null().mean().alias("released_share"),
            pl.len().alias("episodes"),
        )
        .sort(["interaction", "arm"])
    )


def _metrics_path(benchmark_dir: pathlib.Path) -> pathlib.Path:
    for candidate in (benchmark_dir / "combined_metrics.parquet", benchmark_dir / "metrics.parquet"):
        if candidate.exists():
            return candidate
    found = sorted(benchmark_dir.rglob("combined_metrics.parquet"))
    if not found:
        raise FileNotFoundError(f"no combined_metrics.parquet under {benchmark_dir}, run `arena evaluation run` on it first")
    return found[0]


def analyze(benchmark_dir: pathlib.Path, *, n_boot: int = DEFAULT_BOOT, out_dir: pathlib.Path | None = None) -> dict[str, pl.DataFrame]:
    df = load_episodes(pl.read_parquet(_metrics_path(benchmark_dir)))
    if "time_to_goal" in df.columns and "success" in df.columns:
        df = paper_outcomes(df)
    tables = {"family_a": family_table(df, n_boot=n_boot), "windows": window_table(df, n_boot=n_boot), "manipulation_check": manipulation_check(df)}
    out_dir = out_dir or benchmark_dir / "hhi"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table.height:
            table.write_csv(out_dir / f"{name}.csv")
    return tables


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arena_evaluation.experiments.hhi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the scenarios and the paired suite")
    g.add_argument("--scenarios", type=int, default=15, help="scenarios per contact kind (L)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--episodes", type=int, default=EPISODES, help="seeds per scenario")
    g.add_argument("--scenarios-dir", type=pathlib.Path, default=None)
    g.add_argument("--suite", type=pathlib.Path, default=None)
    a = sub.add_parser("analyze", help="paired analysis of a processed run")
    a.add_argument("--benchmark-dir", type=pathlib.Path, required=True)
    a.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    a.add_argument("--out", type=pathlib.Path, default=None)
    args = p.parse_args(argv)
    if args.cmd == "generate":
        suite = generate(args.scenarios, seed=args.seed, scenarios_dir=args.scenarios_dir, suite_path=args.suite, episodes=args.episodes)
        print(f"{len(suite['stages'])} stages x {args.episodes} episodes per contestant")
        return 0
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=40):
        for name, table in analyze(args.benchmark_dir, n_boot=args.boot, out_dir=args.out).items():
            print(f"\n== {name}\n{table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
