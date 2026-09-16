"""Studies A' and C: what changes when a planner can read a social interaction.

Both studies hold the scene fixed and vary only what reaches the planner:

  on   ``gesture_mode=enabled``  + the social cost layer. The interaction is rendered on the skeletons, the
       participants publish a shared interaction id, and the costmap treats them as one social unit.
  off  ``gesture_mode=disabled`` + no social cost layer. The same people stand in the same places doing the
       same thing, but publish no clip and no interaction id, so a planner sees unrelated bodies.

The cost layer stands in for a planner that can recognize an interaction; training a recognizer is out of
scope, so the layer is the consumer that makes the interaction actionable.

Study A' is a two-person contact interaction (hug, handshake); Study C is a three-person conversation. Each
runs in an open room, where the robot has space to choose, and in a corridor, where it may not.

    python -m arena_evaluation.experiments.social generate
    python -m arena_evaluation.experiments.social analyze --on <run_on> --off <run_off>
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl
import yaml

from arena_evaluation.experiments.stats import DEFAULT_BOOT, cluster_bootstrap, holm

WORLD = "hospital_1_bare"
ROBOT = "jackal"
EPISODES = 3
SEED_BASE = 76000
MODELS = ("female_adult_business_02", "male_adult_construction_01", "female_adult_business_02")

_ROOT = pathlib.Path(__file__).resolve().parent

# Layouts measured on the world's map before the runs: the corridor is 3.7 m wide, the ward 7.8 x 14.7 m open,
# the junction a 3.7 m corridor opening into a 9.75 m crossing.
LAYOUTS = {
    "ward": {"centre": (4.0, 17.5), "start": (4.0, 12.0, 1.5708), "goal": (4.0, 23.0, 1.5708), "space": "open room, 7.8 m wide"},
    "corridor": {"centre": (9.2, 23.5), "start": (10.0, 20.0, 1.5708), "goal": (10.0, 27.0, 1.5708), "space": "corridor, 3.7 m wide"},
}
GROUP_LAYOUTS = {
    "ward": {"centre": (4.0, 17.5), "start": (4.0, 12.0, 1.5708), "goal": (4.0, 23.0, 1.5708), "space": "open room, 7.8 m wide"},
    "corridor": {"centre": (10.4, 23.5), "start": (10.0, 20.0, 1.5708), "goal": (10.0, 27.0, 1.5708), "space": "corridor, 3.7 m wide"},
    "junction": {"centre": (10.0, 26.0), "start": (10.0, 22.0, 1.5708), "goal": (10.0, 30.0, 1.5708), "space": "corridor into a 9.75 m crossing"},
}

# interaction -> (participants, separation between partners)
CONTACT = {"hug": (2, 0.30), "handshake": (2, 0.60)}
INTERACTION_TYPES = {"hug": "HUG", "handshake": "SHAKE_HAND"}
GROUP_RADIUS_M = 0.94  # humansim f_formation, three participants


def _world_dir() -> pathlib.Path:
    for parent in _ROOT.parents:
        candidate = parent / "arena_simulation_setup" / "worlds" / WORLD
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"arena_simulation_setup/worlds/{WORLD} not found above {_ROOT}")


def _agent(name: str, interaction: str) -> dict:
    return {
        "name": name,
        "mode": "behavior_tree",
        "extends": "adult",
        "initial_sequence": "seq",
        "sequences": {
            "seq": {
                "steps": {
                    "settle": {"duration": {"mean": 2.0, "std": 0.0}},
                    "interact": {"interaction": interaction, "attention": {"gaze": "partner"}, "duration": {"mean": 10000.0, "std": 0.0}},
                },
            },
        },
    }


def contact_positions(centre: tuple[float, float], separation: float) -> list[tuple[float, float, float]]:
    half = separation / 2.0
    return [(centre[0] - half, centre[1], 0.0), (centre[0] + half, centre[1], 3.14159)]


def group_positions(centre: tuple[float, float], members: int = 3) -> list[tuple[float, float, float]]:
    out = []
    for i in range(members):
        angle = 2.0 * np.pi * i / members
        out.append((float(centre[0] + GROUP_RADIUS_M * np.cos(angle)), float(centre[1] + GROUP_RADIUS_M * np.sin(angle)), float(angle + np.pi)))
    return out


def write_scenario(directory: pathlib.Path, interaction: str, layout: dict, poses: list[tuple[float, float, float]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"person_{i}" for i in range(len(poses))]
    for name in names:
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(_agent(name, interaction), sort_keys=False))
    (directory / "scenario.yaml").write_text(
        yaml.safe_dump(
            {
                "robots": [{"start": list(layout["start"]), "phases": [{"goto": list(layout["goal"])}]}],
                "dynamic": [
                    {
                        "name": name,
                        "model": MODELS[i % len(MODELS)],
                        "pose": [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)],
                        "agent": {"agent_type": f"./{name}.yaml"},
                        "waypoints": [[round(float(p[0]), 3), round(float(p[1]), 3)]],
                    }
                    for i, (name, p) in enumerate(zip(names, poses, strict=True))
                ],
            },
            sort_keys=False,
        ),
    )


def generate(*, episodes: int = EPISODES) -> dict[str, int]:
    scenarios_dir = _world_dir() / "scenarios"
    stages: dict[str, list] = {"on": [], "off": []}
    index = 0
    catalogue = []

    for interaction, (members, separation) in CONTACT.items():
        for layout_name, layout in LAYOUTS.items():
            case = f"contact_{interaction}_{layout_name}"
            poses = contact_positions(layout["centre"], separation)
            write_scenario(scenarios_dir / case, INTERACTION_TYPES[interaction], layout, poses)
            catalogue.append((case, f"{members} people {separation:.2f} m apart in a {layout['space']}"))
            index += 1
            for arm, mode in (("on", "enabled"), ("off", "disabled")):
                stages[arm].append(_stage(case, arm, mode, SEED_BASE + index, episodes))

    for layout_name, layout in GROUP_LAYOUTS.items():
        case = f"group_{layout_name}"
        poses = group_positions(layout["centre"])
        write_scenario(scenarios_dir / case, "GROUP_CONVERSATION", layout, poses)
        catalogue.append((case, f"3 people in conversation in a {layout['space']}"))
        index += 1
        for arm, mode in (("on", "enabled"), ("off", "disabled")):
            stages[arm].append(_stage(case, arm, mode, SEED_BASE + index, episodes))

    header = (
        "# Studies A' and C, generated by arena_evaluation.experiments.social generate.\n"
        "# Each case runs twice on the same seeds; the arm is the suite AND the contest:\n"
        "#   on   suite social_on  + contest paper_planners        (gesture + interaction ids + cost layer)\n"
        "#   off  suite social_off + contest paper_planners_blind  (none of the three)\n"
    )
    for arm in ("on", "off"):
        path = _ROOT.parents[1] / "configs" / "benchmark" / "suites" / f"social_{arm}.yaml"
        path.write_text(header + yaml.safe_dump({"stages": stages[arm]}, sort_keys=False))
    for case, description in catalogue:
        print(f"  {case:28s} {description}")
    print(f"\n{len(stages['on'])} stages per arm x {episodes} episodes per contestant")
    return {arm: len(v) for arm, v in stages.items()}


def _stage(case: str, arm: str, mode: str, seed: int, episodes: int) -> dict:
    return {
        "name": f"{case}_{arm}",
        "map": WORLD,
        "robot": ROBOT,
        "tm_robots": "scenario",
        "tm_obstacles": "scenario",
        "episodes": episodes,
        "seed": seed,
        "timeout": "90s",
        "config": {"scenario": {"file": case, "gesture_mode": mode, "contact_mode": "enabled"}},
    }


# -- analysis -----------------------------------------------------------------------------------

CENTRES = {f"contact_{i}_{k}": v["centre"] for i in CONTACT for k, v in LAYOUTS.items()} | {f"group_{k}": v["centre"] for k, v in GROUP_LAYOUTS.items()}
METRICS = ("min_pedestrian_clearance", "time_in_intimate_zone", "time_in_personal_zone", "path_length", "time_to_goal", "pierced", "success")


def load(run: pathlib.Path, arm: str) -> pl.DataFrame:
    """Episodes of one arm, with the two derived route metrics and without the harness's own casualties.

    An episode the runner marks ``UNRESOLVED`` was cut off when its environment disappeared mid-run, not when
    the robot gave up: counting it as a failure would charge the planner for a crash of the benchmark. Genuine
    give-ups are marked ``FAILED`` and are kept.
    """
    path = run / "combined_metrics.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; process the run first")
    df = pl.read_parquet(path)
    if "result" in df.columns:
        df = df.filter(pl.col("result") != "UNRESOLVED")
    df = df.with_columns(
        pl.col("stage").str.replace(r"_(on|off)$", "").alias("case"),
        pl.lit(arm).alias("arm"),
    )
    pierced, closest = [], []
    for row in df.iter_rows(named=True):
        centre = np.array(CENTRES.get(row["case"], (0.0, 0.0)))
        pts = np.array([[float(p[0]), float(p[1])] for p in (row.get("path") or [])])
        d = np.linalg.norm(pts - centre, axis=1) if len(pts) else np.array([float("nan")])
        radius = GROUP_RADIUS_M if row["case"].startswith("group") else 0.8
        pierced.append(bool(np.nansum(d < radius) > 0))
        closest.append(float(np.nanmin(d)))
    return df.with_columns(
        pl.Series("pierced", pierced, dtype=pl.Float64),
        pl.Series("closest_to_group_m", closest),
        pl.col("success").cast(pl.Float64),
    )


def matched(df: pl.DataFrame) -> pl.DataFrame:
    """Per planner and arm, pooled over the cases that produced episodes in *both* arms.

    Six on-arm steps lost their environment before running an episode, so the two arms do not cover the same
    cases for every planner. Pooling each arm over everything it happens to have would compare DWB's five
    surviving on-arm cases against seven off-arm ones, and the missing two are exactly the ones the blind
    planner walks through -- flattering the result with an accident of the harness. Restricting both arms to
    the cases they share is what the paired differences already do.
    """
    have = df.group_by(["planner", "case", "arm"]).agg(pl.len()).pivot(on="arm", index=["planner", "case"], values="len")
    both = have.filter(pl.col("on").is_not_null() & pl.col("off").is_not_null()).select(["planner", "case"])
    return (
        df.join(both, on=["planner", "case"], how="inner")
        .group_by(["planner", "arm"], maintain_order=True)
        .agg(
            pl.col("case").n_unique().alias("cases"),
            pl.len().alias("n"),
            pl.col("pierced").mean().alias("pierce_rate"),
            pl.col("min_pedestrian_clearance").median().alias("clearance_m"),
            pl.col("success").mean().alias("success"),
        )
        .sort(["planner", "arm"])
    )


def analyze(on: pathlib.Path, off: pathlib.Path, *, n_boot: int = DEFAULT_BOOT, out_dir: pathlib.Path | None = None) -> dict[str, pl.DataFrame]:
    df = pl.concat([load(on, "on"), load(off, "off")], how="diagonal")
    df = df.with_columns((pl.col("episode").rank("ordinal").over(["planner", "stage"]) - 1).cast(pl.Int64).alias("seed"))

    descriptive = df.group_by(["case", "planner", "arm"], maintain_order=True).agg(
        pl.len().alias("n"),
        pl.col("success").mean().alias("success"),
        pl.col("pierced").mean().alias("pierce_rate"),
        pl.col("min_pedestrian_clearance").median().alias("clearance_m"),
        pl.col("time_in_intimate_zone").median().alias("intimate_s"),
        pl.col("time_in_personal_zone").median().alias("personal_s"),
        pl.col("path_length").median().alias("path_m"),
        pl.col("time_to_goal").median().alias("ttg_s"),
    ).sort(["case", "planner", "arm"])

    rows = []
    rng = np.random.default_rng(0)
    for metric in METRICS:
        for (planner,), group in df.group_by(["planner"], maintain_order=True):
            clusters = []
            for (_case,), case_group in group.group_by(["case"], maintain_order=True):
                wide = case_group.filter(pl.col(metric).is_not_null()).pivot(on="arm", index="seed", values=metric, aggregate_function="first")
                if "off" in wide.columns and "on" in wide.columns:
                    diff = (wide["off"] - wide["on"]).drop_nulls().to_numpy()
                    if len(diff):
                        clusters.append(diff)
            if not clusters:
                continue
            r = cluster_bootstrap(clusters, n_boot=n_boot, rng=rng)
            rows.append({"metric": metric, "planner": planner, "off_minus_on": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p, "n_cases": len(clusters), "n_pairs": r.n_pairs})
    paired = pl.DataFrame(rows)
    if len(paired):
        paired = paired.with_columns(pl.Series("p_holm", holm(paired["p"].to_list())))

    tables = {"descriptive": descriptive, "matched": matched(df), "paired": paired}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.write_csv(out_dir / f"{name}.csv")
    for name, table in tables.items():
        print(f"\n== {name}")
        with pl.Config(tbl_rows=60, tbl_width_chars=200):
            print(table)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--episodes", type=int, default=EPISODES)
    ana = sub.add_parser("analyze")
    ana.add_argument("--on", type=pathlib.Path, required=True)
    ana.add_argument("--off", type=pathlib.Path, required=True)
    ana.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    ana.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(episodes=args.episodes)
    else:
        analyze(args.on, args.off, n_boot=args.boot, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
