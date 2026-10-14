"""Study B: a pedestrian calls a patrolling robot for help, with and without the gesture layer.

The robot loops a fixed checkpoint route of the hospital. One pedestrian beckons for help and waits. Under
``gesture_mode=enabled`` the call is rendered on the pedestrian's skeleton and published, so a robot that can
perceive it leaves the loop and serves them; under ``disabled`` the same pedestrian stands in the same place at
the same time and publishes nothing, so they are served only if the route happens to pass. The measured quantity
is how long the pedestrian waits.

    python -m arena_evaluation.experiments.beckon generate
    python -m arena_evaluation.experiments.beckon analyze --benchmark-dir <data_root>/<run_id>
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import polars as pl
import yaml

from arena_evaluation.experiments.stats import DEFAULT_BOOT, cluster_bootstrap, holm

WORLD = "hospital_1_bare"
ROBOT = "jackal"
CALLER = "caller"
CALL_CLIP = "beckon"
CALL_ONSET_S = 5.0
SENSOR_RANGE_M = 12.0
RECOGNITION_DWELL_S = 0.5
CHECKPOINT_RADIUS_M = 1.2
SERVICE_RADIUS_M = 2.5
SERVICE_DWELL_S = 0.0
TIMEOUT_S = 90
EPISODES = 2
SEED_BASE = 72000
MODEL = "female_adult_business_02"

SCENARIO_TO_MAP = (0.0, 0.0)

START = (10.0, 2.0, 1.5708)
ROUTE = [(10.0, 18.0, 1.5708), (10.0, 2.0, -1.5708)]

CALLERS: dict[str, tuple[tuple[float, float, float], str]] = {
    "hall_near": ((9.0, 9.0, 0.0), "beside the hallway, seen over 8 m of the route and passed early"),
    "hall_far": ((9.0, 16.0, 0.0), "beside the hallway 7 m further on, seen from the same stretch but passed late"),
    "ward_door": ((5.0, 16.0, 0.0), "inside the ward, seen through the doorway from one spot at 9 m, never passed"),
}

ARMS = {"on": "enabled", "off": "disabled"}


_ROOT = pathlib.Path(__file__).resolve().parent


def _world_dir() -> pathlib.Path:
    """The source tree's world, so generated scenarios land next to the ones already committed."""
    for parent in _ROOT.parents:
        candidate = parent / "arena_simulation_setup" / "worlds" / WORLD
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"arena_simulation_setup/worlds/{WORLD} not found above {_ROOT}")


# -- generation ---------------------------------------------------------------------------------


class _Map:
    """The world's occupancy image, for validating that a placement is free and a sightline is clear."""

    def __init__(self, world_dir: pathlib.Path) -> None:
        from PIL import Image

        meta = yaml.safe_load((world_dir / "0" / "map.yaml").read_text())
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.grid = np.array(Image.open(world_dir / "0" / meta["image"]).convert("L"))

    def _rc(self, x: float, y: float) -> tuple[int, int]:
        return int(round(self.grid.shape[0] - (y - self.oy) / self.res)), int(round((x - self.ox) / self.res))

    def free(self, x: float, y: float, clearance: float = 0.6) -> bool:
        x, y = x + SCENARIO_TO_MAP[0], y + SCENARIO_TO_MAP[1]
        r, c = self._rc(x, y)
        k = int(clearance / self.res)
        h, w = self.grid.shape
        if not (k <= r < h - k and k <= c < w - k):
            return False
        return bool((self.grid[r - k : r + k + 1, c - k : c + k + 1] > 200).all())

    def route_window(self, samples: list[tuple[float, float]], target: tuple[float, float], sensor_range: float) -> tuple[str, float]:
        """How much of the walked route can see the target, and how close the route passes it."""
        visible = [p for p in samples if math.dist(p, target) <= sensor_range and self.line_of_sight(p, target)]
        closest = min(math.dist(p, target) for p in samples)
        if not visible:
            return "nowhere", closest
        return f"y={min(p[1] for p in visible):.1f}..{max(p[1] for p in visible):.1f}", closest

    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        a = (a[0] + SCENARIO_TO_MAP[0], a[1] + SCENARIO_TO_MAP[1])
        b = (b[0] + SCENARIO_TO_MAP[0], b[1] + SCENARIO_TO_MAP[1])
        steps = int(math.dist(a, b) / (self.res / 2)) + 1
        for i in range(steps + 1):
            x = a[0] + (b[0] - a[0]) * i / steps
            y = a[1] + (b[1] - a[1]) * i / steps
            r, c = self._rc(x, y)
            if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]) or self.grid[r, c] < 100:
                return False
        return True


def _route_samples(step: float = 0.5) -> list[tuple[float, float]]:
    """The patrol line at half-meter spacing, start included, for the visibility report."""
    out = []
    legs = [START, *ROUTE]
    for a, b in zip(legs[:-1], legs[1:], strict=True):
        length = math.dist((a[0], a[1]), (b[0], b[1]))
        steps = max(1, int(length / step))
        out.extend([(a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps) for i in range(steps + 1)])
    return out


def caller_agent_yaml(robot: str = ROBOT) -> dict:
    """Stand, then call and keep calling."""
    return {
        "name": CALLER,
        "mode": "behavior_tree",
        "extends": "adult",
        "initial_sequence": "call_seq",
        "sequences": {
            "call_seq": {
                "steps": {
                    "wait": {
                        "attention": {"posture": "standing"},
                        "duration": {"mean": CALL_ONSET_S, "std": 0.0},
                    },
                    "call": {
                        "attention": {"clip": CALL_CLIP, "gaze": [f"robot:{robot}", "robot:robot_0", "robot"]},
                        "duration": {"mean": 1000.0, "std": 0.0},
                    },
                },
                "then": "call_seq",
            },
        },
    }


def robot_phases() -> list[dict]:
    """Loop the route until the call is seen or the route passes the caller, then go to the caller."""
    called = f"robot sees {CALLER} within {SENSOR_RANGE_M:g} and {CALLER} plays {CALL_CLIP} for {RECOGNITION_DWELL_S:g}"
    passed = f"robot within {SERVICE_RADIUS_M:g} of {CALLER}"
    return [
        {"loop": [{"goto": list(p), "tolerance_radius": CHECKPOINT_RADIUS_M} for p in ROUTE], "until": f"{called} or {passed}"},
        {"goto": CALLER, "tolerance_radius": SERVICE_RADIUS_M, "hold_time": SERVICE_DWELL_S},
    ]


def scenario_yaml(caller_pose: tuple[float, float, float]) -> dict:
    return {
        "robots": [{"start": list(START), "phases": robot_phases()}],
        "dynamic": [
            {
                "name": CALLER,
                "model": MODEL,
                "pose": list(caller_pose),
                "agent": {"agent_type": "./caller.yaml"},
                "waypoints": [[caller_pose[0], caller_pose[1]]],
            },
        ],
    }


def stage_name(case: str, arm: str) -> str:
    return f"beckon_{case}_{arm}"


def generate(*, scenarios_dir: pathlib.Path | None = None, suite_path: pathlib.Path | None = None, episodes: int = EPISODES) -> dict:
    world_dir = _world_dir()
    scenarios_dir = scenarios_dir or (world_dir / "scenarios")
    world_map = _Map(world_dir)

    problems = []
    for label, (x, y, _) in [("start", START), *[(f"route_{i}", p) for i, p in enumerate(ROUTE)], *[(f"caller_{k}", v[0]) for k, v in CALLERS.items()]]:
        if not world_map.free(x, y):
            problems.append(f"{label} at ({x}, {y}) is not free with 0.6 m clearance")
    if problems:
        raise ValueError("; ".join(problems))

    stages = []
    for index, (case, (pose, _description)) in enumerate(CALLERS.items()):
        directory = scenarios_dir / f"beckon_{case}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scenario.yaml").write_text(yaml.safe_dump(scenario_yaml(pose), sort_keys=False, width=120))
        (directory / "caller.yaml").write_text(yaml.safe_dump(caller_agent_yaml(), sort_keys=False, width=120))

        seen, closest = world_map.route_window(_route_samples(), (pose[0], pose[1]), SENSOR_RANGE_M)
        sightlines = f"seen from {seen} of the route, closest approach {closest:.1f} m ({'passed' if closest <= SERVICE_RADIUS_M else 'never passed'})"
        for arm, gesture_mode in ARMS.items():
            stages.append(
                {
                    "name": stage_name(case, arm),
                    "map": WORLD,
                    "robot": ROBOT,
                    "tm_robots": "scenario",
                    "tm_obstacles": "scenario",
                    "episodes": episodes,
                    "seed": SEED_BASE + index,
                    "timeout": f"{TIMEOUT_S}s",
                    "config": {"scenario": {"file": f"beckon_{case}", "gesture_mode": gesture_mode}},
                },
            )
        print(f"beckon_{case:11s} {sightlines}")

    suite = {"stages": stages}
    suite_path = suite_path or (_ROOT.parents[1] / "configs" / "benchmark" / "suites" / "beckon_service.yaml")

    header = (
        "# Study B, a pedestrian calls a patrolling robot for help (EVALUATION.md B).\n"
        "# Generated by arena_evaluation.experiments.beckon generate. Each case runs twice on the same seeds:\n"
        "#   _on   gesture_mode=enabled  the call is rendered and published\n"
        "#   _off  gesture_mode=disabled the same pedestrian stands in the same place and publishes nothing\n"
        "# Run:     arena evaluation benchmark --suite beckon_service --contest paper_planners headless:=true\n"
        "# Analyze: python -m arena_evaluation.experiments.beckon analyze --benchmark-dir <data_root>/<run_id>\n"
    )
    suite_path.write_text(header + yaml.safe_dump(suite, sort_keys=False, width=120))
    for arm in ARMS:
        arm_stages = [stage for stage in stages if stage["name"].endswith(f"_{arm}")]
        (suite_path.parent / f"beckon_{arm}.yaml").write_text(header + yaml.safe_dump({"stages": arm_stages}, sort_keys=False, width=120))
    print(f"\n{len(stages)} stages x {episodes} episodes per contestant -> {suite_path}")
    return suite


# -- analysis -----------------------------------------------------------------------------------


def load_episodes(metrics: pl.DataFrame) -> pl.DataFrame:
    """One row per episode with the case, the arm and the waiting time."""
    arms = "|".join(ARMS)
    parsed = metrics.with_columns(
        pl.col("stage").str.extract(rf"^beckon_(.*)_(?:{arms})$", 1).alias("case"),
        pl.col("stage").str.extract(rf"_({arms})$", 1).alias("arm"),
    ).filter(pl.col("case").is_not_null())
    parsed = parsed.sort("episode").with_columns((pl.col("episode").rank("ordinal").over(["planner", "stage"]) - 1).cast(pl.Int64).alias("seed"))
    return parsed


def paper_outcomes(df: pl.DataFrame) -> pl.DataFrame:
    """Waiting time per episode, censored at the run's own timeout (its longest episode) when the caller was never served."""
    served = pl.col("status_reason").is_null() & (pl.col("success") == 1)
    longest = df.select(pl.col("time_to_goal").max()).item() if "time_to_goal" in df.columns else None
    timeout = max(float(longest), float(TIMEOUT_S)) if longest is not None else float(TIMEOUT_S)
    duration = pl.col("time_to_goal").fill_null(timeout)
    return df.with_columns(
        served.alias("served"),
        pl.when(served).then(duration - CALL_ONSET_S).otherwise(timeout - CALL_ONSET_S).alias("wait_s"),
        (~served).alias("censored"),
        pl.lit(timeout).alias("censor_at_s"),
    )


def case_table(df: pl.DataFrame, *, n_boot: int = DEFAULT_BOOT, seed: int = 0) -> pl.DataFrame:
    """Per planner and case: the waiting time in each arm and their paired difference over seeds."""
    rows = []
    rng = np.random.default_rng(seed)
    for (planner, case), group in df.group_by(["planner", "case"], maintain_order=True):
        on = group.filter(pl.col("arm") == "on")
        off = group.filter(pl.col("arm") == "off")
        pairs = on.join(off, on="seed", how="inner", suffix="_off")
        diffs = (pairs["wait_s"] - pairs["wait_s_off"]).to_numpy()
        result = cluster_bootstrap([diffs], n_boot=n_boot, rng=rng) if len(diffs) else None
        rows.append(
            {
                "planner": planner,
                "case": case,
                "wait_on": float(on["wait_s"].median()) if len(on) else None,
                "wait_off": float(off["wait_s"].median()) if len(off) else None,
                "served_on": float(on["served"].mean()) if len(on) else None,
                "served_off": float(off["served"].mean()) if len(off) else None,
                "diff": None if result is None else result.estimate,
                "ci_low": None if result is None else result.ci_low,
                "ci_high": None if result is None else result.ci_high,
                "p": None if result is None else result.p,
                "n_pairs": len(diffs),
            },
        )
    table = pl.DataFrame(rows)
    finite = [p for p in table["p"].to_list() if p is not None]
    if finite:
        adjusted = dict(zip(finite, holm(finite), strict=True))
        table = table.with_columns(pl.col("p").map_elements(lambda v: adjusted.get(v), return_dtype=pl.Float64).alias("p_holm"))
    return table


def planner_table(df: pl.DataFrame, *, n_boot: int = DEFAULT_BOOT, seed: int = 0) -> pl.DataFrame:
    """Per planner, the paired difference pooled over cases, resampled over cases."""
    rows = []
    rng = np.random.default_rng(seed)
    for (planner,), group in df.group_by(["planner"], maintain_order=True):
        clusters = []
        for (_case,), case_group in group.group_by(["case"], maintain_order=True):
            on = case_group.filter(pl.col("arm") == "on")
            off = case_group.filter(pl.col("arm") == "off")
            pairs = on.join(off, on="seed", how="inner", suffix="_off")
            if len(pairs):
                clusters.append((pairs["wait_s"] - pairs["wait_s_off"]).to_numpy())
        result = cluster_bootstrap(clusters, n_boot=n_boot, rng=rng) if clusters else None
        on_all = group.filter(pl.col("arm") == "on")
        off_all = group.filter(pl.col("arm") == "off")
        rows.append(
            {
                "planner": planner,
                "wait_on": float(on_all["wait_s"].median()) if len(on_all) else None,
                "wait_off": float(off_all["wait_s"].median()) if len(off_all) else None,
                "served_on": float(on_all["served"].mean()) if len(on_all) else None,
                "served_off": float(off_all["served"].mean()) if len(off_all) else None,
                "diff": None if result is None else result.estimate,
                "ci_low": None if result is None else result.ci_low,
                "ci_high": None if result is None else result.ci_high,
                "p": None if result is None else result.p,
                "n_cases": len(clusters),
            },
        )
    return pl.DataFrame(rows)


def _metrics_path(benchmark_dir: pathlib.Path) -> pathlib.Path:
    path = benchmark_dir / "combined_metrics.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `arena evaluation process --benchmark-dir {benchmark_dir}` first")
    return path


def analyze(benchmark_dir: pathlib.Path, *, n_boot: int = DEFAULT_BOOT, out_dir: pathlib.Path | None = None) -> dict[str, pl.DataFrame]:
    episodes = paper_outcomes(load_episodes(pl.read_parquet(_metrics_path(benchmark_dir))))
    tables = {
        "cases": case_table(episodes, n_boot=n_boot),
        "planners": planner_table(episodes, n_boot=n_boot),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.write_csv(out_dir / f"{name}.csv")
    for name, table in tables.items():
        print(f"\n== {name}")
        print(table)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--episodes", type=int, default=EPISODES)

    ana = sub.add_parser("analyze")
    ana.add_argument("--benchmark-dir", type=pathlib.Path, required=True)
    ana.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    ana.add_argument("--out", type=pathlib.Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(episodes=args.episodes)
    else:
        analyze(args.benchmark_dir, n_boot=args.boot, out_dir=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
