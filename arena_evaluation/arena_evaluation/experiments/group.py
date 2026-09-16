"""Study C: does the robot walk through a conversation when the layer cannot show it one?

Three pedestrians hold a group conversation (F-formation) on the robot's straight line to its goal. The arms
differ in everything the animation layer contributes and nothing else:

  on   gesture_mode=enabled  + social cost layer: the clip plays, the interaction id is published, and the
                             costmap fills the space between the participants
  off  gesture_mode=disabled + no social cost layer: the same three people stand in the same places, but
                             publish no clip and no interaction id, so a planner sees three unrelated bodies

The outcome is the route: whether the robot's path passes inside the formation, and how close it comes to a
person while doing so.

    python -m arena_evaluation.experiments.group measure --benchmark-dir <data_root>/<run_id> --centre 4.0 17.5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl

# F-formation radius for n participants, from humansim's f_formation: (base + per_member * (n - 1)) * scale
FORMATION_BASE_M = 0.7
FORMATION_PER_MEMBER_M = 0.12


def formation_radius(members: int, scale: float = 1.0) -> float:
    return (FORMATION_BASE_M + FORMATION_PER_MEMBER_M * max(0, members - 1)) * scale


def episode_routes(metrics: pl.DataFrame, centre: tuple[float, float], radius: float) -> pl.DataFrame:
    """Per episode: how far the path stayed from the conversation, and whether it went through it."""
    rows = []
    origin = np.asarray(centre, dtype=float)
    for row in metrics.iter_rows(named=True):
        path = row.get("path") or []
        points = np.array([[float(p[0]), float(p[1])] for p in path]) if len(path) else np.zeros((0, 2))
        distance = np.linalg.norm(points - origin, axis=1) if len(points) else np.array([float("nan")])
        rows.append(
            {
                "episode": row.get("episode"),
                "planner": row.get("planner"),
                "stage": row.get("stage"),
                "pierced": bool(np.nansum(distance < radius) > 0),
                "ticks_inside": int(np.nansum(distance < radius)),
                "closest_to_group_m": float(np.nanmin(distance)),
                "min_pedestrian_clearance": row.get("min_pedestrian_clearance"),
                "time_in_intimate_zone": row.get("time_in_intimate_zone"),
                "time_in_personal_zone": row.get("time_in_personal_zone"),
                "path_length": row.get("path_length"),
                "time_to_goal": row.get("time_to_goal"),
                "success": row.get("success"),
            },
        )
    return pl.DataFrame(rows)


def summarize(routes: pl.DataFrame) -> pl.DataFrame:
    return routes.group_by(["stage", "planner"], maintain_order=True).agg(
        pl.len().alias("episodes"),
        pl.col("pierced").mean().alias("pierce_rate"),
        pl.col("closest_to_group_m").median().alias("closest_to_group_m"),
        pl.col("min_pedestrian_clearance").median().alias("clearance_m"),
        pl.col("time_in_intimate_zone").median().alias("intimate_s"),
        pl.col("time_in_personal_zone").median().alias("personal_s"),
        pl.col("path_length").median().alias("path_m"),
        pl.col("time_to_goal").median().alias("ttg_s"),
        pl.col("success").mean().alias("success"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--benchmark-dir", type=pathlib.Path, required=True, nargs="+")
    measure.add_argument("--centre", type=float, nargs=2, required=True, metavar=("X", "Y"))
    measure.add_argument("--members", type=int, default=3)
    measure.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    radius = formation_radius(args.members, args.scale)
    frames = []
    for directory in args.benchmark_dir:
        path = directory / "combined_metrics.parquet"
        if not path.is_file():
            raise SystemExit(f"{path} not found; run `arena_evaluation.cli process --benchmark-dir {directory}` first")
        frames.append(episode_routes(pl.read_parquet(path), tuple(args.centre), radius))
    routes = pl.concat(frames)
    print(f"formation radius {radius:.2f} m about ({args.centre[0]}, {args.centre[1]})\n")
    print(routes)
    print()
    print(summarize(routes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
