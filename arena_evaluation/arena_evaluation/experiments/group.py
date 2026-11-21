"""Study C: evaluate robot routing around a group conversation with and without the social cost layer."""

from __future__ import annotations

import argparse
import pathlib
import sys

import polars as pl

from arena_evaluation.experiments.source import episode_summary, robot_tracks

FORMATION_BASE_M = 0.7
FORMATION_PER_MEMBER_M = 0.12


def formation_radius(members: int, scale: float = 1.0) -> float:
    return (FORMATION_BASE_M + FORMATION_PER_MEMBER_M * max(0, members - 1)) * scale


def episode_routes(
    summary: pl.DataFrame,
    tracks: pl.DataFrame | pl.LazyFrame,
    center: tuple[float, float],
    radius: float,
) -> pl.DataFrame:
    """Per episode: how far the robot path stayed from the conversation, and whether it went through it."""
    tracks_lf = tracks.lazy() if isinstance(tracks, pl.DataFrame) else tracks
    offset_x = pl.col("x") - center[0]
    offset_y = pl.col("y") - center[1]
    distance = (offset_x * offset_x + offset_y * offset_y).sqrt()
    per_episode = (
        tracks_lf.group_by("episode_id")
        .agg(
            ticks_inside=(distance < radius).sum().cast(pl.Int64),
            closest_to_group_m=distance.min(),
        )
        .collect()
    )
    routes = summary.join(per_episode, on="episode_id", how="left").with_columns(
        ticks_inside=pl.col("ticks_inside").fill_null(0),
        pierced=pl.col("ticks_inside").fill_null(0) > 0,
    )
    return routes.select(
        "episode_id",
        "planner",
        "stage",
        "pierced",
        "ticks_inside",
        "closest_to_group_m",
        "min_ped_clearance_m",
        "time_in_intimate_zone",
        "time_in_personal_zone",
        "path_length_m",
        "time_to_goal",
        "success",
    )


def summarize(routes: pl.DataFrame) -> pl.DataFrame:
    return routes.group_by(["stage", "planner"], maintain_order=True).agg(
        pl.len().alias("episodes"),
        pl.col("pierced").mean().alias("pierce_rate"),
        pl.col("closest_to_group_m").median().alias("closest_to_group_m"),
        pl.col("min_ped_clearance_m").median().alias("clearance_m"),
        pl.col("time_in_intimate_zone").median().alias("intimate_s"),
        pl.col("time_in_personal_zone").median().alias("personal_s"),
        pl.col("path_length_m").median().alias("path_m"),
        pl.col("time_to_goal").median().alias("ttg_s"),
        pl.col("success").mean().alias("success"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--benchmark-dir", type=pathlib.Path, required=True, nargs="+")
    measure.add_argument(
        "--center", type=float, nargs=2, required=True, metavar=("X", "Y")
    )
    measure.add_argument("--members", type=int, default=3)
    measure.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    radius = formation_radius(args.members, args.scale)
    frames = []
    for directory in args.benchmark_dir:
        try:
            summary = episode_summary(directory)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        frames.append(
            episode_routes(summary, robot_tracks(directory), tuple(args.center), radius)
        )
    routes = pl.concat(frames)
    print(
        f"formation radius {radius:.2f} m about ({args.center[0]}, {args.center[1]})\n"
    )
    print(routes)
    print()
    print(summarize(routes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
