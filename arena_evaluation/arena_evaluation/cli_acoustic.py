"""CLI subcommands for acoustic field listing, animation, and snapshots."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl


def _handle_acoustic(args: argparse.Namespace) -> None:
    """Handle 'evaluation acoustic' subcommands."""
    benchmark_dir = args.benchmark_dir
    if isinstance(benchmark_dir, list):
        benchmark_dir = benchmark_dir[0]
    args.benchmark_dir = benchmark_dir

    if not benchmark_dir.is_dir():
        print(f"Error: benchmark directory does not exist: {benchmark_dir}")
        sys.exit(1)

    df = _load_episode_summary(benchmark_dir)

    match args.acoustic_command:
        case "list":
            _acoustic_list(df)
        case "animate":
            _acoustic_animate(df, args)
        case "snapshot":
            _acoustic_snapshot(df, args)


def _load_episode_summary(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """Canonical Star Schema episode summary of a benchmark, exiting when never processed."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog

    if not (benchmark_dir / "dim_episodes.parquet").is_file():
        print(f"Error: no Star Schema episode summary found in {benchmark_dir}")
        print("Run 'evaluation process --benchmark-dir ...' first.")
        sys.exit(1)
    return BenchmarkCatalog.open(benchmark_dir).get_table("v_episode_summary").collect()


def _map_for_episode(df: pl.DataFrame, episode_id: str) -> str | None:
    """World of one resolved episode from the canonical episode summary."""
    rows = df.filter(pl.col("episode_id") == _episode_number(episode_id))
    if rows.is_empty():
        return None
    world = rows["world"][0]
    return str(world) if world is not None else None


def _episode_number(episode_id: str) -> int:
    return int(episode_id.split("_")[-1])


def _acoustic_list(df: pl.DataFrame) -> None:
    """Print one row per episode with its equivalent continuous sound level and success."""
    work = df.filter(pl.col("l_aeq_t").is_not_null()).sort("l_aeq_t", descending=True)

    print(f"{'EPISODE':<16} {'L_AEQ_T_DBA':>14} {'SUCCESS':>8}")
    print("-" * 38)
    for ep_id, level, success in zip(
        work["episode_id"].to_numpy(),
        work["l_aeq_t"].to_numpy(),
        work["success"].to_numpy(),
        strict=True,
    ):
        print(
            f"{'episode_' + format(int(ep_id), '03d'):<16} "
            f"{float(level):>14.2f} {str(bool(success)):>8}"
        )


def _resolve_episode(
    df: pl.DataFrame, episode_spec: str, benchmark_dir: pathlib.Path
) -> str | None:
    """Resolve an episode specifier ('worst', 'loudest-source', 'max-total', or 'episode_NNN')."""
    match episode_spec:
        case str() if episode_spec.startswith("episode_"):
            episode = _episode_number(episode_spec)
            if df.filter(pl.col("episode_id") == episode).is_empty():
                print(f"Error: {episode_spec} not found in the episode summary.")
                return None
            return episode_spec
        case "worst" | "max-total":
            return _episode_of_max_level(df, episode_spec)
        case "loudest-source":
            return _loudest_source_episode(df, benchmark_dir)
        case _:
            print(f"Error: unknown episode specifier '{episode_spec}'.")
            return None


def _episode_of_max_level(df: pl.DataFrame, label: str) -> str | None:
    """Episode with the largest equivalent continuous sound level."""
    measured = df.filter(pl.col("l_aeq_t").is_not_null())
    if measured.is_empty():
        print("Error: l_aeq_t has no values. Cannot resolve the episode.")
        return None
    best = measured.sort("l_aeq_t", descending=True).row(0, named=True)
    episode = int(best["episode_id"])
    print(
        f"Using {label} episode: episode_{episode:03d} (l_aeq_t={best['l_aeq_t']:.2f} dBA)"
    )
    return f"episode_{episode:03d}"


def _loudest_source_episode(
    df: pl.DataFrame, benchmark_dir: pathlib.Path
) -> str | None:
    """Episode holding the loudest instantaneous source level from its recorded telemetry."""
    from arena_evaluation.presentation.plot_types.acoustic_field import (
        AcousticFieldRenderer,
    )

    measured = df.filter(pl.col("l_aeq_t").is_not_null())
    if measured.is_empty():
        print("Error: no episode has acoustic field data.")
        return None

    best_source = float("-inf")
    best_episode: int | None = None
    for episode in measured["episode_id"]:
        field = AcousticFieldRenderer._load_episode_data(
            benchmark_dir, f"episode_{int(episode):03d}"
        )
        if field is None or field.is_empty() or "source_dba" not in field.schema:
            continue
        peak = field["source_dba"].max()
        if peak is not None and float(peak) > best_source:
            best_source, best_episode = float(peak), int(episode)
    if best_episode is None:
        print("Error: no recorded acoustic field frame found.")
        return None
    print(
        f"Using loudest-source episode: episode_{best_episode:03d} (source={best_source:.1f} dBA)"
    )
    return f"episode_{best_episode:03d}"


def _acoustic_animate(df: pl.DataFrame, args: argparse.Namespace) -> None:
    """Generate an animated acoustic field visualization."""
    episode_id = _resolve_episode(df, args.episode, args.benchmark_dir)
    if episode_id is None:
        sys.exit(1)

    from arena_evaluation.presentation.plot_types.acoustic_field import (
        AcousticFieldRenderer,
    )
    from arena_evaluation.presentation.viz_manifest import PlotSpec
    from arena_evaluation.storage.exceptions import MetricInputError

    renderer = AcousticFieldRenderer(
        PlotSpec(id="acoustic_field", type="acoustic_field", title="")
    )
    renderer.run_dir = args.benchmark_dir

    map_name = _map_for_episode(df, episode_id)
    if not map_name:
        print(
            "Error: could not determine the episode's world from the episode summary."
        )
        sys.exit(1)

    try:
        grid, meta = renderer._load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    except MetricInputError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    resolution = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

    episode_df = AcousticFieldRenderer._load_episode_data(
        args.benchmark_dir, episode_id
    )
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}. Run 'evaluation extract' first.")
        sys.exit(1)

    doors = meta["doors"]
    state_timeline = AcousticFieldRenderer._load_door_timeline(
        args.benchmark_dir, episode_id
    )

    if args.output:
        out_path = args.output
    else:
        plots_dir = args.benchmark_dir / "plots"
        ext = args.format if args.format != "frames" else ""
        out_path = (
            plots_dir / f"{episode_id}_acoustic.{ext}"
            if ext
            else plots_dir / f"{episode_id}_acoustic_frames"
        )

    print(f"Rendering animation for {episode_id} ({len(episode_df)} data frames)...")
    print(f"  fps={args.fps}, max_frames={args.max_frames}, stride={args.stride}")
    print(f"  downsample={args.downsample}, format={args.format}")
    print(
        f"  doors: {len(doors)} found"
        + (f", timeline: {'present' if state_timeline else 'ABSENT'}" if doors else "")
    )

    result_path = renderer.render_animation(
        episode_df,
        grid,
        resolution,
        ox,
        oy,
        doors,
        state_timeline=state_timeline,
        out_path=out_path,
        downsample=args.downsample,
        stride=args.stride,
        max_frames=args.max_frames,
        fps=args.fps,
        dpi=args.dpi,
        vmin=args.vmin,
        vmax=args.vmax,
        robot_trail=args.robot_trail,
        show_doors=not args.no_door_overlay,
        fmt=args.format,
        overlay_trajectories=args.overlay_trajectories,
    )

    if result_path:
        print(f"Animation saved to: {result_path}")
    else:
        print("Animation generation failed.")
        sys.exit(1)


def _acoustic_snapshot(df: pl.DataFrame, args: argparse.Namespace) -> None:
    """Render a single acoustic field frame."""
    episode_id = _resolve_episode(df, args.episode, args.benchmark_dir)
    if episode_id is None:
        sys.exit(1)

    from arena_evaluation.presentation.plot_types.acoustic_field import (
        AcousticFieldRenderer,
    )
    from arena_evaluation.presentation.viz_manifest import PlotSpec
    from arena_evaluation.storage.exceptions import MetricInputError

    renderer = AcousticFieldRenderer(
        PlotSpec(id="acoustic_field", type="acoustic_field", title="")
    )
    renderer.run_dir = args.benchmark_dir

    map_name = _map_for_episode(df, episode_id)
    if not map_name:
        print(
            "Error: could not determine the episode's world from the episode summary."
        )
        sys.exit(1)

    try:
        grid, meta = renderer._load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    except MetricInputError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    resolution = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

    episode_df = AcousticFieldRenderer._load_episode_data(
        args.benchmark_dir, episode_id
    )
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}.")
        sys.exit(1)

    doors = meta["doors"]
    state_timeline = AcousticFieldRenderer._load_door_timeline(
        args.benchmark_dir, episode_id
    )

    rows = episode_df.rows(named=True)
    if args.frame is not None and 0 <= args.frame < len(rows):
        frame_idx = args.frame
        row = rows[frame_idx]
    else:
        best_idx = 0
        best_src = float("-inf")
        for i, r in enumerate(rows):
            s = r.get("source_dba")
            if s is not None and float(s) > best_src:
                best_src = float(s)
                best_idx = i
        frame_idx = best_idx
        row = rows[best_idx]
        print(
            f"Using frame {frame_idx} (source={best_src:.1f} dBA). Use --frame N to pick a specific frame."
        )

    rx_m_raw = row.get("pos_x_gt")
    ry_m_raw = row.get("pos_y_gt")
    src_raw = row.get("source_dba")
    if rx_m_raw is None or ry_m_raw is None or src_raw is None:
        print(
            f"Error: frame {frame_idx} of {episode_id} is missing pose or source_dba."
        )
        sys.exit(1)

    rx_m = float(rx_m_raw)
    ry_m = float(ry_m_raw)
    source_dba = float(src_raw)
    t_ns = int(row.get("t_ns", 0))

    peds_pos = row.get("peds_positions", [])
    peds = AcousticFieldRenderer._parse_pedestrian_positions(peds_pos)

    open_set = (
        state_timeline.open_doors_at(t_ns)
        if state_timeline is not None
        else frozenset()
    )

    out_path = args.output or (
        args.benchmark_dir / "plots" / f"{episode_id}_acoustic_snapshot.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Rendering snapshot: frame {frame_idx}, t={t_ns / 1e9:.1f}s, "
        f"source={source_dba:.0f} dBA, {len(open_set)} doors open"
    )

    traj_data = (
        renderer._extract_trajectory_data(
            episode_df,
            run_dir=args.benchmark_dir,
            episode_id=episode_id,
        )
        if args.overlay_trajectories
        else None
    )

    open_doors = set(open_set) if doors else None
    vmin = args.vmin if args.vmin is not None else 20.0
    vmax = (
        args.vmax
        if args.vmax is not None
        else max(vmin + 20.0, float(np.ceil(source_dba / 5.0) * 5.0))
    )

    ok = renderer._render_cell_png(
        grid,
        resolution,
        ox,
        oy,
        rx_m,
        ry_m,
        source_dba,
        peds,
        title=f"{episode_id}  frame {frame_idx}  t={t_ns / 1e9:.1f}s  {source_dba:.0f} dBA",
        out_path=out_path,
        downsample=args.downsample,
        vmin=vmin,
        vmax=vmax,
        open_doors=open_doors,
        doors=doors if doors else None,
        overlay_trajectories=args.overlay_trajectories,
        trajectory_data=traj_data,
    )

    if ok:
        print(f"Snapshot saved to: {out_path}")
    else:
        print("Snapshot generation failed.")
        sys.exit(1)


def setup_acoustic_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add acoustic subcommands to the main parser."""
    acoustic_parser = subparsers.add_parser(
        "acoustic",
        help="Acoustic field visualization: list episodes, animate fields, render snapshots.",
    )
    acoustic_sub = acoustic_parser.add_subparsers(
        dest="acoustic_command", required=True
    )

    acoustic_list = acoustic_sub.add_parser(
        "list", help="List episodes with acoustic metrics."
    )
    acoustic_list.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        required=True,
        metavar="DIR",
        help="Path to benchmark directory.",
    )

    acoustic_anim = acoustic_sub.add_parser(
        "animate", help="Generate animated GIF/MP4 of acoustic field."
    )
    acoustic_anim.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        required=True,
        metavar="DIR",
        help="Path to benchmark directory.",
    )
    acoustic_anim.add_argument(
        "--episode",
        type=str,
        default="worst",
        metavar="EP",
        help="Episode ID, 'worst', 'loudest-source', or 'max-total' (default: worst).",
    )
    acoustic_anim.add_argument(
        "--fps", type=int, default=10, help="Output frame rate (default: 10)."
    )
    acoustic_anim.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Cap rendered frames (default: 120).",
    )
    acoustic_anim.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every Nth data frame (default: 1).",
    )
    acoustic_anim.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Solver grid downsample (default: 2).",
    )
    acoustic_anim.add_argument(
        "--format",
        type=str,
        default="gif",
        choices=["gif", "mp4", "frames"],
        help="Output format (default: gif).",
    )
    acoustic_anim.add_argument(
        "--dpi", type=int, default=150, help="Output resolution (default: 150)."
    )
    acoustic_anim.add_argument(
        "--vmin",
        type=float,
        default=20.0,
        help="Color-scale floor in dBA (default: 20).",
    )
    acoustic_anim.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color-scale ceiling in dBA (default: auto).",
    )
    acoustic_anim.add_argument(
        "--no-door-overlay", action="store_true", help="Hide door contours."
    )
    acoustic_anim.add_argument(
        "--no-trajectories",
        action="store_false",
        dest="overlay_trajectories",
        help="Disable trajectory overlay.",
    )
    acoustic_anim.add_argument(
        "--robot-trail",
        type=int,
        default=0,
        help="Show past N robot positions as trail (default: 0).",
    )
    acoustic_anim.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Override output path (default: plots/{episode}_acoustic.{format}).",
    )

    acoustic_snap = acoustic_sub.add_parser(
        "snapshot", help="Render a single acoustic field frame."
    )
    acoustic_snap.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        required=True,
        metavar="DIR",
        help="Path to benchmark directory.",
    )
    acoustic_snap.add_argument(
        "--episode",
        type=str,
        default="worst",
        metavar="EP",
        help="Episode ID or keyword (default: worst).",
    )
    acoustic_snap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index (default: worst-case frame).",
    )
    acoustic_snap.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Solver grid downsample (default: 2).",
    )
    acoustic_snap.add_argument(
        "--vmin",
        type=float,
        default=20.0,
        help="Color-scale floor in dBA (default: 20).",
    )
    acoustic_snap.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color-scale ceiling in dBA (default: auto).",
    )
    acoustic_snap.add_argument(
        "--no-trajectories",
        action="store_false",
        dest="overlay_trajectories",
        help="Disable trajectory overlay.",
    )
    acoustic_snap.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Override output path (default: plots/{episode}_acoustic_snapshot.png).",
    )
