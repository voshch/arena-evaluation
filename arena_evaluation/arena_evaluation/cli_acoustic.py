import argparse
import pathlib
import sys

import numpy as np
import polars as pl


def _handle_acoustic(args: argparse.Namespace) -> None:
    """Handle 'evaluation acoustic' subcommands."""
    from arena_evaluation.processing.parquet_store import ParquetStore

    benchmark_dir = args.benchmark_dir
    if isinstance(benchmark_dir, list):
        benchmark_dir = benchmark_dir[0]
    args.benchmark_dir = benchmark_dir

    if not benchmark_dir.is_dir():
        print(f"Error: benchmark directory does not exist: {benchmark_dir}")
        sys.exit(1)

    metrics_path = benchmark_dir / "combined_metrics.parquet"
    if not metrics_path.exists():
        metrics_path = benchmark_dir / "metrics.parquet"
    if not metrics_path.exists():
        print(f"Error: no metrics.parquet or combined_metrics.parquet found in {benchmark_dir}")
        print("Run 'evaluation process --benchmark-dir ...' first.")
        sys.exit(1)

    df, _ = ParquetStore.read(metrics_path)

    if args.acoustic_command == "list":
        _acoustic_list(df)
    elif args.acoustic_command == "animate":
        _acoustic_animate(df, args)
    elif args.acoustic_command == "snapshot":
        _acoustic_snapshot(df, args)


def _acoustic_list(df: pl.DataFrame) -> None:
    """Print a table of episodes with acoustic metrics."""
    import polars as pl

    cols = []
    if "episode" in df.columns:
        cols.append("episode")
    if "ped_max_exposure_dba" in df.columns:
        cols.append("ped_max_exposure_dba")
    if "ped_leq_exposure_dba" in df.columns:
        cols.append("ped_leq_exposure_dba")

    if not cols:
        print("No acoustic metric columns found in metrics file.")
        return

    work = df.select(cols).sort("ped_max_exposure_dba", descending=True) if "ped_max_exposure_dba" in df.columns else df.select(cols)

    # Compute total exposure if timeseries available
    has_total = "timeseries_acoustic_exposure_dba" in df.columns

    print(f"{'EPISODE':<16} {'MAX_DBA':>10} {'LEQ_DBA':>10}", end="")
    if has_total:
        print(f" {'TOTAL_EXP':>12}", end="")
    print()
    print("-" * (16 + 10 + 10 + (12 if has_total else 0)))

    for row in work.iter_rows(named=True):
        ep = str(row.get("episode", "?"))
        mx = row.get("ped_max_exposure_dba")
        leq = row.get("ped_leq_exposure_dba")
        mx_str = f"{mx:.1f}" if mx is not None else "N/A"
        leq_str = f"{leq:.1f}" if leq is not None else "N/A"
        print(f"{ep:<16} {mx_str:>10} {leq_str:>10}", end="")
        if has_total:
            ts = df.filter(pl.col("episode") == row.get("episode")).select("timeseries_acoustic_exposure_dba").row(0)[0] if "episode" in df.columns else None
            if ts is not None:
                total = sum(sum(f) for f in ts if f)
                print(f" {total:>12.1f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()


def _resolve_episode(df: pl.DataFrame, episode_spec: str) -> str | None:
    """Resolve an episode specifier ('worst', 'loudest-source', 'max-total', or 'episode_NNN')."""
    import polars as pl

    if episode_spec.startswith("episode_"):
        ep_str = episode_spec
        if "episode" in df.columns:
            ep_int = int(episode_spec.split("_")[-1])
            if df.filter(pl.col("episode") == ep_int).is_empty():
                print(f"Error: {episode_spec} not found in metrics.")
                return None
        return ep_str

    if "ped_max_exposure_dba" not in df.columns:
        print("Error: ped_max_exposure_dba not found in metrics. Cannot resolve episode by metric.")
        return None

    if episode_spec == "worst":
        best = df.filter(pl.col("ped_max_exposure_dba").is_not_null()).sort("ped_max_exposure_dba", descending=True).row(0, named=True)
        ep = int(best.get("episode", 0))
        print(f"Using worst episode: episode_{ep:03d} (max={best['ped_max_exposure_dba']:.1f} dBA)")
        return f"episode_{ep:03d}"

    if episode_spec == "loudest-source":
        if "worst_case_acoustic_frame" not in df.columns:
            print("Error: worst_case_acoustic_frame not in metrics.")
            return None
        best_source = -1.0
        best_ep = 0
        for row in df.iter_rows(named=True):
            wf = row.get("worst_case_acoustic_frame")
            if wf is None:
                continue
            if isinstance(wf, str):
                import json

                try:
                    wf = json.loads(wf)
                except Exception:
                    continue
            src = float(wf.get("source_dba", 0))
            if src > best_source:
                best_source = src
                best_ep = int(row.get("episode", 0))
        print(f"Using loudest-source episode: episode_{best_ep:03d} (source={best_source:.1f} dBA)")
        return f"episode_{best_ep:03d}"

    if episode_spec == "max-total":
        if "timeseries_acoustic_exposure_dba" not in df.columns:
            print("Error: timeseries_acoustic_exposure_dba not in metrics.")
            return None
        best_total = -1.0
        best_ep = 0
        for row in df.iter_rows(named=True):
            ts = row.get("timeseries_acoustic_exposure_dba")
            if ts is None:
                continue
            total = sum(sum(f) for f in ts if f)
            if total > best_total:
                best_total = total
                best_ep = int(row.get("episode", 0))
        print(f"Using max-total episode: episode_{best_ep:03d} (total={best_total:.1f})")
        return f"episode_{best_ep:03d}"

    print(f"Error: unknown episode specifier '{episode_spec}'.")
    return None


def _acoustic_animate(df: pl.DataFrame, args: argparse.Namespace) -> None:
    """Generate an animated acoustic field visualization."""
    episode_id = _resolve_episode(df, args.episode)
    if episode_id is None:
        sys.exit(1)

    import polars as pl

    from arena_evaluation.presentation.plot_types.acoustic_field import AcousticFieldRenderer
    from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline

    renderer = AcousticFieldRenderer(None)
    renderer.run_dir = args.benchmark_dir

    # Load map
    map_name = None
    metrics_path = args.benchmark_dir / "combined_metrics.parquet"
    if not metrics_path.exists():
        metrics_path = args.benchmark_dir / "metrics.parquet"
    if metrics_path.exists():
        from arena_evaluation.processing.parquet_store import ParquetStore

        metrics_df, _ = ParquetStore.read(metrics_path)
        if "map" in metrics_df.columns and len(metrics_df) > 0:
            map_name = metrics_df["map"][0]

    if not map_name:
        print("Error: could not determine map name from metrics.")
        sys.exit(1)

    result = renderer._load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    if result is None:
        print(f"Error: could not load map '{map_name}'.")
        sys.exit(1)

    grid, meta = result
    resolution = meta["resolution"]
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

    # Load episode data
    episode_df = AcousticFieldRenderer._load_episode_data(args.benchmark_dir, episode_id)
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}. Run 'evaluation extract' first.")
        sys.exit(1)

    # Load doors
    from arena_evaluation.processing.acoustics.door_map import door_segments

    doors = door_segments(map_name, grid, resolution, (ox, oy, 0.0), run_dir=args.benchmark_dir)

    # Load door state timeline
    state_timeline = None
    semantic_path = args.benchmark_dir / "episodes" / episode_id / "topics" / "semantic_snapshot.parquet"
    if semantic_path.exists():
        semantic_df = pl.read_parquet(semantic_path)
        state_timeline = DoorStateTimeline.from_semantic_frame(semantic_df)

    # Output path
    if args.output:
        out_path = args.output
    else:
        plots_dir = args.benchmark_dir / "plots"
        ext = "gif" if args.format != "frames" else ""
        out_path = plots_dir / f"{episode_id}_acoustic.{ext}" if ext else plots_dir / f"{episode_id}_acoustic_frames"

    print(f"Rendering animation for {episode_id} ({len(episode_df)} data frames)...")
    print(f"  fps={args.fps}, max_frames={args.max_frames}, stride={args.stride}")
    print(f"  downsample={args.downsample}, format={args.format}")
    print(f"  doors: {len(doors)} found" + (f", timeline: {'present' if state_timeline else 'ABSENT'}" if doors else ""))

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
    episode_id = _resolve_episode(df, args.episode)
    if episode_id is None:
        sys.exit(1)

    import polars as pl

    from arena_evaluation.presentation.plot_types.acoustic_field import AcousticFieldRenderer
    from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline

    renderer = AcousticFieldRenderer(None)
    renderer.run_dir = args.benchmark_dir

    map_name = None
    metrics_path = args.benchmark_dir / "combined_metrics.parquet"
    if not metrics_path.exists():
        metrics_path = args.benchmark_dir / "metrics.parquet"
    if metrics_path.exists():
        from arena_evaluation.processing.parquet_store import ParquetStore

        metrics_df, _ = ParquetStore.read(metrics_path)
        if "map" in metrics_df.columns and len(metrics_df) > 0:
            map_name = metrics_df["map"][0]

    if not map_name:
        print("Error: could not determine map name.")
        sys.exit(1)

    result = renderer._load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    if result is None:
        print(f"Error: could not load map '{map_name}'.")
        sys.exit(1)

    grid, meta = result
    resolution = meta["resolution"]
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

    episode_df = AcousticFieldRenderer._load_episode_data(args.benchmark_dir, episode_id)
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}.")
        sys.exit(1)

    from arena_evaluation.processing.acoustics.door_map import door_segments

    doors = door_segments(map_name, grid, resolution, (ox, oy, 0.0), run_dir=args.benchmark_dir)

    state_timeline = None
    semantic_path = args.benchmark_dir / "episodes" / episode_id / "topics" / "semantic_snapshot.parquet"
    if semantic_path.exists():
        semantic_df = pl.read_parquet(semantic_path)
        state_timeline = DoorStateTimeline.from_semantic_frame(semantic_df)

    # Determine which frame
    rows = episode_df.rows(named=True)
    if args.frame is not None and 0 <= args.frame < len(rows):
        frame_idx = args.frame
        row = rows[frame_idx]
    else:
        # Use the frame with highest source_dba
        best_idx = 0
        best_src = -1
        for i, r in enumerate(rows):
            s = r.get("source_dba", 0) or 0
            if s > best_src:
                best_src = s
                best_idx = i
        frame_idx = best_idx
        row = rows[best_idx]
        print(f"Using frame {frame_idx} (source={best_src:.1f} dBA). Use --frame N to pick a specific frame.")

    rx_m = float(row.get("pos_x_gt", 0) or 0)
    ry_m = float(row.get("pos_y_gt", 0) or 0)
    source_dba = float(row.get("source_dba", 42.0) or 42.0)
    time_ns = int(row.get("time_ns", 0))

    peds_pos = row.get("peds_positions", [])
    peds = AcousticFieldRenderer._parse_pedestrian_positions(peds_pos)

    open_set = frozenset()
    if state_timeline is not None:
        open_set = state_timeline.open_doors_at(time_ns)

    from arena_evaluation.processing.acoustics.door_map import build_pixel_tl

    pixel_tl = build_pixel_tl(grid, doors, open_doors=set(open_set)) if doors else None

    out_path = args.output or (args.benchmark_dir / "plots" / f"{episode_id}_acoustic_snapshot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering snapshot: frame {frame_idx}, t={time_ns / 1e9:.1f}s, source={source_dba:.0f} dBA, {len(open_set)} doors open")

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
    vmax = args.vmax if args.vmax is not None else max(vmin + 20.0, float(np.ceil(source_dba / 5.0) * 5.0))

    ok = renderer._render_cell_png(
        grid,
        resolution,
        ox,
        oy,
        rx_m,
        ry_m,
        source_dba,
        peds,
        title=f"{episode_id}  frame {frame_idx}  t={time_ns / 1e9:.1f}s  {source_dba:.0f} dBA",
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
    acoustic_sub = acoustic_parser.add_subparsers(dest="acoustic_command", required=True)

    # acoustic list
    acoustic_list = acoustic_sub.add_parser("list", help="List episodes with acoustic metrics.")
    acoustic_list.add_argument("--benchmark-dir", type=pathlib.Path, required=True, metavar="DIR", help="Path to benchmark directory.")

    # acoustic animate
    acoustic_anim = acoustic_sub.add_parser("animate", help="Generate animated GIF/MP4 of acoustic field.")
    acoustic_anim.add_argument("--benchmark-dir", type=pathlib.Path, required=True, metavar="DIR", help="Path to benchmark directory.")
    acoustic_anim.add_argument("--episode", type=str, default="worst", metavar="EP", help="Episode ID, 'worst', 'loudest-source', or 'max-total' (default: worst).")
    acoustic_anim.add_argument("--fps", type=int, default=10, help="Output frame rate (default: 10).")
    acoustic_anim.add_argument("--max-frames", type=int, default=120, help="Cap rendered frames (default: 120).")
    acoustic_anim.add_argument("--stride", type=int, default=1, help="Render every Nth data frame (default: 1).")
    acoustic_anim.add_argument("--downsample", type=int, default=2, help="Solver grid downsample (default: 2).")
    acoustic_anim.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "frames"], help="Output format (default: gif).")
    acoustic_anim.add_argument("--dpi", type=int, default=150, help="Output resolution (default: 150).")
    acoustic_anim.add_argument("--vmin", type=float, default=20.0, help="Color-scale floor in dBA (default: 20).")
    acoustic_anim.add_argument("--vmax", type=float, default=None, help="Color-scale ceiling in dBA (default: auto).")
    acoustic_anim.add_argument("--no-door-overlay", action="store_true", help="Hide door contours.")
    acoustic_anim.add_argument("--no-trajectories", action="store_false", dest="overlay_trajectories", help="Disable trajectory overlay.")
    acoustic_anim.add_argument("--robot-trail", type=int, default=0, help="Show past N robot positions as trail (default: 0).")
    acoustic_anim.add_argument("--output", type=pathlib.Path, default=None, metavar="PATH", help="Override output path (default: plots/{episode}_acoustic.{format}).")

    # acoustic snapshot
    acoustic_snap = acoustic_sub.add_parser("snapshot", help="Render a single acoustic field frame.")
    acoustic_snap.add_argument("--benchmark-dir", type=pathlib.Path, required=True, metavar="DIR", help="Path to benchmark directory.")
    acoustic_snap.add_argument("--episode", type=str, default="worst", metavar="EP", help="Episode ID or keyword (default: worst).")
    acoustic_snap.add_argument("--frame", type=int, default=None, help="Frame index (default: worst-case frame).")
    acoustic_snap.add_argument("--downsample", type=int, default=2, help="Solver grid downsample (default: 2).")
    acoustic_snap.add_argument("--vmin", type=float, default=20.0, help="Color-scale floor in dBA (default: 20).")
    acoustic_snap.add_argument("--vmax", type=float, default=None, help="Color-scale ceiling in dBA (default: auto).")
    acoustic_snap.add_argument("--no-trajectories", action="store_false", dest="overlay_trajectories", help="Disable trajectory overlay.")
    acoustic_snap.add_argument("--output", type=pathlib.Path, default=None, metavar="PATH", help="Override output path (default: plots/{episode}_acoustic_snapshot.png).")
