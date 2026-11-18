"""CLI subcommands for acoustic field listing, animation, and snapshots.

``list`` prints the per-episode pedestrian exposure roll-up (max, Leq and summed max exposure)
beside the existing acoustic level and success columns. ``animate`` and ``snapshot`` obtain their
renderer through the public ``prepare_renderer`` dispatch and call only its public methods.
``snapshot --frame`` defaults, in order, to the frame nearest the max-exposure pedestrian's
``worst_case_acoustic_t_ns`` from ``fact_pedestrians``, then to the frame of maximum exposure in
``series_acoustic_exposure``, then to the frame with the highest recorded source level.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import polars as pl

from arena_evaluation.presentation.viz_manifest import PlotSpec
from arena_evaluation.storage.exceptions import MetricInputError

_ACOUSTIC_SPEC_ID = "acoustic_field"
_EXPOSURE_COLUMNS = ("ped_max_exposure_dba", "ped_leq_exposure_dba")


def handle_acoustic(args: argparse.Namespace) -> None:
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
            _acoustic_list(df, benchmark_dir)
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


def _load_pedestrian_exposure(benchmark_dir: pathlib.Path) -> pl.DataFrame | None:
    """Per-episode pedestrian exposure roll-up, None when the facts lack the exposure columns."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog

    catalog = BenchmarkCatalog.open(benchmark_dir)
    table = catalog.get_table("fact_pedestrians")
    names = set(table.collect_schema().names())
    if not {"episode_id", *_EXPOSURE_COLUMNS} <= names:
        return None
    measured = pl.col("ped_max_exposure_dba").is_not_null()
    return (
        table.group_by("episode_id")
        .agg(
            pl.col("ped_max_exposure_dba").max().alias("max_exposure_dba"),
            pl.col("ped_leq_exposure_dba").max().alias("leq_exposure_dba"),
            pl.when(measured.any())
            .then(pl.col("ped_max_exposure_dba").sum())
            .otherwise(None)
            .alias("total_exposure_dba"),
        )
        .collect()
    )


def _format_level(value: object) -> str:
    """One exposure level for the text table, '-' when it was not measured."""
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.2f}"


def _map_for_episode(df: pl.DataFrame, episode_id: str) -> str | None:
    """World of one resolved episode from the canonical episode summary."""
    rows = df.filter(pl.col("episode_id") == _episode_number(episode_id))
    if rows.is_empty():
        return None
    world = rows["world"][0]
    return str(world) if world is not None else None


def _episode_number(episode_id: str) -> int:
    """The integer number of an episode id written as ``episode_NNN``."""
    return int(episode_id.split("_")[-1])


def _acoustic_list(df: pl.DataFrame, benchmark_dir: pathlib.Path) -> None:
    """Print one row per episode with its pedestrian exposure and equivalent sound level."""
    exposure = _load_pedestrian_exposure(benchmark_dir)
    if exposure is None:
        print(
            "Notice: fact_pedestrians has no pedestrian acoustic exposure columns; "
            "the exposure columns are left blank."
        )
        work = df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("max_exposure_dba"),
            pl.lit(None, dtype=pl.Float64).alias("leq_exposure_dba"),
            pl.lit(None, dtype=pl.Float64).alias("total_exposure_dba"),
        )
    else:
        work = df.join(exposure, on="episode_id", how="left")
    work = work.filter(pl.col("l_aeq_t").is_not_null()).sort("l_aeq_t", descending=True)

    print(
        f"{'EPISODE':<16} {'MAX_DBA':>10} {'LEQ_DBA':>10} {'TOTAL_EXP':>10} "
        f"{'L_AEQ_T_DBA':>14} {'SUCCESS':>8}"
    )
    print("-" * 74)
    for (
        ep_id,
        max_exposure,
        leq_exposure,
        total_exposure,
        level,
        success,
    ) in zip(
        work["episode_id"].to_numpy(),
        work["max_exposure_dba"].to_numpy(),
        work["leq_exposure_dba"].to_numpy(),
        work["total_exposure_dba"].to_numpy(),
        work["l_aeq_t"].to_numpy(),
        work["success"].to_numpy(),
        strict=True,
    ):
        print(
            f"{'episode_' + format(int(ep_id), '03d'):<16} "
            f"{_format_level(max_exposure):>10} {_format_level(leq_exposure):>10} "
            f"{_format_level(total_exposure):>10} {float(level):>14.2f} "
            f"{str(bool(success)):>8}"
        )


def _resolve_episode(
    df: pl.DataFrame, episode_spec: str, benchmark_dir: pathlib.Path
) -> str | None:
    """Resolve an episode specifier ('worst', 'max-total', 'loudest-source', or 'episode_NNN')."""
    match episode_spec:
        case str() if episode_spec.startswith("episode_"):
            episode = _episode_number(episode_spec)
            if df.filter(pl.col("episode_id") == episode).is_empty():
                print(f"Error: {episode_spec} not found in the episode summary.")
                return None
            return episode_spec
        case "worst":
            return _episode_of_max_exposure(
                benchmark_dir,
                column="max_exposure_dba",
                metric_label="max exposure",
                selector="worst",
            )
        case "max-total":
            return _episode_of_max_exposure(
                benchmark_dir,
                column="total_exposure_dba",
                metric_label="total exposure",
                selector="max-total",
            )
        case "loudest-source":
            return _loudest_source_episode(df, benchmark_dir)
        case _:
            print(f"Error: unknown episode specifier '{episode_spec}'.")
            return None


def _episode_of_max_exposure(
    benchmark_dir: pathlib.Path,
    *,
    column: str,
    metric_label: str,
    selector: str,
) -> str | None:
    """Episode with the largest pedestrian exposure roll-up value in one column."""
    exposure = _load_pedestrian_exposure(benchmark_dir)
    if exposure is None:
        print(
            "Error: fact_pedestrians has no pedestrian acoustic exposure columns. "
            f"Cannot resolve the '{selector}' episode."
        )
        return None
    measured = exposure.filter(pl.col(column).is_not_null())
    if measured.is_empty():
        print(
            f"Error: fact_pedestrians has no {metric_label} values. "
            f"Cannot resolve the '{selector}' episode."
        )
        return None
    best = measured.sort(column, descending=True).row(0, named=True)
    episode = int(best["episode_id"])
    print(
        f"Using {selector} episode: episode_{episode:03d} "
        f"({metric_label}={float(best[column]):.2f} dBA)"
    )
    return f"episode_{episode:03d}"


def _loudest_source_episode(
    df: pl.DataFrame, benchmark_dir: pathlib.Path
) -> str | None:
    """Episode holding the loudest instantaneous source level from its recorded telemetry."""
    from arena_evaluation.processing.acoustics.episode_series import (
        load_aligned_episode,
    )

    measured = df.filter(pl.col("l_aeq_t").is_not_null())
    if measured.is_empty():
        print("Error: no episode has acoustic field data.")
        return None

    best_source = float("-inf")
    best_episode: int | None = None
    for episode in measured["episode_id"]:
        field = load_aligned_episode(benchmark_dir, f"episode_{int(episode):03d}")
        if field is None or field.is_empty():
            continue
        peak = field["source_dba"].max() if "source_dba" in field.schema else None
        if peak is not None and np.isfinite(float(peak)) and float(peak) > best_source:
            best_source, best_episode = float(peak), int(episode)
    if best_episode is None:
        print("Error: no recorded acoustic field frame found.")
        return None
    print(
        f"Using loudest-source episode: episode_{best_episode:03d} (source={best_source:.1f} dBA)"
    )
    return f"episode_{best_episode:03d}"


def _animation_spec(args: argparse.Namespace) -> PlotSpec:
    """The single PlotSpec holding every argparse option of the animate command."""
    options: dict[str, object] = {
        "format": args.format,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "stride": args.stride,
        "downsample": args.downsample,
        "dpi": args.dpi,
        "vmin": args.vmin,
        "robot_trail": args.robot_trail,
        "show_doors": not args.no_door_overlay,
        "overlay_trajectories": args.overlay_trajectories,
    }
    if args.vmax is not None:
        options["vmax"] = args.vmax
    return PlotSpec(
        id=_ACOUSTIC_SPEC_ID, type="acoustic_field", title="", options=options
    )


def _snapshot_spec(args: argparse.Namespace) -> PlotSpec:
    """The single PlotSpec holding every argparse option of the snapshot command."""
    options: dict[str, object] = {
        "downsample": args.downsample,
        "vmin": args.vmin,
        "overlay_trajectories": args.overlay_trajectories,
    }
    if args.vmax is not None:
        options["vmax"] = args.vmax
    return PlotSpec(
        id=_ACOUSTIC_SPEC_ID, type="acoustic_field", title="", options=options
    )


def _acoustic_animate(df: pl.DataFrame, args: argparse.Namespace) -> None:
    """Generate an animated acoustic field visualization."""
    from arena_evaluation.presentation.plot_types import prepare_renderer
    from arena_evaluation.processing.acoustics.episode_series import (
        load_aligned_episode,
    )

    episode_id = _resolve_episode(df, args.episode, args.benchmark_dir)
    if episode_id is None:
        sys.exit(1)

    map_name = _map_for_episode(df, episode_id)
    if not map_name:
        print(
            "Error: could not determine the episode's world from the episode summary."
        )
        sys.exit(1)

    episode_df = load_aligned_episode(args.benchmark_dir, episode_id)
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}. Run 'evaluation extract' first.")
        sys.exit(1)

    spec = _animation_spec(args)
    try:
        renderer = prepare_renderer(spec, episode_df, run_dir=args.benchmark_dir)
        grid, meta = renderer.load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    except MetricInputError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    resolution = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    doors = meta["doors"]
    state_timeline = renderer.load_door_timeline(args.benchmark_dir, episode_id)

    if args.output:
        out_path = args.output
    else:
        plots_dir = args.benchmark_dir / "plots"
        extension = args.format if args.format != "frames" else ""
        out_path = (
            plots_dir / f"{episode_id}_acoustic.{extension}"
            if extension
            else plots_dir / f"{episode_id}_acoustic_frames"
        )

    print(f"Rendering animation for {episode_id} ({len(episode_df)} data frames)...")
    print(f"  fps={args.fps}, max_frames={args.max_frames}, stride={args.stride}")
    print(f"  downsample={args.downsample}, format={args.format}")
    print(
        f"  doors: {len(doors)} found"
        + (f", timeline: {'present' if state_timeline else 'ABSENT'}" if doors else "")
    )

    try:
        result_path = renderer.render_animation(
            episode_df,
            grid,
            resolution,
            ox,
            oy,
            doors,
            state_timeline=state_timeline,
            out_path=out_path,
            downsample=int(spec.options["downsample"]),
            stride=int(spec.options["stride"]),
            max_frames=int(spec.options["max_frames"]),
            fps=int(spec.options["fps"]),
            dpi=int(spec.options["dpi"]),
            vmin=float(spec.options["vmin"]),
            vmax=(float(spec.options["vmax"]) if "vmax" in spec.options else None),
            robot_trail=int(spec.options["robot_trail"]),
            show_doors=bool(spec.options["show_doors"]),
            format=str(spec.options["format"]),
            overlay_trajectories=bool(spec.options["overlay_trajectories"]),
        )
    except MetricInputError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if result_path:
        print(f"Animation saved to: {result_path}")
    else:
        print("Animation generation failed.")
        sys.exit(1)


def _pedestrian_worst_case_t_ns(
    benchmark_dir: pathlib.Path, episode_id: str
) -> int | None:
    """The worst-exposure pedestrian's own worst acoustic instant, None when not published."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog

    catalog = BenchmarkCatalog.open(benchmark_dir)
    table = catalog.get_table("fact_pedestrians")
    names = set(table.collect_schema().names())
    if not {"episode_id", *_EXPOSURE_COLUMNS, "worst_case_acoustic_t_ns"} <= names:
        return None
    rows = (
        table.filter(
            (pl.col("episode_id") == _episode_number(episode_id))
            & pl.col("ped_max_exposure_dba").is_not_null()
            & pl.col("worst_case_acoustic_t_ns").is_not_null()
        )
        .sort("ped_max_exposure_dba", descending=True)
        .select("worst_case_acoustic_t_ns")
        .limit(1)
        .collect()
    )
    if rows.is_empty():
        return None
    return int(rows["worst_case_acoustic_t_ns"][0])


def _series_worst_exposure_t_ns(
    benchmark_dir: pathlib.Path, episode_id: str
) -> int | None:
    """The instant of maximum pedestrian exposure in the recorded exposure series."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog

    catalog = BenchmarkCatalog.open(benchmark_dir)
    if "series_acoustic_exposure" not in catalog.table_names():
        return None
    rows = (
        catalog.get_table("series_acoustic_exposure")
        .filter(pl.col("episode_id") == _episode_number(episode_id))
        .select(["t_ns", "exposure_dba"])
        .filter(pl.col("exposure_dba").is_not_null())
        .sort("exposure_dba", descending=True)
        .limit(1)
        .collect()
    )
    if rows.is_empty():
        return None
    return int(rows["t_ns"][0])


def _nearest_frame_index(episode_df: pl.DataFrame, t_ns: int) -> int:
    """The episode frame nearest one instant."""
    return int((episode_df["t_ns"] - int(t_ns)).abs().arg_min() or 0)


def _default_snapshot_frame(
    episode_df: pl.DataFrame,
    rows: list[dict[str, object]],
    benchmark_dir: pathlib.Path,
    episode_id: str,
) -> tuple[int, dict[str, object], str]:
    """The documented default frame and the evidence that selected it."""
    target_ns = _pedestrian_worst_case_t_ns(benchmark_dir, episode_id)
    if target_ns is not None:
        index = _nearest_frame_index(episode_df, target_ns)
        return (
            index,
            rows[index],
            f"nearest worst-case pedestrian exposure at t={target_ns / 1e9:.1f}s",
        )
    target_ns = _series_worst_exposure_t_ns(benchmark_dir, episode_id)
    if target_ns is not None:
        index = _nearest_frame_index(episode_df, target_ns)
        return (
            index,
            rows[index],
            f"maximum exposure in series_acoustic_exposure at t={target_ns / 1e9:.1f}s",
        )
    levels = episode_df["source_dba"].to_numpy()
    finite = np.isfinite(levels)
    index = int(np.argmax(np.where(finite, levels, -np.inf))) if finite.any() else 0
    return index, rows[index], "highest recorded source level"


def _acoustic_snapshot(df: pl.DataFrame, args: argparse.Namespace) -> None:
    """Render a single acoustic field frame."""
    from arena_evaluation.presentation.plot_types import prepare_renderer
    from arena_evaluation.processing.acoustics.episode_series import (
        load_aligned_episode,
    )

    episode_id = _resolve_episode(df, args.episode, args.benchmark_dir)
    if episode_id is None:
        sys.exit(1)

    map_name = _map_for_episode(df, episode_id)
    if not map_name:
        print(
            "Error: could not determine the episode's world from the episode summary."
        )
        sys.exit(1)

    episode_df = load_aligned_episode(args.benchmark_dir, episode_id)
    if episode_df is None:
        print(f"Error: no topic data for {episode_id}.")
        sys.exit(1)

    spec = _snapshot_spec(args)
    try:
        renderer = prepare_renderer(spec, episode_df, run_dir=args.benchmark_dir)
        grid, meta = renderer.load_grid_and_meta(map_name, run_dir=args.benchmark_dir)
    except MetricInputError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    resolution = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    doors = meta["doors"]
    state_timeline = renderer.load_door_timeline(args.benchmark_dir, episode_id)

    rows = episode_df.rows(named=True)
    if args.frame is not None and 0 <= args.frame < len(rows):
        frame_idx = args.frame
        row = rows[frame_idx]
    else:
        frame_idx, row, reason = _default_snapshot_frame(
            episode_df, rows, args.benchmark_dir, episode_id
        )
        print(
            f"Using frame {frame_idx} ({reason}). "
            "Use --frame N to pick a specific frame."
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

    peds = renderer.parse_pedestrian_positions(row.get("peds_positions"))

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
        renderer.trajectory_data(
            episode_df,
            run_dir=args.benchmark_dir,
            episode_id=episode_id,
        )
        if args.overlay_trajectories
        else None
    )

    open_doors = set(open_set) if doors else None
    vmin = float(spec.options["vmin"])
    vmax = (
        float(spec.options["vmax"])
        if "vmax" in spec.options
        else max(vmin + 20.0, float(np.ceil(source_dba / 5.0) * 5.0))
    )

    ok = renderer.render_cell_png(
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
        downsample=int(spec.options["downsample"]),
        vmin=vmin,
        vmax=vmax,
        open_doors=open_doors,
        doors=doors if doors else None,
        overlay_trajectories=bool(spec.options["overlay_trajectories"]),
        trajectory_data=traj_data,
    )

    if ok:
        print(f"Snapshot saved to: {out_path}")
    else:
        print("Snapshot generation failed.")
        sys.exit(1)


def add_acoustic_commands(subparsers: argparse._SubParsersAction) -> None:
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

    episode_help = (
        "Episode ID or selector: 'worst' (max pedestrian exposure), "
        "'max-total' (largest summed pedestrian exposure), "
        "'loudest-source' (highest recorded source level) (default: worst)."
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
        help=episode_help,
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
        help="Output format (default: gif); mp4 requires ffmpeg on PATH.",
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
        help=episode_help,
    )
    acoustic_snap.add_argument(
        "--frame",
        type=int,
        default=None,
        help=(
            "Frame index. Default resolution order: (1) the frame nearest the "
            "max-exposure pedestrian's worst_case_acoustic_t_ns from fact_pedestrians, "
            "(2) the frame of maximum exposure in series_acoustic_exposure, "
            "(3) the frame with the highest recorded source level."
        ),
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
