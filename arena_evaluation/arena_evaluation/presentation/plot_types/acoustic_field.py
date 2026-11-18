"""Acoustic field spatial and animated plot renderers."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import typing
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    label_values,
)
from arena_evaluation.processing.acoustics.constants import (
    MIC_DISTANCE_M,
    ROBOT_ACOUSTICS_TOPIC,
    WALL_TRANSMISSION_LOSS_DB,
)
from arena_evaluation.processing.acoustics.door_map import (
    _entity_matches_door,
    barrier_door_tl,
    build_pixel_tl,
    load_barrier_map,
)
from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
from arena_evaluation.processing.acoustics.episode_series import (
    episode_number,
    load_aligned_episode,
)
from arena_evaluation.processing.acoustics.impedance_grid import (
    compute_acoustic_field,
    downsample_mask,
    downsample_occupancy,
)
from arena_evaluation.storage.exceptions import MetricInputError

if typing.TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_CELL_FIGSIZE = (5, 4)
_CELL_DPI = 150
# Colour-scale floor only; a missing measurement is never replaced by it.
_FIELD_VMIN_DBA = 20.0


def _cell_extent(
    origin_m: float, cells: int, cell_m: float, source_res_m: float
) -> tuple[float, float]:
    """Edges of one grid axis starting half a source cell below the origin."""
    lower = origin_m - source_res_m / 2
    return lower, lower + cells * cell_m


def _cell_centres(lower_m: float, cells: int, cell_m: float) -> np.ndarray:
    """Centres of the cells along one axis for wall and door contours."""
    return lower_m + (np.arange(cells) + 0.5) * cell_m


@dataclasses.dataclass(frozen=True, slots=True)
class _AnimationLayers:
    """The static figure elements an acoustic-field animation updates each frame."""

    fig: typing.Any
    ax: typing.Any
    im: typing.Any
    extent: tuple[float, float, float, float]
    cx: np.ndarray
    cy: np.ndarray
    wall_outline: np.ndarray
    grid_ds: np.ndarray
    door_mask_all: np.ndarray
    open_door_masks: list[np.ndarray]
    render_grids: list[np.ndarray]
    show_doors: bool
    h: int
    w: int
    vmin: float
    vmax: float
    open_door_contour: typing.Any
    robot_dot: typing.Any
    ped_dots: typing.Any
    trail_line: typing.Any
    title_text: typing.Any


@dataclasses.dataclass(frozen=True, slots=True)
class _AnimationFrames:
    """Per-rendered-frame bookkeeping: source rows, pedestrian positions and collision flags."""

    rows: list[dict[str, typing.Any]]
    indices: list[int]
    total_rows: int
    peds_per_frame: list[list[tuple[float, float]]]
    collision_per_frame: list[bool]


class AcousticFieldRenderer(BasePlotRenderer):
    """Renders 2D barrier-aware acoustic propagation fields."""

    PLOT_TYPE = "acoustic_field"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = (
        BasePlotRenderer.SUPPORTED_OPTIONS
        | frozenset(
            {
                "include_reference",
                "downsample",
                "vmin",
                "vmax",
                "mode",
                "overlay_trajectories",
                "fps",
                "max_frames",
                "stride",
                "dpi",
                "robot_trail",
                "show_doors",
                "format",
                "per_episode",
            }
        )
    )

    @staticmethod
    def load_grid_and_meta(
        world_name: str, run_dir: pathlib.Path | None = None
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Occupancy grid and geometry of a world; raises MetricInputError if missing."""
        barrier = load_barrier_map(world_name, run_dir=run_dir)
        meta: dict[str, typing.Any] = {
            "resolution": barrier.resolution,
            "origin": barrier.origin,
            "doors": barrier_door_tl(barrier),
        }
        return barrier.occupancy, meta

    def _compute_full_field(
        self,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        rx_m: float,
        ry_m: float,
        source_dba: float,
        downsample: int = 1,
        pixel_tl: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        if downsample > 1:
            grid = downsample_occupancy(grid, downsample)
            resolution = resolution * downsample

        h, w = grid.shape
        rx_px = (rx_m - ox) / resolution
        ry_px = (ry_m - oy) / resolution

        att_grid = compute_acoustic_field(
            grid,
            resolution,
            rx_px,
            ry_px,
            wall_tl=WALL_TRANSMISSION_LOSS_DB,
            mic_distance=MIC_DISTANCE_M,
            pixel_tl=pixel_tl,
        )
        field_dba = source_dba - att_grid
        return field_dba, resolution, (h, w)

    def render_cell_png(
        self,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        rx_m: float,
        ry_m: float,
        source_dba: float,
        peds: Sequence[Sequence[float]],
        title: str,
        out_path: pathlib.Path,
        downsample: int = 1,
        vmin: float = _FIELD_VMIN_DBA,
        vmax: float = 90.0,
        open_doors: set[str] | None = None,
        doors: dict[str, tuple[np.ndarray, float]] | None = None,
        overlay_trajectories: bool = False,
        trajectory_data: dict[str, typing.Any] | None = None,
    ) -> bool:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eff_res = resolution
        grid_eval = grid
        doors_eval = doors

        if downsample > 1:
            grid_eval = downsample_occupancy(grid, downsample)
            eff_res = resolution * downsample
            if doors:
                doors_eval = {
                    name: (downsample_mask(mask, downsample), tl_db)
                    for name, (mask, tl_db) in doors.items()
                }

        pixel_tl = (
            build_pixel_tl(grid_eval, doors_eval, open_doors=open_doors)
            if doors_eval
            else None
        )
        field_dba, _, (h, w) = self._compute_full_field(
            grid_eval,
            eff_res,
            ox,
            oy,
            rx_m,
            ry_m,
            source_dba,
            downsample=1,
            pixel_tl=pixel_tl,
        )

        door_mask = np.zeros_like(grid_eval, dtype=bool)
        open_door_mask = np.zeros_like(grid_eval, dtype=bool)
        if doors_eval:
            open_set = open_doors or set()
            for name, (mask, _tl) in doors_eval.items():
                door_mask |= mask
                if any(_entity_matches_door(name, e) for e in open_set):
                    open_door_mask |= mask

        wall_or_closed = (grid_eval == 1) & ~open_door_mask
        render_grid = np.flipud(
            np.where(wall_or_closed | np.isinf(field_dba), np.nan, field_dba)
        )
        door_overlay = door_mask if door_mask.any() else None

        plt.figure(figsize=_CELL_FIGSIZE)
        ax = plt.gca()
        ax.set_facecolor("black")
        ax.grid(False)
        left, right = _cell_extent(ox, w, eff_res, resolution)
        bottom, top = _cell_extent(oy, h, eff_res, resolution)
        extent = (left, right, bottom, top)

        im = plt.imshow(
            render_grid,
            cmap="inferno",
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        cy = _cell_centres(bottom, h, eff_res)
        cx = _cell_centres(left, w, eff_res)
        wall_outline = (grid_eval == 1).astype(np.uint8)
        ax.contour(
            cx,
            cy,
            wall_outline,
            levels=[0.5],
            colors=["#ffffff"],
            linewidths=0.3,
            alpha=0.25,
        )
        if door_overlay is not None:
            ax.contour(
                cx,
                cy,
                door_overlay.astype(np.uint8),
                levels=[0.5],
                colors=["#64748b"],
                linewidths=0.9,
                alpha=0.6,
                linestyles="dashed",
            )
            if open_door_mask.any():
                ax.contour(
                    cx,
                    cy,
                    open_door_mask.astype(np.uint8),
                    levels=[0.5],
                    colors=["#10b981"],
                    linewidths=1.4,
                    alpha=0.75,
                )
        plt.colorbar(im, label="dBA", fraction=0.046, pad=0.04)

        if overlay_trajectories and trajectory_data:
            self._draw_trajectories_overlay(ax, trajectory_data)

        plt.plot(
            rx_m,
            ry_m,
            marker="o",
            color="#00f0ff",
            markeredgecolor="#ffffff",
            markeredgewidth=1.2,
            markersize=6.5,
            linestyle="None",
            zorder=10,
            label="Robot (Position)",
        )
        if peds:
            px = [p[0] for p in peds if len(p) >= 2]
            py = [p[1] for p in peds if len(p) >= 2]
            if px:
                plt.plot(
                    px,
                    py,
                    marker="o",
                    color="#ff5722",
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.8,
                    markersize=4.8,
                    linestyle="None",
                    zorder=10,
                    label="Peds (Position)",
                )

        plt.title(title, fontsize=9)
        plt.xlabel("X (m)", fontsize=8)
        plt.ylabel("Y (m)", fontsize=8)
        plt.tick_params(labelsize=7)
        plt.legend(
            fontsize=6.5,
            loc="upper right",
            framealpha=0.75,
            facecolor="#181825",
            edgecolor="#313244",
            labelcolor="#ffffff",
        )
        plt.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=_CELL_DPI, bbox_inches="tight")
        plt.close()
        return True

    def pick_worst_row(self, pdf: pl.DataFrame) -> dict[str, typing.Any] | None:
        """Worst-exposure row with its recorded pose, source level and pedestrian facts."""
        if pdf.is_empty():
            return None
        schema = pdf.schema
        if "ped_max_exposure_dba" not in schema:
            return None
        max_idx = pdf["ped_max_exposure_dba"].arg_max()
        if max_idx is None:
            return None
        row = pdf.row(max_idx, named=True)
        exposure = float(row["ped_max_exposure_dba"])
        ep_id = row.get("episode_id")
        t_ns = row.get("worst_case_acoustic_t_ns")
        rx = row.get("robot_x")
        ry = row.get("robot_y")
        src_dba = row.get("source_dba")
        peds: list[tuple[float, float]] = []
        door_states: dict[str, str] = {}

        if (
            (rx is None or ry is None or src_dba is None)
            and self.run_dir is not None
            and ep_id is not None
        ):
            ep_df = self.load_episode_data(
                pathlib.Path(self.run_dir), f"episode_{int(ep_id):03d}"
            )
            if ep_df is not None and not ep_df.is_empty():
                if t_ns is not None:
                    frame_idx = int((ep_df["t_ns"] - int(t_ns)).abs().arg_min() or 0)
                elif "source_dba" in ep_df.schema:
                    frame_idx = int(ep_df["source_dba"].arg_max() or 0)
                else:
                    frame_idx = 0
                ep_row = ep_df.row(frame_idx, named=True)
                if rx is None:
                    rx = ep_row.get("pos_x_gt")
                if ry is None:
                    ry = ep_row.get("pos_y_gt")
                if src_dba is None:
                    src_dba = ep_row.get("source_dba")
                if t_ns is None:
                    t_ns = ep_row.get("t_ns")
                peds = self.parse_pedestrian_positions(ep_row.get("peds_positions"))
                if t_ns is not None:
                    timeline = self.load_door_timeline(
                        pathlib.Path(self.run_dir), int(ep_id)
                    )
                    if timeline is not None:
                        door_states = {
                            name: "open" for name in timeline.open_doors_at(int(t_ns))
                        }

        if rx is None or ry is None:
            return None
        if src_dba is None or not np.isfinite(float(src_dba)):
            episode = (
                f"episode_{int(ep_id):03d}" if ep_id is not None else "the plot frame"
            )
            raise MetricInputError(
                f"{self.PLOT_TYPE}: {episode} has no recorded "
                f"{ROBOT_ACOUSTICS_TOPIC} total_level_af_dba; refusing to invent a source level."
            )

        return {
            "robot_x": float(rx),
            "robot_y": float(ry),
            "source_dba": float(src_dba),
            "pedestrians": peds,
            "door_states": door_states,
            "ped_max_exposure_dba": exposure,
            "planner": str(row.get("planner") or row.get("local_planner") or ""),
            "stage": str(row.get("stage") or ""),
            "episode_id": ep_id,
        }

    def _prepared_df(self, df: pl.DataFrame) -> pl.DataFrame:
        include_ref = bool(self.spec.options.get("include_reference", False))
        if not include_ref and "is_reference" in df.schema:
            return df.filter(
                pl.col("is_reference").is_null() | (~pl.col("is_reference"))
            )
        return df

    def _facet_axes(
        self, work_df: pl.DataFrame
    ) -> tuple[str | None, str | None, list[str], list[str]]:
        row_col = self.resolve_channel("facet_row")
        col_col = self.resolve_channel("facet_col")
        row_values = (
            list(dict.fromkeys(label_values(work_df, row_col).tolist()))
            if row_col is not None
            else [""]
        )
        col_values = (
            list(dict.fromkeys(label_values(work_df, col_col).tolist()))
            if col_col is not None
            else [""]
        )
        return row_col, col_col, row_values, col_values

    def _filter_facet(
        self,
        work_df: pl.DataFrame,
        row_col: str | None,
        col_col: str | None,
        rv: str,
        cv: str,
    ) -> pl.DataFrame:
        pdf = work_df
        if row_col is not None:
            pdf = pdf.filter(pl.col(row_col).cast(pl.Utf8) == rv)
        if col_col is not None:
            pdf = pdf.filter(pl.col(col_col).cast(pl.Utf8) == cv)
        return pdf

    def render_plotly(self, df: pl.DataFrame) -> str | list[str]:
        """Return an HTML string embedding the acoustic-field image(s)."""
        self.validate_frame(df)
        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return ""

        world_name = work_df["world"][0] if "world" in work_df.schema else None
        if not world_name:
            return ""

        try:
            grid, meta = self.load_grid_and_meta(str(world_name), run_dir=self.run_dir)
        except MetricInputError as exc:
            self.add_notice(str(exc))
            return ""

        resolution = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        downsample = int(self.spec.options.get("downsample", 1))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))
        mode = str(self.spec.options.get("mode", "grid")).lower()
        doors = meta["doors"]
        plots_dir = (
            pathlib.Path(self.run_dir) / "plots"
            if self.run_dir
            else pathlib.Path("plots")
        )

        if mode == "single":
            worst = self.pick_worst_row(work_df)
            if worst is None:
                return ""
            row_label = str(worst["planner"])
            col_label = str(worst["stage"])
            safe = self.spec.id.replace("/", "_").replace(" ", "_")
            png_path = plots_dir / f"{safe}.png"
            src_dba = float(worst["source_dba"])
            vmax = float(np.ceil(src_dba / 10.0) * 10.0)
            if vmax <= vmin:
                vmax = vmin + 20.0
            door_states = worst.get("door_states") or {}
            open_doors = (
                {n for n, st in door_states.items() if st == "open"} if doors else None
            )
            overlay_trajectories = self._overlay_trajectories()
            traj_data = (
                self._cell_trajectories(worst, work_df)
                if overlay_trajectories
                else None
            )
            ok = self.render_cell_png(
                grid,
                resolution,
                ox,
                oy,
                worst["robot_x"],
                worst["robot_y"],
                src_dba,
                worst["pedestrians"],
                title=f"{row_label} / {col_label}  |  {worst['ped_max_exposure_dba']:.0f} dBA",
                out_path=png_path,
                downsample=downsample,
                vmin=vmin,
                vmax=vmax,
                open_doors=open_doors,
                doors=doors,
                overlay_trajectories=overlay_trajectories,
                trajectory_data=traj_data,
            )
            if not ok:
                return ""
            return (
                f'<div style="text-align:center;">'
                f'<img src="plots/{png_path.name}" style="max-width:100%;border-radius:4px;" alt="{self.spec.title}">'
                f'<br><span style="font-size:0.78em;color:#475569;">'
                f"{row_label} / {col_label} &mdash; {worst['ped_max_exposure_dba']:.0f} dBA</span></div>"
            )

        row_col, col_col, row_values, col_values = self._facet_axes(work_df)
        entries: list[tuple[dict[str, typing.Any], str, str]] = []
        for rv in row_values:
            for cv in col_values:
                pdf = self._filter_facet(work_df, row_col, col_col, rv, cv)
                if len(pdf) == 0:
                    continue
                worst = self.pick_worst_row(pdf)
                if worst is None:
                    continue
                entries.append((worst, str(rv), str(cv)))

        if not entries:
            return ""

        global_max = max((float(w["source_dba"]) for w, _, _ in entries), default=80.0)
        global_max = float(np.ceil(global_max / 10.0) * 10.0)
        if global_max <= vmin:
            global_max = vmin + 20.0

        overlay_trajectories = self._overlay_trajectories()
        cells = []
        for worst, row_label, col_label in entries:
            safe = self.spec.id
            if row_label:
                safe += f"_{row_label}"
            if col_label:
                safe += f"_{col_label}"
            safe = safe.replace("/", "_").replace(" ", "_")
            png_path = plots_dir / f"{safe}.png"
            door_states = worst.get("door_states") or {}
            open_doors = (
                {n for n, st in door_states.items() if st == "open"} if doors else None
            )
            traj_data = (
                self._cell_trajectories(worst, work_df)
                if overlay_trajectories
                else None
            )
            ok = self.render_cell_png(
                grid,
                resolution,
                ox,
                oy,
                worst["robot_x"],
                worst["robot_y"],
                float(worst["source_dba"]),
                worst["pedestrians"],
                title=f"{row_label} / {col_label}  |  {worst['ped_max_exposure_dba']:.0f} dBA",
                out_path=png_path,
                downsample=downsample,
                vmin=vmin,
                vmax=global_max,
                open_doors=open_doors,
                doors=doors,
                overlay_trajectories=overlay_trajectories,
                trajectory_data=traj_data,
            )
            if ok:
                cells.append(
                    {
                        "row_val": row_label,
                        "col_label": col_label,
                        "img_rel_path": f"plots/{png_path.name}",
                        "exposure": worst["ped_max_exposure_dba"],
                    }
                )

        if not cells:
            return ""

        ncols = max(len(col_values), 1)
        html = f'<div style="display:grid;grid-template-columns:repeat({ncols},1fr);gap:12px;margin-top:8px;">'
        for c in cells:
            html += (
                f'<div style="text-align:center;font-size:0.78em;color:#475569;">'
                f'<img src="{c["img_rel_path"]}" style="width:100%;border-radius:4px;" alt="{c["row_val"]}/{c["col_label"]}">'
                f"<br>{c['row_val']} / {c['col_label']} - {c['exposure']:.0f} dBA</div>"
            )
        html += "</div>"
        return html

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Dispatch the static frame to the seaborn renderer (D-06)."""
        self.render_seaborn(df, out_path)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render the worst-case acoustic field snapshot as a static image."""
        self.validate_frame(df)
        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return

        world_name = work_df["world"][0] if "world" in work_df.schema else None
        if not world_name:
            raise MetricInputError(
                f"{self.PLOT_TYPE}: the plot frame declares no world."
            )

        grid, meta = self.load_grid_and_meta(str(world_name), run_dir=self.run_dir)
        resolution = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        downsample = int(self.spec.options.get("downsample", 1))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))
        doors = meta["doors"]

        worst = self.pick_worst_row(work_df)
        if worst is None:
            return

        src_dba = float(worst["source_dba"])
        vmax = float(np.ceil(src_dba / 10.0) * 10.0)
        if vmax <= vmin:
            vmax = vmin + 20.0

        door_states = worst.get("door_states") or {}
        open_doors = (
            {n for n, st in door_states.items() if st == "open"} if doors else None
        )
        overlay_trajectories = self._overlay_trajectories()
        traj_data = (
            self._cell_trajectories(worst, work_df) if overlay_trajectories else None
        )

        self.render_cell_png(
            grid,
            resolution,
            ox,
            oy,
            worst["robot_x"],
            worst["robot_y"],
            src_dba,
            worst["pedestrians"],
            title=f"{self.spec.title} (Robot: {src_dba:.0f} dBA)",
            out_path=out_path,
            downsample=downsample,
            vmin=vmin,
            vmax=vmax,
            open_doors=open_doors,
            doors=doors,
            overlay_trajectories=overlay_trajectories,
            trajectory_data=traj_data,
        )

    @staticmethod
    def parse_pedestrian_positions(positions: object) -> list[tuple[float, float]]:
        """Parse one frame's pedestrian positions as (x, y) pairs; other shapes yield none."""
        if not isinstance(positions, (list, tuple, np.ndarray)):
            return []
        pts: list[tuple[float, float]] = []
        for item in positions:
            if not isinstance(item, (list, tuple, np.ndarray)) or len(item) < 2:
                continue
            x = float(item[0])
            y = float(item[1])
            if not (np.isnan(x) or np.isnan(y)):
                pts.append((x, y))
        return pts

    def _overlay_trajectories(self) -> bool:
        return bool(self.spec.options.get("overlay_trajectories"))

    def _cell_trajectories(
        self, worst: dict[str, typing.Any], work_df: pl.DataFrame
    ) -> dict[str, typing.Any]:
        ep_val = worst.get("episode_id")
        if ep_val is not None and self.run_dir:
            ep_df = self.load_episode_data(
                pathlib.Path(self.run_dir), f"episode_{int(ep_val):03d}"
            )
            if ep_df is not None:
                return self.trajectory_data(
                    ep_df, run_dir=self.run_dir, episode_id=ep_val
                )
        return self.trajectory_data(work_df, run_dir=self.run_dir, episode_id=ep_val)

    @staticmethod
    def trajectory_data(
        df: pl.DataFrame | None = None,
        run_dir: pathlib.Path | str | None = None,
        episode_id: str | int | None = None,
    ) -> dict[str, typing.Any]:
        traj_data: dict[str, typing.Any] = {
            "robot_path": None,
            "robot_start": None,
            "robot_goal": None,
            "collision_pos": None,
            "ped_trajectories": {},
            "ped_waypoints": {},
        }
        schema = df.schema if df is not None else {}
        rx_col = (
            "pos_x_gt"
            if "pos_x_gt" in schema
            else ("pos_x" if "pos_x" in schema else None)
        )
        ry_col = (
            "pos_y_gt"
            if "pos_y_gt" in schema
            else ("pos_y" if "pos_y" in schema else None)
        )

        if rx_col and ry_col and df is not None:
            rx = df[rx_col].to_numpy()
            ry = df[ry_col].to_numpy()
            valid = ~np.isnan(rx) & ~np.isnan(ry)
            if np.any(valid):
                rx_clean = rx[valid]
                ry_clean = ry[valid]
                traj_data["robot_path"] = (rx_clean, ry_clean)
                start_val = (
                    df["start"][0]
                    if "start" in schema
                    else (df["start_pos"][0] if "start_pos" in schema else None)
                )
                if (
                    isinstance(start_val, (list, tuple, np.ndarray))
                    and len(start_val) >= 2
                ):
                    traj_data["robot_start"] = (
                        float(start_val[0]),
                        float(start_val[1]),
                    )
                elif len(rx_clean) > 0:
                    traj_data["robot_start"] = (
                        float(rx_clean[0]),
                        float(ry_clean[0]),
                    )
                goal_val = (
                    df["goal"][0]
                    if "goal" in schema
                    else (df["goal_pos"][0] if "goal_pos" in schema else None)
                )
                if (
                    isinstance(goal_val, (list, tuple, np.ndarray))
                    and len(goal_val) >= 2
                ):
                    traj_data["robot_goal"] = (
                        float(goal_val[0]),
                        float(goal_val[1]),
                    )
                elif len(rx_clean) > 0:
                    traj_data["robot_goal"] = (
                        float(rx_clean[-1]),
                        float(ry_clean[-1]),
                    )

        if "has_collision" in schema and rx_col and ry_col and df is not None:
            coll_df = df.filter(pl.col("has_collision"))
            if len(coll_df) > 0:
                cx = coll_df[rx_col][0]
                cy = coll_df[ry_col][0]
                if (
                    cx is not None
                    and cy is not None
                    and not np.isnan(cx)
                    and not np.isnan(cy)
                ):
                    traj_data["collision_pos"] = (float(cx), float(cy))

        ped_trajs: dict[int, list[tuple[float, float]]] = {}
        if df is not None and len(df) > 0:
            if "pedestrian_path" in schema:
                ped_trajs = AcousticFieldRenderer._ped_tracks_from_paths(
                    df["pedestrian_path"][0]
                )
            if not ped_trajs:
                peds_col = (
                    "peds_positions"
                    if "peds_positions" in schema
                    else ("peds" if "peds" in schema else None)
                )
                if peds_col:
                    ped_trajs = AcousticFieldRenderer._ped_tracks_from_positions(
                        list(df[peds_col])
                    )
        if not ped_trajs and run_dir is not None and episode_id is not None:
            ped_trajs = AcousticFieldRenderer._ped_tracks_from_catalog(
                pathlib.Path(run_dir), episode_id
            )

        traj_data["ped_trajectories"] = ped_trajs
        ped_wps = {}
        for pid, pts in ped_trajs.items():
            if len(pts) >= 1:
                wp1 = pts[0]
                wp2 = pts[-1]
                if len(pts) > 2:
                    dists = [np.hypot(p[0] - wp1[0], p[1] - wp1[1]) for p in pts]
                    max_i = int(np.argmax(dists))
                    if dists[max_i] > np.hypot(wp2[0] - wp1[0], wp2[1] - wp1[1]):
                        wp2 = pts[max_i]
                ped_wps[pid] = (wp1, wp2)
        traj_data["ped_waypoints"] = ped_wps
        return traj_data

    @staticmethod
    def _draw_trajectories_overlay(ax: Axes, traj_data: dict[str, typing.Any]) -> None:
        """Draw trajectory tracks under the start, goal, waypoint, and collision markers."""
        if not traj_data:
            return

        robot_path = traj_data.get("robot_path")
        if robot_path is not None and len(robot_path[0]) > 0:
            rx, ry = robot_path
            ax.plot(
                rx,
                ry,
                color="#00f0ff",
                linestyle="-",
                linewidth=1.3,
                alpha=0.75,
                zorder=3,
                label="Robot Track",
            )

        ped_trajs = traj_data.get("ped_trajectories") or {}
        first_ped_labeled = False
        for pts in ped_trajs.values():
            if not pts or len(pts) < 2:
                continue
            px = [p[0] for p in pts]
            py = [p[1] for p in pts]
            lbl = "Pedestrians" if not first_ped_labeled else None
            ax.plot(
                px,
                py,
                color="#c084fc",
                linestyle=":",
                linewidth=1.2,
                alpha=0.6,
                zorder=3,
                label=lbl,
            )
            first_ped_labeled = True

        ped_wps = traj_data.get("ped_waypoints") or {}
        first_wp_labeled = False
        for wp1, wp2 in ped_wps.values():
            if wp1 is not None and len(wp1) >= 2:
                lbl = "Ped Waypoints" if not first_wp_labeled else None
                ax.scatter(
                    wp1[0],
                    wp1[1],
                    marker="d",
                    color="#c084fc",
                    edgecolor="#ffffff",
                    linewidth=0.5,
                    s=14,
                    zorder=4,
                    label=lbl,
                )
                first_wp_labeled = True
            if wp2 is not None and len(wp2) >= 2:
                ax.scatter(
                    wp2[0],
                    wp2[1],
                    marker="d",
                    color="#c084fc",
                    edgecolor="#ffffff",
                    linewidth=0.5,
                    s=14,
                    zorder=4,
                )

        start = traj_data.get("robot_start")
        if start is not None and len(start) >= 2:
            ax.scatter(
                start[0],
                start[1],
                marker="o",
                color="#00f0ff",
                edgecolor="#ffffff",
                linewidth=0.8,
                s=24,
                zorder=5,
                label="Start",
            )

        goal = traj_data.get("robot_goal")
        if goal is not None and len(goal) >= 2:
            ax.scatter(
                goal[0],
                goal[1],
                marker="*",
                color="#fbbf24",
                edgecolor="#ffffff",
                linewidth=0.6,
                s=55,
                zorder=5,
                label="Goal",
            )

        col = traj_data.get("collision_pos")
        if col is not None and len(col) >= 2:
            ax.scatter(
                col[0],
                col[1],
                marker="X",
                color="#ef4444",
                edgecolor="#ffffff",
                linewidth=0.5,
                s=45,
                zorder=6,
                label="Collision",
            )

    @staticmethod
    def _ped_tracks_from_paths(
        p_paths: object,
    ) -> dict[int, list[tuple[float, float]]]:
        tracks: dict[int, list[tuple[float, float]]] = {}
        if not isinstance(p_paths, (list, tuple)):
            return tracks
        for k, p_arr in enumerate(p_paths):
            if p_arr is None or len(p_arr) == 0:
                continue
            arr = np.array(p_arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                pts = [
                    (float(pt[0]), float(pt[1]))
                    for pt in arr
                    if not np.isnan(pt[0]) and not np.isnan(pt[1])
                ]
                if pts:
                    tracks[k] = pts
        return tracks

    @staticmethod
    def _ped_tracks_from_positions(
        raw_pos: list[typing.Any],
    ) -> dict[int, list[tuple[float, float]]]:
        frames: list[list[list[float]]] = []
        for frame_raw in raw_pos:
            if frame_raw is None or len(frame_raw) == 0:
                frames.append([])
            elif isinstance(frame_raw[0], (list, tuple, np.ndarray)):
                frames.append(
                    [[float(p[0]), float(p[1])] for p in frame_raw if len(p) >= 2]
                )
            elif isinstance(frame_raw[0], dict):
                frames.append(
                    [
                        [float(p["x"]), float(p["y"])]
                        for p in frame_raw
                        if "x" in p and "y" in p
                    ]
                )
            else:
                stride = 3 if len(frame_raw) % 3 == 0 and len(frame_raw) >= 3 else 2
                frames.append(
                    [
                        [float(frame_raw[j]), float(frame_raw[j + 1])]
                        for j in range(0, len(frame_raw) - 1, stride)
                    ]
                )
        tracks: dict[int, list[tuple[float, float]]] = {}
        for p_idx in range(max((len(f) for f in frames), default=0)):
            pts = [
                (f[p_idx][0], f[p_idx][1])
                for f in frames
                if len(f) > p_idx
                and not np.isnan(f[p_idx][0])
                and not np.isnan(f[p_idx][1])
            ]
            if len(pts) > 1:
                tracks[p_idx] = pts
        return tracks

    @staticmethod
    def _ped_tracks_from_catalog(
        benchmark_dir: pathlib.Path, episode_id: str | int
    ) -> dict[int, list[tuple[float, float]]]:
        catalog = BenchmarkCatalog.open(benchmark_dir)
        names = set(catalog.table_names())
        if "pedestrian_arena_peds" not in names:
            return {}
        ep_num = episode_number(episode_id)
        p_df = (
            catalog.get_table("pedestrian_arena_peds")
            .filter(pl.col("episode_id") == ep_num)
            .select(["t_ns", "ped_id", "x", "y"])
            .sort("t_ns")
            .collect()
        )
        if p_df.is_empty():
            return {}
        parts = p_df.partition_by("ped_id", maintain_order=True)
        tracks: dict[int, list[tuple[float, float]]] = {}
        for idx, part in enumerate(parts):
            # agent-contracts: ignore AC004 - Matplotlib rendering boundary
            xs = part["x"].to_list()
            # agent-contracts: ignore AC004 - Matplotlib rendering boundary
            ys = part["y"].to_list()
            tracks[idx] = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
        return tracks

    @staticmethod
    def load_door_timeline(
        benchmark_dir: pathlib.Path, episode_id: str | int
    ) -> DoorStateTimeline | None:
        """Door-state timeline of one episode, None when its world semantics were not recorded."""
        catalog = BenchmarkCatalog.open(benchmark_dir)
        if "world_semantics" not in catalog.table_names():
            return None
        ep_num = episode_number(episode_id)
        sem_df = (
            catalog.get_table("world_semantics")
            .filter(pl.col("episode_id") == ep_num)
            .collect()
        )
        if sem_df.is_empty():
            return None
        return DoorStateTimeline.from_semantic_frame(sem_df)

    @staticmethod
    def load_episode_data(
        benchmark_dir: pathlib.Path, episode_id: str | int
    ) -> pl.DataFrame | None:
        """Episode telemetry aligned to ground-truth pose by the processing alignment module."""
        return load_aligned_episode(benchmark_dir, episode_id)

    def compute_field_timeseries(
        self,
        df: pl.DataFrame,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        doors: dict[str, tuple[np.ndarray, float]],
        state_timeline: DoorStateTimeline | None = None,
        downsample: int = 1,
        stride: int = 1,
        max_frames: int = 120,
    ) -> list[tuple[np.ndarray, float, tuple[int, int], frozenset[str], float] | None]:
        """Compute the 2D acoustic field for a sequence of episode frames."""
        if downsample > 1:
            grid = downsample_occupancy(grid, downsample)
            resolution = resolution * downsample

        h, w = grid.shape
        doors_ds = (
            {
                name: (downsample_mask(mask, downsample), tl_db)
                for name, (mask, tl_db) in doors.items()
            }
            if doors
            else {}
        )

        total_rows = len(df)
        indices = list(range(0, total_rows, stride))
        if len(indices) > max_frames:
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]

        tl_cache: dict[tuple[str, ...], np.ndarray] = {}
        results: list[
            tuple[np.ndarray, float, tuple[int, int], frozenset[str], float] | None
        ] = []
        # agent-contracts: ignore AC004 - Matplotlib rendering boundary
        rows = df.rows(named=True)
        missing_source = False

        for frame_idx in indices:
            row = rows[frame_idx]
            rx_m = row.get("pos_x_gt")
            ry_m = row.get("pos_y_gt")
            source_dba = row.get("source_dba")
            t_ns = int(row.get("t_ns", 0))

            if rx_m is None or ry_m is None or np.isnan(rx_m) or np.isnan(ry_m):
                results.append(None)
                continue
            if source_dba is None or np.isnan(source_dba):
                missing_source = True
                results.append(None)
                continue

            rx_px = (float(rx_m) - ox) / resolution
            ry_px = (float(ry_m) - oy) / resolution
            open_set = (
                state_timeline.open_doors_at(t_ns)
                if state_timeline is not None
                else frozenset()
            )
            tl_key = tuple(sorted(open_set))
            pixel_tl = tl_cache.get(tl_key)
            if pixel_tl is None:
                pixel_tl = build_pixel_tl(grid, doors_ds, open_doors=set(open_set))
                tl_cache[tl_key] = pixel_tl

            att_grid = compute_acoustic_field(
                occupancy_grid=grid,
                resolution=resolution,
                start_x_px=rx_px,
                start_y_px=ry_px,
                wall_tl=WALL_TRANSMISSION_LOSS_DB,
                mic_distance=MIC_DISTANCE_M,
                pixel_tl=pixel_tl,
            )
            field_dba = float(source_dba) - att_grid
            results.append((field_dba, resolution, (h, w), open_set, float(source_dba)))

        if missing_source:
            self.add_notice(
                "acoustic_field: frames without a recorded "
                f"{ROBOT_ACOUSTICS_TOPIC} source level were skipped"
            )
        return results

    def render_animation(
        self,
        df: pl.DataFrame,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        doors: dict[str, tuple[np.ndarray, float]],
        state_timeline: DoorStateTimeline | None = None,
        out_path: pathlib.Path | None = None,
        downsample: int = 1,
        stride: int = 1,
        max_frames: int = 120,
        fps: int = 10,
        dpi: int = 150,
        vmin: float | None = None,
        vmax: float | None = None,
        robot_trail: int = 0,
        show_doors: bool = True,
        format: str = "gif",
        overlay_trajectories: bool = False,
        first_frame_path: pathlib.Path | None = None,
    ) -> pathlib.Path | None:
        """Render an animated GIF/MP4/PNG sequence, plus the first frame when asked."""
        valid = self._animation_fields(
            df,
            grid,
            resolution,
            ox,
            oy,
            doors,
            state_timeline=state_timeline,
            downsample=downsample,
            stride=stride,
            max_frames=max_frames,
        )
        if not valid:
            return None
        vmin, vmax = self._animation_color_scale(valid, vmin, vmax)
        layers = self._animation_layers(
            df,
            valid,
            grid,
            resolution,
            ox,
            oy,
            doors,
            downsample,
            vmin,
            vmax,
            show_doors,
            overlay_trajectories,
            robot_trail,
        )
        frames = self._animation_metadata(df, valid, stride, max_frames)
        anim = self._build_animation(layers, frames, valid, fps, robot_trail)
        return self._write_animation(
            anim,
            layers,
            frames,
            valid,
            out_path,
            format,
            fps,
            dpi,
            first_frame_path,
        )

    def _animation_fields(
        self,
        df: pl.DataFrame,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        doors: dict[str, tuple[np.ndarray, float]],
        *,
        state_timeline: DoorStateTimeline | None,
        downsample: int,
        stride: int,
        max_frames: int,
    ) -> list[
        tuple[int, tuple[np.ndarray, float, tuple[int, int], frozenset[str], float]]
    ]:
        """Compute each sampled frame's field, dropping frames with no recorded source level."""
        fields = self.compute_field_timeseries(
            df,
            grid,
            resolution,
            ox,
            oy,
            doors,
            state_timeline=state_timeline,
            downsample=downsample,
            stride=stride,
            max_frames=max_frames,
        )
        return [
            (index, field) for index, field in enumerate(fields) if field is not None
        ]

    def _animation_color_scale(
        self,
        valid: list[
            tuple[int, tuple[np.ndarray, float, tuple[int, int], frozenset[str], float]]
        ],
        vmin: float | None,
        vmax: float | None,
    ) -> tuple[float, float]:
        """Resolve the documented colour scale, the measured level range when unset."""
        if vmin is None:
            vmin = _FIELD_VMIN_DBA
        if vmax is None:
            field_maxima = np.sort(
                np.array([np.nanmax(field[0]) for _, field in valid], dtype=float)
            )
            p99_idx = int(0.99 * (len(field_maxima) - 1))
            all_max = float(field_maxima[p99_idx]) if len(field_maxima) > 0 else vmin
            vmax = float(np.ceil(all_max / 10.0) * 10.0)
            if vmax <= vmin:
                vmax = vmin + 20.0
        return float(vmin), float(vmax)

    def _animation_layers(
        self,
        df: pl.DataFrame,
        valid: list[
            tuple[int, tuple[np.ndarray, float, tuple[int, int], frozenset[str], float]]
        ],
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        doors: dict[str, tuple[np.ndarray, float]],
        downsample: int,
        vmin: float,
        vmax: float,
        show_doors: bool,
        overlay_trajectories: bool,
        robot_trail: int,
    ) -> _AnimationLayers:
        """Build the static figure: field layers, contours, trajectories and empty artists."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, eff_res, (h, w), _, _ = valid[0][1]
        grid_ds = downsample_occupancy(grid, downsample) if downsample > 1 else grid
        doors_ds: dict[str, tuple[np.ndarray, float]] = {}
        door_mask_all = np.zeros((h, w), dtype=bool)
        if doors:
            for name, (mask, tl_db) in doors.items():
                mask_ds = downsample_mask(mask, downsample)
                doors_ds[name] = (mask_ds, tl_db)
                door_mask_all |= mask_ds

        fig, ax = plt.subplots(figsize=_CELL_FIGSIZE)
        ax.set_facecolor("black")
        ax.grid(False)
        left, right = _cell_extent(ox, w, eff_res, resolution)
        bottom, top = _cell_extent(oy, h, eff_res, resolution)
        extent = (left, right, bottom, top)

        render_grids: list[np.ndarray] = []
        open_door_masks: list[np.ndarray] = []
        for _, field_data in valid:
            field_dba, _, _, open_set, _ = field_data
            open_dm = np.zeros((h, w), dtype=bool)
            if doors_ds:
                pixel_tl = build_pixel_tl(grid_ds, doors_ds, open_doors=set(open_set))
                for _name, (mask_ds, _tl) in doors_ds.items():
                    open_dm |= mask_ds & (pixel_tl == 0.0)
            wall_or_closed = (grid_ds == 1) & ~open_dm
            render_grids.append(
                np.flipud(
                    np.where(wall_or_closed | np.isinf(field_dba), np.nan, field_dba)
                )
            )
            open_door_masks.append(open_dm)

        im = ax.imshow(
            render_grids[0],
            cmap="inferno",
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, label="dBA", fraction=0.046, pad=0.04)

        cy = _cell_centres(bottom, h, eff_res)
        cx = _cell_centres(left, w, eff_res)
        wall_outline = (grid_ds == 1).astype(np.uint8)
        ax.contour(
            cx,
            cy,
            wall_outline,
            levels=[0.5],
            colors=["#ffffff"],
            linewidths=0.3,
            alpha=0.25,
        )

        open_door_contour = None
        if show_doors and door_mask_all.any():
            ax.contour(
                cx,
                cy,
                door_mask_all.astype(np.uint8),
                levels=[0.5],
                colors=["#64748b"],
                linewidths=0.9,
                alpha=0.6,
                linestyles="dashed",
            )
            if open_door_masks[0].any():
                open_door_contour = ax.contour(
                    cx,
                    cy,
                    open_door_masks[0].astype(np.uint8),
                    levels=[0.5],
                    colors=["#10b981"],
                    linewidths=1.4,
                    alpha=0.75,
                )

        if overlay_trajectories:
            self._draw_trajectories_overlay(
                ax, self.trajectory_data(df, run_dir=self.run_dir)
            )

        (robot_dot,) = ax.plot(
            [],
            [],
            marker="o",
            color="#00f0ff",
            markeredgecolor="#ffffff",
            markeredgewidth=1.2,
            markersize=6.5,
            linestyle="None",
            zorder=10,
            label="Robot (Position)",
        )
        (ped_dots,) = ax.plot(
            [],
            [],
            marker="o",
            color="#ff5722",
            markeredgecolor="#ffffff",
            markeredgewidth=0.8,
            markersize=4.8,
            linestyle="None",
            zorder=10,
            label="Peds (Position)",
        )
        ax.legend(
            fontsize=6.5,
            loc="upper right",
            framealpha=0.75,
            facecolor="#181825",
            edgecolor="#313244",
            labelcolor="#ffffff",
        )

        trail_line = None
        if robot_trail > 0:
            (trail_line,) = ax.plot([], [], "g-", linewidth=0.8, alpha=0.4)

        title_text = ax.set_title("", fontsize=9)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)

        return _AnimationLayers(
            fig=fig,
            ax=ax,
            im=im,
            extent=extent,
            cx=cx,
            cy=cy,
            wall_outline=wall_outline,
            grid_ds=grid_ds,
            door_mask_all=door_mask_all,
            open_door_masks=open_door_masks,
            render_grids=render_grids,
            show_doors=show_doors,
            h=h,
            w=w,
            vmin=vmin,
            vmax=vmax,
            open_door_contour=open_door_contour,
            robot_dot=robot_dot,
            ped_dots=ped_dots,
            trail_line=trail_line,
            title_text=title_text,
        )

    def _animation_metadata(
        self,
        df: pl.DataFrame,
        valid: list[
            tuple[int, tuple[np.ndarray, float, tuple[int, int], frozenset[str], float]]
        ],
        stride: int,
        max_frames: int,
    ) -> _AnimationFrames:
        """Collect the source row, pedestrian positions and collision flag of each frame."""
        # agent-contracts: ignore AC004 - Matplotlib rendering boundary
        rows = df.rows(named=True)
        total_rows = len(df)
        indices = list(range(0, total_rows, stride))
        if len(indices) > max_frames:
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]

        peds_per_frame: list[list[tuple[float, float]]] = []
        collision_per_frame: list[bool] = []
        for orig_idx, _ in valid:
            row = rows[min(indices[orig_idx], total_rows - 1)]
            peds_per_frame.append(
                self.parse_pedestrian_positions(row.get("peds_positions"))
            )
            collision_per_frame.append(row.get("has_collision") is True)
        return _AnimationFrames(
            rows=rows,
            indices=indices,
            total_rows=total_rows,
            peds_per_frame=peds_per_frame,
            collision_per_frame=collision_per_frame,
        )

    def _build_animation(
        self,
        layers: _AnimationLayers,
        frames: _AnimationFrames,
        valid: list[
            tuple[int, tuple[np.ndarray, float, tuple[int, int], frozenset[str], float]]
        ],
        fps: int,
        robot_trail: int,
    ) -> typing.Any:
        """Wire the per-frame artist updates into a blitting ``FuncAnimation``."""
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        collision_flash = plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=layers.ax.transAxes,
            facecolor="red",
            alpha=0.0,
            zorder=10,
        )
        layers.ax.add_patch(collision_flash)
        prev_collision = [False]
        open_door_contour = [layers.open_door_contour]
        robot_xs: list[float] = []
        robot_ys: list[float] = []

        def update(
            enum_tuple: tuple[
                int,
                tuple[
                    int,
                    tuple[np.ndarray, float, tuple[int, int], frozenset[str], float],
                ],
            ],
        ) -> list[Artist]:
            rg_idx, (orig_idx, field_data) = enum_tuple
            _field_dba, _, _, open_set, source_dba = field_data
            layers.im.set_data(layers.render_grids[rg_idx])

            row = frames.rows[min(frames.indices[orig_idx], frames.total_rows - 1)]
            rx_m = float(row.get("pos_x_gt", 0) or 0)
            ry_m = float(row.get("pos_y_gt", 0) or 0)
            layers.robot_dot.set_data([rx_m], [ry_m])
            robot_xs.append(rx_m)
            robot_ys.append(ry_m)

            if layers.trail_line is not None:
                layers.trail_line.set_data(
                    robot_xs[-robot_trail:], robot_ys[-robot_trail:]
                )

            peds = frames.peds_per_frame[rg_idx]
            if peds:
                layers.ped_dots.set_data([p[0] for p in peds], [p[1] for p in peds])
            else:
                layers.ped_dots.set_data([], [])

            is_collision = frames.collision_per_frame[rg_idx]
            if is_collision != prev_collision[0]:
                collision_flash.set_alpha(0.25 if is_collision else 0.0)
                prev_collision[0] = is_collision

            if layers.show_doors and layers.open_door_masks[rg_idx].any():
                if open_door_contour[0] is not None:
                    open_door_contour[0].remove()
                open_door_contour[0] = layers.ax.contour(
                    layers.cx,
                    layers.cy,
                    layers.open_door_masks[rg_idx].astype(np.uint8),
                    levels=[0.5],
                    colors=["#00ff00"],
                    linewidths=2.0,
                    alpha=0.9,
                )

            collision_label = " COLLISION!" if is_collision else ""
            layers.title_text.set_text(
                f"t={row.get('t_ns', 0) / 1e9:.1f}s  |  source={source_dba:.0f} dBA  |  "
                f"{len(open_set)} doors open{collision_label}"
            )
            artists: list[Artist] = [
                layers.im,
                layers.robot_dot,
                layers.ped_dots,
                layers.title_text,
            ]
            if layers.trail_line is not None:
                artists.append(layers.trail_line)
            return artists

        return animation.FuncAnimation(
            layers.fig,
            update,
            frames=list(enumerate(valid)),
            interval=max(1, 1000 // fps),
            blit=True,
        )

    def _write_animation(
        self,
        anim: typing.Any,
        layers: _AnimationLayers,
        frames: _AnimationFrames,
        valid: list[
            tuple[int, tuple[np.ndarray, float, tuple[int, int], frozenset[str], float]]
        ],
        out_path: pathlib.Path | None,
        output_format: str,
        fps: int,
        dpi: int,
        first_frame_path: pathlib.Path | None,
    ) -> pathlib.Path:
        """Write the requested animation artefact and, when asked, the first frame PNG."""
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        if out_path is None:
            out_path = pathlib.Path("plots") / "acoustic_animation.gif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if first_frame_path is not None:
            self._save_animation_frame(
                layers,
                frames,
                valid[0][0],
                0,
                valid[0][1],
                first_frame_path,
                True,
                dpi,
            )

        if output_format == "frames":
            frames_dir = out_path.with_suffix("") if out_path.suffix else out_path
            frames_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, (orig_idx, field_data) in enumerate(valid):
                self._save_animation_frame(
                    layers,
                    frames,
                    orig_idx,
                    frame_index,
                    field_data,
                    frames_dir / f"frame_{frame_index:04d}.png",
                    True,
                    dpi,
                )
            plt.close(layers.fig)
            return frames_dir

        if output_format == "mp4":
            if not animation.FFMpegWriter.isAvailable():
                raise MetricInputError(
                    "acoustic_field_animation: mp4 output needs ffmpeg on PATH; "
                    "install ffmpeg or request --format gif."
                )
            anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
        else:
            anim.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
        plt.close(layers.fig)
        return out_path

    def _save_animation_frame(
        self,
        layers: _AnimationLayers,
        frames: _AnimationFrames,
        orig_idx: int,
        frame_index: int,
        field_data: tuple[np.ndarray, float, tuple[int, int], frozenset[str], float],
        path: pathlib.Path,
        show_doors: bool,
        dpi: int,
    ) -> None:
        """Draw one standalone field frame with its pose, pedestrians and collision flag."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        field_dba, _, _, open_set, source_dba = field_data
        open_dm = (
            layers.open_door_masks[frame_index]
            if show_doors
            else np.zeros((layers.h, layers.w), dtype=bool)
        )
        wall_or_closed = (layers.grid_ds == 1) & ~open_dm
        rg = np.flipud(
            np.where(wall_or_closed | np.isinf(field_dba), np.nan, field_dba)
        )

        fig_frame, ax_frame = plt.subplots(figsize=_CELL_FIGSIZE)
        ax_frame.set_facecolor("black")
        ax_frame.grid(False)
        im_frame = ax_frame.imshow(
            rg,
            cmap="inferno",
            origin="upper",
            extent=layers.extent,
            vmin=layers.vmin,
            vmax=layers.vmax,
        )
        plt.colorbar(im_frame, ax=ax_frame, label="dBA", fraction=0.046, pad=0.04)
        ax_frame.contour(
            layers.cx,
            layers.cy,
            layers.wall_outline,
            levels=[0.5],
            colors=["#ffffff"],
            linewidths=0.3,
            alpha=0.25,
        )
        row = frames.rows[min(frames.indices[orig_idx], frames.total_rows - 1)]
        ax_frame.plot(
            float(row.get("pos_x_gt", 0) or 0),
            float(row.get("pos_y_gt", 0) or 0),
            marker="o",
            color="#00e5ff",
            markeredgecolor="#ffffff",
            markeredgewidth=0.7,
            markersize=5.5,
            linestyle="None",
            label="Robot",
        )
        peds = frames.peds_per_frame[frame_index]
        if peds:
            ax_frame.plot(
                [p[0] for p in peds],
                [p[1] for p in peds],
                marker="o",
                color="#ff5722",
                markeredgecolor="#ffffff",
                markeredgewidth=0.8,
                markersize=4.8,
                linestyle="None",
                zorder=10,
                label="Peds",
            )
        if show_doors and layers.door_mask_all.any():
            ax_frame.contour(
                layers.cx,
                layers.cy,
                layers.door_mask_all.astype(np.uint8),
                levels=[0.5],
                colors=["#64748b"],
                linewidths=0.9,
                alpha=0.6,
                linestyles="dashed",
            )
            if open_dm.any():
                ax_frame.contour(
                    layers.cx,
                    layers.cy,
                    open_dm.astype(np.uint8),
                    levels=[0.5],
                    colors=["#10b981"],
                    linewidths=1.4,
                    alpha=0.75,
                )
        collision_label = (
            "  COLLISION!" if frames.collision_per_frame[frame_index] else ""
        )
        ax_frame.set_title(
            f"t={float(row.get('t_ns', 0)) / 1e9:.1f}s | src={source_dba:.0f} dBA | "
            f"{len(open_set)} doors open{collision_label}",
            fontsize=8,
        )
        ax_frame.set_xlabel("X (m)", fontsize=7)
        ax_frame.set_ylabel("Y (m)", fontsize=7)
        ax_frame.tick_params(labelsize=6)
        fig_frame.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_frame.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_frame)


class AcousticFieldAnimationRenderer(AcousticFieldRenderer):
    """Manifest-driven acoustic field animation (GIF/MP4 during report)."""

    PLOT_TYPE = "acoustic_field_animation"

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Dispatch the static artifact to the seaborn renderer (D-06)."""
        self.render_seaborn(df, out_path)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Write the first animation frame to the static path and the animation beside it."""
        self.validate_frame(df)
        work_df = self._prepared_df(df)
        if work_df.is_empty():
            return
        world_name = work_df["world"][0] if "world" in work_df.schema else None
        if not world_name:
            raise MetricInputError(
                "acoustic_field_animation: the plot frame declares no world."
            )

        grid, meta = self.load_grid_and_meta(str(world_name), run_dir=self.run_dir)
        resolution = float(meta["resolution"])
        ox = float(meta["origin"][0])
        oy = float(meta["origin"][1])
        downsample = int(self.spec.options.get("downsample", 2))
        fps = int(self.spec.options.get("fps", 10))
        max_frames = int(self.spec.options.get("max_frames", 120))
        stride = int(self.spec.options.get("stride", 1))
        dpi = int(self.spec.options.get("dpi", 150))
        output_format = str(self.spec.options.get("format", "gif"))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))
        doors = meta["doors"]
        per_episode = bool(self.spec.options.get("per_episode", False))
        static_image = out_path.suffix.lower() in {".png", ".svg"}
        artefact_path = self._animation_artefact_path(out_path, output_format)
        episode_ids = self._animation_episode_ids(work_df)
        if not episode_ids:
            raise MetricInputError(
                "acoustic_field_animation: the plot frame has no episode_id."
            )

        static_written = False
        for episode_id in episode_ids:
            episode_dir_name = f"episode_{int(episode_id):03d}"
            benchmark_dir = self._episode_benchmark_dir(episode_dir_name)
            episode_df = self.load_episode_data(benchmark_dir, episode_dir_name)
            if episode_df is None:
                continue
            state_timeline = self.load_door_timeline(benchmark_dir, int(episode_id))
            target = (
                out_path.with_name(
                    f"{out_path.stem}_{episode_dir_name}.{output_format}"
                )
                if per_episode
                else artefact_path
            )
            result = self.render_animation(
                episode_df,
                grid,
                resolution,
                ox,
                oy,
                doors,
                state_timeline=state_timeline,
                out_path=target,
                downsample=downsample,
                stride=stride,
                max_frames=max_frames,
                fps=fps,
                dpi=dpi,
                vmin=vmin,
                vmax=None,
                show_doors=bool(doors),
                format=output_format,
                overlay_trajectories=self._overlay_trajectories(),
                first_frame_path=(
                    out_path if static_image and not static_written else None
                ),
            )
            if result is not None and static_image:
                static_written = True

        if static_image and not static_written:
            raise MetricInputError(
                "acoustic_field_animation: no recorded acoustic field frame for "
                f"{world_name}; the static report image cannot be rendered."
            )

    @staticmethod
    def _animation_artefact_path(
        out_path: pathlib.Path, output_format: str
    ) -> pathlib.Path:
        """The animation artefact path a static output path implies."""
        if output_format == "frames":
            return out_path.with_suffix("") if out_path.suffix else out_path
        return out_path.with_suffix(f".{output_format}")

    def _animation_episode_ids(self, work_df: pl.DataFrame) -> list[int]:
        """Episodes to animate: every row when per_episode, else the worst-exposure episode."""
        if "episode_id" not in work_df.schema:
            raise MetricInputError(
                "acoustic_field_animation: the plot frame must carry an episode_id column."
            )
        if bool(self.spec.options.get("per_episode", False)):
            return sorted(int(value) for value in work_df["episode_id"].unique())
        if "ped_max_exposure_dba" in work_df.schema:
            measured = work_df.filter(pl.col("ped_max_exposure_dba").is_not_null())
            if not measured.is_empty():
                return [
                    int(
                        measured.sort("ped_max_exposure_dba", descending=True).row(
                            0, named=True
                        )["episode_id"]
                    )
                ]
        self.add_notice(
            "acoustic_field_animation: no pedestrian exposure row; animating the first episode."
        )
        return [int(work_df["episode_id"][0])]

    def _episode_benchmark_dir(self, episode_dir_name: str) -> pathlib.Path:
        """The benchmark root holding an episode directory, searched up from the run dir."""
        benchmark_dir = (
            pathlib.Path(self.run_dir) if self.run_dir else pathlib.Path(".")
        )
        if (benchmark_dir / "episodes" / episode_dir_name).is_dir():
            return benchmark_dir
        for parent in benchmark_dir.parents:
            if (parent / "episodes" / episode_dir_name).is_dir():
                return parent
        return benchmark_dir

    def render_plotly(self, df: pl.DataFrame) -> str | list[str]:
        """Return HTML snippet(s) embedding the animation(s) in the report."""
        self.validate_frame(df)
        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return ""

        output_format = str(self.spec.options.get("format", "gif"))
        if output_format not in ("gif", "mp4"):
            return ""

        def _media(rel: str) -> str:
            if output_format == "mp4":
                return f'<video src="{rel}" controls loop style="max-width:100%;border-radius:4px;"></video>'
            return f'<img src="{rel}" style="max-width:100%;border-radius:4px;" alt="{self.spec.title}">'

        per_episode = bool(self.spec.options.get("per_episode", False))
        if per_episode and "episode_id" in work_df.schema:
            chunks = []
            # agent-contracts: ignore AC004 - Matplotlib rendering boundary
            ep_ids = sorted(int(v) for v in work_df["episode_id"].unique().to_list())
            for ep_id in ep_ids:
                rel = f"plots/{self.spec.id}_episode_{ep_id:03d}.{output_format}"
                caption = f"{self.spec.title} - episode_{ep_id:03d}"
                chunks.append(
                    f'<div style="text-align:center;">{_media(rel)}<br>'
                    f'<span style="font-size:0.78em;color:#475569;">{caption}</span></div>'
                )
            return chunks

        rel = f"plots/{self.spec.id}.{output_format}"
        return (
            f'<div style="text-align:center;">{_media(rel)}<br>'
            f'<span style="font-size:0.78em;color:#475569;">{self.spec.title}</span></div>'
        )
