"""Acoustic field spatial and animated plot renderers."""

from __future__ import annotations

import json
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
from arena_evaluation.processing.acoustics.door_map import (
    _entity_matches_door,
    barrier_door_tl,
    build_pixel_tl,
    load_barrier_map,
)
from arena_evaluation.processing.acoustics.door_state import DoorStateTimeline
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
_DEFAULT_RESOLUTION_M = 0.05
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


def _parse_episode_int(episode_id: str | int) -> int:
    if isinstance(episode_id, int):
        return episode_id
    text = str(episode_id)
    return int(text.removeprefix("episode_"))


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
                "fmt",
                "format",
                "per_episode",
            }
        )
    )

    @staticmethod
    def _load_grid_and_meta(
        world_name: str, run_dir: pathlib.Path | None = None
    ) -> tuple[np.ndarray, dict[str, typing.Any]]:
        """Occupancy grid and geometry of a world; raises MetricInputError if missing."""
        barrier = load_barrier_map(
            world_name, run_dir=run_dir, resolution=_DEFAULT_RESOLUTION_M
        )
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
            wall_tl=47.0,
            mic_distance=1.0,
            pixel_tl=pixel_tl,
        )
        field_dba = np.clip(source_dba - att_grid, 0, None)
        return field_dba, resolution, (h, w)

    def _render_cell_png(
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

    def _pick_worst_row(self, pdf: pl.DataFrame) -> dict[str, typing.Any] | None:
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
            (rx is None or ry is None)
            and self.run_dir is not None
            and ep_id is not None
        ):
            ep_df = self._load_episode_data(
                pathlib.Path(self.run_dir), f"episode_{int(ep_id):03d}"
            )
            if ep_df is not None and not ep_df.is_empty():
                if t_ns is not None:
                    nearest_idx = int((ep_df["t_ns"] - int(t_ns)).abs().arg_min() or 0)
                    ep_row = ep_df.row(nearest_idx, named=True)
                else:
                    src_series = ep_df["source_dba"]
                    best_idx = int(src_series.arg_max() or 0)
                    ep_row = ep_df.row(best_idx, named=True)
                    t_ns = ep_row.get("t_ns")
                rx = ep_row.get("pos_x_gt")
                ry = ep_row.get("pos_y_gt")
                if src_dba is None:
                    src_dba = ep_row.get("source_dba")
                peds = self._parse_pedestrian_positions(
                    ep_row.get("peds_positions", [])
                )
                if t_ns is not None:
                    timeline = self._load_door_timeline(
                        pathlib.Path(self.run_dir), int(ep_id)
                    )
                    if timeline is not None:
                        door_states = {
                            name: "open" for name in timeline.open_doors_at(int(t_ns))
                        }

        if rx is None or ry is None:
            return None

        return {
            "robot_x": float(rx),
            "robot_y": float(ry),
            "source_dba": float(src_dba) if src_dba is not None else 60.0,
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
            grid, meta = self._load_grid_and_meta(str(world_name), run_dir=self.run_dir)
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
            worst = self._pick_worst_row(work_df)
            if worst is None:
                return ""
            row_label = str(worst["planner"])
            col_label = str(worst["stage"])
            safe = self.spec.id.replace("/", "_").replace(" ", "_")
            png_path = plots_dir / f"{safe}.png"
            src_dba = float(worst["source_dba"])
            if float(worst["ped_max_exposure_dba"]) >= 100.0 and src_dba < 100.0:
                src_dba = 100.0
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
            ok = self._render_cell_png(
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
                worst = self._pick_worst_row(pdf)
                if worst is None:
                    continue
                entries.append((worst, str(rv), str(cv)))

        if not entries:
            return ""

        def _eff_src(w: dict[str, typing.Any]) -> float:
            s = float(w["source_dba"])
            return (
                100.0 if float(w["ped_max_exposure_dba"]) >= 100.0 and s < 100.0 else s
            )

        global_max = max((_eff_src(w) for w, _, _ in entries), default=80.0)
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
            ok = self._render_cell_png(
                grid,
                resolution,
                ox,
                oy,
                worst["robot_x"],
                worst["robot_y"],
                _eff_src(worst),
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
        """Render the worst-case acoustic field snapshot as a static image."""
        self.validate_frame(df)
        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return

        world_name = work_df["world"][0] if "world" in work_df.schema else None
        if not world_name:
            return

        try:
            grid, meta = self._load_grid_and_meta(str(world_name), run_dir=self.run_dir)
        except MetricInputError as exc:
            self.add_notice(str(exc))
            return

        resolution = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        downsample = int(self.spec.options.get("downsample", 1))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))
        doors = meta["doors"]

        worst = self._pick_worst_row(work_df)
        if worst is None:
            return

        src_dba = float(worst["source_dba"])
        if float(worst["ped_max_exposure_dba"]) >= 100.0 and src_dba < 100.0:
            src_dba = 100.0
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

        self._render_cell_png(
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
    def _parse_pedestrian_positions(row: object) -> list[tuple[float, float]]:
        """Parse one frame's pedestrian positions (flat or nested schema)."""
        pts: list[tuple[float, float]] = []
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except json.JSONDecodeError:
                return pts
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) == 0:
            return pts
        if isinstance(row[0], dict):
            for item in row:
                if "x" in item and "y" in item:
                    x, y = item["x"], item["y"]
                    if (
                        x is not None
                        and y is not None
                        and not np.isnan(x)
                        and not np.isnan(y)
                    ):
                        pts.append((float(x), float(y)))
        elif isinstance(row[0], (list, tuple, np.ndarray)):
            for item in row:
                if len(item) >= 2 and not np.isnan(item[0]) and not np.isnan(item[1]):
                    pts.append((float(item[0]), float(item[1])))
        else:
            for j in range(0, len(row), 3):
                if (
                    j + 1 < len(row)
                    and not np.isnan(row[j])
                    and not np.isnan(row[j + 1])
                ):
                    pts.append((float(row[j]), float(row[j + 1])))
        return pts

    def _overlay_trajectories(self) -> bool:
        return bool(self.spec.options.get("overlay_trajectories"))

    def _cell_trajectories(
        self, worst: dict[str, typing.Any], work_df: pl.DataFrame
    ) -> dict[str, typing.Any]:
        ep_val = worst.get("episode_id")
        if ep_val is not None and self.run_dir:
            ep_df = self._load_episode_data(
                pathlib.Path(self.run_dir), f"episode_{int(ep_val):03d}"
            )
            if ep_df is not None:
                return self._extract_trajectory_data(
                    ep_df, run_dir=self.run_dir, episode_id=ep_val
                )
        return self._extract_trajectory_data(
            work_df, run_dir=self.run_dir, episode_id=ep_val
        )

    @staticmethod
    def _extract_trajectory_data(
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
        ep_num = _parse_episode_int(episode_id)
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
    def _load_door_timeline(
        benchmark_dir: pathlib.Path, episode_id: str | int
    ) -> DoorStateTimeline | None:
        catalog = BenchmarkCatalog.open(benchmark_dir)
        if "world_semantics" not in catalog.table_names():
            return None
        ep_num = _parse_episode_int(episode_id)
        sem_df = (
            catalog.get_table("world_semantics")
            .filter(pl.col("episode_id") == ep_num)
            .collect()
        )
        if sem_df.is_empty():
            return None
        return DoorStateTimeline.from_semantic_frame(sem_df)

    @staticmethod
    def _load_episode_data(
        benchmark_dir: pathlib.Path, episode_id: str | int
    ) -> pl.DataFrame | None:
        """Load the time-aligned episode DataFrame from BenchmarkCatalog telemetry tables."""
        catalog = BenchmarkCatalog.open(benchmark_dir)
        names = set(catalog.table_names())
        ep_num = _parse_episode_int(episode_id)

        base_lf: pl.LazyFrame | None = None
        if "robot_tracks" in names:
            rt = catalog.get_table("robot_tracks").filter(
                pl.col("episode_id") == ep_num
            )
            if not rt.limit(1).collect().is_empty():
                base_lf = rt.select(
                    [
                        "t_ns",
                        pl.col("x").alias("pos_x_gt"),
                        pl.col("y").alias("pos_y_gt"),
                    ]
                )
        if base_lf is None and "robot_odom" in names:
            ro = catalog.get_table("robot_odom").filter(pl.col("episode_id") == ep_num)
            if not ro.limit(1).collect().is_empty():
                base_lf = ro.select(
                    [
                        "t_ns",
                        pl.col("odom_x").alias("pos_x_gt"),
                        pl.col("odom_y").alias("pos_y_gt"),
                    ]
                )
        if base_lf is None:
            return None

        base_lf = base_lf.sort("t_ns")

        if "robot_acoustics" in names:
            ra = (
                catalog.get_table("robot_acoustics")
                .filter(pl.col("episode_id") == ep_num)
                .select(
                    [
                        "t_ns",
                        pl.col("total_level_af_dba").alias("source_dba"),
                        "operating_state",
                    ]
                )
                .sort("t_ns")
            )
            if not ra.limit(1).collect().is_empty():
                base_lf = base_lf.join_asof(ra, on="t_ns", strategy="nearest")

        if "pedestrian_arena_peds" in names:
            pt_df = (
                catalog.get_table("pedestrian_arena_peds")
                .filter(pl.col("episode_id") == ep_num)
                .select(["t_ns", "x", "y"])
                .sort("t_ns")
                .collect()
            )
            if not pt_df.is_empty():
                ped_rows: list[dict[str, typing.Any]] = []
                for part in pt_df.partition_by("t_ns", maintain_order=True):
                    t_val = int(part["t_ns"][0])
                    pts = [
                        (float(x), float(y))
                        for x, y in zip(
                            part["x"].to_numpy(),
                            part["y"].to_numpy(),
                            strict=True,
                        )
                    ]
                    ped_rows.append({"t_ns": t_val, "peds_positions": pts})
                peds_grouped = pl.DataFrame(ped_rows).lazy().sort("t_ns")
                base_lf = base_lf.join_asof(peds_grouped, on="t_ns", strategy="nearest")

        if "robot_collision_events" in names:
            ce = (
                catalog.get_table("robot_collision_events")
                .filter(pl.col("episode_id") == ep_num)
                .select(["t_ns", (pl.col("event_count") > 0).alias("has_collision")])
                .sort("t_ns")
            )
            if not ce.limit(1).collect().is_empty():
                base_lf = base_lf.join_asof(ce, on="t_ns", strategy="nearest")

        return base_lf.collect()

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

        for frame_idx in indices:
            row = rows[frame_idx]
            rx_m = row.get("pos_x_gt")
            ry_m = row.get("pos_y_gt")
            source_dba = (
                row.get("total_level_af_dba")
                or row.get("source_dba")
                or _FIELD_VMIN_DBA
            )
            t_ns = int(row.get("t_ns", 0))

            if rx_m is None or ry_m is None or np.isnan(rx_m) or np.isnan(ry_m):
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

            if source_dba is None or np.isnan(source_dba):
                source_dba = _FIELD_VMIN_DBA

            att_grid = compute_acoustic_field(
                occupancy_grid=grid,
                resolution=resolution,
                start_x_px=rx_px,
                start_y_px=ry_px,
                wall_tl=47.0,
                mic_distance=1.0,
                pixel_tl=pixel_tl,
            )
            field_dba = np.clip(float(source_dba) - att_grid, 0, None)
            results.append((field_dba, resolution, (h, w), open_set, float(source_dba)))

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
        fmt: str = "gif",
        overlay_trajectories: bool = False,
    ) -> pathlib.Path | None:
        """Render an animated GIF/MP4/PNG-sequence of the acoustic field timeseries."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        if vmin is None:
            vmin = _FIELD_VMIN_DBA

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
        valid = [(i, f) for i, f in enumerate(fields) if f is not None]
        if not valid:
            return None

        if vmax is None:
            field_maxima = np.sort(
                np.array([np.nanmax(f[0]) for _, f in valid], dtype=float)
            )
            p99_idx = int(0.99 * (len(field_maxima) - 1))
            all_max = float(field_maxima[p99_idx]) if len(field_maxima) > 0 else vmin
            vmax = float(np.ceil(all_max / 10.0) * 10.0)
            if vmax <= vmin:
                vmax = vmin + 20.0

        _, eff_res, (h, w), _, _ = valid[0][1]
        grid_ds = downsample_occupancy(grid, downsample) if downsample > 1 else grid
        doors_ds: dict[str, tuple[np.ndarray, float]] = {}
        door_mask_all = np.zeros((h, w), dtype=bool)
        if doors:
            for name, (m, tl_db) in doors.items():
                m_ds = downsample_mask(m, downsample)
                doors_ds[name] = (m_ds, tl_db)
                door_mask_all |= m_ds

        fig, ax = plt.subplots(figsize=_CELL_FIGSIZE)
        ax.set_facecolor("black")
        ax.grid(False)
        left, right = _cell_extent(ox, w, eff_res, resolution)
        bottom, top = _cell_extent(oy, h, eff_res, resolution)
        extent = (left, right, bottom, top)

        render_grids = []
        open_door_masks = []
        for _, field_data in valid:
            field_dba, _, _, open_set, _ = field_data
            open_dm = np.zeros((h, w), dtype=bool)
            if doors_ds:
                pixel_tl = build_pixel_tl(grid_ds, doors_ds, open_doors=set(open_set))
                for _name, (m_ds, _tl) in doors_ds.items():
                    open_dm |= m_ds & (pixel_tl == 0.0)
            wall_or_closed = (grid_ds == 1) & ~open_dm
            rg = np.flipud(
                np.where(wall_or_closed | np.isinf(field_dba), np.nan, field_dba)
            )
            render_grids.append(rg)
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
            traj_data = self._extract_trajectory_data(df, run_dir=self.run_dir)
            self._draw_trajectories_overlay(ax, traj_data)

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

        robot_xs: list[float] = []
        robot_ys: list[float] = []
        # agent-contracts: ignore AC004 - Matplotlib rendering boundary
        rows = df.rows(named=True)
        total_rows = len(df)

        indices = list(range(0, total_rows, stride))
        if len(indices) > max_frames:
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]

        peds_per_frame = []
        collision_per_frame = []
        for orig_idx, _ in valid:
            row_idx = indices[orig_idx]
            row = rows[min(row_idx, total_rows - 1)]
            peds_per_frame.append(
                self._parse_pedestrian_positions(row.get("peds_positions", []))
            )
            src_val = float(
                row.get("total_level_af_dba") or row.get("source_dba") or 0.0
            )
            coll = (src_val >= 99.0) or (row.get("operating_state") == "collision")
            collision_per_frame.append(coll)

        collision_flash = plt.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes, facecolor="red", alpha=0.0, zorder=10
        )
        ax.add_patch(collision_flash)
        prev_collision = [False]

        def update(
            enum_tuple: tuple[
                int,
                tuple[
                    int,
                    tuple[np.ndarray, float, tuple[int, int], frozenset[str], float],
                ],
            ],
        ) -> list[Artist]:
            nonlocal open_door_contour
            rg_idx, (orig_idx, field_data) = enum_tuple
            _field_dba, _, _, open_set, source_dba = field_data
            im.set_data(render_grids[rg_idx])

            row = rows[min(indices[orig_idx], total_rows - 1)]
            rx_m = float(row.get("pos_x_gt", 0) or 0)
            ry_m = float(row.get("pos_y_gt", 0) or 0)
            robot_dot.set_data([rx_m], [ry_m])
            robot_xs.append(rx_m)
            robot_ys.append(ry_m)

            if trail_line is not None:
                trail_line.set_data(robot_xs[-robot_trail:], robot_ys[-robot_trail:])

            peds = peds_per_frame[rg_idx]
            if peds:
                ped_dots.set_data([p[0] for p in peds], [p[1] for p in peds])
            else:
                ped_dots.set_data([], [])

            is_collision = collision_per_frame[rg_idx]
            if is_collision != prev_collision[0]:
                collision_flash.set_alpha(0.25 if is_collision else 0.0)
                prev_collision[0] = is_collision

            if show_doors and open_door_masks[rg_idx].any():
                if open_door_contour is not None:
                    open_door_contour.remove()
                open_door_contour = ax.contour(
                    cx,
                    cy,
                    open_door_masks[rg_idx].astype(np.uint8),
                    levels=[0.5],
                    colors=["#00ff00"],
                    linewidths=2.0,
                    alpha=0.9,
                )

            collision_label = " COLLISION 100dBA!" if is_collision else ""
            title_text.set_text(
                f"t={row.get('t_ns', 0) / 1e9:.1f}s  |  source={source_dba:.0f} dBA  |  {len(open_set)} doors open{collision_label}"
            )
            artists: list[Artist] = [im, robot_dot, ped_dots, title_text]
            if trail_line is not None:
                artists.append(trail_line)
            return artists

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=list(enumerate(valid)),
            interval=max(1, 1000 // fps),
            blit=True,
        )

        if out_path is None:
            out_path = pathlib.Path("plots") / "acoustic_animation.gif"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "frames":
            frames_dir = out_path.with_suffix("") if out_path.suffix else out_path
            frames_dir.mkdir(parents=True, exist_ok=True)
            for fi, (orig_idx, field_data) in enumerate(valid):
                frame_path = frames_dir / f"frame_{fi:04d}.png"
                field_dba, _, _, open_set, source_dba = field_data
                open_dm = (
                    open_door_masks[fi] if show_doors else np.zeros((h, w), dtype=bool)
                )
                wall_or_closed = (grid_ds == 1) & ~open_dm
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
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.colorbar(
                    im_frame, ax=ax_frame, label="dBA", fraction=0.046, pad=0.04
                )
                ax_frame.contour(
                    cx,
                    cy,
                    wall_outline,
                    levels=[0.5],
                    colors=["#ffffff"],
                    linewidths=0.3,
                    alpha=0.25,
                )
                row = rows[min(indices[orig_idx], total_rows - 1)]
                rx_frame = float(row.get("pos_x_gt", 0) or 0)
                ry_frame = float(row.get("pos_y_gt", 0) or 0)
                t_s = float(row.get("t_ns", 0)) / 1e9
                ax_frame.plot(
                    rx_frame,
                    ry_frame,
                    marker="o",
                    color="#00e5ff",
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.7,
                    markersize=5.5,
                    linestyle="None",
                    label="Robot",
                )
                peds = peds_per_frame[fi]
                if peds:
                    px_f = [p[0] for p in peds if len(p) >= 2]
                    py_f = [p[1] for p in peds if len(p) >= 2]
                    if px_f:
                        ax_frame.plot(
                            px_f,
                            py_f,
                            marker="o",
                            color="#ff5722",
                            markeredgecolor="#ffffff",
                            markeredgewidth=0.8,
                            markersize=4.8,
                            linestyle="None",
                            zorder=10,
                            label="Peds",
                        )
                if show_doors and door_mask_all.any():
                    ax_frame.contour(
                        cx,
                        cy,
                        door_mask_all.astype(np.uint8),
                        levels=[0.5],
                        colors=["#64748b"],
                        linewidths=0.9,
                        alpha=0.6,
                        linestyles="dashed",
                    )
                    if open_dm.any():
                        ax_frame.contour(
                            cx,
                            cy,
                            open_dm.astype(np.uint8),
                            levels=[0.5],
                            colors=["#10b981"],
                            linewidths=1.4,
                            alpha=0.75,
                        )
                collision_label = "  COLLISION!" if collision_per_frame[fi] else ""
                ax_frame.set_title(
                    f"t={t_s:.1f}s | src={source_dba:.0f} dBA | {len(open_set)} doors open{collision_label}",
                    fontsize=8,
                )
                ax_frame.set_xlabel("X (m)", fontsize=7)
                ax_frame.set_ylabel("Y (m)", fontsize=7)
                ax_frame.tick_params(labelsize=6)
                fig_frame.tight_layout()
                fig_frame.savefig(frame_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig_frame)
            plt.close(fig)
            return frames_dir

        if fmt == "mp4":
            ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
        else:
            ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
        return out_path


class AcousticFieldAnimationRenderer(AcousticFieldRenderer):
    """Manifest-driven acoustic field animation (GIF/MP4 during report)."""

    PLOT_TYPE = "acoustic_field_animation"

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Generate acoustic-field animations."""
        self.validate_frame(df)
        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return

        world_name = work_df["world"][0] if "world" in work_df.schema else None
        if not world_name:
            return

        try:
            grid, meta = self._load_grid_and_meta(str(world_name), run_dir=self.run_dir)
        except MetricInputError as exc:
            self.add_notice(str(exc))
            return

        resolution = float(meta["resolution"])
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        downsample = int(self.spec.options.get("downsample", 2))
        fps = int(self.spec.options.get("fps", 10))
        max_frames = int(self.spec.options.get("max_frames", 120))
        stride = int(self.spec.options.get("stride", 1))
        fmt = str(self.spec.options.get("format", "gif"))
        vmin = float(self.spec.options.get("vmin", _FIELD_VMIN_DBA))
        doors = meta["doors"]
        benchmark_dir = (
            pathlib.Path(self.run_dir) if self.run_dir else pathlib.Path(".")
        )

        self._rendered_gifs: list[str] = []
        per_episode = bool(self.spec.options.get("per_episode", False))

        if per_episode and "episode_id" in work_df.schema:
            # agent-contracts: ignore AC004 - Matplotlib rendering boundary
            ep_ids = sorted(int(v) for v in work_df["episode_id"].unique().to_list())
            for ep_id in ep_ids:
                gif_path = self._render_episode_gif(
                    ep_id,
                    benchmark_dir,
                    out_path,
                    grid,
                    resolution,
                    ox,
                    oy,
                    doors,
                    downsample,
                    fps,
                    max_frames,
                    stride,
                    fmt,
                    vmin,
                    per_episode=True,
                )
                if gif_path is not None:
                    self._rendered_gifs.append(f"plots/{gif_path.name}")
            return

        episode_id: int | None = None
        if "episode_id" in work_df.schema and len(work_df) > 0:
            if "ped_max_exposure_dba" in work_df.schema and work_df[
                "ped_max_exposure_dba"
            ].null_count() < len(work_df):
                max_idx = work_df["ped_max_exposure_dba"].arg_max()
                if max_idx is not None:
                    episode_id = int(work_df.row(max_idx, named=True)["episode_id"])
            if episode_id is None:
                episode_id = int(work_df["episode_id"][0])
        if episode_id is None:
            return

        gif_path = self._render_episode_gif(
            episode_id,
            benchmark_dir,
            out_path,
            grid,
            resolution,
            ox,
            oy,
            doors,
            downsample,
            fps,
            max_frames,
            stride,
            fmt,
            vmin,
            per_episode=False,
        )
        if gif_path is not None:
            self._rendered_gifs.append(f"plots/{gif_path.name}")

    def _render_episode_gif(
        self,
        episode_id: int,
        benchmark_dir: pathlib.Path,
        out_path: pathlib.Path,
        grid: np.ndarray,
        resolution: float,
        ox: float,
        oy: float,
        doors: dict[str, tuple[np.ndarray, float]],
        downsample: int,
        fps: int,
        max_frames: int,
        stride: int,
        fmt: str,
        vmin: float,
        per_episode: bool = False,
    ) -> pathlib.Path | None:
        episode_dir_name = f"episode_{int(episode_id):03d}"
        bdir = benchmark_dir
        if not (bdir / "episodes" / episode_dir_name).is_dir():
            for parent in bdir.parents:
                if (parent / "episodes" / episode_dir_name).is_dir():
                    bdir = parent
                    break

        episode_df = self._load_episode_data(bdir, episode_dir_name)
        if episode_df is None:
            return None

        state_timeline = self._load_door_timeline(bdir, episode_id)
        if per_episode:
            gif_path = out_path.with_name(f"{out_path.stem}_{episode_dir_name}.{fmt}")
        elif fmt == "frames":
            gif_path = out_path.with_suffix("")
        else:
            gif_path = out_path.with_suffix(f".{fmt}")

        return self.render_animation(
            episode_df,
            grid,
            resolution,
            ox,
            oy,
            doors,
            state_timeline=state_timeline,
            out_path=gif_path,
            downsample=downsample,
            stride=stride,
            max_frames=max_frames,
            fps=fps,
            vmin=vmin,
            vmax=None,
            show_doors=bool(doors),
            fmt=fmt,
            overlay_trajectories=self._overlay_trajectories(),
        )

    def render_plotly(self, df: pl.DataFrame) -> str | list[str]:
        """Return HTML snippet(s) embedding the animation(s) in the report."""
        self.validate_frame(df)
        work_df = self._prepared_df(df)
        if len(work_df) == 0:
            return ""

        fmt = str(self.spec.options.get("format", "gif"))
        if fmt not in ("gif", "mp4"):
            return ""

        def _media(rel: str) -> str:
            if fmt == "mp4":
                return f'<video src="{rel}" controls loop style="max-width:100%;border-radius:4px;"></video>'
            return f'<img src="{rel}" style="max-width:100%;border-radius:4px;" alt="{self.spec.title}">'

        per_episode = bool(self.spec.options.get("per_episode", False))
        if per_episode and "episode_id" in work_df.schema:
            chunks = []
            # agent-contracts: ignore AC004 - Matplotlib rendering boundary
            ep_ids = sorted(int(v) for v in work_df["episode_id"].unique().to_list())
            for ep_id in ep_ids:
                rel = f"plots/{self.spec.id}_episode_{ep_id:03d}.{fmt}"
                caption = f"{self.spec.title} - episode_{ep_id:03d}"
                chunks.append(
                    f'<div style="text-align:center;">{_media(rel)}<br>'
                    f'<span style="font-size:0.78em;color:#475569;">{caption}</span></div>'
                )
            return chunks

        rel = f"plots/{self.spec.id}.{fmt}"
        return (
            f'<div style="text-align:center;">{_media(rel)}<br>'
            f'<span style="font-size:0.78em;color:#475569;">{self.spec.title}</span></div>'
        )
