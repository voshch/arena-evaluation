from __future__ import annotations

import base64
import math
import pathlib
import types
import typing

import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer
from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.processing.metrics.ecological.mechanism_common import DoorGeom, ElevatorGeom, MechanismWorld, load_mechanism_world
from arena_evaluation.processing.pose_segments import teleport_jumps

if typing.TYPE_CHECKING:
    import pandas as pd
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

LEGACY_JUMP_M = 3.0
GIF_FRAMES = 100
GIF_FPS = 20
PED_TAIL_S = 5.0
SUCCESS_RESULTS = frozenset({"SUCCESS", "success", "reached", "goal_reached", "GOAL_REACHED"})
COLLISION_RESULTS = frozenset({"COLLISION", "collision", "CRASH"})
DOOR_COLORS = {"locked": "#d946ef", "closed": "#d3273e", "moving": "#f59e0b"}
CABIN_COLOR = "#06b6d4"


def _sub_paths(path: object) -> list[np.ndarray]:
    """Path cell as a list of (N, >=2) arrays: one per agent for multi-agent cells, else one."""
    if not isinstance(path, (list, tuple, np.ndarray, pl.Series)) or len(path) == 0:
        return []
    first = path[0]
    nested = isinstance(first, (list, tuple, np.ndarray)) and len(first) > 0 and isinstance(first[0], (list, tuple, np.ndarray))
    groups = [p for p in path if p is not None and len(p) > 0] if nested else [path]
    out = []
    for group in groups:
        try:
            arr = np.array([list(p) for p in group], dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 2 and arr.shape[1] >= 2:
            out.append(arr)
    return out


def _row_time_ns(row: typing.Mapping | pd.Series, n: int) -> np.ndarray | None:
    """The row's `time` samples when they line up with its path, else None."""
    time_ns = row.get("time")
    if time_ns is None or len(time_ns) != n:
        return None
    return np.asarray(time_ns, dtype=np.int64)


def _jump_indices(x: np.ndarray, y: np.ndarray, time_ns: np.ndarray | None) -> np.ndarray:
    """Steps that are teleports (resets, elevator rides)."""
    if time_ns is not None:
        return teleport_jumps(x, y, time_ns)
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.where(dists > LEGACY_JUMP_M)[0]


def _break_at_teleports(x: np.ndarray, y: np.ndarray, time_ns: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Copies of x, y with NaN at every sample that lands a teleport, so lines stop there."""
    bx, by = x.astype(float).copy(), y.astype(float).copy()
    jumps = _jump_indices(x, y, time_ns)
    if len(jumps):
        bx[jumps + 1] = np.nan
        by[jumps + 1] = np.nan
    return bx, by


def _first_moving_index(x: np.ndarray, y: np.ndarray, time_ns: np.ndarray | None) -> int:
    """Index where the first segment longer than 0.1 m starts (spawn jumps skipped)."""
    dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    jumps = _jump_indices(x, y, time_ns)
    starts = np.concatenate([[0], jumps + 1]).astype(int)
    ends = np.concatenate([jumps, [len(dists)]]).astype(int)
    for s_idx, e_idx in zip(starts, ends, strict=True):
        if e_idx > s_idx and np.sum(dists[s_idx:e_idx]) > 0.1:
            return int(s_idx)
    return int(starts[-1])


def _is_success(row: typing.Mapping) -> bool:
    return row.get("result") in SUCCESS_RESULTS or bool(row.get("success"))


def _is_collision(row: typing.Mapping | pd.Series) -> bool:
    return row.get("result") in COLLISION_RESULTS or (row.get("collision_amount") or 0) > 0


def _door_leaf(geom: DoorGeom | ElevatorGeom, progress: float) -> tuple[list[float], list[float]]:
    if isinstance(geom, DoorGeom):
        (ax, ay), (bx, by) = geom.leaf(progress)
    else:
        (ax, ay), (ex, ey) = geom.door_start, geom.door_end
        bx, by = ax + (ex - ax) * (1.0 - progress), ay + (ey - ay) * (1.0 - progress)
    return [ax, bx], [ay, by]


def _door_color(progress: float, locked: bool) -> str:
    if locked:
        return DOOR_COLORS["locked"]
    return DOOR_COLORS["closed"] if progress < 0.02 else DOOR_COLORS["moving"]


def _state_geometry(world: MechanismWorld, state: typing.Mapping) -> DoorGeom | ElevatorGeom | None:
    entity = f"{state['entity']}/{state['level']}" if state.get("level") else state["entity"]
    return world.door(entity) if state.get("kind") == "door" else world.elevator(entity)


class TrajectoryRenderer(BasePlotRenderer):
    """Robot paths over the world map."""

    PLOT_TYPE = "trajectory"
    generate_gifs: bool = False
    _worlds: typing.ClassVar[dict[str, MechanismWorld | None]] = {}

    def _load_map_image(self, map_name: str, run_dir: pathlib.Path | None = None) -> dict | None:
        return MapRegistry.get_map(map_name, run_dir=run_dir)

    def _mechanism_world(self, map_name: str | None) -> MechanismWorld | None:
        if not map_name or not self.spec.options.get("overlay_mechanisms", True):
            return None
        if map_name not in self._worlds:
            try:
                self._worlds[map_name] = load_mechanism_world(map_name)
            except Exception as e:
                print(f"No mechanism geometry for {map_name}: {e}")
                self._worlds[map_name] = None
        return self._worlds[map_name]

    @staticmethod
    def _extract_start_goal(r: dict) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        s_raw = r.get("start") or r.get("start_pos")
        g_raw = r.get("goal") or r.get("goal_pos")

        s_pt = None
        g_pt = None

        if s_raw is not None and isinstance(s_raw, (list, tuple, np.ndarray)) and len(s_raw) >= 2:
            try:
                s_pt = (float(s_raw[0]), float(s_raw[1]))
            except (ValueError, TypeError):
                pass

        if g_raw is not None and isinstance(g_raw, (list, tuple, np.ndarray)) and len(g_raw) >= 2:
            try:
                g_pt = (float(g_raw[0]), float(g_raw[1]))
            except (ValueError, TypeError):
                pass

        path_val = r.get("path")
        if (s_pt is None or g_pt is None) and path_val is not None and len(path_val) > 0:
            flat_pts = []
            try:
                if isinstance(path_val[0], (list, tuple, np.ndarray)) and len(path_val[0]) > 0 and isinstance(path_val[0][0], (list, tuple, np.ndarray)):
                    for sub in path_val:
                        for pt in sub:
                            if len(pt) >= 2:
                                flat_pts.append((float(pt[0]), float(pt[1])))
                else:
                    for pt in path_val:
                        if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                            flat_pts.append((float(pt[0]), float(pt[1])))
            except Exception:
                pass

            if flat_pts:
                if s_pt is None:
                    s_pt = flat_pts[0]
                if g_pt is None:
                    g_pt = flat_pts[-1]

        return s_pt, g_pt

    def _render_single_plot(self, pdf: pd.DataFrame, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None = None) -> str:
        fig = go.Figure()
        diff_col = self.spec.differentiate or "planner"

        map_meta = None
        if self.spec.options.get("show_map", True) and map_name:
            map_meta = self._load_map_image(map_name, run_dir=run_dir)

        # Separate evaluated dynamic episodes from reference baseline episodes
        if "is_reference" in pdf.columns:
            pdf_eval = pdf[~pdf["is_reference"].fillna(False)]
            pdf_ref = pdf[pdf["is_reference"].fillna(False)]
        else:
            pdf_eval = pdf
            pdf_ref = pdf.iloc[0:0]

        planners = pdf_eval[diff_col].unique() if diff_col in pdf_eval.columns else ["unknown"]  # pyright: ignore[reportAttributeAccessIssue]
        seen_planners = set()

        for planner in planners:
            if diff_col in pdf_eval.columns:
                planner_df = pdf_eval[pdf_eval[diff_col] == planner]
            else:
                planner_df = pdf_eval

            if "episode" in planner_df.columns:  # pyright: ignore[reportAttributeAccessIssue]
                planner_df = planner_df.sort_values("episode")  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]

            overlay_markers = self.spec.options.get("overlay_markers", True)

            all_x_by_agent = []
            all_y_by_agent = []

            starts_x_by_agent = []
            starts_y_by_agent = []
            starts_idx_by_agent = []
            goals_x_by_agent = []
            goals_y_by_agent = []
            goals_idx_by_agent = []
            col_x_by_agent = []
            col_y_by_agent = []
            col_idx_by_agent = []

            data_key = self.spec.data_key or "path"
            for _, row in planner_df.iterrows():  # pyright: ignore[reportAttributeAccessIssue]
                path = row.get(data_key)
                if path is None or len(path) == 0:
                    continue

                paths_to_process = _sub_paths(path)

                for k, path_arr in enumerate(paths_to_process):
                    time_ns = _row_time_ns(row, len(path_arr))

                    while len(all_x_by_agent) <= k:
                        all_x_by_agent.append([])
                        all_y_by_agent.append([])
                        if overlay_markers:
                            starts_x_by_agent.append([])
                            starts_y_by_agent.append([])
                            starts_idx_by_agent.append([])
                            goals_x_by_agent.append([])
                            goals_y_by_agent.append([])
                            goals_idx_by_agent.append([])
                            col_x_by_agent.append([])
                            col_y_by_agent.append([])
                            col_idx_by_agent.append([])

                    if overlay_markers:
                        current_len = len(all_x_by_agent[k])
                        valid_mask = ~np.isnan(path_arr[:, 0])
                        if np.any(valid_mask):
                            clean_indices = np.where(valid_mask)[0]
                            clean_t = time_ns[clean_indices] if time_ns is not None else None
                            first_idx = int(clean_indices[_first_moving_index(path_arr[clean_indices, 0], path_arr[clean_indices, 1], clean_t)])
                            last_idx = int(clean_indices[-1])

                            starts_x_by_agent[k].append(path_arr[first_idx, 0])
                            starts_y_by_agent[k].append(path_arr[first_idx, 1])
                            starts_idx_by_agent[k].append(current_len + first_idx)

                            if _is_collision(row):
                                col_x_by_agent[k].append(path_arr[last_idx, 0])
                                col_y_by_agent[k].append(path_arr[last_idx, 1])
                                col_idx_by_agent[k].append(current_len + last_idx)
                            else:
                                goals_x_by_agent[k].append(path_arr[last_idx, 0])
                                goals_y_by_agent[k].append(path_arr[last_idx, 1])
                                goals_idx_by_agent[k].append(current_len + last_idx)

                    if len(all_x_by_agent[k]) > 0:
                        all_x_by_agent[k].append(None)
                        all_y_by_agent[k].append(None)
                    bx, by = _break_at_teleports(path_arr[:, 0], path_arr[:, 1], time_ns)
                    all_x_by_agent[k].extend([None if np.isnan(v) else float(v) for v in bx])
                    all_y_by_agent[k].extend([None if np.isnan(v) else float(v) for v in by])

            palette = get_color_palette()
            planner_idx = list(planners).index(planner) if planner in planners else 0
            planner_color = palette[planner_idx % len(palette)]

            for k in range(len(all_x_by_agent)):
                if not all_x_by_agent[k]:
                    continue
                if len(all_x_by_agent) > 1:
                    trace_name = f"{planner} - Agent {k}"
                    legendgroup = planner
                    showlegend = planner not in seen_planners
                    seen_planners.add(planner)
                    opacity = 0.5
                else:
                    trace_name = planner
                    legendgroup = planner
                    showlegend = planner not in seen_planners
                    seen_planners.add(planner)
                    opacity = 0.8

                customdata = []
                idx_counter = 0
                for val in all_x_by_agent[k]:
                    if val is None:
                        customdata.append(idx_counter)
                        idx_counter += 1
                    else:
                        customdata.append(idx_counter)
                        idx_counter += 1

                fig.add_trace(
                    go.Scatter(
                        x=all_x_by_agent[k], y=all_y_by_agent[k], mode='lines', name=trace_name, legendgroup=legendgroup, showlegend=showlegend, line=dict(color=planner_color), opacity=opacity, hovertemplate=f"<b>{planner}</b><br>{trace_name}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>", customdata=customdata
                    )
                )

                if overlay_markers:
                    if starts_x_by_agent[k]:
                        fig.add_trace(
                            go.Scatter(
                                x=starts_x_by_agent[k],
                                y=starts_y_by_agent[k],
                                mode='markers',
                                marker=dict(symbol='circle', size=8, color='#10b981', line=dict(width=1, color='#065f46')),
                                name=f"{planner} Starts",
                                legendgroup=planner,
                                showlegend=False,
                                opacity=0.9,
                                hovertemplate=f"<b>{planner}</b><br>Start<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                                customdata=starts_idx_by_agent[k],
                            )
                        )
                    if goals_x_by_agent[k]:
                        fig.add_trace(
                            go.Scatter(
                                x=goals_x_by_agent[k],
                                y=goals_y_by_agent[k],
                                mode='markers',
                                marker=dict(symbol='star', size=12, color='#ffc845'),
                                name=f"{planner} Goals",
                                legendgroup=planner,
                                showlegend=False,
                                opacity=0.9,
                                hovertemplate=f"<b>{planner}</b><br>Goal<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                                customdata=goals_idx_by_agent[k],
                            )
                        )
                    if col_x_by_agent[k]:
                        fig.add_trace(
                            go.Scatter(
                                x=col_x_by_agent[k],
                                y=col_y_by_agent[k],
                                mode='markers',
                                marker=dict(symbol='x', size=10, color='#d3273e', line=dict(width=2)),
                                name=f"{planner} Collisions",
                                legendgroup=planner,
                                showlegend=False,
                                opacity=1.0,
                                hovertemplate=f"<b>{planner}</b><br>Collision<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                                customdata=col_idx_by_agent[k],
                            )
                        )

        # 1. Overlay Theta* Synthetic Demonstration Baseline Path
        if self.spec.options.get("overlay_theta_star", False) or self.spec.options.get("show_theta_star", False) or (self.spec.data_key and "theta" in self.spec.data_key):
            try:
                from arena_evaluation.processing.path.theta_star import compute_theta_star_for_episode

                theta_rendered = False
                seen_theta_endpoints = set()
                # Iterate through evaluated dynamic episodes to render matching Theta* baseline for each stage
                for _, r in pdf_eval.iterrows():
                    m_name = map_name or r.get("map")
                    s_pt, g_pt = self._extract_start_goal(r.to_dict())

                    if m_name and s_pt is not None and g_pt is not None:
                        ep_key = (str(m_name), round(s_pt[0], 2), round(s_pt[1], 2), round(g_pt[0], 2), round(g_pt[1], 2))
                        if ep_key in seen_theta_endpoints:
                            continue
                        seen_theta_endpoints.add(ep_key)

                        try:
                            res = compute_theta_star_for_episode(str(m_name), s_pt, g_pt, run_dir=str(run_dir) if run_dir else None)
                            if res and res.success and len(res.path_x) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=res.path_x.tolist(),
                                        y=res.path_y.tolist(),
                                        mode='lines+markers',
                                        name='Theta* Optimal Demonstration',
                                        legendgroup='theta_star',
                                        line=dict(color='#d97706', width=3, dash='dash'),
                                        marker=dict(size=7, symbol='diamond', color='#d97706'),
                                        opacity=0.95,
                                        hovertemplate="<b>Theta* Optimal Path</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                                        showlegend=not theta_rendered,
                                    )
                                )
                                theta_rendered = True
                        except Exception:
                            pass
            except Exception:
                pass

        # 2. Overlay Pedestrian Paths (Human Interaction Flows)
        if self.spec.options.get("overlay_pedestrians", False) or self.spec.options.get("show_peds", False) or (self.spec.data_key and "ped" in self.spec.data_key):
            peds_rendered = False
            for _, r in pdf_eval.iterrows():
                p_paths = r.get("pedestrian_path")
                # Fallback: extract directly from episode topics/peds.parquet if available
                if p_paths is None and run_dir is not None and "episode" in r:
                    try:
                        ep_id = int(r["episode"])  # pyright: ignore[reportArgumentType]
                        peds_pq = pathlib.Path(run_dir) / "episodes" / f"episode_{ep_id:03d}" / "topics" / "peds.parquet"
                        if peds_pq.exists():
                            p_df = pl.read_parquet(peds_pq)
                            if "peds_positions" in p_df.columns:
                                raw_pos = p_df["peds_positions"].to_list()
                                parsed_frames = []
                                for frame_raw in raw_pos:
                                    if frame_raw is not None and len(frame_raw) > 0:
                                        if isinstance(frame_raw[0], (list, tuple, np.ndarray)):
                                            parsed_frames.append([[float(p[0]), float(p[1])] for p in frame_raw if len(p) >= 2])
                                        else:
                                            pts = []
                                            for j in range(0, len(frame_raw), 3):
                                                if j + 1 < len(frame_raw):
                                                    pts.append([float(frame_raw[j]), float(frame_raw[j + 1])])
                                            parsed_frames.append(pts)
                                    else:
                                        parsed_frames.append([])
                                if parsed_frames:
                                    max_peds = max(len(f) for f in parsed_frames) if parsed_frames else 0
                                    p_paths = []
                                    for p_idx in range(max_peds):
                                        ped_traj = [f[p_idx] for f in parsed_frames if len(f) > p_idx]
                                        if len(ped_traj) > 1:
                                            p_paths.append(ped_traj)
                    except Exception:
                        pass

                if p_paths is not None and isinstance(p_paths, (list, tuple)):
                    for p_idx, single_ped_path in enumerate(p_paths):
                        if single_ped_path is not None and len(single_ped_path) > 1:
                            arr = np.array(single_ped_path)
                            if arr.ndim == 2 and arr.shape[1] >= 2:
                                fig.add_trace(
                                    go.Scatter(
                                        x=arr[:, 0].tolist(),
                                        y=arr[:, 1].tolist(),
                                        mode='lines',
                                        name='Pedestrian Flows' if not peds_rendered else f'Pedestrian {p_idx + 1}',
                                        legendgroup='pedestrians',
                                        line=dict(color='#8b5cf6', width=2, dash='dot'),
                                        opacity=0.7,
                                        hovertemplate=f"<b>Pedestrian {p_idx + 1}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                                        showlegend=not peds_rendered,
                                    )
                                )
                                peds_rendered = True

        # 3. Overlay Unobstructed Robot Reference Baseline Path
        if self.spec.options.get("overlay_reference", False) or (self.spec.data_key and "mar" in self.spec.data_key):
            ref_rendered = False
            for _, r in pdf.iterrows():
                is_ref = bool(r.get("is_reference")) if r.get("is_reference") is not None else False
                ref_type = str(r.get("reference_type", "")).lower()
                stage_name = str(r.get("stage", "")).lower()
                planner_name = str(r.get("planner", "")).lower()

                # Exclude unhindered_peds
                if "unhindered" in ref_type or "unhindered" in planner_name:
                    continue

                if (ref_type == "unobstructed_robot") or ("unobstructed" in ref_type) or ("unobstructed" in stage_name) or (is_ref and ref_type != "unhindered_peds"):
                    r_path = r.get("path")
                    if r_path is not None:
                        paths_to_process = []
                        first_elem = r_path[0] if len(r_path) > 0 else None
                        if first_elem is not None and isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) > 0 and isinstance(first_elem[0], (list, tuple, np.ndarray)):
                            for sub_p in r_path:
                                paths_to_process.append(np.array(sub_p))
                        else:
                            paths_to_process.append(np.array(r_path))

                        for r_arr in paths_to_process:
                            if r_arr.ndim == 2 and r_arr.shape[1] >= 2:
                                # Ensure it's not a dummy spawn location
                                if np.any(np.abs(r_arr[:, 0]) > 500) or np.any(np.abs(r_arr[:, 1]) > 500):
                                    continue
                                fig.add_trace(
                                    go.Scatter(
                                        x=r_arr[:, 0].tolist(),
                                        y=r_arr[:, 1].tolist(),
                                        mode='lines',
                                        name='Unobstructed Reference Robot',
                                        legendgroup='unobstructed_ref',
                                        line=dict(color='#06b6d4', width=3, dash='dashdot'),
                                        opacity=0.9,
                                        hovertemplate="<b>Unobstructed Reference</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                                        showlegend=not ref_rendered,
                                    )
                                )
                                ref_rendered = True

        world = self._mechanism_world(map_name or (pdf["map"].iloc[0] if "map" in pdf.columns and not pdf.empty else None))
        if world is not None:
            for i, door in enumerate(world.doors):
                fig.add_trace(go.Scatter(x=[door.start[0], door.end[0]], y=[door.start[1], door.end[1]], mode="lines", name="Doors", legendgroup="mechanisms", showlegend=i == 0, line=dict(color=DOOR_COLORS["closed"], width=4), opacity=0.6, hovertemplate=f"<b>door {door.name}</b> ({door.kind})<extra></extra>"))
            for i, elevator in enumerate(world.elevators):
                xs = [c[0] for c in elevator.polygon] + [elevator.polygon[0][0]]
                ys = [c[1] for c in elevator.polygon] + [elevator.polygon[0][1]]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Elevator cabins", legendgroup="mechanisms", showlegend=i == 0, line=dict(color=CABIN_COLOR, width=1.5, dash="dot"), hovertemplate=f"<b>elevator {elevator.name}</b> level {elevator.level}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[elevator.door_start[0], elevator.door_end[0]], y=[elevator.door_start[1], elevator.door_end[1]], mode="lines", legendgroup="mechanisms", showlegend=False, line=dict(color=CABIN_COLOR, width=4), opacity=0.6, hoverinfo="skip"))

        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"

        layout_args: dict[str, object] = dict(title=title, xaxis_title="X [m]", yaxis_title="Y [m]", yaxis=dict(scaleanchor="x", scaleratio=1), template="plotly_white", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))

        if map_meta:
            png_path = map_meta["png_path"]
            try:
                with open(png_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")

                res = map_meta["resolution"]
                w_m = map_meta["width"] * res
                h_m = map_meta["height"] * res
                ox = float(map_meta.get("origin", [0.0, 0.0, 0.0])[0])
                oy = float(map_meta.get("origin", [0.0, 0.0, 0.0])[1])

                fig.add_layout_image(dict(source=f"data:image/png;base64,{encoded}", xref="x", yref="y", x=ox, y=oy + h_m, sizex=w_m, sizey=h_m, xanchor="left", yanchor="top", sizing="fill", opacity=0.45, layer="below"))
                layout_args["xaxis"] = dict(range=[ox, ox + w_m], title="X [m]")
                layout_args["yaxis"] = dict(range=[oy, oy + h_m], title="Y [m]", scaleanchor="x", scaleratio=1)
            except Exception as e:
                print(f"Failed to overlay map {map_name}: {e}")

        fig.update_layout(layout_args)
        return self._inject_slider_js(fig)

    def _inject_slider_js(self, fig: go.Figure) -> str:
        import uuid

        plot_id = f"traj_plot_{uuid.uuid4().hex}"
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=plot_id, config={'responsive': True})

        custom_js = f"""
        <div style="margin-top: 15px; padding: 10px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <label for="slider_{plot_id}" style="font-weight: 600; color: #1e293b;">Time Progression (s)</label>
                <span style="font-family: monospace; font-weight: bold; background: #e2e8f0; padding: 2px 8px; border-radius: 4px;" id="val_{plot_id}">Max</span>
            </div>
            <input type="range" id="slider_{plot_id}" min="0" max="100" value="100" style="width: 100%; cursor: pointer;">
        </div>
        <script>
            setTimeout(function() {{
                var plotDiv = document.getElementById("{plot_id}");
                if (!plotDiv || !plotDiv.data) return;

                plotDiv._originalData = plotDiv.data.map(t => ({{
                    x: t.x ? Array.from(t.x) : [],
                    y: t.y ? Array.from(t.y) : []
                }}));

                var maxLen = 0;
                plotDiv.data.forEach(t => {{
                    if (t.customdata && t.customdata.length > 0) {{
                        let cdata = Array.isArray(t.customdata[0]) ? t.customdata.map(d => d[0]) : t.customdata;
                        let m = Math.max(...cdata);
                        if (m > maxLen) maxLen = m;
                    }} else if (t.x && t.x.length > maxLen) {{
                        maxLen = t.x.length;
                    }}
                }});

                var slider = document.getElementById("slider_{plot_id}");
                var label = document.getElementById("val_{plot_id}");

                function formatTime(steps) {{
                    var totalMs = steps * 100; // assuming dt = 0.1s
                    var h = Math.floor(totalMs / 3600000);
                    var m = Math.floor((totalMs % 3600000) / 60000);
                    var s = Math.floor((totalMs % 60000) / 1000);
                    var ms = totalMs % 1000;
                    return String(h).padStart(2, '0') + ':' +
                           String(m).padStart(2, '0') + ':' +
                           String(s).padStart(2, '0') + ':' +
                           String(ms).padStart(3, '0');
                }}

                if (maxLen > 0) {{
                    slider.max = maxLen;
                    slider.value = maxLen;
                    var maxStr = formatTime(maxLen);
                    label.innerText = maxStr + " / " + maxStr;

                    slider.addEventListener("input", function(e) {{
                        var limit = parseInt(e.target.value);
                        label.innerText = formatTime(limit) + " / " + maxStr;

                        // Mutate directly and redraw for 100% reliability
                        plotDiv.data.forEach((trace, i) => {{
                            let isStaticRef = trace.name && (trace.name.includes("Theta*") || trace.name.includes("Unobstructed") || trace.legendgroup === "theta_star" || trace.legendgroup === "unobstructed_ref");
                            if (isStaticRef) {{
                                trace.x = plotDiv._originalData[i].x;
                                trace.y = plotDiv._originalData[i].y;
                            }} else if (trace.customdata && trace.customdata.length > 0) {{
                                let new_x = [];
                                let new_y = [];
                                let cdata = Array.isArray(trace.customdata[0]) ? trace.customdata.map(d => d[0]) : trace.customdata;
                                for (let j = 0; j < cdata.length; j++) {{
                                    if (limit >= cdata[j]) {{
                                        new_x.push(plotDiv._originalData[i].x[j]);
                                        new_y.push(plotDiv._originalData[i].y[j]);
                                    }}
                                }}
                                trace.x = new_x;
                                trace.y = new_y;
                            }} else {{
                                trace.x = plotDiv._originalData[i].x.slice(0, limit);
                                trace.y = plotDiv._originalData[i].y.slice(0, limit);
                            }}
                        }});

                        Plotly.redraw(plotDiv);
                    }});
                }} else {{
                    slider.disabled = true;
                }}
            }}, 500);
        </script>
        """
        return fig_html + custom_js

    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        df_filtered = self._apply_filters(df)
        data_key = self.spec.data_key or "path"
        if data_key not in df_filtered.columns:
            return None

        run_dir = self.run_dir

        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]

            valid_groups = [c for c in group_cols if c in df_filtered.columns]
            if not valid_groups:
                return self._render_single_plot(df_filtered.to_pandas(), "", None, run_dir=run_dir)

            htmls = []
            for name, group_df in df_filtered.group_by(valid_groups):
                pdf = group_df.to_pandas()
                if isinstance(name, tuple):
                    suffix = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name, strict=True))
                else:
                    suffix = f"{valid_groups[0]}: {name}"

                map_name = None
                if "map" in valid_groups:
                    map_idx = valid_groups.index("map")
                    map_name = name[map_idx] if isinstance(name, tuple) else name
                elif "map" in pdf.columns:
                    map_name = pdf["map"].iloc[0] if not pdf.empty else None

                html = self._render_single_plot(pdf, suffix, map_name, run_dir=run_dir)
                if html:
                    htmls.append(html)
            return htmls
        else:
            pdf = df_filtered.to_pandas()
            map_name = pdf["map"].iloc[0] if "map" in pdf.columns and not pdf.empty else None
            return self._render_single_plot(pdf, "", map_name, run_dir=run_dir)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        data_key = self.spec.data_key or "path"
        if data_key not in df_filtered.columns:
            return

        run_dir = self.run_dir

        if self.spec.group_by:
            group_cols = self.spec.group_by
            if isinstance(group_cols, str):
                group_cols = [group_cols]

            valid_groups = [c for c in group_cols if c in df_filtered.columns]
            if not valid_groups:
                pdf = df_filtered.to_pandas()
                if not pdf.empty:
                    self.render_seaborn_single(pdf, out_path, "", None, run_dir)
                return

            for name, group_df in df_filtered.group_by(valid_groups):
                pdf = group_df.to_pandas()
                if isinstance(name, tuple):
                    suffix = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name, strict=True))
                    file_suffix = "_".join(str(n).replace(" ", "_").replace("/", "_") for n in name)
                else:
                    suffix = f"{valid_groups[0]}: {name}"
                    file_suffix = str(name).replace(" ", "_").replace("/", "_")

                map_name = None
                if "map" in valid_groups:
                    map_idx = valid_groups.index("map")
                    map_name = name[map_idx] if isinstance(name, tuple) else name
                elif "map" in pdf.columns:
                    map_name = pdf["map"].iloc[0] if not pdf.empty else None

                group_out_path = out_path.with_name(f"{out_path.stem}_{file_suffix}{out_path.suffix}")
                self.render_seaborn_single(pdf, group_out_path, suffix, map_name, run_dir)
        else:
            pdf = df_filtered.to_pandas()
            if not pdf.empty:
                map_name = pdf["map"].iloc[0] if "map" in pdf.columns and not pdf.empty else None
                self.render_seaborn_single(pdf, out_path, "", map_name, run_dir)

    def render_seaborn_single(self, pdf: pd.DataFrame, out_path: pathlib.Path, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        import matplotlib.pyplot as plt

        diff_col = self.spec.differentiate or "planner"
        facet_col = self.spec.options.get("facet")
        if facet_col not in pdf.columns:
            facet_col = None
        crop = bool(self.spec.options.get("crop", facet_col is not None))
        data_key = self.spec.data_key or "path"
        if map_name is None and "map" in pdf.columns and not pdf.empty:
            map_name = pdf["map"].iloc[0]

        map_meta = self._load_map_image(map_name, run_dir=run_dir) if self.spec.options.get("show_map", True) and map_name else None
        world = self._mechanism_world(map_name)
        values = list(pdf[diff_col].unique()) if diff_col in pdf.columns else ["unknown"]
        palette = get_color_palette()
        colors = {v: palette[i % len(palette)] for i, v in enumerate(values)}

        groups = [(str(v), pdf.loc[pdf[facet_col] == v]) for v in sorted(pdf[facet_col].unique(), key=str)] if facet_col else [(None, pdf)]
        panels = [_Panel(name, frame, _tracks(frame, data_key, diff_col, colors)) for name, frame in groups]
        panels = [panel for panel in panels if panel.tracks] or panels[:1]

        fig, axes = self._figure(plt, panels, map_meta, crop)
        for panel, ax in zip(panels, axes, strict=True):
            panel.ax = ax
            self._draw_map(ax, map_meta)
            self._draw_static(ax, panel, world, map_name, run_dir)
            if panel.name is not None:
                ax.set_title(panel.title(), fontsize=10)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal")

        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
        if len(panels) > 1:
            fig.suptitle(title)
        else:
            axes[0].set_title(title)

        handles, labels = [], []
        for ax in axes:
            h, lab = ax.get_legend_handles_labels()
            handles += h
            labels += lab
        by_label = dict(zip(labels, handles, strict=True))
        if len(panels) > 1:
            fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(len(by_label), 6), fontsize=9)
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        else:
            axes[0].legend(by_label.values(), by_label.keys())
            fig.tight_layout()
        fig.savefig(out_path, dpi=300)

        if self.generate_gifs:
            self._generate_gif(fig, panels, out_path.with_suffix(".gif"))

        plt.close(fig)

    @staticmethod
    def _figure(plt: types.ModuleType, panels: list[_Panel], map_meta: dict | None, crop: bool) -> tuple[Figure, list[Axes]]:
        """One axes per panel, sized from the viewed area, and the shared view limits stored on each panel."""
        if map_meta and not crop:
            origin = map_meta.get("origin", [0.0, 0.0, 0.0])
            ox, oy = float(origin[0]), float(origin[1])
            x0, x1 = ox, ox + map_meta["width"] * map_meta["resolution"]
            y0, y1 = oy, oy + map_meta["height"] * map_meta["resolution"]
        else:
            points = np.vstack([pt for panel in panels for pt in panel.points()] or [np.zeros((1, 2))])
            x0, x1 = float(np.nanmin(points[:, 0])) - 2.0, float(np.nanmax(points[:, 0])) + 2.0
            y0, y1 = float(np.nanmin(points[:, 1])) - 2.0, float(np.nanmax(points[:, 1])) + 2.0
        for panel in panels:
            panel.limits = (x0, x1, y0, y1)
        n = len(panels)
        if n == 1:
            fig, ax = plt.subplots(figsize=(10, 10))
            return fig, [ax]
        w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        ncols = n if h > 1.5 * w else (2 if w > 1.5 * h else 3)
        ncols = min(ncols, n)
        nrows = math.ceil(n / ncols)
        scale = min(16.0 / (ncols * w), 9.0 / (nrows * h))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w * scale + 1.0, nrows * h * scale + 1.5), squeeze=False)
        for ax in axes.flat[n:]:
            ax.axis("off")
        return fig, list(axes.flat[:n])

    @staticmethod
    def _draw_map(ax: Axes, map_meta: dict | None) -> None:
        if not map_meta:
            return
        import matplotlib.image as mpimg

        try:
            img = mpimg.imread(map_meta["png_path"])
            res = map_meta["resolution"]
            origin = map_meta.get("origin", [0.0, 0.0, 0.0])
            ox, oy = float(origin[0]), float(origin[1])
            ax.imshow(img, extent=(ox, ox + map_meta["width"] * res, oy, oy + map_meta["height"] * res), alpha=0.5, origin="upper", zorder=0)
        except Exception as e:
            print(f"Failed to overlay map in seaborn: {e}")

    def _draw_static(self, ax: Axes, panel: _Panel, world: MechanismWorld | None, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        x0, x1, y0, y1 = panel.limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        overlay_markers = self.spec.options.get("overlay_markers", True)

        if world is not None:
            self._draw_mechanisms(ax, panel, world)

        for track in panel.tracks:
            (track.line,) = ax.plot(track.x, track.y, label=track.label, alpha=track.alpha, color=track.color, zorder=4)

        if overlay_markers:
            starts = [t.start for t in panel.tracks if t.start is not None]
            goals = [t.end for t in panel.tracks if t.end is not None and not t.collision]
            cols = [t.end for t in panel.tracks if t.end is not None and t.collision]
            if starts:
                ax.scatter([p[0] for p in starts], [p[1] for p in starts], marker="o", color="#00bfb2", s=30, zorder=5, label="Start")
            if goals:
                panel.goal_scatter = ax.scatter([p[0] for p in goals], [p[1] for p in goals], marker="*", color="#ffc845", s=60, zorder=5, label="Goal")
            if cols:
                panel.collision_scatter = ax.scatter([p[0] for p in cols], [p[1] for p in cols], marker="x", color="#d3273e", s=50, linewidths=2, zorder=5, label="Collision")

        if self.spec.options.get("overlay_theta_star", False) or self.spec.options.get("show_theta_star", False) or (self.spec.data_key and "theta" in self.spec.data_key):
            self._draw_theta_star(ax, panel.frame, map_name, run_dir)

        if self.spec.options.get("overlay_pedestrians", False) or self.spec.options.get("show_peds", False) or (self.spec.data_key and "ped" in self.spec.data_key):
            first = True
            for track in panel.tracks:
                for ped in track.peds:
                    (ped.line,) = ax.plot(ped.xy[:, 0], ped.xy[:, 1], color="#8b5cf6", linestyle=":", linewidth=1.5, alpha=0.6, label="Pedestrians" if first else None, zorder=3)
                    (ped.dot,) = ax.plot([], [], "o", color="#8b5cf6", markersize=5, zorder=3)
                    first = False

        if self.spec.options.get("overlay_reference", False) or (self.spec.data_key and "mar" in self.spec.data_key):
            ref_rendered = False
            for _, r in panel.frame.iterrows():
                if r.get("is_reference") and r.get("reference_type") == "unobstructed_robot":
                    for r_arr in _sub_paths(r.get("path")):
                        ax.plot(r_arr[:, 0], r_arr[:, 1], color="#06b6d4", linestyle="-.", linewidth=2.5, alpha=0.85, label="Unobstructed Reference" if not ref_rendered else None, zorder=4)
                        ref_rendered = True

    @staticmethod
    def _draw_theta_star(ax: Axes, frame: pd.DataFrame, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        try:
            from arena_evaluation.processing.path.theta_star import compute_theta_star_for_episode
        except ImportError:
            return
        for _, r in frame.iterrows():
            m_name = map_name or r.get("map")
            s_pt, g_pt = r.get("start"), r.get("goal")
            if not m_name or s_pt is None or g_pt is None:
                continue
            try:
                res = compute_theta_star_for_episode(str(m_name), (float(s_pt[0]), float(s_pt[1])), (float(g_pt[0]), float(g_pt[1])), run_dir=str(run_dir) if run_dir else None)
            except Exception:
                continue
            if res and res.success and len(res.path_x) > 0:
                ax.plot(res.path_x, res.path_y, color="#d97706", linestyle="--", linewidth=2.5, marker="d", markersize=5, label="Theta* Reference", zorder=4)
                return

    @staticmethod
    def _draw_mechanisms(ax: Axes, panel: _Panel, world: MechanismWorld) -> None:
        """Doors as their closed slab and cabins as dotted boxes."""
        states = {(st["entity"], st.get("level")): st for st in panel.states()}
        for i, door in enumerate(world.doors):
            xs, ys = _door_leaf(door, 0.0)
            (line,) = ax.plot(xs, ys, color=DOOR_COLORS["closed"], linewidth=4, alpha=0.7, solid_capstyle="butt", label="Doors" if i == 0 else None, zorder=2)
            panel.leaves.append(_Leaf(door, line, states.get((door.name, door.level)) or states.get((door.name, None))))
        for i, elevator in enumerate(world.elevators):
            xs = [c[0] for c in elevator.polygon] + [elevator.polygon[0][0]]
            ys = [c[1] for c in elevator.polygon] + [elevator.polygon[0][1]]
            ax.plot(xs, ys, color=CABIN_COLOR, linewidth=1.2, linestyle=":", label="Elevator cabins" if i == 0 else None, zorder=2)
            lx, ly = _door_leaf(elevator, 0.0)
            (line,) = ax.plot(lx, ly, color=CABIN_COLOR, linewidth=4, alpha=0.7, solid_capstyle="butt", zorder=2)
            panel.leaves.append(_Leaf(elevator, line, states.get((elevator.name, elevator.level)) or states.get((elevator.name, None))))

    @staticmethod
    def _generate_gif(fig: Figure, panels: list[_Panel], out_path: pathlib.Path) -> None:
        """Replay every panel on one sim clock."""
        import matplotlib.animation as animation

        t_end = max((t.t[-1] for panel in panels for t in panel.tracks if len(t.t)), default=0.0)
        if t_end <= 0.0:
            return
        times = np.linspace(0.0, t_end, GIF_FRAMES + 1)
        clock = fig.text(0.01, 0.01, "", fontsize=11, family="monospace")

        def update(now: float) -> list:
            artists: list[Artist] = [clock]
            for panel in panels:
                goals, cols = [], []
                for track in panel.tracks:
                    k = int(np.searchsorted(track.t, now, side="right"))
                    if track.line is not None:
                        track.line.set_data(track.x[:k], track.y[:k])
                        artists.append(track.line)
                    if track.end is not None and now >= track.t[-1]:
                        (cols if track.collision else goals).append(track.end)
                    for ped in track.peds:
                        if ped.line is None or ped.dot is None:
                            continue
                        j = int(np.searchsorted(track.t, now - PED_TAIL_S, side="left"))
                        ped.line.set_data(ped.xy[j:k, 0], ped.xy[j:k, 1])
                        here = ped.xy[k - 1] if k > 0 else None
                        ped.dot.set_data(([here[0]], [here[1]]) if here is not None and np.isfinite(here[0]) else ([], []))
                        artists += [ped.line, ped.dot]
                for leaf in panel.leaves:
                    if leaf.state is None:
                        continue
                    k = min(int(np.searchsorted(leaf.t, now, side="right")), len(leaf.progress)) - 1
                    progress = float(leaf.progress[k]) if k >= 0 and np.isfinite(leaf.progress[k]) else 0.0
                    xs, ys = _door_leaf(leaf.geom, min(max(progress, 0.0), 1.0))
                    leaf.line.set_data(xs, ys)
                    leaf.line.set_color(_door_color(progress, bool(leaf.locked[k]) if k >= 0 else False))
                    artists.append(leaf.line)
                if panel.goal_scatter is not None:
                    panel.goal_scatter.set_offsets(np.array(goals) if goals else np.empty((0, 2)))
                    artists.append(panel.goal_scatter)
                if panel.collision_scatter is not None:
                    panel.collision_scatter.set_offsets(np.array(cols) if cols else np.empty((0, 2)))
                    artists.append(panel.collision_scatter)
            clock.set_text(f"t = {now:6.1f} s")
            return artists

        ani = animation.FuncAnimation(fig, update, frames=times, interval=1000 // GIF_FPS, blit=True)
        try:
            ani.save(out_path, writer="pillow", fps=GIF_FPS)
        except Exception as e:
            print(f"Failed to save GIF {out_path}: {e}")
        finally:
            clock.remove()


class _Ped:
    def __init__(self, xy: np.ndarray) -> None:
        self.xy = xy
        self.line: Line2D | None = None
        self.dot: Line2D | None = None


class _Leaf:
    """A door slab artist bound to the recorded state series that moves it."""

    def __init__(self, geom: DoorGeom | ElevatorGeom, line: Line2D, state: typing.Mapping | None) -> None:
        self.geom = geom
        self.line = line
        self.state = state
        if state is not None:
            self.t = np.asarray(state["t"], dtype=float)
            self.progress = np.asarray(state["progress"], dtype=float)
            self.locked = np.asarray(state["locked"], dtype=bool)


class _Track:
    """One episode path on a panel: broken at teleports, on its own relative sim clock."""

    def __init__(self, row: typing.Mapping | pd.Series, arr: np.ndarray, label: str | None, color: str, alpha: float) -> None:
        time_ns = _row_time_ns(row, len(arr))
        self.x, self.y = _break_at_teleports(arr[:, 0], arr[:, 1], time_ns)
        self.t = (time_ns - time_ns[0]) / 1e9 if time_ns is not None else np.arange(len(arr), dtype=float)
        self.label = label
        self.color = color
        self.alpha = alpha
        self.collision = _is_collision(row)
        valid = np.where(~np.isnan(arr[:, 0]))[0]
        self.start: tuple[float, float] | None = None
        self.end: tuple[float, float] | None = None
        if len(valid):
            start = row.get("start")
            goal = row.get("goal")
            first = int(valid[_first_moving_index(arr[valid, 0], arr[valid, 1], time_ns[valid] if time_ns is not None else None)])
            last = int(valid[-1])
            self.start = (float(start[0]), float(start[1])) if isinstance(start, (list, tuple, np.ndarray)) and len(start) >= 2 else (float(arr[first, 0]), float(arr[first, 1]))
            if not self.collision and isinstance(goal, (list, tuple, np.ndarray)) and len(goal) >= 2:
                self.end = (float(goal[0]), float(goal[1]))
            else:
                self.end = (float(arr[last, 0]), float(arr[last, 1]))
        self.peds: list[_Ped] = []
        peds = row.get("pedestrian_path")
        for ped in peds if peds is not None else []:
            if ped is not None and len(ped) == len(arr):
                xy = np.array([list(p[:2]) for p in ped], dtype=float)
                if np.isfinite(xy[:, 0]).any():
                    self.peds.append(_Ped(xy))
        self.states: list[dict] = []
        states = row.get("mechanism_states")
        for state in states if states is not None else []:
            if len(state["progress"]) == len(arr):
                self.states.append({**state, "t": self.t})
        self.line: Line2D | None = None


class _Panel:
    def __init__(self, name: str | None, frame: pd.DataFrame, tracks: list[_Track]) -> None:
        self.name = name
        self.frame = frame
        self.tracks = tracks
        self.ax: Axes | None = None
        self.limits = (0.0, 1.0, 0.0, 1.0)
        self.leaves: list[_Leaf] = []
        self.goal_scatter: PathCollection | None = None
        self.collision_scatter: PathCollection | None = None

    def points(self) -> list[np.ndarray]:
        pts = [np.column_stack([t.x, t.y]) for t in self.tracks]
        pts += [p.xy for t in self.tracks for p in t.peds]
        return [p[np.isfinite(p[:, 0])] for p in pts if np.isfinite(p[:, 0]).any()]

    def states(self) -> list[dict]:
        """Mechanism states to animate for the single episode on this panel, none when it overlays several."""
        with_states = [t for t in self.tracks if t.states]
        return with_states[0].states if len(with_states) == 1 else []

    def title(self) -> str:
        results = sorted({str(r) for r in (self.frame["result"] if "result" in self.frame.columns else []) if r is not None and r == r})
        return f"{self.name}: {', '.join(results)}" if results else str(self.name)


def _tracks(frame: pd.DataFrame, data_key: str, diff_col: str, colors: dict) -> list[_Track]:
    tracks: list[_Track] = []
    seen: set[object] = set()
    ordered = frame.sort_values("episode") if "episode" in frame.columns else frame
    for _, row in ordered.iterrows():
        value = row[diff_col] if diff_col in frame.columns else "unknown"
        arrays = _sub_paths(row.get(data_key))
        for arr in arrays:
            label = None if value in seen else str(value)
            seen.add(value)
            tracks.append(_Track(row, arr, label, colors.get(value, "#41b6e6"), 0.5 if len(arrays) > 1 else 0.8))
    return tracks
