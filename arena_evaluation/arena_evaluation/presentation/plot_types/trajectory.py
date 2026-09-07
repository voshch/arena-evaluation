from __future__ import annotations

import pathlib
import base64
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer
from arena_evaluation.processing.map_registry import MapRegistry


class TrajectoryRenderer(BasePlotRenderer):
    PLOT_TYPE = "trajectory"
    generate_gifs: bool = False

    def _load_map_image(self, map_name: str, run_dir: pathlib.Path | None = None):
        return MapRegistry.get_map(map_name, run_dir=run_dir)

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

    def _render_single_plot(self, pdf, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None = None) -> str:
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

        planners = pdf_eval[diff_col].unique() if diff_col in pdf_eval.columns else ["unknown"]
        seen_planners = set()
        
        for planner in planners:
            if diff_col in pdf_eval.columns:
                planner_df = pdf_eval[pdf_eval[diff_col] == planner]
            else:
                planner_df = pdf_eval
                
            if "episode" in planner_df.columns:
                planner_df = planner_df.sort_values("episode")
                
            overlay_markers = self.spec.options.get("overlay_markers", True)
            
            all_x_by_agent = []
            all_y_by_agent = []
            
            if overlay_markers:
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
            for _, row in planner_df.iterrows():
                path = row.get(data_key)
                if path is None or len(path) == 0:
                    continue
                    
                is_collision = row.get("result") == "COLLISION"
                    
                paths_to_process = []
                try:
                    first_elem = path[0]
                    if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) > 0 and isinstance(first_elem[0], (list, tuple, np.ndarray)):
                        for sub_path in path:
                            if sub_path is not None and len(sub_path) > 0:
                                paths_to_process.append(np.array([list(p) for p in sub_path]))
                    else:
                        paths_to_process.append(np.array([list(p) for p in path]))
                except Exception:
                    continue
                    
                for k, path_arr in enumerate(paths_to_process):
                    if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                        continue
                        
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
                            
                            dists = np.sqrt(np.diff(path_arr[clean_indices, 0])**2 + np.diff(path_arr[clean_indices, 1])**2)
                            jumps = np.where(dists > 3.0)[0]
                            segment_starts = np.concatenate([[0], jumps + 1])
                            segment_ends = np.concatenate([jumps, [len(dists)]])
                            
                            first_clean_idx = segment_starts[-1]
                            for s_idx, e_idx in zip(segment_starts, segment_ends):
                                if e_idx > s_idx:
                                    segment_length = np.sum(dists[s_idx:e_idx])
                                    if segment_length > 0.1:
                                        first_clean_idx = s_idx
                                        break
                                
                            first_idx = int(clean_indices[first_clean_idx])
                            last_idx = int(clean_indices[-1])
                            
                            starts_x_by_agent[k].append(path_arr[first_idx, 0])
                            starts_y_by_agent[k].append(path_arr[first_idx, 1])
                            starts_idx_by_agent[k].append(current_len + first_idx)
                            
                            is_success = (row.get("result") in ("SUCCESS", "success", "reached", "goal_reached")) or bool(row.get("success"))
                            is_coll = (row.get("result") in ("COLLISION", "collision", "CRASH")) or ((row.get("collision_amount") or 0) > 0)
                            
                            if is_coll:
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
                    all_x_by_agent[k].extend(path_arr[:, 0].tolist())
                    all_y_by_agent[k].extend(path_arr[:, 1].tolist())

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
                
                fig.add_trace(go.Scatter(
                    x=all_x_by_agent[k],
                    y=all_y_by_agent[k],
                    mode='lines',
                    name=trace_name,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    line=dict(color=planner_color),
                    opacity=opacity,
                    hovertemplate=f"<b>{planner}</b><br>{trace_name}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                    customdata=customdata
                ))
                
                if overlay_markers:
                    if starts_x_by_agent[k]:
                        fig.add_trace(go.Scatter(
                            x=starts_x_by_agent[k], y=starts_y_by_agent[k],
                            mode='markers',
                            marker=dict(symbol='circle', size=8, color='#10b981', line=dict(width=1, color='#065f46')),
                            name=f"{planner} Starts",
                            legendgroup=planner,
                            showlegend=False,
                            opacity=0.9,
                            hovertemplate=f"<b>{planner}</b><br>Start<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                            customdata=starts_idx_by_agent[k]
                        ))
                    if goals_x_by_agent[k]:
                        fig.add_trace(go.Scatter(
                            x=goals_x_by_agent[k], y=goals_y_by_agent[k],
                            mode='markers',
                            marker=dict(symbol='star', size=12, color='#ffc845'),
                            name=f"{planner} Goals",
                            legendgroup=planner,
                            showlegend=False,
                            opacity=0.9,
                            hovertemplate=f"<b>{planner}</b><br>Goal<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                            customdata=goals_idx_by_agent[k]
                        ))
                    if col_x_by_agent[k]:
                        fig.add_trace(go.Scatter(
                            x=col_x_by_agent[k], y=col_y_by_agent[k],
                            mode='markers',
                            marker=dict(symbol='x', size=10, color='#d3273e', line=dict(width=2)),
                            name=f"{planner} Collisions",
                            legendgroup=planner,
                            showlegend=False,
                            opacity=1.0,
                            hovertemplate=f"<b>{planner}</b><br>Collision<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                            customdata=col_idx_by_agent[k]
                        ))

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
                                fig.add_trace(go.Scatter(
                                    x=res.path_x.tolist(),
                                    y=res.path_y.tolist(),
                                    mode='lines+markers',
                                    name='Theta* Optimal Demonstration',
                                    legendgroup='theta_star',
                                    line=dict(color='#d97706', width=3, dash='dash'),
                                    marker=dict(size=7, symbol='diamond', color='#d97706'),
                                    opacity=0.95,
                                    hovertemplate="<b>Theta* Optimal Path</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                                    showlegend=not theta_rendered
                                ))
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
                        ep_id = int(r["episode"])
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
                                                    pts.append([float(frame_raw[j]), float(frame_raw[j+1])])
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
                                fig.add_trace(go.Scatter(
                                    x=arr[:, 0].tolist(),
                                    y=arr[:, 1].tolist(),
                                    mode='lines',
                                    name='Pedestrian Flows' if not peds_rendered else f'Pedestrian {p_idx+1}',
                                    legendgroup='pedestrians',
                                    line=dict(color='#8b5cf6', width=2, dash='dot'),
                                    opacity=0.7,
                                    hovertemplate=f"<b>Pedestrian {p_idx+1}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                                    showlegend=not peds_rendered
                                ))
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
                                fig.add_trace(go.Scatter(
                                    x=r_arr[:, 0].tolist(),
                                    y=r_arr[:, 1].tolist(),
                                    mode='lines',
                                    name='Unobstructed Reference Robot',
                                    legendgroup='unobstructed_ref',
                                    line=dict(color='#06b6d4', width=3, dash='dashdot'),
                                    opacity=0.9,
                                    hovertemplate="<b>Unobstructed Reference</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                                    showlegend=not ref_rendered
                                ))
                                ref_rendered = True
            
        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
            
        layout_args = dict(
            title=title,
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        
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
                
                fig.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{encoded}",
                        xref="x",
                        yref="y",
                        x=ox,
                        y=oy + h_m,
                        sizex=w_m,
                        sizey=h_m,
                        xanchor="left",
                        yanchor="top",
                        sizing="fill",
                        opacity=0.45,
                        layer="below"
                    )
                )
                layout_args["xaxis"] = dict(range=[ox, ox + w_m], title="X [m]")
                layout_args["yaxis"] = dict(range=[oy, oy + h_m], title="Y [m]", scaleanchor="x", scaleratio=1)
            except Exception as e:
                print(f"Failed to overlay map {map_name}: {e}")
                
        fig.update_layout(**layout_args)
        return self._inject_slider_js(fig)

    def _generate_gif(self, fig, lines_data, markers_info, out_path: pathlib.Path):
        import matplotlib.animation as animation
        
        max_len = max([len(data[1]) for data in lines_data]) if lines_data else 0
        if max_len == 0:
            return
            
        step = max(1, max_len // 100)
        frames = list(range(0, max_len, step))
        if frames[-1] != max_len:
            frames.append(max_len)
            
        def update(frame):
            artists = []
            for line, x, y in lines_data:
                line.set_data(x[:frame], y[:frame])
                artists.append(line)
                
            if markers_info:
                data = markers_info["data"]
                
                s_pts = [[m["x"], m["y"]] for m in data if m["type"] == "start" and frame >= m["frame"]]
                if markers_info.get("starts"):
                    markers_info["starts"].set_offsets(s_pts if s_pts else np.empty((0, 2)))
                    artists.append(markers_info["starts"])
                    
                g_pts = [[m["x"], m["y"]] for m in data if m["type"] == "goal" and frame >= m["frame"]]
                if markers_info.get("goals"):
                    markers_info["goals"].set_offsets(g_pts if g_pts else np.empty((0, 2)))
                    artists.append(markers_info["goals"])
                    
                c_pts = [[m["x"], m["y"]] for m in data if m["type"] == "collision" and frame >= m["frame"]]
                if markers_info.get("cols"):
                    markers_info["cols"].set_offsets(c_pts if c_pts else np.empty((0, 2)))
                    artists.append(markers_info["cols"])
                    
            return artists
            
        for line, _, _ in lines_data:
            line.set_data([], [])
            
        if markers_info:
            if markers_info.get("starts"): markers_info["starts"].set_offsets(np.empty((0, 2)))
            if markers_info.get("goals"): markers_info["goals"].set_offsets(np.empty((0, 2)))
            if markers_info.get("cols"): markers_info["cols"].set_offsets(np.empty((0, 2)))
            
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
        try:
            ani.save(out_path, writer='pillow', fps=20)
        except Exception as e:
            print(f"Failed to save GIF {out_path}: {e}")

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
                    suffix = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name))
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
                    suffix = " | ".join(f"{c}: {n}" for c, n in zip(valid_groups, name))
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

    def render_seaborn_single(self, pdf, out_path: pathlib.Path, title_suffix: str, map_name: str | None, run_dir: pathlib.Path | None) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        plt.figure(figsize=(10, 10))
        
        map_meta = None
        if self.spec.options.get("show_map", True) and map_name:
            map_meta = self._load_map_image(map_name, run_dir=run_dir)
            
        if map_meta:
            try:
                img = mpimg.imread(map_meta["png_path"])
                res = map_meta["resolution"]
                w_m = map_meta["width"] * res
                h_m = map_meta["height"] * res
                plt.imshow(img, extent=[0, w_m, 0, h_m], alpha=0.5, origin='upper', zorder=0)
                plt.xlim(0, w_m)
                plt.ylim(0, h_m)
            except Exception as e:
                print(f"Failed to overlay map {map_name} in seaborn: {e}")
                
        diff_col = self.spec.differentiate or "planner"
        
        overlay_markers = self.spec.options.get("overlay_markers", True)
        
        lines_data = []
        markers_data = []
        
        all_x_by_agent = {}
        starts_x, starts_y = [], []
        goals_x, goals_y = [], []
        col_x, col_y = [], []
        
        data_key = self.spec.data_key or "path"
        
        planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
        seen_planners = set()
        
        for planner in planners:
            if diff_col in pdf.columns:
                planner_df = pdf[pdf[diff_col] == planner]
            else:
                planner_df = pdf
                
            if "episode" in planner_df.columns:
                planner_df = planner_df.sort_values("episode")
                
            if planner not in all_x_by_agent:
                all_x_by_agent[planner] = []
                
            for _, row in planner_df.iterrows():
                path = row.get(data_key)
                if path is None or len(path) == 0:
                    continue
                    
                is_collision = row.get("result") == "COLLISION"
                    
                paths_to_process = []
                try:
                    first_elem = path[0]
                    if isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) > 0 and isinstance(first_elem[0], (list, tuple, np.ndarray)):
                        for sub_path in path:
                            if sub_path is not None and len(sub_path) > 0:
                                paths_to_process.append(np.array([list(p) for p in sub_path]))
                    else:
                        paths_to_process.append(np.array([list(p) for p in path]))
                except Exception:
                    continue
                    
                for k, path_arr in enumerate(paths_to_process):
                    if path_arr.ndim != 2 or path_arr.shape[1] < 2:
                        continue
                        
                    while len(all_x_by_agent[planner]) <= k:
                        all_x_by_agent[planner].append({"x": [], "y": []})
                        
                    current_len = len(all_x_by_agent[planner][k]["x"])
                    
                    if overlay_markers:
                        valid_mask = ~np.isnan(path_arr[:, 0])
                        if np.any(valid_mask):
                            clean_indices = np.where(valid_mask)[0]
                            
                            # Find actual start avoiding initial spawn jumps
                            dists = np.sqrt(np.diff(path_arr[clean_indices, 0])**2 + np.diff(path_arr[clean_indices, 1])**2)
                            jumps = np.where(dists > 3.0)[0]
                            segment_starts = np.concatenate([[0], jumps + 1])
                            segment_ends = np.concatenate([jumps, [len(dists)]])
                            
                            first_clean_idx = segment_starts[-1]
                            for s_idx, e_idx in zip(segment_starts, segment_ends):
                                if e_idx > s_idx:
                                    segment_length = np.sum(dists[s_idx:e_idx])
                                    if segment_length > 0.1:
                                        first_clean_idx = s_idx
                                        break
                                
                            first_idx = int(clean_indices[first_clean_idx])
                            last_idx = int(clean_indices[-1])
                            
                            # True start/goal from dataframe
                            true_start = row.get("start")
                            true_goal = row.get("goal")
                            
                            if isinstance(true_start, (list, tuple, np.ndarray)) and len(true_start) >= 2:
                                starts_x.append(true_start[0])
                                starts_y.append(true_start[1])
                                markers_data.append({"frame": current_len + first_idx, "x": true_start[0], "y": true_start[1], "type": "start"})
                            else:
                                starts_x.append(path_arr[first_idx, 0])
                                starts_y.append(path_arr[first_idx, 1])
                                markers_data.append({"frame": current_len + first_idx, "x": path_arr[first_idx, 0], "y": path_arr[first_idx, 1], "type": "start"})
                            
                            if is_collision:
                                col_x.append(path_arr[last_idx, 0])
                                col_y.append(path_arr[last_idx, 1])
                                markers_data.append({"frame": current_len + last_idx, "x": path_arr[last_idx, 0], "y": path_arr[last_idx, 1], "type": "collision"})
                            else:
                                if isinstance(true_goal, (list, tuple, np.ndarray)) and len(true_goal) >= 2:
                                    goals_x.append(true_goal[0])
                                    goals_y.append(true_goal[1])
                                    markers_data.append({"frame": current_len + last_idx, "x": true_goal[0], "y": true_goal[1], "type": "goal"})
                                else:
                                    goals_x.append(path_arr[last_idx, 0])
                                    goals_y.append(path_arr[last_idx, 1])
                                    markers_data.append({"frame": current_len + last_idx, "x": path_arr[last_idx, 0], "y": path_arr[last_idx, 1], "type": "goal"})
                                
                    all_x_by_agent[planner][k]["x"].extend(path_arr[:, 0])
                    all_x_by_agent[planner][k]["x"].append(np.nan) # Separator for seaborn
                    all_x_by_agent[planner][k]["y"].extend(path_arr[:, 1])
                    all_x_by_agent[planner][k]["y"].append(np.nan)
                    
        palette = get_color_palette()
        planners_list = list(planners)
        
        for planner, agents in all_x_by_agent.items():
            planner_idx = planners_list.index(planner) if planner in planners_list else 0
            planner_color = palette[planner_idx % len(palette)]
            
            for k, agent_data in enumerate(agents):
                if not agent_data["x"]:
                    continue
                    
                x_arr = np.array(agent_data["x"], dtype=float)
                y_arr = np.array(agent_data["y"], dtype=float)
                
                dists = np.sqrt(np.diff(x_arr)**2 + np.diff(y_arr)**2)
                jumps = np.where((dists > 3.0) & ~np.isnan(dists))[0]
                split_indices = jumps + 1
                
                final_x = x_arr.copy()
                final_y = y_arr.copy()
                if len(split_indices) > 0:
                    final_x[split_indices] = np.nan
                    final_y[split_indices] = np.nan
                
                if len(agents) > 1:
                    opacity = 0.5
                    label = planner if k == 0 else None
                else:
                    opacity = 0.8
                    label = planner if k == 0 else None
                    
                line, = plt.plot(final_x, final_y, label=label, alpha=opacity, color=planner_color)
                lines_data.append((line, final_x, final_y))
                
        scat_starts = None
        scat_goals = None
        scat_cols = None
        
        if overlay_markers:
            if starts_x:
                scat_starts = plt.scatter(starts_x, starts_y, marker='o', color='#00bfb2', s=30, zorder=5, label='Start')
            if goals_x:
                scat_goals = plt.scatter(goals_x, goals_y, marker='*', color='#ffc845', s=60, zorder=5, label='Goal')
            if col_x:
                scat_cols = plt.scatter(col_x, col_y, marker='x', color='#d3273e', s=50, linewidths=2, zorder=5, label='Collision')

        # 1. Overlay Theta* Synthetic Demonstration Baseline
        if self.spec.options.get("overlay_theta_star", False) or self.spec.options.get("show_theta_star", False) or (self.spec.data_key and "theta" in self.spec.data_key):
            try:
                from arena_evaluation.processing.path.theta_star import compute_theta_star_for_episode
                for _, r in pdf.iterrows():
                    m_name = map_name or r.get("map")
                    s_pt = r.get("start")
                    g_pt = r.get("goal")
                    if m_name and s_pt is not None and g_pt is not None:
                        try:
                            s_coords = (float(s_pt[0]), float(s_pt[1]))
                            g_coords = (float(g_pt[0]), float(g_pt[1]))
                            res = compute_theta_star_for_episode(str(m_name), s_coords, g_coords, run_dir=str(run_dir) if run_dir else None)
                            if res and res.success and len(res.path_x) > 0:
                                plt.plot(res.path_x, res.path_y, color='#d97706', linestyle='--', linewidth=2.5, marker='d', markersize=5, label='Theta* Reference', zorder=4)
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        # 2. Overlay Pedestrian Paths
        if self.spec.options.get("overlay_pedestrians", False) or self.spec.options.get("show_peds", False) or (self.spec.data_key and "ped" in self.spec.data_key):
            peds_rendered = False
            for _, r in pdf.iterrows():
                p_paths = r.get("pedestrian_path")
                if p_paths is not None and isinstance(p_paths, (list, tuple)):
                    for single_ped_path in p_paths:
                        if single_ped_path is not None and len(single_ped_path) > 1:
                            arr = np.array(single_ped_path)
                            if arr.ndim == 2 and arr.shape[1] >= 2:
                                plt.plot(arr[:, 0], arr[:, 1], color='#8b5cf6', linestyle=':', linewidth=1.5, alpha=0.6, label='Pedestrians' if not peds_rendered else None, zorder=3)
                                peds_rendered = True

        # 3. Overlay Unobstructed Robot Reference Baseline
        if self.spec.options.get("overlay_reference", False) or (self.spec.data_key and "mar" in self.spec.data_key):
            ref_rendered = False
            for _, r in pdf.iterrows():
                if r.get("is_reference") and r.get("reference_type") == "unobstructed_robot":
                    r_path = r.get("path")
                    if r_path is not None:
                        r_arr = np.array(r_path)
                        if r_arr.ndim == 2 and r_arr.shape[1] >= 2:
                            plt.plot(r_arr[:, 0], r_arr[:, 1], color='#06b6d4', linestyle='-.', linewidth=2.5, alpha=0.85, label='Unobstructed Reference' if not ref_rendered else None, zorder=4)
                            ref_rendered = True
            
        title = self.spec.title
        if title_suffix:
            title = f"{title} - {title_suffix}"
        plt.title(title)
        
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis('equal')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        
        if self.generate_gifs:
            markers_info = None
            if overlay_markers:
                markers_info = {
                    "data": markers_data,
                    "starts": scat_starts,
                    "goals": scat_goals,
                    "cols": scat_cols
                }
                
            gif_path = out_path.with_suffix(".gif")
            self._generate_gif(plt.gcf(), lines_data, markers_info, gif_path)
        
        plt.close()
