from __future__ import annotations

import json
import typing
import yaml
import polars as pl

from arena_evaluation.processing.topic_aligner import TopicAligner
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, TopicBundle


def _parse_conditions(raw: str | None) -> list[dict] | None:
    """Parse the recorded episode_record `conditions` JSON string into a list of dicts."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def _env_offset(tf_static: pl.DataFrame | None) -> tuple[float, float] | None:
    """Env packing translation from the `map -> env_<n>/map` transform, None if multi-env ambiguous."""
    if tf_static is None or len(tf_static) == 0:
        return (0.0, 0.0)
    rows = tf_static.filter(
        (pl.col("frame_id") == "map") & pl.col("child_frame_id").str.contains(r"^env_\d+/map$")
    ).select("trans_x", "trans_y").unique()
    if len(rows) == 0:
        return (0.0, 0.0)
    if len(rows) > 1:
        return None
    return (float(rows["trans_x"][0]), float(rows["trans_y"][0]))


def _episode_snapshot(snapshot: pl.DataFrame | None, start_time: int, end_time: int) -> pl.DataFrame | None:
    """Episode-scoped snapshot rows: in-window rows plus the seed, all rows of the latest
    snapshot stamp at-or-before `start_time`, or the first stamp inside the window if none."""
    if snapshot is None or len(snapshot) == 0:
        return None

    window = snapshot.filter((pl.col("time_ns") >= start_time) & (pl.col("time_ns") <= end_time))

    seed_candidates = snapshot.filter(pl.col("time_ns") <= start_time)
    if len(seed_candidates) > 0:
        seed_time = seed_candidates["time_ns"].max()
    elif len(window) > 0:
        seed_time = window["time_ns"].min()
    else:
        return None

    seed = snapshot.filter(pl.col("time_ns") == seed_time)
    return pl.concat([seed, window]).unique().sort("time_ns")


class EpisodeSplitter:
    """
    Splits continuous topic data into discrete episodes using EpisodeRecord messages.
    """
    def __init__(self, aligner: TopicAligner, min_episode_frames: int = 5):
        self.aligner = aligner
        self.min_episode_frames = min_episode_frames

    def split(self, bundle: TopicBundle, robot_name: str | None = None) -> list[AlignedEpisodeBundle]:
        """
        Split the raw bundle into aligned episodes.
        """
        if bundle.odom is None:
            return []

        def _to_df(lf_or_df):
            if isinstance(lf_or_df, pl.LazyFrame):
                return lf_or_df.collect()
            return lf_or_df

        odom_is_empty = False
        if isinstance(bundle.odom, pl.LazyFrame):
            odom_is_empty = bundle.odom.limit(1).collect().height == 0
        else:
            odom_is_empty = len(bundle.odom) == 0

        if odom_is_empty:
            return []

        record_df = _to_df(bundle.episode_record)
        initialpose_df = _to_df(bundle.initialpose)
        plan_df = _to_df(bundle.plan)
        semantic_snapshot_df = _to_df(bundle.semantic_snapshot)
        env_offset = _env_offset(_to_df(bundle.tf_static))

        episodes = []

        if record_df is None or len(record_df) == 0:
            aligned_df = self.aligner.align(bundle)
            aligned_df = _to_df(aligned_df)
            if aligned_df is not None and len(aligned_df) >= self.min_episode_frames:
                start_pos, goal_pos = [], []
                first_row = aligned_df.row(0, named=True)
                last_row = aligned_df.row(-1, named=True)
                if "pos_x" in first_row and "pos_y" in first_row and first_row["pos_x"] is not None:
                    start_pos = [first_row["pos_x"], first_row["pos_y"], first_row.get("yaw", 0.0)]
                if "pos_x" in last_row and "pos_y" in last_row and last_row["pos_x"] is not None:
                    goal_pos = [last_row["pos_x"], last_row["pos_y"], last_row.get("yaw", 0.0)]
                
                episodes.append(
                    AlignedEpisodeBundle(
                        episode_id=0,
                        data=aligned_df,
                        start_pos=start_pos,
                        goal_pos=goal_pos,
                        num_pedestrians=self._estimate_peds(aligned_df),
                        robot_name=robot_name,
                        semantic_snapshot=semantic_snapshot_df,
                        env_offset=env_offset,
                    )
                )
            return episodes


        rows = list(record_df.iter_rows(named=True))
        
        i = 0
        while i < len(rows):
            row = rows[i]
            start_time = row["time_ns"]
            
            if i + 1 < len(rows) and rows[i+1]["episode_id"] == row["episode_id"]:
                end_time = rows[i+1]["time_ns"]
                i += 2
            else:
                if i + 1 < len(rows):
                    end_time = rows[i + 1]["time_ns"] - 1
                else:
                    if isinstance(bundle.odom, pl.LazyFrame):
                        end_time = bundle.odom.select(pl.col("time_ns").max()).collect().item()
                    else:
                        end_time = bundle.odom.select(pl.col("time_ns").max()).item()
                i += 1
                
            aligned_df = self.aligner.align(bundle, start_time, end_time)
            aligned_df = _to_df(aligned_df)
            
            if aligned_df is None or len(aligned_df) < self.min_episode_frames:
                continue
                
            start_pos = []
            goal_pos = []
            
            try:
                params_str = row["robots_params"]
                if params_str:
                    params_dict = yaml.safe_load(params_str)
                    for robot_params in params_dict.values():
                        if "start" in robot_params:
                            start_pos = robot_params["start"]
                        if "goal" in robot_params:
                            goal_pos = robot_params["goal"]
                        break
            except Exception:
                pass
                
            if not start_pos and initialpose_df is not None and len(initialpose_df) > 0:
                df_init = initialpose_df.filter(pl.col("time_ns") >= start_time)
                if len(df_init) > 0:
                    row_init = df_init.row(0, named=True)
                else:
                    row_init = initialpose_df.row(-1, named=True)
                start_pos = [row_init["pos_x"], row_init["pos_y"], row_init["yaw"]]
                
            if start_pos and len(start_pos) == 3 and plan_df is not None and len(plan_df) > 0:
                df_plan = plan_df.filter(pl.col("time_ns") >= start_time)
                if len(df_plan) > 0:
                    row_plan = df_plan.row(0, named=True)
                    if "poses_yaw" in row_plan and len(row_plan["poses_yaw"]) > 0:
                        plan_yaw = row_plan["poses_yaw"][0]
                        if plan_yaw != 0.0 and abs(start_pos[2] - plan_yaw) > 1.0:
                            start_pos[2] = plan_yaw

            if not start_pos and aligned_df is not None and len(aligned_df) > 0:
                first_row = aligned_df.row(0, named=True)
                if "pos_x" in first_row and "pos_y" in first_row:
                    start_pos = [first_row["pos_x"], first_row["pos_y"], first_row.get("yaw", 0.0)]

            if not goal_pos and plan_df is not None and len(plan_df) > 0:
                df_plan_ep = plan_df.filter((pl.col("time_ns") >= start_time) & (pl.col("time_ns") <= end_time))
                if len(df_plan_ep) > 0:
                    row_plan = df_plan_ep.row(0, named=True)
                    if "poses_x" in row_plan and len(row_plan["poses_x"]) > 0:
                        goal_pos = [row_plan["poses_x"][-1], row_plan["poses_y"][-1], row_plan["poses_yaw"][-1]]

            if not goal_pos and aligned_df is not None and len(aligned_df) > 0:
                last_row = aligned_df.row(-1, named=True)
                if "pos_x" in last_row and "pos_y" in last_row:
                    goal_pos = [last_row["pos_x"], last_row["pos_y"], last_row.get("yaw", 0.0)]

            episodes.append(
                AlignedEpisodeBundle(
                    episode_id=row["episode_id"],
                    data=aligned_df,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    num_pedestrians=self._estimate_peds(aligned_df),
                    robot_name=robot_name,
                    semantic_snapshot=_episode_snapshot(semantic_snapshot_df, start_time, end_time),
                    conditions=_parse_conditions(row.get("conditions")),
                    env_offset=env_offset,
                )
            )
            
        return episodes

    def _estimate_peds(self, df: pl.DataFrame) -> int:
        if "num_pedestrians" in df.columns:
            peds = df["num_pedestrians"].drop_nulls()
            if len(peds) > 0:
                return int(peds.max())
        return 0
