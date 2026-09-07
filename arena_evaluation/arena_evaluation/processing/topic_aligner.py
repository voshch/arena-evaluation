from __future__ import annotations

import polars as pl
import typing

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle, TopicBundle


class TopicAligner:
    """Aligns asynchronous topics onto the odom time axis using asof joins."""
    def __init__(self, tolerance_ns: int = 100_000_000):
        self.tolerance_ns = tolerance_ns

    def align(
        self,
        bundle: TopicBundle,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Align all available topics onto the odom time axis, optionally filtered to [start_time_ns, end_time_ns]."""
        if bundle.odom is None:
            return None

        def is_empty(frame):
            if isinstance(frame, pl.LazyFrame):
                return frame.limit(1).collect().height == 0
            return len(frame) == 0

        if is_empty(bundle.odom):
            return None

        df = bundle.odom
        should_collect = not isinstance(df, pl.LazyFrame)
        
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        if start_time_ns is not None:
            df = df.filter(pl.col("time_ns") >= start_time_ns)
        if end_time_ns is not None:
            df = df.filter(pl.col("time_ns") <= end_time_ns)
            
        if is_empty(df):
            return None

        df = df.sort("time_ns")
        
        def join_topic(primary: pl.LazyFrame, secondary: pl.DataFrame | pl.LazyFrame | None, prefix: str) -> pl.LazyFrame:
            if secondary is None or is_empty(secondary):
                return primary
                
            sec_df = secondary
            if isinstance(sec_df, pl.DataFrame):
                sec_df = sec_df.lazy()

            if start_time_ns is not None:
                sec_df = sec_df.filter(pl.col("time_ns") >= start_time_ns - self.tolerance_ns)
            if end_time_ns is not None:
                sec_df = sec_df.filter(pl.col("time_ns") <= end_time_ns + self.tolerance_ns)
                
            if is_empty(sec_df):
                return primary
                
            return primary.join_asof(
                sec_df.sort("time_ns"),
                on="time_ns",
                strategy="backward",
                tolerance=self.tolerance_ns
            )

        df = join_topic(df, bundle.scan, "scan")
        df = join_topic(df, bundle.cmd_vel, "cmd")
        df = join_topic(df, bundle.joint_states, "joint")
        df = join_topic(df, bundle.peds, "peds")
        df = join_topic(df, bundle.collision_events, "col")
        df = join_topic(df, bundle.collision_monitor_state, "cms")
        df = join_topic(df, bundle.power, "power")
        df = join_topic(df, bundle.energy, "energy")
        df = join_topic(df, bundle.acoustics, "acous")
        df = join_topic(df, bundle.tf_gt, "tf_gt")

        if bundle.characterization_phase is not None and not is_empty(bundle.characterization_phase):
            markers = bundle.characterization_phase
            if isinstance(markers, pl.LazyFrame):
                markers = markers.select(["time_ns", "label"]).sort("time_ns")
            else:
                markers = markers.select(["time_ns", "label"]).sort("time_ns").lazy()
            df = df.join_asof(markers, on="time_ns", strategy="backward")

        if should_collect:
            return df.collect()
        return df
