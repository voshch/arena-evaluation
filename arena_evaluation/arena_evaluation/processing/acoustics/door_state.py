"""Door state timeline constructed from SemanticSnapshot messages."""

from __future__ import annotations

import bisect

import numpy as np
import polars as pl

from arena_evaluation.processing.acoustics.constants import (
    DOOR_OPEN_PROGRESS_THRESHOLD as OPEN_PROGRESS_THRESHOLD,
)
from arena_evaluation.storage.exceptions import MetricInputError


class DoorStateTimeline:
    """Sorted (time, open-doors frozenset) lookup with binary search."""

    def __init__(self, times_ns: np.ndarray, open_sets: list[frozenset[str]]):
        self.times_ns = times_ns
        self.open_sets = open_sets

    @classmethod
    def from_semantic_frame(
        cls,
        semantic: pl.DataFrame | None,
        *,
        open_progress_threshold: float = OPEN_PROGRESS_THRESHOLD,
    ) -> DoorStateTimeline | None:
        """Build the timeline from the flattened semantic snapshot table.

        A door is open while its progress (0 closed, 1 open) is strictly above
        ``open_progress_threshold`` (default ``OPEN_PROGRESS_THRESHOLD``, declared in
        ``constants.DOOR_OPEN_PROGRESS_THRESHOLD``), whether it is opening or closing. State
        labels are not used. A door reading without a progress row is an error.
        """
        if semantic is None:
            return None
        if isinstance(semantic, pl.LazyFrame):
            semantic = semantic.collect()
        if len(semantic) == 0:
            return None

        df = semantic
        missing = {"t_ns", "entity_name", "kind", "field_name", "value_num"} - set(df.columns)
        if missing:
            raise MetricInputError(f"AcousticExposureCalculator: semantic table lacks columns {sorted(missing)}; re-extract world_semantics for the episode (has {df.columns})")
        df = df.filter(pl.col("kind") == "door")
        if len(df) == 0:
            return None

        progress = df.filter(pl.col("field_name") == "progress")
        readings = df.select("t_ns", "entity_name").unique()
        unmeasured = readings.join(
            progress.select("t_ns", "entity_name").unique(),
            on=["t_ns", "entity_name"],
            how="anti",
        )
        if len(unmeasured) > 0:
            raise MetricInputError(f"AcousticExposureCalculator: {len(unmeasured)} door readings have no 'progress' row (first {unmeasured.row(0)}); re-extract world_semantics for the episode")

        is_open_expr = pl.col("value_num").is_not_nan() & (pl.col("value_num") > open_progress_threshold)
        all_timestamps = df.select("t_ns").unique().sort("t_ns")
        open_by_ts = progress.filter(is_open_expr).group_by("t_ns").agg(pl.col("entity_name").unique().alias("open_doors"))
        timeline_df = all_timestamps.join(open_by_ts, on="t_ns", how="left").sort("t_ns")
        timestamps = timeline_df["t_ns"].to_numpy()
        open_sets = [frozenset(doors) if doors is not None else frozenset() for doors in timeline_df["open_doors"]]
        return cls(timestamps, open_sets)

    def open_doors_at(self, t_ns: int) -> frozenset[str]:
        """Backward-asof: door state at or before *t_ns* (empty before first)."""
        idx = bisect.bisect_right(self.times_ns, t_ns) - 1
        if idx < 0:
            return frozenset()
        return self.open_sets[idx]
