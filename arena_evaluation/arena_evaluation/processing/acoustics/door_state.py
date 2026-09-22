"""Door state timeline constructed from SemanticSnapshot messages."""

from __future__ import annotations

import bisect

import numpy as np
import polars as pl

OPEN_PROGRESS_THRESHOLD = 0.5


class DoorStateTimeline:
    """Sorted (time, open-doors frozenset) lookup with binary search."""

    def __init__(self, times_ns: np.ndarray, open_sets: list[frozenset[str]]):
        self.times_ns = times_ns
        self.open_sets = open_sets

    @classmethod
    def from_semantic_frame(cls, semantic: pl.DataFrame | None) -> DoorStateTimeline | None:
        """Build the timeline from the flattened semantic snapshot table."""
        if semantic is None:
            return None
        if isinstance(semantic, pl.LazyFrame):
            semantic = semantic.collect()  # lazy -> eager
        if len(semantic) == 0:
            return None

        df = semantic
        if "kind" not in df.columns:
            return None
        df = df.filter(pl.col("kind") == "door")
        if len(df) == 0:
            return None

        # per timestamp: build {entity: field -> value}
        timestamps = sorted(df["time_ns"].unique().to_list())
        open_sets: list[frozenset[str]] = []
        for ts in timestamps:
            frame = df.filter(pl.col("time_ns") == ts)
            open_doors: set[str] = set()
            entities = frame["entity"].unique().to_list()
            for ent in entities:
                erows = frame.filter(pl.col("entity") == ent)
                is_open = False
                for row in erows.iter_rows(named=True):
                    field = row.get("field", "")
                    if field == "open" and row.get("value_bool") is True:
                        is_open = True
                    elif field == "state" and str(row.get("value_str", "")) == "open":
                        is_open = True
                    elif field == "progress":
                        v = row.get("value_num")
                        if v is not None and not np.isnan(float(v)) and float(v) > OPEN_PROGRESS_THRESHOLD:
                            is_open = True
                if is_open:
                    open_doors.add(ent)
            open_sets.append(frozenset(open_doors))

        return cls(np.asarray(timestamps, dtype=np.int64), open_sets)

    def open_doors_at(self, time_ns: int) -> frozenset[str]:
        """Backward-asof: door state at or before *time_ns* (empty before first)."""
        idx = bisect.bisect_right(self.times_ns, time_ns) - 1
        if idx < 0:
            return frozenset()
        return self.open_sets[idx]
