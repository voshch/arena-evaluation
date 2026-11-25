"""Which ground-truth pose stream an episode is evaluated on: dense, sparse or absent."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import polars as pl

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import TopicBundle

TIME_COLUMN = "t_ns"
DENSE_GAP_NS = 200_000_000
NS_PER_S = 1_000_000_000


@dataclasses.dataclass(frozen=True)
class PoseSource:
    """The ground-truth stream of an episode: its kind (tf, tf_sparse or absent), sample count and median gap."""

    kind: str
    samples: int
    median_gap_s: float | None


def resolve_pose_source(bundle: TopicBundle, window: tuple[int | None, int | None]) -> tuple[pl.DataFrame | None, PoseSource]:
    """The ground-truth track inside the window, kept as recorded, with how it was classified."""
    track = _windowed(bundle.robot_tracks, window)
    if track is None or track.height == 0:
        return None, PoseSource("absent", 0, None)
    gap_ns = None if track.height < 2 else _median_gap_ns(track)
    dense = gap_ns is not None and gap_ns <= DENSE_GAP_NS
    return track, PoseSource(
        "tf" if dense else "tf_sparse",
        track.height,
        None if gap_ns is None else gap_ns / NS_PER_S,
    )


def pose_frame(
    transforms: pl.DataFrame | pl.LazyFrame | None,
    window: tuple[int | None, int | None],
) -> str | None:
    """The TF parent the robot pose came from: the most frequent parent_frame inside the episode window."""
    frame = _windowed(transforms, window)
    if frame is None:
        return None
    parents = frame.get_column("parent_frame").drop_nulls()
    return None if parents.len() == 0 else str(parents.mode().sort()[0])


def _median_gap_ns(frame: pl.DataFrame) -> float:
    stamps = np.sort(frame.get_column(TIME_COLUMN).to_numpy())
    return float(np.median(np.diff(stamps)))


def _windowed(
    frame: pl.DataFrame | pl.LazyFrame | None,
    window: tuple[int | None, int | None],
) -> pl.DataFrame | None:
    """The collected frame restricted to the window, or None when there is no frame."""
    if frame is None:
        return None
    kept = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    start_ns, end_ns = window
    if start_ns is not None:
        kept = kept.filter(pl.col(TIME_COLUMN) >= start_ns)
    if end_ns is not None:
        kept = kept.filter(pl.col(TIME_COLUMN) <= end_ns)
    return kept
