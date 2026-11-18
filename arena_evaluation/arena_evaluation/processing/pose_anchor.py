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


def classify(track: pl.DataFrame | pl.LazyFrame | None) -> str:
    """dense | sparse | absent, on the median gap between the native time stamps of the track."""
    frame = _collect(track)
    if frame is None or frame.height == 0:
        return "absent"
    if frame.height < 2:
        return "sparse"
    return "dense" if _median_gap_ns(frame) <= DENSE_GAP_NS else "sparse"


def resolve_pose_source(bundle: TopicBundle, window: tuple[int | None, int | None]) -> tuple[pl.DataFrame | None, PoseSource]:
    """The ground-truth track inside the window, kept as recorded, with how it was classified."""
    track = _windowed(_collect(bundle.robot_tracks), window)
    kind = classify(track)
    if track is None or kind == "absent":
        return None, PoseSource("absent", 0, None)
    median_gap_s = None if track.height < 2 else _median_gap_ns(track) / NS_PER_S
    return track, PoseSource("tf" if kind == "dense" else "tf_sparse", track.height, median_gap_s)


def _median_gap_ns(frame: pl.DataFrame) -> float:
    stamps = np.sort(frame.get_column(TIME_COLUMN).to_numpy())
    return float(np.median(np.diff(stamps)))


def _collect(frame: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    if frame is None:
        return None
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _windowed(frame: pl.DataFrame | None, window: tuple[int | None, int | None]) -> pl.DataFrame | None:
    if frame is None:
        return None
    start_ns, end_ns = window
    kept = frame
    if start_ns is not None:
        kept = kept.filter(pl.col(TIME_COLUMN) >= start_ns)
    if end_ns is not None:
        kept = kept.filter(pl.col(TIME_COLUMN) <= end_ns)
    return kept
