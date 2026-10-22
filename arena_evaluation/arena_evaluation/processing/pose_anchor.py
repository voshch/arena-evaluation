"""Rigid anchoring of the dense odom track onto a sparse map-frame pose stream."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import polars as pl

from .pose_segments import teleport_jumps

if typing.TYPE_CHECKING:
    from ..storage.schemas import TopicBundle

DENSE_GAP_NS = 200_000_000
PAIR_TOLERANCE_NS = 50_000_000
MAX_RESIDUAL_M = 0.5
MIN_SEGMENT_M = 0.2


@dataclasses.dataclass(frozen=True)
class RigidFit:
    """Planar transform (yaw, translation) from the odom frame to the world frame."""

    theta: float
    tx: float
    ty: float
    residual_m: float
    n: int


@dataclasses.dataclass(frozen=True)
class PoseSource:
    """Which pose stream an episode was evaluated on."""

    kind: str  # tf | anchored | tf_sparse | odom
    samples: int
    residual_m: float | None


def _collect(frame: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _windowed(frame: pl.DataFrame | None, window: tuple[int | None, int | None]) -> pl.DataFrame | None:
    """Rows inside the episode window, on log time."""
    if frame is None:
        return None
    start_ns, end_ns = window
    if start_ns is not None:
        frame = frame.filter(pl.col("time_ns") >= start_ns)
    if end_ns is not None:
        frame = frame.filter(pl.col("time_ns") <= end_ns)
    return frame


def _wrap(yaw: np.ndarray) -> np.ndarray:
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def _stamps(df: pl.DataFrame, col: str) -> pl.Series:
    """Header stamps, log time for frames recorded without them."""
    return df[col] if col in df.columns else df["time_ns"]


_ODOM_POSE = ("pos_x", "pos_y", "yaw")
_GT_POSE = ("pos_x_gt", "pos_y_gt", "yaw_gt")


def longest_segment(odom: pl.DataFrame | pl.LazyFrame | None) -> tuple[int, int]:
    """Row bounds [lo, hi) of the longest teleport-free run of the odom track."""
    df = _collect(odom)
    if df is None:
        return 0, 0
    if len(df) < 2:
        return 0, len(df)

    x = df["pos_x"].to_numpy()
    y = df["pos_y"].to_numpy()
    jumps = teleport_jumps(x, y, _stamps(df, "stamp_ns").to_numpy())
    if len(jumps) == 0:
        return 0, len(df)

    bounds = [0, *(int(j) + 1 for j in jumps), len(df)]
    best = (0, len(df))
    best_len = -1.0
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        seg_len = float(np.sum(np.hypot(np.diff(x[lo:hi]), np.diff(y[lo:hi])))) if hi - lo >= 2 else 0.0
        if seg_len >= MIN_SEGMENT_M and seg_len > best_len:
            best_len = seg_len
            best = (lo, hi)
    return best


def pair_samples(
    odom: pl.DataFrame | pl.LazyFrame | None,
    tf_gt: pl.DataFrame | pl.LazyFrame | None,
    lo: int,
    hi: int,
    tol_ns: int = PAIR_TOLERANCE_NS,
) -> pl.DataFrame:
    """Ground-truth samples joined to the nearest odom row by header stamp, odom restricted to [lo, hi)."""
    empty = pl.DataFrame(schema={"stamp_ns_gt": pl.Int64, "pos_x_gt": pl.Float64, "pos_y_gt": pl.Float64, "yaw_gt": pl.Float64, "stamp_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64, "yaw": pl.Float64})
    od_df = _collect(odom)
    gt_df = _collect(tf_gt)
    if od_df is None or gt_df is None or not set(_ODOM_POSE) <= set(od_df.columns) or not set(_GT_POSE) <= set(gt_df.columns):
        return empty

    od = od_df.slice(lo, hi - lo).select(_stamps(od_df, "stamp_ns").slice(lo, hi - lo).alias("stamp_ns"), *_ODOM_POSE).sort("stamp_ns")
    gt = gt_df.select(_stamps(gt_df, "stamp_ns_gt").alias("stamp_ns_gt"), *_GT_POSE).sort("stamp_ns_gt")
    if len(od) == 0 or len(gt) == 0:
        return empty

    paired = gt.join_asof(od, left_on="stamp_ns_gt", right_on="stamp_ns", strategy="nearest", tolerance=tol_ns)
    return paired.drop_nulls(["pos_x", "pos_y", "yaw"])


def fit_rigid(pairs: pl.DataFrame) -> RigidFit:
    """Yaw and translation carrying the paired odom poses onto their ground-truth poses."""
    n = len(pairs)
    if n == 0:
        return RigidFit(0.0, 0.0, 0.0, 0.0, 0)

    x = pairs["pos_x"].to_numpy()
    y = pairs["pos_y"].to_numpy()
    yaw = pairs["yaw"].to_numpy()
    gx = pairs["pos_x_gt"].to_numpy()
    gy = pairs["pos_y_gt"].to_numpy()
    gyaw = pairs["yaw_gt"].to_numpy()

    dyaw = gyaw - yaw
    theta = float(np.arctan2(np.mean(np.sin(dyaw)), np.mean(np.cos(dyaw))))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rx = cos_t * x - sin_t * y
    ry = sin_t * x + cos_t * y
    tx = float(np.mean(gx - rx))
    ty = float(np.mean(gy - ry))
    residual = 0.0 if n == 1 else float(np.max(np.hypot(gx - (rx + tx), gy - (ry + ty))))
    return RigidFit(theta, tx, ty, residual, n)


def anchor_odom(odom: pl.DataFrame | pl.LazyFrame, fit: RigidFit) -> pl.DataFrame:
    """The odom track mapped through ``fit``, in the schema of an extracted tf_gt frame."""
    df = _collect(odom)
    assert df is not None
    x = df["pos_x"].to_numpy()
    y = df["pos_y"].to_numpy()
    cos_t, sin_t = np.cos(fit.theta), np.sin(fit.theta)
    return pl.DataFrame(
        {
            "time_ns": df["time_ns"],
            "stamp_ns_gt": _stamps(df, "stamp_ns"),
            "pos_x_gt": fit.tx + cos_t * x - sin_t * y,
            "pos_y_gt": fit.ty + sin_t * x + cos_t * y,
            "yaw_gt": _wrap(df["yaw"].to_numpy() + fit.theta),
            "frame_id": pl.Series("frame_id", ["anchored"] * len(df), dtype=pl.String),
        }
    )


def classify(tf_gt: pl.DataFrame | pl.LazyFrame | None) -> str:
    """dense | sparse | absent, on the median gap between ground-truth header stamps."""
    df = _collect(tf_gt)
    if df is None or len(df) == 0:
        return "absent"
    if len(df) < 2:
        return "sparse"
    gaps = np.diff(np.sort(_stamps(df, "stamp_ns_gt").to_numpy()))
    return "dense" if float(np.median(gaps)) <= DENSE_GAP_NS else "sparse"  # pyright: ignore[reportCallIssue, reportArgumentType]


def resolve_pose_source(bundle: TopicBundle, window: tuple[int | None, int | None]) -> tuple[pl.DataFrame | None, PoseSource]:
    """The pose stream the episode is evaluated on, anchoring odom when ground truth is too sparse to measure."""
    tf_gt = _collect(bundle.tf_gt)
    in_window = _windowed(tf_gt, window)
    kind = classify(in_window)
    if in_window is None:
        return None, PoseSource("odom", 0, None)

    if kind == "dense":
        return tf_gt, PoseSource("tf", len(in_window), None)
    if kind == "absent":
        return None, PoseSource("odom", 0, None)

    odom = _collect(bundle.odom)
    if odom is None or len(odom) == 0:
        return None, PoseSource("odom", 0, None)

    lo, hi = longest_segment(odom)
    pairs = pair_samples(odom, in_window, lo, hi)
    if len(pairs) == 0:
        has_yaw = set(_GT_POSE) <= set(in_window.columns)
        return (None, PoseSource("odom", 0, None)) if has_yaw else (tf_gt, PoseSource("tf_sparse", len(in_window), None))

    fit = fit_rigid(pairs)
    if fit.n >= 2 and fit.residual_m > MAX_RESIDUAL_M:
        return tf_gt, PoseSource("tf_sparse", fit.n, fit.residual_m)
    return anchor_odom(odom, fit), PoseSource("anchored", fit.n, None if fit.n < 2 else fit.residual_m)
