"""Shared flat-telemetry trajectory, encounter and pedestrian-track helpers.

Unmeasured kinematics stay NaN: a missing velocity is never a zero, and an empty trajectory
resamples to an undefined position rather than the origin.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.common.alignment import interpolate_values
from arena_evaluation.processing.metrics.ecological.mechanism_common import entity_name
from arena_evaluation.processing.windows import (
    ACTOR_PEDESTRIAN,
    DEFAULT_DETECTOR_PARAMS,
    ENCOUNTER_EDGE_DISTANCE_M,
    ENCOUNTER_MERGE_GAP_S,
    DetectorParams,
    detect_encounter_windows_lf,
    empty_encounter_windows,
    fleet_pairs_lf,
    pedestrian_pairs_lf,
)

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

DEFAULT_PED_RADIUS_M: Final[float] = 0.3
PEDESTRIAN_ID_PREFIXES: Final[tuple[str, ...]] = ("ped", "pedestrian")
UNIDENTIFIED_ROBOT_ID: Final[str] = "robot"
NS_PER_S: Final[float] = 1e9
ROBOT_TRACKS_TOPIC: Final[str] = "robot_tracks"
PEDESTRIAN_TRACKS_TOPIC: Final[str] = "pedestrian_tracks"
PEDESTRIAN_PHYSICS_TOPIC: Final[str] = "pedestrian_tracks_physics"

NATIVE_ROBOT_COLUMNS: Final[tuple[str, ...]] = ("t_ns", "x", "y", "yaw", "vx", "vy")


def native_robot_frame(episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
    """The bundle's own robot track, one row per sample: nanosecond time, planar pose and planar velocity."""
    tracks = episode.topics.get(ROBOT_TRACKS_TOPIC) if episode.topics else None
    if tracks is None:
        return None
    return (
        tracks.select(
            pl.col("t_ns").cast(pl.Int64),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("yaw").cast(pl.Float64),
            pl.col("vx").cast(pl.Float64),
            pl.col("vy").cast(pl.Float64),
        )
        .drop_nulls(list(NATIVE_ROBOT_COLUMNS))
        .sort("t_ns")
    )


@dataclass(frozen=True, slots=True)
class Trajectory2D:
    """Planar trajectory (x, y, vx, vy) on a sorted nanosecond time axis."""

    t_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    radius_m: float = DEFAULT_PED_RADIUS_M

    @property
    def xy(self) -> np.ndarray:
        """Positions as an (N, 2) float64 array."""
        return np.column_stack((self.x, self.y))

    @property
    def vel(self) -> np.ndarray:
        """Velocities as an (N, 2) float64 array."""
        return np.column_stack((self.vx, self.vy))

    @property
    def times_s(self) -> np.ndarray:
        """Timestamps in seconds."""
        return self.t_ns.astype(np.float64) / NS_PER_S

    def slice_ns(self, t_start_ns: int, t_end_ns: int) -> Trajectory2D:
        """Sub-trajectory within [t_start_ns, t_end_ns]; the nearest sample when the window holds none."""
        mask = (self.t_ns >= t_start_ns) & (self.t_ns <= t_end_ns)
        if not np.any(mask) and len(self.t_ns) > 0:
            nearest = int(np.argmin(np.abs(self.t_ns - t_start_ns)))
            mask = np.zeros(len(self.t_ns), dtype=bool)
            mask[nearest] = True
        return Trajectory2D(
            t_ns=self.t_ns[mask],
            x=self.x[mask],
            y=self.y[mask],
            vx=self.vx[mask],
            vy=self.vy[mask],
            radius_m=self.radius_m,
        )

    def resample_onto(self, query_time_ns: np.ndarray) -> Trajectory2D:
        """Linearly interpolate trajectory onto query_time_ns; an empty source stays undefined (NaN), never the origin."""
        if len(self.t_ns) == 0 or len(query_time_ns) == 0:
            undefined = np.full(len(query_time_ns), np.nan)
            return Trajectory2D(query_time_ns, undefined, undefined.copy(), undefined.copy(), undefined.copy(), self.radius_m)
        t_src = self.t_ns.astype(np.float64)
        t_dst = query_time_ns.astype(np.float64)
        return Trajectory2D(
            t_ns=query_time_ns,
            x=interpolate_values(t_src, self.x, t_dst),
            y=interpolate_values(t_src, self.y, t_dst),
            vx=interpolate_values(t_src, self.vx, t_dst),
            vy=interpolate_values(t_src, self.vy, t_dst),
            radius_m=self.radius_m,
        )


@dataclass(frozen=True, slots=True)
class PedTrack:
    """Identified pedestrian with its 2D trajectory."""

    ped_id: str
    ped_name: str
    trajectory: Trajectory2D
    radius_m: float | None = None
    desired_speed: float | None = None

    @property
    def t_ns(self) -> np.ndarray:
        """Sorted nanosecond timestamps."""
        return self.trajectory.t_ns

    @property
    def x(self) -> np.ndarray:
        """Planar X coordinates in meters."""
        return self.trajectory.x

    @property
    def y(self) -> np.ndarray:
        """Planar Y coordinates in meters."""
        return self.trajectory.y

    @property
    def vx(self) -> np.ndarray:
        """Planar X velocities in m/s."""
        return self.trajectory.vx

    @property
    def vy(self) -> np.ndarray:
        """Planar Y velocities in m/s."""
        return self.trajectory.vy


def episode_table(episode: AlignedEpisodeBundle, topic: str) -> pl.DataFrame | None:
    """Collected flat table of an episode topic, None when the recording carries none."""
    topics = episode.topics
    if topics is None:
        return None
    return topics.get(topic)


def resolve_robot_trajectory(calc: BaseMetricCalculator, episode: AlignedEpisodeBundle) -> tuple[Trajectory2D, np.ndarray]:
    """Robot ground-truth Trajectory2D and yaw array from the flat robot_tracks table.

    A missing velocity or yaw sample stays NaN: the recording holds no measurement for it.
    """
    radius_m = float(calc.robot_params.robot_radius)
    empty = np.array([], dtype=np.float64)
    empty_trajectory = Trajectory2D(t_ns=np.array([], dtype=np.int64), x=empty, y=empty.copy(), vx=empty.copy(), vy=empty.copy(), radius_m=radius_m)
    tracks = episode_table(episode, ROBOT_TRACKS_TOPIC)
    if tracks is None:
        return empty_trajectory, empty

    rows = tracks.select("t_ns", "x", "y", "yaw", "vx", "vy").drop_nulls(["t_ns", "x", "y"]).sort("t_ns")
    if len(rows) == 0:
        return empty_trajectory, empty

    return (
        Trajectory2D(
            t_ns=rows["t_ns"].to_numpy().astype(np.int64),
            x=rows["x"].to_numpy().astype(np.float64),
            y=rows["y"].to_numpy().astype(np.float64),
            vx=_measured(rows["vx"].to_numpy().astype(np.float64)),
            vy=_measured(rows["vy"].to_numpy().astype(np.float64)),
            radius_m=radius_m,
        ),
        rows["yaw"].to_numpy().astype(np.float64),
    )


def extract_episode_ped_tracks(episode: AlignedEpisodeBundle) -> dict[str, PedTrack]:
    """Pedestrian tracks of the episode keyed by canonical ped_id, read from the flat pedestrian_tracks table."""
    return extract_ped_tracks(episode_table(episode, PEDESTRIAN_TRACKS_TOPIC))


def extract_ped_tracks(peds: pl.DataFrame | None) -> dict[str, PedTrack]:
    """Pedestrian tracks keyed by canonical ped_id from a flat pedestrian_tracks frame."""
    if peds is None or len(peds) == 0:
        return {}

    rows = peds.select("t_ns", "ped_id", "ped_name", "x", "y", "vx", "vy", "radius_m", "desired_speed").drop_nulls(["t_ns", "ped_id"]).filter(pl.col("x").is_finite() & pl.col("y").is_finite()).sort("t_ns")
    if len(rows) == 0:
        return {}

    return {str(frame["ped_id"][0]): _track_from_frame(frame) for frame in rows.partition_by("ped_id", maintain_order=True)}


def resolve_ped_track(tracks: Mapping[str, PedTrack], target_agent_id: int | str | None, target_agent_name: str | None = None) -> PedTrack | None:
    """Track addressed by a configured agent id (canonical ped_id, name, numeric or ``ped_<n>`` alias), then by name."""
    if target_agent_id is not None:
        wanted = _identity_aliases(str(target_agent_id))
        for track in tracks.values():
            if wanted & _identity_aliases(track.ped_id) or wanted & _identity_aliases(track.ped_name):
                return track
    if target_agent_name is not None:
        wanted_name = target_agent_name.strip().lower()
        for track in tracks.values():
            if track.ped_name.strip().lower() == wanted_name or track.ped_id.strip().lower() == wanted_name:
                return track
    return None


def _identity_aliases(value: str) -> set[str]:
    """Case-insensitive identity tokens of a pedestrian id/name; another actor kind never gains a ``ped_<n>`` alias."""
    normalized = value.strip().lower()
    if normalized.isdigit():
        return {normalized, f"ped_{normalized}", f"pedestrian_{normalized}"}
    prefix, separator, suffix = normalized.rpartition("_")
    if separator and suffix.isdigit() and prefix in PEDESTRIAN_ID_PREFIXES:
        return {normalized, suffix, f"ped_{suffix}", f"pedestrian_{suffix}"}
    return {normalized}


def detector_params_from_config(metrics_config: Mapping[str, typing.Any]) -> DetectorParams:
    """Detector parameters of a metrics config, only the edge distance and merge gap being metric-configurable."""
    return DetectorParams(
        encounter_edge_distance_m=float(metrics_config.get("encounter_edge_distance_m", ENCOUNTER_EDGE_DISTANCE_M)),
        encounter_merge_gap_s=float(metrics_config.get("encounter_merge_gap_s", ENCOUNTER_MERGE_GAP_S)),
    )


def episode_encounter_windows(
    episode: AlignedEpisodeBundle,
    robot_radius_m: float,
    params: DetectorParams = DEFAULT_DETECTOR_PARAMS,
) -> pl.DataFrame:
    """Encounter windows of the episode robot with its pedestrians and fleet robots from ``windows.py``."""
    robot_tracks = episode_table(episode, ROBOT_TRACKS_TOPIC)
    if robot_tracks is None or len(robot_tracks) == 0:
        return empty_encounter_windows()
    robot = robot_tracks.select(
        pl.col("t_ns").cast(pl.Int64),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("yaw").cast(pl.Float64),
        pl.col("vx").cast(pl.Float64),
        pl.col("vy").cast(pl.Float64),
    ).with_columns(robot_id=pl.lit(episode.robot_name or UNIDENTIFIED_ROBOT_ID))
    pairs: list[pl.LazyFrame] = []
    pedestrian_tracks = episode_table(episode, PEDESTRIAN_TRACKS_TOPIC)
    if pedestrian_tracks is not None and len(pedestrian_tracks) > 0:
        pairs.append(pedestrian_pairs_lf(robot, pedestrian_tracks, robot_radius_m, params))
    if episode.fleet:
        pairs.append(fleet_pairs_lf(robot, episode.fleet, robot_radius_m, params))
    if not pairs:
        return empty_encounter_windows()
    pairs_lf = pairs[0] if len(pairs) == 1 else pl.concat(pairs)
    return detect_encounter_windows_lf(pairs_lf, params).collect()


def pedestrian_encounter_spans(
    episode: AlignedEpisodeBundle,
    robot_radius_m: float,
    params: DetectorParams = DEFAULT_DETECTOR_PARAMS,
) -> list[tuple[str, int, int]]:
    """(pedestrian id, start, end) of every pedestrian encounter window, robot actors excluded."""
    windows = episode_encounter_windows(episode, robot_radius_m, params)
    pedestrian = windows.filter(pl.col("actor_kind") == ACTOR_PEDESTRIAN)
    return list(
        zip(
            pedestrian.get_column("actor_id").to_numpy(),
            pedestrian.get_column("t_start_ns").to_numpy().astype(np.int64),
            pedestrian.get_column("t_end_ns").to_numpy().astype(np.int64),
            strict=True,
        )
    )


def _track_from_frame(frame: pl.DataFrame) -> PedTrack:
    """PedTrack of one pedestrian's frames, in time order."""
    ped_id = str(frame["ped_id"][0])
    t_ns = frame["t_ns"].to_numpy().astype(np.int64)
    x = frame["x"].to_numpy().astype(np.float64)
    y = frame["y"].to_numpy().astype(np.float64)
    vx = _measured(frame["vx"].to_numpy().astype(np.float64))
    vy = _measured(frame["vy"].to_numpy().astype(np.float64))

    twist_is_zero = not bool(np.any(np.hypot(vx, vy) > 1e-9))
    position_moved = len(x) >= 2 and bool(np.any(np.hypot(np.diff(x), np.diff(y)) > 1e-6))
    if twist_is_zero and position_moved:
        vx, vy = BaseMetricCalculator.velocity_from_pose(x, y, t_ns)
        vx[0], vy[0] = np.nan, np.nan

    radius_m = _first_float(frame["radius_m"])
    if radius_m is None or not np.isfinite(radius_m) or radius_m <= 0.0:
        radius_m = None
    effective_radius_m = radius_m if radius_m is not None else DEFAULT_PED_RADIUS_M
    trajectory = Trajectory2D(t_ns=t_ns, x=x, y=y, vx=vx, vy=vy, radius_m=effective_radius_m)
    return PedTrack(
        ped_id=ped_id,
        ped_name=_first_text(frame["ped_name"], ped_id),
        trajectory=trajectory,
        radius_m=radius_m,
        desired_speed=_first_float(frame["desired_speed"]),
    )


def _first_text(series: pl.Series, default: str) -> str:
    """First non-empty authored name of a text series, the entity-stripped default when every value is blank."""
    for value in series.drop_nulls().to_numpy():
        text = entity_name(str(value).strip())
        if text:
            return text
    return entity_name(default.strip()) or default


def _measured(values: np.ndarray) -> np.ndarray:
    """Velocity samples with a non-finite entry published as no measurement (NaN), never as a zero."""
    return np.where(np.isfinite(values), values, np.nan)


def _first_float(series: pl.Series) -> float | None:
    """First non-null value of a float series, None when every value is null."""
    present = series.drop_nulls()
    return float(present[0]) if len(present) > 0 else None


def contiguous_spans(mask: np.ndarray, t_ns: np.ndarray, merge_gap_ns: int) -> list[tuple[int, int]]:
    """Inclusive index spans of the True runs of a mask, bridging gaps of at most merge_gap_ns."""
    hits = np.flatnonzero(mask)
    if len(hits) == 0:
        return []
    breaks = np.flatnonzero(np.diff(t_ns[hits]) > merge_gap_ns) + 1
    edges = np.concatenate(([0], breaks, [len(hits)]))
    return [(int(hits[start]), int(hits[end - 1])) for start, end in zip(edges[:-1], edges[1:], strict=True)]


def merge_intervals_ns(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent [t_start_ns, t_end_ns] intervals into disjoint sorted intervals."""
    valid = sorted((int(start), int(end)) for start, end in intervals if end >= start)
    if not valid:
        return []
    merged: list[tuple[int, int]] = [valid[0]]
    for start, end in valid[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged
