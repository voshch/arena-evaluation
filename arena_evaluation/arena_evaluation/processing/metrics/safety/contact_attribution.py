"""Contact attribution as measured scalars: closing speeds, impact speeds and frozen-robot clearance."""

from __future__ import annotations

import re
import typing
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.ecological.mechanism_common import entity_name
from arena_evaluation.processing.metrics.social.interaction_common import (
    NS_PER_S,
    ROBOT_TRACKS_TOPIC,
    PedTrack,
    Trajectory2D,
    contiguous_spans,
    episode_table,
    extract_episode_ped_tracks,
    merge_intervals_ns,
    resolve_ped_track,
    resolve_robot_trajectory,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import (
    Metric,
    MetricScope,
    NullReason,
    define_metric,
)

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PEDESTRIAN_KIND: Final[str] = "pedestrian"
ROBOT_KIND: Final[str] = "robot"
#: Collider kinds that move on their own and whose speed is therefore recorded on a track.
TRACKED_KINDS: Final[frozenset[str]] = frozenset({PEDESTRIAN_KIND, ROBOT_KIND})
PEDESTRIAN_KIND_PATTERN: Final[str] = "ped|human"
ROBOT_COLLISION_CONTACTS_TOPIC: Final[str] = "robot_collision_contacts"
ROBOT_COLLISION_EVENTS_TOPIC: Final[str] = "robot_collision_events"
PRE_CONTACT_WINDOW_S: Final[float] = 0.5
POST_CONTACT_WINDOW_S: Final[float] = 0.5
#: Wall-clock gap, seconds, that still merges two recorded contact samples of the same target into one contact.
CONTACT_MERGE_GAP_S: Final[float] = 0.25


def _metric(
    name: str,
    dtype: pl.DataType | type[pl.DataType],
    scope: MetricScope,
    unit: str,
    description: str,
) -> Metric:
    return define_metric(
        name,
        dtype,
        scopes=(scope,),
        unit=unit,
        description=description,
        null_reason=True,
    )


V_CLOSE_ROBOT = _metric(
    "v_close_robot_mps",
    pl.Float64,
    MetricScope.CONTACT,
    "m/s",
    "Mean part of the robot's velocity pointing at the other body along the line of sight over the 0.5 s before contact; positive when the robot closes in.",
)
V_CLOSE_TARGET = _metric(
    "v_close_target_mps",
    pl.Float64,
    MetricScope.CONTACT,
    "m/s",
    "Mean part of the other body's velocity pointing at the robot along the line of sight over the 0.5 s before contact; positive when the other body closes in.",
)
V_ROBOT_IMPACT = _metric(
    "v_robot_impact_mps",
    pl.Float64,
    MetricScope.CONTACT,
    "m/s",
    "Robot speed at contact onset.",
)
V_TARGET_IMPACT = _metric(
    "v_target_impact_mps",
    pl.Float64,
    MetricScope.CONTACT,
    "m/s",
    "Speed of the other body at contact onset; zero for a wall or static collider.",
)
RELATIVE_IMPACT_SPEED = _metric(
    "relative_impact_speed_mps",
    pl.Float64,
    MetricScope.CONTACT,
    "m/s",
    "Relative speed of the robot and the other body at contact onset.",
)
FREEZE_CLEARANCE = _metric(
    "freeze_clearance_m",
    pl.Float64,
    MetricScope.CONTACT,
    "m",
    "Closest approach of the pedestrian's recorded path to the robot frozen at t0 minus the pre-contact window, over [t0 - pre, t0 + post] (0.5 s each by default), minus both body radii; zero or negative means the contact would have happened anyway.",
)
CONTACT_COUNT = _metric(
    "contact_count",
    pl.Int64,
    MetricScope.EPISODE,
    "count",
    "Number of contacts detected in the episode; NULL when no collision stream was recorded and no contact could be detected.",
)
CONTACT_COUNT_PEDESTRIAN = _metric(
    "contact_count_pedestrian",
    pl.Int64,
    MetricScope.EPISODE,
    "count",
    "Contacts of the episode whose other body is a pedestrian; from the same detector as contact_count.",
)
CONTACT_COUNT_ROBOT = _metric(
    "contact_count_robot",
    pl.Int64,
    MetricScope.EPISODE,
    "count",
    "Contacts of the episode whose other body is another robot; from the same detector as contact_count.",
)
CONTACT_COUNT_WALL = _metric(
    "contact_count_wall",
    pl.Int64,
    MetricScope.EPISODE,
    "count",
    "Contacts of the episode with a wall; from the same detector as contact_count.",
)
CONTACT_COUNT_STATIC = _metric(
    "contact_count_static",
    pl.Int64,
    MetricScope.EPISODE,
    "count",
    "Contacts of the episode with a static obstacle; from the same detector as contact_count.",
)
CONTACT_COUNT_BY_KIND: Final[tuple[tuple[str, Metric], ...]] = (
    (PEDESTRIAN_KIND, CONTACT_COUNT_PEDESTRIAN),
    (ROBOT_KIND, CONTACT_COUNT_ROBOT),
    ("wall", CONTACT_COUNT_WALL),
    ("static", CONTACT_COUNT_STATIC),
)
_SCALAR_METRICS: Final[tuple[Metric, ...]] = (
    V_CLOSE_ROBOT,
    V_CLOSE_TARGET,
    V_ROBOT_IMPACT,
    V_TARGET_IMPACT,
    RELATIVE_IMPACT_SPEED,
    FREEZE_CLEARANCE,
)
_COUNT_METRICS: Final[tuple[Metric, ...]] = (
    CONTACT_COUNT,
    *(metric for _, metric in CONTACT_COUNT_BY_KIND),
)
_SCALAR_NAMES: Final[tuple[str, ...]] = tuple(metric.name for metric in _SCALAR_METRICS)
_COUNT_NAMES: Final[tuple[str, ...]] = tuple(metric.name for metric in _COUNT_METRICS)


def _undefined_metrics(names: Iterable[str], reason: str) -> dict[str, typing.Any]:
    """Every named metric NULL with one shared reason and its null-reason companion."""
    undefined: dict[str, typing.Any] = dict.fromkeys(names)
    undefined.update({f"{name}_null_reason": reason for name in undefined})
    return undefined


@dataclass(frozen=True, slots=True)
class ContactEvent:
    """Detected contact interval before kinematic attribution."""

    t0_ns: int
    t_end_ns: int
    obstacle_kind: str
    obstacle_id: str
    track: PedTrack | None = None
    static_point: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class ContactContext:
    """Episode-level context needed to attribute a ContactEvent."""

    robot: Trajectory2D
    pre_window_s: float = PRE_CONTACT_WINDOW_S
    post_window_s: float = POST_CONTACT_WINDOW_S


@dataclass(frozen=True, slots=True)
class ContactAttributionRecord:
    """Measured attribution scalars of a single contact event.

    The two closing shares are views of ``v_close_robot_mps`` and ``v_close_target_mps``: computed
    properties, not stored fields, so the ratio is never duplicated in storage.
    """

    contact_id: int
    t0_ns: int
    t_end_ns: int
    t0_s: float
    duration_s: float
    obstacle_kind: str
    obstacle_id: str
    target_ped_id: str | None
    target_ped_name: str | None
    v_close_robot_mps: float | None
    v_close_target_mps: float | None
    v_robot_impact_mps: float | None
    v_target_impact_mps: float | None
    relative_impact_speed_mps: float | None
    freeze_clearance_m: float | None

    @property
    def robot_closing_share(self) -> float | None:
        """Robot share of the closing speed, a view of the two stored closing speeds."""
        return closing_shares(self.v_close_robot_mps, self.v_close_target_mps)[0]

    @property
    def target_closing_share(self) -> float | None:
        """Target share of the closing speed, one minus the robot share."""
        return closing_shares(self.v_close_robot_mps, self.v_close_target_mps)[1]


def closing_shares(
    v_close_robot_mps: float | None,
    v_close_target_mps: float | None,
) -> tuple[float | None, float | None]:
    """Robot and target closing shares in 0..1; both None when a speed is unmeasured or no body closes."""
    if v_close_robot_mps is None or v_close_target_mps is None or not np.isfinite(v_close_robot_mps) or not np.isfinite(v_close_target_mps):
        return None, None
    closing_robot = max(0.0, float(v_close_robot_mps))
    closing_target = max(0.0, float(v_close_target_mps))
    total = closing_robot + closing_target
    if total <= 1e-9:
        return None, None
    share = closing_robot / total
    return float(share), float(1.0 - share)


def _finite_or_none(value: float) -> float | None:
    """The value when it is a finite measurement, None when the quantity is undefined."""
    return float(value) if np.isfinite(value) else None


def _sample_at(trajectory: Trajectory2D, t_ns: int) -> Trajectory2D:
    """The trajectory linearly interpolated at one nanosecond stamp."""
    return trajectory.resample_onto(np.asarray([t_ns], dtype=np.int64))


def _closing_components(
    dx: np.ndarray,
    dy: np.ndarray,
    robot: Trajectory2D,
    other_vx: np.ndarray | None = None,
    other_vy: np.ndarray | None = None,
) -> tuple[float, float]:
    """Mean robot and other-body closing speed along the line of sight of each valid sample."""
    distance = np.hypot(dx, dy)
    valid = (distance > 1e-6) & np.isfinite(distance) & np.isfinite(robot.vx) & np.isfinite(robot.vy)
    if other_vx is not None and other_vy is not None:
        valid &= np.isfinite(other_vx) & np.isfinite(other_vy)
    if not np.any(valid):
        return float("nan"), float("nan")
    ux, uy = dx[valid] / distance[valid], dy[valid] / distance[valid]
    robot_closing = float(np.mean(robot.vx[valid] * ux + robot.vy[valid] * uy))
    target_closing = float(-np.mean(other_vx[valid] * ux + other_vy[valid] * uy)) if other_vx is not None and other_vy is not None else 0.0
    return robot_closing, target_closing


def compute_closing_speeds(event: ContactEvent, context: ContactContext) -> tuple[float, float]:
    """Mean robot and target closing speeds (m/s) over [t0 - pre_window_s, t0]; NaN when unmeasured."""
    win_start_ns = int(event.t0_ns - round(context.pre_window_s * NS_PER_S))
    robot = context.robot.slice_ns(win_start_ns, event.t0_ns)
    if len(robot.t_ns) == 0:
        return float("nan"), float("nan")

    track = event.track
    covers_onset = track is not None and len(track.t_ns) > 0 and int(track.t_ns[0]) <= event.t0_ns <= int(track.t_ns[-1])
    if covers_onset:
        pedestrian = track.trajectory.resample_onto(robot.t_ns)
        inside = (robot.t_ns >= track.t_ns[0]) & (robot.t_ns <= track.t_ns[-1])
        return _closing_components(
            np.where(inside, pedestrian.x - robot.x, np.nan),
            np.where(inside, pedestrian.y - robot.y, np.nan),
            robot,
            pedestrian.vx,
            pedestrian.vy,
        )

    target_closing = 0.0 if event.obstacle_kind not in TRACKED_KINDS else float("nan")
    if event.static_point is None:
        return float("nan"), target_closing
    robot_closing, _ = _closing_components(
        float(event.static_point[0]) - robot.x,
        float(event.static_point[1]) - robot.y,
        robot,
    )
    return robot_closing, target_closing


def attribute_single_contact(
    contact_id: int,
    event: ContactEvent,
    context: ContactContext,
) -> ContactAttributionRecord:
    """Compute the attribution scalars of a single ContactEvent."""
    v_close_r, v_close_t = compute_closing_speeds(event, context)
    at_t0 = _sample_at(context.robot, event.t0_ns)
    rvx_0, rvy_0 = float(at_t0.vx[0]), float(at_t0.vy[0])
    v_robot_impact = float(np.hypot(rvx_0, rvy_0))
    track = event.track

    if event.obstacle_kind in TRACKED_KINDS:
        v_target_impact = float("nan")
        rel_impact = float("nan")
        if track is not None and len(track.t_ns) > 0 and int(track.t_ns[0]) <= event.t0_ns <= int(track.t_ns[-1]):
            ped_at_t0 = _sample_at(track.trajectory, event.t0_ns)
            pvx_0, pvy_0 = float(ped_at_t0.vx[0]), float(ped_at_t0.vy[0])
            v_target_impact = float(np.hypot(pvx_0, pvy_0))
            rel_impact = float(np.hypot(rvx_0 - pvx_0, rvy_0 - pvy_0))
    else:
        # A wall or static collider is at rest by definition, so the relative speed is the robot's.
        v_target_impact = 0.0
        rel_impact = v_robot_impact

    freeze_clearance_m: float | None = None
    if event.obstacle_kind == PEDESTRIAN_KIND and track is not None and track.radius_m is not None and np.isfinite(track.radius_m) and track.radius_m > 0.0 and len(track.t_ns) > 0 and len(context.robot.t_ns) > 0:
        win_start_ns = int(event.t0_ns - round(context.pre_window_s * NS_PER_S))
        win_end_ns = int(event.t0_ns + round(context.post_window_s * NS_PER_S))
        frozen = _sample_at(context.robot, max(int(context.robot.t_ns[0]), win_start_ns))
        freeze_x, freeze_y = float(frozen.x[0]), float(frozen.y[0])
        measured = (track.t_ns >= win_start_ns) & (track.t_ns <= win_end_ns) & np.isfinite(track.x) & np.isfinite(track.y)
        if np.isfinite(freeze_x) and np.isfinite(freeze_y) and np.any(measured):
            distances = np.hypot(track.x[measured] - freeze_x, track.y[measured] - freeze_y)
            freeze_clearance_m = float(np.min(distances)) - (float(context.robot.radius_m) + float(track.radius_m))

    ep_start_ns = int(context.robot.t_ns[0]) if len(context.robot.t_ns) else event.t0_ns
    return ContactAttributionRecord(
        contact_id=contact_id,
        t0_ns=int(event.t0_ns),
        t_end_ns=int(max(event.t0_ns, event.t_end_ns)),
        t0_s=float(event.t0_ns - ep_start_ns) / NS_PER_S,
        duration_s=float(max(0, event.t_end_ns - event.t0_ns)) / NS_PER_S,
        obstacle_kind=event.obstacle_kind,
        obstacle_id=event.obstacle_id,
        target_ped_id=event.track.ped_id if event.track is not None else None,
        target_ped_name=event.track.ped_name if event.track is not None else None,
        v_close_robot_mps=_finite_or_none(v_close_r),
        v_close_target_mps=_finite_or_none(v_close_t),
        v_robot_impact_mps=_finite_or_none(v_robot_impact),
        v_target_impact_mps=_finite_or_none(v_target_impact),
        relative_impact_speed_mps=_finite_or_none(rel_impact),
        freeze_clearance_m=freeze_clearance_m,
    )


def contact_scalars(record: ContactAttributionRecord, event: ContactEvent) -> dict[str, typing.Any]:
    """The six stored contact scalars plus a null reason beside each undefined one."""
    values = (
        record.v_close_robot_mps,
        record.v_close_target_mps,
        record.v_robot_impact_mps,
        record.v_target_impact_mps,
        record.relative_impact_speed_mps,
        record.freeze_clearance_m,
    )
    scalars: dict[str, typing.Any] = dict(zip(_SCALAR_NAMES, values, strict=True))
    scalars.update({f"{name}_null_reason": _null_reason(name, event) for name, value in zip(_SCALAR_NAMES, values, strict=True) if value is None})
    return scalars


def _null_reason(key: str, event: ContactEvent) -> str:
    """Why a contact scalar is NULL: inapplicable to the collider kind, an unrecorded radius, or no usable sample in the window."""
    if event.obstacle_kind == ROBOT_KIND and key != V_ROBOT_IMPACT.name:
        return NullReason.NOT_APPLICABLE.value
    if key == FREEZE_CLEARANCE.name:
        if event.obstacle_kind != PEDESTRIAN_KIND:
            return NullReason.NOT_APPLICABLE.value
        if event.track is not None and event.track.radius_m is None:
            return NullReason.NOT_RECORDED.value
    return NullReason.NO_SAMPLES.value


def static_point_of(obstacle_x: float | None, obstacle_y: float | None) -> tuple[float, float] | None:
    """Static obstacle position of a collision row; None when the recording carries none, only the (0, 0) sentinel or a non-finite value."""
    if obstacle_x is None or obstacle_y is None or not (np.isfinite(obstacle_x) and np.isfinite(obstacle_y)):
        return None
    point = (float(obstacle_x), float(obstacle_y))
    return None if point == (0.0, 0.0) else point


def _as_point(static_point: Sequence[float] | None) -> tuple[float, float] | None:
    """The two coordinates of a configured point, None when it carries fewer."""
    return None if static_point is None or len(static_point) < 2 else (float(static_point[0]), float(static_point[1]))


def _numeric_suffix(name: str) -> str | None:
    """Trailing number of an entity name (``Pedestrian_7`` -> ``7``), None when it has none."""
    digits = re.findall(r"\d+", name)
    return digits[-1] if digits else None


def _match_track(obstacle_id: str, tracks: Mapping[str, PedTrack], robot_at_t0: Trajectory2D) -> PedTrack | None:
    """Resolve the PedTrack of a contact: exact ped id/name, then numeric suffix, then nearest pose."""
    if not tracks:
        return None
    normalized = obstacle_id.strip()
    base = entity_name(normalized)
    if normalized:
        for track in tracks.values():
            if track.ped_id in (normalized, base) or track.ped_name in (
                normalized,
                base,
            ):
                return track
        suffix = _numeric_suffix(base)
        if suffix is not None:
            for key, track in tracks.items():
                if suffix in (_numeric_suffix(key), _numeric_suffix(track.ped_name)):
                    return track
    if len(robot_at_t0.t_ns) == 0:
        return None
    candidates = [track for track in tracks.values() if len(track.t_ns) > 0]
    if not candidates:
        return None
    robot_x, robot_y = float(robot_at_t0.x[0]), float(robot_at_t0.y[0])

    def distance(track: PedTrack) -> float:
        pedestrian = track.trajectory.resample_onto(robot_at_t0.t_ns)
        return float(np.hypot(pedestrian.x[0] - robot_x, pedestrian.y[0] - robot_y))

    return min(candidates, key=distance)


def _resolved_id(row: Mapping[str, typing.Any], robot: Trajectory2D, tracks: Mapping[str, PedTrack]) -> str:
    """Canonical ped_id of a contact row, or the raw id when no recorded track matches."""
    raw_id = str(row["obstacle_id"])
    if row["obstacle_kind"] != PEDESTRIAN_KIND:
        return raw_id
    track = _match_track(raw_id, tracks, _sample_at(robot, int(row["t_ns"])))
    return track.ped_id if track is not None else (raw_id or PEDESTRIAN_KIND)


def _contacts_from_table(
    *,
    cc_df: pl.DataFrame,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """ContactEvent rows grouped per resolved target from a flat robot_collision_contacts table."""
    raw_kind = pl.col("kind").fill_null("").str.to_lowercase()
    frame = cc_df.with_columns(
        pl.when(raw_kind == "").then(pl.lit("static")).when(raw_kind.str.contains(PEDESTRIAN_KIND_PATTERN)).then(pl.lit(PEDESTRIAN_KIND)).otherwise(raw_kind).alias("obstacle_kind"),
        pl.col("t_ns").cast(pl.Int64),
        pl.col("obstacle_id").fill_null("").alias("obstacle_id"),
        pl.col("obstacle_x").cast(pl.Float64),
        pl.col("obstacle_y").cast(pl.Float64),
    ).select("t_ns", "obstacle_id", "obstacle_kind", "obstacle_x", "obstacle_y")
    if not frame.is_empty() and tracks:
        frame = frame.with_columns(
            pl.Series(
                "obstacle_id",
                [
                    _resolved_id(row, robot, tracks)
                    # agent-contracts: ignore AC004 - matches contact rows against Python PedTrack dict
                    for row in frame.iter_rows(named=True)
                ],
                dtype=pl.String,
            )
        )
    grouped = (
        frame.with_columns((pl.col("obstacle_kind") + "|" + pl.col("obstacle_id")).alias("target_key"))
        .sort("target_key", "t_ns")
        .with_columns((~(pl.col("target_key").shift(1) == pl.col("target_key")).fill_null(False) | (pl.col("t_ns").diff() > merge_gap_ns).fill_null(True)).cum_sum().alias("group_id"))
        .group_by("group_id")
        .agg(
            t0_ns=pl.col("t_ns").min(),
            t_end_ns=pl.col("t_ns").max(),
            obstacle_kind=pl.col("obstacle_kind").first(),
            obstacle_id=pl.col("obstacle_id").first(),
            obstacle_x=pl.col("obstacle_x").first(),
            obstacle_y=pl.col("obstacle_y").first(),
        )
        .sort("t0_ns", "obstacle_kind", "obstacle_id")
    )
    return [
        ContactEvent(
            t0_ns=int(row["t0_ns"]),
            t_end_ns=int(row["t_end_ns"]),
            obstacle_kind=row["obstacle_kind"],
            obstacle_id=str(row["obstacle_id"]) or (PEDESTRIAN_KIND if row["obstacle_kind"] == PEDESTRIAN_KIND else ""),
            track=tracks.get(str(row["obstacle_id"])) if row["obstacle_kind"] == PEDESTRIAN_KIND else None,
            static_point=None if row["obstacle_kind"] in TRACKED_KINDS else static_point_of(row["obstacle_x"], row["obstacle_y"]),
        )
        # agent-contracts: ignore AC004 - materializes ContactEvent dataclasses
        for row in grouped.iter_rows(named=True)
    ]


def detect_contact_events(
    episode: AlignedEpisodeBundle,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    *,
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """Contact intervals of one episode, each lasting at least one robot sample period.

    A contact seen at a single robot sample would otherwise have an empty window; its end is moved one period later.
    """
    bouts = _contact_bouts(episode, robot, tracks, merge_gap_ns=merge_gap_ns)
    if not bouts:
        return bouts
    period_ns = _sample_period_ns(robot)
    return [replace(bout, t_end_ns=max(bout.t_end_ns, bout.t0_ns + period_ns)) for bout in bouts]


def _sample_period_ns(robot: Trajectory2D) -> int:
    """Median spacing of the robot samples, in ns."""
    if len(robot.t_ns) < 2:
        raise MetricInputError("contact intervals need at least two robot samples to set the sample period")
    return int(np.median(np.diff(robot.t_ns)))


def _contact_bouts(
    episode: AlignedEpisodeBundle,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    *,
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """Contact intervals of the recorded collision stream; the clearance and bumper streams stand in when it is absent or empty."""
    contact_rows = episode_table(episode, ROBOT_COLLISION_CONTACTS_TOPIC)
    if contact_rows is not None and not contact_rows.is_empty():
        return _contacts_from_table(cc_df=contact_rows, robot=robot, tracks=tracks, merge_gap_ns=merge_gap_ns)

    events: list[ContactEvent] = []
    for track in tracks.values():
        if len(track.t_ns) == 0 or len(robot.t_ns) == 0:
            continue
        t_lo = max(int(robot.t_ns[0]), int(track.t_ns[0]))
        t_hi = min(int(robot.t_ns[-1]), int(track.t_ns[-1]))
        if t_hi < t_lo:
            continue
        robot_window = robot.slice_ns(t_lo, t_hi)
        pedestrian = track.trajectory.resample_onto(robot_window.t_ns)
        radius_m = track.radius_m
        if radius_m is None or not np.isfinite(radius_m) or radius_m <= 0.0:
            raise MetricInputError(f"contact attribution needs the recorded radius_m of pedestrian {track.ped_id!r}")
        clearance = np.hypot(pedestrian.x - robot_window.x, pedestrian.y - robot_window.y) - (robot.radius_m + radius_m)
        events.extend(
            ContactEvent(
                t0_ns=int(robot_window.t_ns[start]),
                t_end_ns=int(robot_window.t_ns[end]),
                obstacle_kind=PEDESTRIAN_KIND,
                obstacle_id=track.ped_id,
                track=track,
            )
            for start, end in contiguous_spans(clearance <= 0.0, robot_window.t_ns, merge_gap_ns)
        )

    bumper = episode_table(episode, ROBOT_COLLISION_EVENTS_TOPIC)
    if bumper is not None and not bumper.is_empty():
        bumper_hits = bumper.select("t_ns", "wall_count", "static_count").filter((pl.col("wall_count") > 0) | (pl.col("static_count") > 0)).sort("t_ns")
        # agent-contracts: ignore AC004 - materializes ContactEvent dataclasses
        for row in bumper_hits.iter_rows(named=True):
            kind = "wall" if int(row["wall_count"]) > 0 else "static"
            events.append(
                ContactEvent(
                    t0_ns=int(row["t_ns"]),
                    t_end_ns=int(row["t_ns"]),
                    obstacle_kind=kind,
                    obstacle_id=kind,
                )
            )

    by_target: dict[tuple[str, str], list[ContactEvent]] = {}
    for event in events:
        by_target.setdefault((event.obstacle_kind, event.obstacle_id), []).append(event)
    merged = [replace(target[0], t0_ns=start_ns, t_end_ns=end_ns) for target in by_target.values() for start_ns, end_ns in merge_intervals_ns([(event.t0_ns, event.t_end_ns) for event in target])]
    return sorted(merged, key=lambda event: (event.t0_ns, event.obstacle_kind, event.obstacle_id))


def _event_from_fields(
    *,
    t0_ns: int,
    t_end_ns: int,
    obstacle_kind: str,
    obstacle_id: str,
    tracks: Mapping[str, PedTrack],
    robot: Trajectory2D,
    static_point: tuple[float, float] | None = None,
    target_agent_id: int | str | None = None,
) -> ContactEvent:
    """Build a ContactEvent from a configured or scoped target, resolving its pedestrian track."""
    kind = obstacle_kind.lower()
    track = resolve_ped_track(tracks, target_agent_id) if kind == PEDESTRIAN_KIND else None
    if track is None and kind == PEDESTRIAN_KIND:
        track = _match_track(obstacle_id, tracks, _sample_at(robot, t0_ns))
    return ContactEvent(
        t0_ns=t0_ns,
        t_end_ns=t_end_ns,
        obstacle_kind=kind,
        obstacle_id=obstacle_id or (track.ped_id if track is not None else kind),
        track=track,
        static_point=None if kind in TRACKED_KINDS else static_point,
    )


class ContactAttributionCalculator(BaseMetricCalculator):
    """Per-contact closing speeds, impact speeds and frozen-robot clearance; no label and no cut.

    The keys contact_t0_ns, contact_t_end_ns, obstacle_kind, obstacle_id, target_agent_id and static_point are
    debug hooks that select one contact: metrics_config reads them, and they are not PARAMS.
    """

    NAME = "contact_attribution"
    CATEGORY = "safety"
    REQUIRED_TOPICS = [ROBOT_TRACKS_TOPIC]
    PARAMS = ("contact_merge_gap_s", "pre_contact_window_s", "post_contact_window_s")
    OUTPUTS = (*_SCALAR_METRICS, *_COUNT_METRICS)
    DIRECTIONS = {metric.name: Direction.LOWER for metric in _COUNT_METRICS}
    PRIMARY_OUTPUTS = [
        "v_close_robot_mps",
        "v_close_target_mps",
        "freeze_clearance_m",
        "contact_count",
    ]

    def _window_s(self, key: str, default: float, override: float | None) -> float:
        """Configured window length, or the contact-scope override when the slice supplies one."""
        return float(override if override is not None else self.metrics_config.get(key, default))

    def _configured_contact(self, robot: Trajectory2D, tracks: Mapping[str, PedTrack]) -> list[ContactEvent] | None:
        """Build a single ContactEvent if contact_t0_ns is explicitly set in metrics_config."""
        cfg_t0_ns = self.metrics_config.get("contact_t0_ns")
        if cfg_t0_ns is None:
            return None
        t0_ns = int(cfg_t0_ns)
        return [
            _event_from_fields(
                t0_ns=t0_ns,
                t_end_ns=int(self.metrics_config.get("contact_t_end_ns", t0_ns)),
                obstacle_kind=str(self.metrics_config.get("obstacle_kind", PEDESTRIAN_KIND)),
                obstacle_id=str(self.metrics_config.get("obstacle_id", "")),
                tracks=tracks,
                robot=robot,
                static_point=_as_point(self.metrics_config.get("static_point")),
                target_agent_id=self.metrics_config.get("target_agent_id"),
            )
        ]

    def _detect_contacts(
        self,
        episode: AlignedEpisodeBundle,
        robot: Trajectory2D,
        tracks: Mapping[str, PedTrack],
    ) -> list[ContactEvent]:
        """Detect all contact intervals in the episode, or the one configured in metrics_config."""
        configured = self._configured_contact(robot, tracks)
        if configured is not None:
            return configured
        merge_gap_s = float(self.metrics_config.get("contact_merge_gap_s", CONTACT_MERGE_GAP_S))
        return detect_contact_events(episode, robot, tracks, merge_gap_ns=int(round(merge_gap_s * NS_PER_S)))

    def _context(
        self,
        robot: Trajectory2D,
        pre_window_s: float | None = None,
        post_window_s: float | None = None,
    ) -> ContactContext:
        """Attribution context of the episode, with the window lengths of the contact scope when given."""
        return ContactContext(
            robot=robot,
            pre_window_s=self._window_s("pre_contact_window_s", PRE_CONTACT_WINDOW_S, pre_window_s),
            post_window_s=self._window_s("post_contact_window_s", POST_CONTACT_WINDOW_S, post_window_s),
        )

    def _attribute_contacts(self, episode: AlignedEpisodeBundle, robot: Trajectory2D) -> list[ContactAttributionRecord]:
        """Detect and attribute every contact of an episode whose robot track is known."""
        if len(robot.t_ns) == 0:
            return []
        tracks = extract_episode_ped_tracks(episode)
        detected = self._detect_contacts(episode, robot, tracks)
        if not detected:
            return []
        context = self._context(robot)
        return [attribute_single_contact(contact_id, event, context) for contact_id, event in enumerate(detected)]

    def attribute_all_contacts(self, episode: AlignedEpisodeBundle) -> list[ContactAttributionRecord]:
        """Detect and attribute every contact in the episode."""
        robot, _yaw = resolve_robot_trajectory(self, episode)
        return self._attribute_contacts(episode, robot)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
        """Episode scope: the number of detected contacts; NULL when no stream was recorded."""
        del prior_results
        robot, _yaw = resolve_robot_trajectory(self, episode)
        if len(robot.t_ns) == 0:
            return _undefined_metrics(_COUNT_NAMES, f"{NullReason.TOPIC_MISSING.value}:{ROBOT_TRACKS_TOPIC}")
        records = self._attribute_contacts(episode, robot)
        counts = Counter(record.obstacle_kind for record in records)
        stream_recorded = episode_table(episode, ROBOT_COLLISION_CONTACTS_TOPIC) is not None or episode_table(episode, ROBOT_COLLISION_EVENTS_TOPIC) is not None
        if records or stream_recorded:
            return {
                CONTACT_COUNT.name: len(records),
                **{metric.name: counts[kind] for kind, metric in CONTACT_COUNT_BY_KIND},
            }
        return _undefined_metrics(
            _COUNT_NAMES,
            f"{NullReason.TOPIC_MISSING.value}:{ROBOT_COLLISION_CONTACTS_TOPIC}",
        )

    def _event_from_slice(
        self,
        episode: AlignedEpisodeBundle,
        scope_slice: ScopeSlice,
        robot: Trajectory2D,
    ) -> ContactEvent:
        """ContactEvent addressed by a contact scope slice; a slice without an identity raises."""
        identity = scope_slice.identity
        kind_value = identity.get("collider_kind")
        if kind_value is None:
            raise MetricInputError("a contact scope slice carries no collider_kind")
        id_value = identity.get("obstacle_id")
        if id_value is None:
            raise MetricInputError("a contact scope slice carries no obstacle_id")
        obstacle_id = str(id_value)
        return _event_from_fields(
            t0_ns=int(scope_slice.t_start_ns),
            t_end_ns=int(scope_slice.t_end_ns),
            obstacle_kind=str(kind_value),
            obstacle_id=obstacle_id,
            tracks=extract_episode_ped_tracks(episode),
            robot=robot,
            static_point=_as_point(identity.get("static_point")),
            target_agent_id=obstacle_id,
        )

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """Compute the six contact scalars for a contact scope slice without mutating calculator state."""
        if scope_slice is None or scope_slice.scope is not MetricScope.CONTACT:
            return self.calculate(episode, prior_results)
        robot, _yaw = resolve_robot_trajectory(self, episode)
        event = self._event_from_slice(episode, scope_slice, robot)
        if len(robot.t_ns) == 0:
            return _undefined_metrics(_SCALAR_NAMES, f"{NullReason.TOPIC_MISSING.value}:{ROBOT_TRACKS_TOPIC}")
        context = self._context(
            robot,
            pre_window_s=scope_slice.identity.get("pre_contact_window_s"),
            post_window_s=scope_slice.identity.get("post_contact_window_s"),
        )
        return contact_scalars(attribute_single_contact(0, event, context), event)
