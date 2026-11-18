"""Contact attribution as measured scalars: closing speeds, impact speeds and the frozen-robot clearance.

Nothing here declares fault, a label or a cut. The robot and target closing shares are views of the
two stored closing speeds (:func:`closing_shares`), never stored copies, so each metric is stored
once and an analysis compares the scalars instead of reading a verdict.
"""

from __future__ import annotations

import re
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from arena_evaluation.storage.registry import MetricScope, NullReason, define_metric

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

V_CLOSE_ROBOT = define_metric(
    "v_close_robot_mps",
    pl.Float64,
    scopes=(MetricScope.CONTACT,),
    unit="m/s",
    description="Mean part of the robot's velocity pointing at the other body along the line of sight over the 0.5 s before contact; positive when the robot closes in.",
    null_reason=True,
)
V_CLOSE_TARGET = define_metric(
    "v_close_target_mps",
    pl.Float64,
    scopes=(MetricScope.CONTACT,),
    unit="m/s",
    description="Mean part of the other body's velocity pointing at the robot along the line of sight over the 0.5 s before contact; positive when the other body closes in.",
    null_reason=True,
)
V_ROBOT_IMPACT = define_metric(
    "v_robot_impact_mps",
    pl.Float64,
    scopes=(MetricScope.CONTACT,),
    unit="m/s",
    description="Robot speed at contact onset.",
    null_reason=True,
)
V_TARGET_IMPACT = define_metric(
    "v_target_impact_mps",
    pl.Float64,
    scopes=(MetricScope.CONTACT,),
    unit="m/s",
    description="Speed of the other body at contact onset; zero for a wall or static collider.",
    null_reason=True,
)
RELATIVE_IMPACT_SPEED = define_metric(
    "relative_impact_speed_mps",
    pl.Float64,
    scopes=(MetricScope.CONTACT,),
    unit="m/s",
    description="Relative speed of the robot and the other body at contact onset.",
    null_reason=True,
)
FREEZE_CLEARANCE = define_metric(
    "freeze_clearance_m",
    pl.Float64,
    scopes=(MetricScope.CONTACT,),
    unit="m",
    description="Closest approach of the pedestrian's recorded path to the robot frozen at t0 minus the pre-contact window, over [t0 - pre, t0 + post] (0.5 s each by default), minus both body radii; zero or negative means the contact would have happened anyway.",
    null_reason=True,
)
CONTACT_COUNT = define_metric(
    "contact_count",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    unit="count",
    description="Number of contacts detected in the episode; NULL when no collision stream was recorded and no contact could be detected.",
    null_reason=True,
)

PRE_CONTACT_WINDOW_S: Final[float] = 0.5
POST_CONTACT_WINDOW_S: Final[float] = 0.5
#: Wall-clock gap, seconds, that still merges two recorded contact samples of the same target into one contact.
CONTACT_MERGE_GAP_S: Final[float] = 0.25
PEDESTRIAN_KIND: Final[str] = "pedestrian"
ROBOT_KIND: Final[str] = "robot"
#: Collider kinds that move on their own and whose speed is therefore recorded on a track.
TRACKED_KINDS: Final[frozenset[str]] = frozenset({PEDESTRIAN_KIND, ROBOT_KIND})
PEDESTRIAN_KIND_PATTERN: Final[str] = "ped|human"
ROBOT_COLLISION_CONTACTS_TOPIC: Final[str] = "robot_collision_contacts"
ROBOT_COLLISION_EVENTS_TOPIC: Final[str] = "robot_collision_events"

CONTACT_COUNT_PEDESTRIAN = define_metric(
    "contact_count_pedestrian",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    unit="count",
    description="Contacts of the episode whose other body is a pedestrian; from the same detector as contact_count.",
    null_reason=True,
)
CONTACT_COUNT_ROBOT = define_metric(
    "contact_count_robot",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    unit="count",
    description="Contacts of the episode whose other body is another robot; from the same detector as contact_count.",
    null_reason=True,
)
CONTACT_COUNT_WALL = define_metric(
    "contact_count_wall",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    unit="count",
    description="Contacts of the episode with a wall; from the same detector as contact_count.",
    null_reason=True,
)
CONTACT_COUNT_STATIC = define_metric(
    "contact_count_static",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    unit="count",
    description="Contacts of the episode with a static obstacle; from the same detector as contact_count.",
    null_reason=True,
)
CONTACT_COUNT_BY_KIND: Final[tuple[tuple[str, typing.Any], ...]] = (
    (PEDESTRIAN_KIND, CONTACT_COUNT_PEDESTRIAN),
    (ROBOT_KIND, CONTACT_COUNT_ROBOT),
    ("wall", CONTACT_COUNT_WALL),
    ("static", CONTACT_COUNT_STATIC),
)


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
    """Robot and target closing shares in 0..1, computed from the two stored closing speeds.

    Each speed is clipped at 0 and the two shares add to 1. When neither body closes there is no
    denominator and both shares are None; when either speed is unmeasured both are None.
    """
    if v_close_robot_mps is None or v_close_target_mps is None:
        return None, None
    if not (np.isfinite(v_close_robot_mps) and np.isfinite(v_close_target_mps)):
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


def _covers_onset(track: PedTrack, t0_ns: int) -> bool:
    """Whether the contact onset lies inside the recorded span of the pedestrian track."""
    return len(track.t_ns) > 0 and int(track.t_ns[0]) <= t0_ns <= int(track.t_ns[-1])


def compute_closing_speeds(
    event: ContactEvent, context: ContactContext
) -> tuple[float, float]:
    """Mean robot and target closing speeds (m/s) over [t0 - pre_window_s, t0].

    A closing speed needs a line of sight and both velocity samples; where either is missing the
    result is NaN, which the caller publishes as NULL.
    """
    win_start_ns = int(event.t0_ns - round(context.pre_window_s * NS_PER_S))
    r_win = context.robot.slice_ns(win_start_ns, event.t0_ns)
    if len(r_win.t_ns) == 0:
        return float("nan"), float("nan")

    if event.track is not None and _covers_onset(event.track, event.t0_ns):
        p_win = event.track.trajectory.resample_onto(r_win.t_ns)
        dx = p_win.x - r_win.x
        dy = p_win.y - r_win.y
        dist = np.hypot(dx, dy)
        inside_track = (r_win.t_ns >= event.track.t_ns[0]) & (
            r_win.t_ns <= event.track.t_ns[-1]
        )
        valid = (
            inside_track
            & (dist > 1e-6)
            & np.isfinite(dist)
            & np.isfinite(r_win.vx)
            & np.isfinite(r_win.vy)
            & np.isfinite(p_win.vx)
            & np.isfinite(p_win.vy)
        )
        if not np.any(valid):
            return float("nan"), float("nan")
        ux, uy = dx[valid] / dist[valid], dy[valid] / dist[valid]
        v_close_r = r_win.vx[valid] * ux + r_win.vy[valid] * uy
        v_close_t = -(p_win.vx[valid] * ux + p_win.vy[valid] * uy)
        return float(np.mean(v_close_r)), float(np.mean(v_close_t))

    if event.static_point is not None:
        dx = float(event.static_point[0]) - r_win.x
        dy = float(event.static_point[1]) - r_win.y
        dist = np.hypot(dx, dy)
        valid = (
            (dist > 1e-6)
            & np.isfinite(dist)
            & np.isfinite(r_win.vx)
            & np.isfinite(r_win.vy)
        )
        target_closing = (
            0.0 if event.obstacle_kind not in TRACKED_KINDS else float("nan")
        )
        if np.any(valid):
            ux, uy = dx[valid] / dist[valid], dy[valid] / dist[valid]
            return float(
                np.mean(r_win.vx[valid] * ux + r_win.vy[valid] * uy)
            ), target_closing
        return float("nan"), target_closing

    # A wall or static body does not move, so its speed is measured zero; the robot's own closing
    # speed needs a position to define the line of sight and stays undefined without one.
    if event.obstacle_kind not in TRACKED_KINDS:
        return float("nan"), 0.0
    return float("nan"), float("nan")


def frozen_robot_clearance(
    event: ContactEvent, context: ContactContext
) -> float | None:
    """Frozen-robot clearance (m) over [t0 - pre_window_s, t0 + post_window_s].

    The robot is held where it was ``pre_window_s`` before contact and the pedestrian's recorded
    path is replayed against it; the result is the closest centre distance minus both body radii.
    Only a pedestrian yields it: a wall does not move and another robot reacts to a stopped robot.
    """
    track = event.track
    if (
        event.obstacle_kind != PEDESTRIAN_KIND
        or track is None
        or track.radius_m is None
        or not np.isfinite(track.radius_m)
        or track.radius_m <= 0.0
        or len(track.t_ns) == 0
        or len(context.robot.t_ns) == 0
    ):
        return None

    win_start_ns = int(event.t0_ns - round(context.pre_window_s * NS_PER_S))
    win_end_ns = int(event.t0_ns + round(context.post_window_s * NS_PER_S))
    ped_in_win = (track.t_ns >= win_start_ns) & (track.t_ns <= win_end_ns)
    if not np.any(ped_in_win):
        return None

    t_freeze_ns = max(int(context.robot.t_ns[0]), win_start_ns)
    frozen_pose = context.robot.resample_onto(np.asarray([t_freeze_ns], dtype=np.int64))
    freeze_x, freeze_y = float(frozen_pose.x[0]), float(frozen_pose.y[0])
    if not (np.isfinite(freeze_x) and np.isfinite(freeze_y)):
        return None

    ped_x = track.x[ped_in_win]
    ped_y = track.y[ped_in_win]
    measured = np.isfinite(ped_x) & np.isfinite(ped_y)
    if not np.any(measured):
        return None
    dists = np.hypot(ped_x[measured] - freeze_x, ped_y[measured] - freeze_y)

    return float(np.min(dists)) - (
        float(context.robot.radius_m) + float(track.radius_m)
    )


def attribute_single_contact(
    contact_id: int,
    event: ContactEvent,
    context: ContactContext,
) -> ContactAttributionRecord:
    """Compute the attribution scalars of a single ContactEvent."""
    v_close_r, v_close_t = compute_closing_speeds(event, context)

    t0_arr = np.asarray([event.t0_ns], dtype=np.int64)
    r_at_t0 = context.robot.resample_onto(t0_arr)
    rvx_0, rvy_0 = float(r_at_t0.vx[0]), float(r_at_t0.vy[0])
    v_robot_impact = float(np.hypot(rvx_0, rvy_0))

    if event.obstacle_kind in TRACKED_KINDS:
        if event.track is not None and _covers_onset(event.track, event.t0_ns):
            p_at_t0 = event.track.trajectory.resample_onto(t0_arr)
            pvx_0, pvy_0 = float(p_at_t0.vx[0]), float(p_at_t0.vy[0])
            v_target_impact = float(np.hypot(pvx_0, pvy_0))
            rel_impact = float(np.hypot(rvx_0 - pvx_0, rvy_0 - pvy_0))
        else:
            v_target_impact = float("nan")
            rel_impact = float("nan")
    else:
        # A wall or static collider is at rest by definition, so the relative speed is the robot's.
        v_target_impact = 0.0
        rel_impact = v_robot_impact

    ep_start_ns = (
        int(context.robot.t_ns[0]) if len(context.robot.t_ns) > 0 else event.t0_ns
    )
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
        freeze_clearance_m=frozen_robot_clearance(event, context),
    )


def contact_scalars(
    record: ContactAttributionRecord, event: ContactEvent
) -> dict[str, typing.Any]:
    """The six stored contact scalars plus a null reason beside each undefined one."""
    scalars: dict[str, typing.Any] = {
        V_CLOSE_ROBOT.name: record.v_close_robot_mps,
        V_CLOSE_TARGET.name: record.v_close_target_mps,
        V_ROBOT_IMPACT.name: record.v_robot_impact_mps,
        V_TARGET_IMPACT.name: record.v_target_impact_mps,
        RELATIVE_IMPACT_SPEED.name: record.relative_impact_speed_mps,
        FREEZE_CLEARANCE.name: record.freeze_clearance_m,
    }
    for key, value in tuple(scalars.items()):
        if value is None:
            scalars[f"{key}_null_reason"] = _null_reason(key, event)
    return scalars


def _null_reason(key: str, event: ContactEvent) -> str:
    """Why a contact scalar is NULL: inapplicable to the collider kind, an unrecorded radius, or no usable sample in the window."""
    if event.obstacle_kind == ROBOT_KIND and key != V_ROBOT_IMPACT.name:
        return NullReason.NOT_APPLICABLE.value
    if key == FREEZE_CLEARANCE.name and event.obstacle_kind != PEDESTRIAN_KIND:
        return NullReason.NOT_APPLICABLE.value
    if (
        key == FREEZE_CLEARANCE.name
        and event.track is not None
        and event.track.radius_m is None
    ):
        return NullReason.NOT_RECORDED.value
    return NullReason.NO_SAMPLES.value


def _recorded_ped_radius_m(track: PedTrack) -> float:
    """The pedestrian's recorded body radius; a missing or non-positive one is refused, never replaced by a default."""
    if (
        track.radius_m is None
        or not np.isfinite(track.radius_m)
        or track.radius_m <= 0.0
    ):
        raise MetricInputError(
            f"contact attribution needs the recorded radius_m of pedestrian {track.ped_id!r}"
        )
    return float(track.radius_m)


def static_point_of(
    obstacle_x: float | None, obstacle_y: float | None
) -> tuple[float, float] | None:
    """Static obstacle position of a collision row; None when the recording carries none, only the (0, 0) sentinel or a non-finite value."""
    if (
        obstacle_x is None
        or obstacle_y is None
        or not (np.isfinite(obstacle_x) and np.isfinite(obstacle_y))
    ):
        return None
    point = (float(obstacle_x), float(obstacle_y))
    return None if point == (0.0, 0.0) else point


def _numeric_suffix(name: str) -> str | None:
    """Trailing number of an entity name (``Pedestrian_7`` -> ``7``), None when it has none."""
    digits = re.findall(r"\d+", name)
    return digits[-1] if digits else None


def _track_by_name_or_id(
    obstacle_id: str, tracks: Mapping[str, PedTrack]
) -> PedTrack | None:
    """Track whose ped_id or ped_name equals the obstacle id or its env-stripped entity name."""
    normalized = obstacle_id.strip()
    if not normalized:
        return None
    base = entity_name(normalized)
    for track in tracks.values():
        if track.ped_id in (normalized, base) or track.ped_name in (normalized, base):
            return track
    return None


def _track_by_numeric_suffix(
    base: str, tracks: Mapping[str, PedTrack]
) -> PedTrack | None:
    """Track whose ped_id or ped_name ends in the same number as the obstacle id."""
    suffix = _numeric_suffix(base)
    if suffix is None:
        return None
    for key, track in tracks.items():
        if _numeric_suffix(key) == suffix or _numeric_suffix(track.ped_name) == suffix:
            return track
    return None


def _nearest_track(
    tracks: Mapping[str, PedTrack], robot_at_t0: Trajectory2D
) -> PedTrack | None:
    """Pedestrian whose resampled position is nearest to the robot pose at t0."""
    if len(robot_at_t0.t_ns) == 0:
        return None
    robot_x, robot_y = float(robot_at_t0.x[0]), float(robot_at_t0.y[0])
    nearest: PedTrack | None = None
    nearest_distance_m = float("inf")
    for track in tracks.values():
        if len(track.t_ns) == 0:
            continue
        pedestrian_at_t0 = track.trajectory.resample_onto(robot_at_t0.t_ns)
        distance_m = float(
            np.hypot(pedestrian_at_t0.x[0] - robot_x, pedestrian_at_t0.y[0] - robot_y)
        )
        if distance_m < nearest_distance_m:
            nearest_distance_m = distance_m
            nearest = track
    return nearest


def _match_track(
    obstacle_id: str, tracks: Mapping[str, PedTrack], robot_at_t0: Trajectory2D
) -> PedTrack | None:
    """Resolve the PedTrack of a contact: exact ped id/name, then numeric suffix, then nearest pose."""
    if not tracks:
        return None
    track = _track_by_name_or_id(obstacle_id, tracks)
    if track is not None:
        return track
    track = _track_by_numeric_suffix(entity_name(obstacle_id.strip()), tracks)
    if track is not None:
        return track
    return _nearest_track(tracks, robot_at_t0)


def _group_contact_rows(cc_df: pl.DataFrame, merge_gap_ns: int) -> pl.DataFrame:
    """One row per contiguous contact run of the same resolved target, within the merge gap."""
    contacts = (
        cc_df.select(
            pl.col("t_ns").cast(pl.Int64),
            pl.col("obstacle_id"),
            pl.col("obstacle_kind"),
            pl.col("obstacle_x"),
            pl.col("obstacle_y"),
        )
        .with_columns(
            (pl.col("obstacle_kind") + pl.lit("|") + pl.col("obstacle_id")).alias(
                "target_key"
            )
        )
        .sort("target_key", "t_ns")
    )

    same_target = pl.col("target_key").shift(1) == pl.col("target_key")
    beyond_gap = pl.col("t_ns").diff() > merge_gap_ns
    starts_group = ~same_target.fill_null(False) | beyond_gap.fill_null(True)

    return (
        contacts.with_columns(starts_group.cum_sum().alias("group_id"))
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


def _resolve_contact_targets(
    cc_df: pl.DataFrame, robot: Trajectory2D, tracks: Mapping[str, PedTrack]
) -> pl.DataFrame:
    """Raw contact rows whose pedestrian obstacle ids are canonical ``ped_id`` values, so later grouping never splits one track."""
    frame = _groupable_contacts_frame(cc_df)
    if frame.is_empty() or not tracks:
        return frame
    times_ns = frame.get_column("t_ns").to_numpy().astype(np.int64)
    kinds = frame.get_column("obstacle_kind").to_numpy()
    raw_ids = frame.get_column("obstacle_id").to_numpy()
    pose_by_time = {
        int(stamp): robot.resample_onto(np.asarray([stamp], dtype=np.int64))
        for stamp in np.unique(times_ns)
    }
    resolved: list[str] = []
    for stamp, kind, raw_id in zip(times_ns, kinds, raw_ids, strict=True):
        if kind != PEDESTRIAN_KIND:
            resolved.append(str(raw_id))
            continue
        track = _track_by_name_or_id(str(raw_id), tracks) or _track_by_numeric_suffix(
            entity_name(str(raw_id).strip()), tracks
        )
        if track is None:
            track = _nearest_track(tracks, pose_by_time[int(stamp)])
        resolved.append(track.ped_id if track is not None else str(raw_id))
    return frame.with_columns(
        pl.Series("obstacle_id", resolved, dtype=pl.String)
    ).select("t_ns", "obstacle_id", "obstacle_kind", "obstacle_x", "obstacle_y")


def _groupable_contacts_frame(cc_df: pl.DataFrame) -> pl.DataFrame:
    """Contact rows with the normalized obstacle kind and their grouping columns."""
    raw_kind = pl.col("kind").fill_null("").str.to_lowercase()
    obstacle_kind = (
        pl.when(raw_kind == "")
        .then(pl.lit("static"))
        .when(raw_kind.str.contains(PEDESTRIAN_KIND_PATTERN))
        .then(pl.lit(PEDESTRIAN_KIND))
        .otherwise(raw_kind)
        .alias("obstacle_kind")
    )
    return cc_df.with_columns(obstacle_kind).select(
        pl.col("t_ns").cast(pl.Int64),
        pl.col("obstacle_id").fill_null("").alias("obstacle_id"),
        pl.col("obstacle_kind"),
        pl.col("obstacle_x").cast(pl.Float64),
        pl.col("obstacle_y").cast(pl.Float64),
    )


def _contacts_from_table(
    *,
    cc_df: pl.DataFrame,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """ContactEvent rows grouped per resolved target from a flat robot_collision_contacts table."""
    grouped = _group_contact_rows(
        _resolve_contact_targets(cc_df, robot, tracks), merge_gap_ns
    )
    t0_values = grouped.get_column("t0_ns").to_numpy()
    end_values = grouped.get_column("t_end_ns").to_numpy()
    kind_values = grouped.get_column("obstacle_kind").to_numpy()
    id_values = grouped.get_column("obstacle_id").to_numpy()
    x_values = grouped.get_column("obstacle_x").to_numpy()
    y_values = grouped.get_column("obstacle_y").to_numpy()
    events: list[ContactEvent] = []
    for t0_ns, t_end_ns, obstacle_kind, obstacle_id, obstacle_x, obstacle_y in zip(
        t0_values, end_values, kind_values, id_values, x_values, y_values, strict=True
    ):
        robot_at_t0 = robot.resample_onto(np.asarray([int(t0_ns)], dtype=np.int64))
        is_pedestrian = obstacle_kind == PEDESTRIAN_KIND
        track = (
            _match_track(str(obstacle_id), tracks, robot_at_t0)
            if is_pedestrian
            else None
        )
        resolved_id = str(obstacle_id) or (
            track.ped_id if track is not None else PEDESTRIAN_KIND
        )
        events.append(
            ContactEvent(
                t0_ns=int(t0_ns),
                t_end_ns=int(t_end_ns),
                obstacle_kind=str(obstacle_kind),
                obstacle_id=resolved_id,
                track=track,
                static_point=None
                if obstacle_kind in TRACKED_KINDS
                else static_point_of(obstacle_x, obstacle_y),
            )
        )
    return events


def _contacts_from_clearance(
    *,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """Detect pedestrian contacts from surface-to-surface clearance <= 0."""
    events: list[ContactEvent] = []
    for track in tracks.values():
        if len(track.t_ns) == 0 or len(robot.t_ns) == 0:
            continue
        t_lo = max(int(robot.t_ns[0]), int(track.t_ns[0]))
        t_hi = min(int(robot.t_ns[-1]), int(track.t_ns[-1]))
        if t_hi < t_lo:
            continue

        robot_window = robot.slice_ns(t_lo, t_hi)
        pedestrian_window = track.trajectory.resample_onto(robot_window.t_ns)
        ped_radius_m = _recorded_ped_radius_m(track)
        # Surface-to-surface (edge) clearance in metres: the centre distance minus both body radii.
        edge_clearance_m = np.hypot(
            pedestrian_window.x - robot_window.x, pedestrian_window.y - robot_window.y
        ) - (robot.radius_m + ped_radius_m)
        spans = contiguous_spans(
            edge_clearance_m <= 0.0, robot_window.t_ns, merge_gap_ns
        )
        events.extend(
            ContactEvent(
                t0_ns=int(robot_window.t_ns[start]),
                t_end_ns=int(robot_window.t_ns[end]),
                obstacle_kind=PEDESTRIAN_KIND,
                obstacle_id=track.ped_id,
                track=track,
            )
            for start, end in spans
        )
    return events


def _contacts_from_bumper_events(ce_df: pl.DataFrame | None) -> list[ContactEvent]:
    """ContactEvent rows for wall and static counts in the flat robot_collision_events table."""
    if ce_df is None or len(ce_df) == 0:
        return []
    bumper_rows = (
        ce_df.select("t_ns", "wall_count", "static_count")
        .filter((pl.col("wall_count") > 0) | (pl.col("static_count") > 0))
        .sort("t_ns")
    )
    events: list[ContactEvent] = []
    for t_ns, wall_count in zip(
        bumper_rows["t_ns"].to_numpy(),
        bumper_rows["wall_count"].to_numpy(),
        strict=True,
    ):
        kind = "wall" if int(wall_count) > 0 else "static"
        events.append(
            ContactEvent(
                t0_ns=int(t_ns),
                t_end_ns=int(t_ns),
                obstacle_kind=kind,
                obstacle_id=kind,
            )
        )
    return events


def _merge_contact_events(events: Sequence[ContactEvent]) -> list[ContactEvent]:
    """Union overlapping intervals of the same target across the clearance and bumper paths."""
    by_target: dict[tuple[str, str], list[ContactEvent]] = {}
    for event in events:
        by_target.setdefault((event.obstacle_kind, event.obstacle_id), []).append(event)

    merged: list[ContactEvent] = []
    for target_events in by_target.values():
        first = target_events[0]
        for start_ns, end_ns in merge_intervals_ns(
            [(event.t0_ns, event.t_end_ns) for event in target_events]
        ):
            merged.append(
                ContactEvent(
                    t0_ns=start_ns,
                    t_end_ns=end_ns,
                    obstacle_kind=first.obstacle_kind,
                    obstacle_id=first.obstacle_id,
                    track=first.track,
                    static_point=first.static_point,
                )
            )
    merged.sort(key=lambda event: (event.t0_ns, event.obstacle_kind, event.obstacle_id))
    return merged


def detect_contact_events(
    episode: AlignedEpisodeBundle,
    robot: Trajectory2D,
    tracks: Mapping[str, PedTrack],
    *,
    merge_gap_ns: int,
) -> list[ContactEvent]:
    """Contact intervals of the recorded collision stream; the clearance and bumper streams stand in when it is absent or empty."""
    contact_rows = episode_table(episode, ROBOT_COLLISION_CONTACTS_TOPIC)
    if contact_rows is not None and len(contact_rows) > 0:
        return _contacts_from_table(
            cc_df=contact_rows, robot=robot, tracks=tracks, merge_gap_ns=merge_gap_ns
        )

    events = _contacts_from_clearance(
        robot=robot, tracks=tracks, merge_gap_ns=merge_gap_ns
    )
    events.extend(
        _contacts_from_bumper_events(
            episode_table(episode, ROBOT_COLLISION_EVENTS_TOPIC)
        )
    )
    events.sort(key=lambda event: (event.t0_ns, event.obstacle_kind, event.obstacle_id))
    return _merge_contact_events(events)


def contact_stream_recorded(episode: AlignedEpisodeBundle) -> bool:
    """Whether the recording carries a collision stream: an empty stream means no contact, an absent one means unknown."""
    return (
        episode_table(episode, ROBOT_COLLISION_CONTACTS_TOPIC) is not None
        or episode_table(episode, ROBOT_COLLISION_EVENTS_TOPIC) is not None
    )


def _configured_static_point(
    static_point: Sequence[float] | None, obstacle_kind: str
) -> tuple[float, float] | None:
    """Static obstacle position of a configured wall or static contact."""
    if obstacle_kind in TRACKED_KINDS or static_point is None or len(static_point) < 2:
        return None
    return (float(static_point[0]), float(static_point[1]))


class ContactAttributionCalculator(BaseMetricCalculator):
    """Per-contact closing speeds, impact speeds and frozen-robot clearance; no label and no cut.

    The keys contact_t0_ns, contact_t_end_ns, obstacle_kind, obstacle_id, target_agent_id and static_point are
    debug hooks that select one contact: metrics_config reads them, and they are not PARAMS.
    """

    NAME = "contact_attribution"
    CATEGORY = "safety"
    REQUIRED_TOPICS = [ROBOT_TRACKS_TOPIC]
    PARAMS = (
        "contact_merge_gap_s",
        "pre_contact_window_s",
        "post_contact_window_s",
    )
    OUTPUTS = (
        V_CLOSE_ROBOT,
        V_CLOSE_TARGET,
        V_ROBOT_IMPACT,
        V_TARGET_IMPACT,
        RELATIVE_IMPACT_SPEED,
        FREEZE_CLEARANCE,
        CONTACT_COUNT,
        CONTACT_COUNT_PEDESTRIAN,
        CONTACT_COUNT_ROBOT,
        CONTACT_COUNT_WALL,
        CONTACT_COUNT_STATIC,
    )
    DIRECTIONS = {
        "contact_count": Direction.LOWER,
        "contact_count_pedestrian": Direction.LOWER,
        "contact_count_robot": Direction.LOWER,
        "contact_count_wall": Direction.LOWER,
        "contact_count_static": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = [
        "v_close_robot_mps",
        "v_close_target_mps",
        "freeze_clearance_m",
        "contact_count",
    ]

    def _configured_contact(
        self, robot: Trajectory2D, tracks: Mapping[str, PedTrack]
    ) -> list[ContactEvent] | None:
        """Build a single ContactEvent if contact_t0_ns is explicitly set in metrics_config."""
        cfg_t0_ns = self.metrics_config.get("contact_t0_ns")
        if cfg_t0_ns is None:
            return None
        t0_ns = int(cfg_t0_ns)
        t_end_ns = int(self.metrics_config.get("contact_t_end_ns", t0_ns))
        obstacle_kind = str(
            self.metrics_config.get("obstacle_kind", PEDESTRIAN_KIND)
        ).lower()
        obstacle_id = str(self.metrics_config.get("obstacle_id", ""))

        track = (
            resolve_ped_track(tracks, self.metrics_config.get("target_agent_id"))
            if obstacle_kind == PEDESTRIAN_KIND
            else None
        )
        if track is None and obstacle_kind == PEDESTRIAN_KIND:
            robot_at_t0 = robot.resample_onto(np.asarray([t0_ns], dtype=np.int64))
            track = _match_track(obstacle_id, tracks, robot_at_t0)
        resolved_id = obstacle_id or (
            track.ped_name if track is not None else obstacle_kind
        )
        return [
            ContactEvent(
                t0_ns=t0_ns,
                t_end_ns=t_end_ns,
                obstacle_kind=obstacle_kind,
                obstacle_id=resolved_id,
                track=track,
                static_point=_configured_static_point(
                    self.metrics_config.get("static_point"), obstacle_kind
                ),
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
        merge_gap_ns = int(
            round(
                float(
                    self.metrics_config.get("contact_merge_gap_s", CONTACT_MERGE_GAP_S)
                )
                * NS_PER_S
            )
        )
        return detect_contact_events(episode, robot, tracks, merge_gap_ns=merge_gap_ns)

    def _context(
        self,
        robot: Trajectory2D,
        pre_window_s: float | None = None,
        post_window_s: float | None = None,
    ) -> ContactContext:
        """Attribution context of the episode, with the window lengths of the contact scope when given."""
        return ContactContext(
            robot=robot,
            pre_window_s=float(
                self.metrics_config.get("pre_contact_window_s", PRE_CONTACT_WINDOW_S)
            )
            if pre_window_s is None
            else float(pre_window_s),
            post_window_s=float(
                self.metrics_config.get("post_contact_window_s", POST_CONTACT_WINDOW_S)
            )
            if post_window_s is None
            else float(post_window_s),
        )

    def _attribute_contacts(
        self, episode: AlignedEpisodeBundle, robot: Trajectory2D
    ) -> list[ContactAttributionRecord]:
        """Detect and attribute every contact of an episode whose robot track is known."""
        if len(robot.t_ns) == 0:
            return []
        tracks = extract_episode_ped_tracks(episode)
        detected = self._detect_contacts(episode, robot, tracks)
        if not detected:
            return []
        context = self._context(robot)
        return [
            attribute_single_contact(contact_id, event, context)
            for contact_id, event in enumerate(detected)
        ]

    def attribute_all_contacts(
        self, episode: AlignedEpisodeBundle
    ) -> list[ContactAttributionRecord]:
        """Detect and attribute every contact in the episode."""
        robot, _yaw = resolve_robot_trajectory(self, episode)
        return self._attribute_contacts(episode, robot)

    @staticmethod
    def _undefined_counts(reason: str) -> dict[str, typing.Any]:
        """Every episode-scope count NULL with one shared reason."""
        undefined: dict[str, typing.Any] = {
            CONTACT_COUNT.name: None,
            **{metric.name: None for _, metric in CONTACT_COUNT_BY_KIND},
        }
        undefined.update({f"{name}_null_reason": reason for name in undefined})
        return undefined

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Episode scope: the number of detected contacts; NULL when no stream was recorded."""
        del prior_results
        robot, _yaw = resolve_robot_trajectory(self, episode)
        if len(robot.t_ns) == 0:
            return self._undefined_counts(
                f"{NullReason.TOPIC_MISSING.value}:{ROBOT_TRACKS_TOPIC}"
            )
        records = self._attribute_contacts(episode, robot)
        by_kind = {metric.name: 0 for _, metric in CONTACT_COUNT_BY_KIND}
        for record in records:
            for kind, metric in CONTACT_COUNT_BY_KIND:
                if record.obstacle_kind == kind:
                    by_kind[metric.name] += 1
                    break
        if records:
            return {CONTACT_COUNT.name: len(records), **by_kind}
        if contact_stream_recorded(episode):
            return {CONTACT_COUNT.name: 0, **by_kind}
        return self._undefined_counts(
            f"{NullReason.TOPIC_MISSING.value}:{ROBOT_COLLISION_CONTACTS_TOPIC}"
        )

    def _event_from_slice(
        self,
        episode: AlignedEpisodeBundle,
        scope_slice: ScopeSlice,
        robot: Trajectory2D,
    ) -> ContactEvent:
        """ContactEvent addressed by a contact scope slice; a slice without an identity raises."""
        kind_value = scope_slice.identity.get("collider_kind")
        if kind_value is None:
            raise MetricInputError("a contact scope slice carries no collider_kind")
        obstacle_kind = str(kind_value).lower()
        id_value = scope_slice.identity.get("obstacle_id")
        if id_value is None:
            raise MetricInputError("a contact scope slice carries no obstacle_id")
        obstacle_id = str(id_value)

        tracks = extract_episode_ped_tracks(episode)
        pedestrian = obstacle_kind == PEDESTRIAN_KIND
        track = resolve_ped_track(tracks, obstacle_id) if pedestrian else None
        if track is None and pedestrian:
            robot_at_t0 = robot.resample_onto(
                np.asarray([int(scope_slice.t_start_ns)], dtype=np.int64)
            )
            track = _match_track(obstacle_id, tracks, robot_at_t0)

        resolved_id = obstacle_id or (
            track.ped_id if track is not None else obstacle_kind
        )
        return ContactEvent(
            t0_ns=int(scope_slice.t_start_ns),
            t_end_ns=int(scope_slice.t_end_ns),
            obstacle_kind=obstacle_kind,
            obstacle_id=resolved_id,
            track=track,
            static_point=_configured_static_point(
                scope_slice.identity.get("static_point"), obstacle_kind
            ),
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
            undefined = dict.fromkeys(
                (
                    V_CLOSE_ROBOT.name,
                    V_CLOSE_TARGET.name,
                    V_ROBOT_IMPACT.name,
                    V_TARGET_IMPACT.name,
                    RELATIVE_IMPACT_SPEED.name,
                    FREEZE_CLEARANCE.name,
                )
            )
            undefined.update(
                {
                    f"{key}_null_reason": f"{NullReason.TOPIC_MISSING.value}:{ROBOT_TRACKS_TOPIC}"
                    for key in undefined
                }
            )
            return undefined

        context = self._context(
            robot,
            pre_window_s=scope_slice.identity.get("pre_contact_window_s"),
            post_window_s=scope_slice.identity.get("post_contact_window_s"),
        )
        return contact_scalars(attribute_single_contact(0, event, context), event)
