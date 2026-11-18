from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.common.alignment import asof_position
from arena_evaluation.processing.metrics.common.caching import load_world
from arena_evaluation.processing.metrics.common.events import reconstruct_events
from arena_evaluation.processing.metrics.common.zones import (
    DoorGeometry,
    ZoneGeometry,
    extract_door_geometry,
    extract_zone_geometry,
    zone_membership,
)
from arena_evaluation.processing.metrics.ecological.mechanism_common import entity_name, robot_env_snapshot
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

logger = logging.getLogger(__name__)

SPEED_ZONE_VIOLATIONS = define_metric("speed_zone_violations", pl.Int64, scopes=(MetricScope.ROBOT,), description="ecological metric of the compliance_metrics calculator", null_reason=True)
SPEED_ZONE_VIOLATION_SECONDS = define_metric("speed_zone_violation_seconds", pl.Float64, scopes=(MetricScope.ROBOT,), unit="s", description="ecological metric of the compliance_metrics calculator", null_reason=True)
QUIET_ZONE_DWELL_SECONDS = define_metric("quiet_zone_dwell_seconds", pl.Float64, scopes=(MetricScope.ROBOT,), unit="s", description="ecological metric of the compliance_metrics calculator", null_reason=True)
RESTRICTED_ZONE_ENTRIES = define_metric("restricted_zone_entries", pl.Int64, scopes=(MetricScope.ROBOT,), description="ecological metric of the compliance_metrics calculator", null_reason=True)
DOORWAY_BLOCKING_TIME = define_metric("doorway_blocking_time", pl.Float64, scopes=(MetricScope.ROBOT,), unit="s", description="ecological metric of the compliance_metrics calculator", null_reason=True)
PASSING_RULE_COMPLIANCE = define_metric("passing_rule_compliance", pl.Float64, scopes=(MetricScope.ROBOT,), unit="1", description="ecological metric of the compliance_metrics calculator", null_reason=True)

# The source helper names stay as module aliases: calculators and tests import them by these names.
_DoorGeometry = DoorGeometry
_ZoneGeometry = ZoneGeometry
_extract_door_geometry = extract_door_geometry
_extract_zone_geometry = extract_zone_geometry
_zone_membership = zone_membership
_reconstruct_events = reconstruct_events

#: Head-on pairing radius for the passing-rule head-on test, centre-to-centre metres (no body radii
#: removed), so it is 2 m larger than the 2.0 m edge encounter distance of `processing/windows.py`.
HEAD_ON_CENTRE_DISTANCE_M: float = 4.0


def _offset_zones(zones: list[ZoneGeometry], offset: tuple[float, float]) -> list[ZoneGeometry]:
    """World-frame zones translated into the recorded map frame by the env packing offset."""
    if offset == (0.0, 0.0):
        return zones
    import shapely.affinity

    return [dataclasses.replace(zone, polygon=shapely.affinity.translate(zone.polygon, offset[0], offset[1])) for zone in zones]


def _offset_doors(doors: list[DoorGeometry], offset: tuple[float, float]) -> list[DoorGeometry]:
    """World-frame door centers translated into the recorded map frame."""
    if offset == (0.0, 0.0):
        return doors
    return [dataclasses.replace(door, center_x=door.center_x + offset[0], center_y=door.center_y + offset[1]) for door in doors]


_ZONE_OUTPUT_KEYS: typing.Final[tuple[str, ...]] = (
    "doorway_blocking_time",
    "quiet_zone_dwell_seconds",
    "restricted_zone_entries",
    "speed_zone_violation_seconds",
    "speed_zone_violations",
)


def _speed_zone_metrics(zone_idx: np.ndarray, speed: np.ndarray, dt: np.ndarray, zones: list[ZoneGeometry]) -> tuple[int, float]:
    mask = np.zeros(len(zone_idx), dtype=bool)
    for i, j in enumerate(zone_idx):
        if j == -1:
            continue
        max_speed = zones[j].max_speed
        if max_speed is not None and speed[i] > max_speed:
            mask[i] = True

    violations = 0
    prev = False
    for flag in mask:
        if flag and not prev:
            violations += 1
        prev = flag

    seconds = float(sum(dt[i] for i in range(len(dt)) if mask[i]))
    return violations, seconds


def _quiet_zone_dwell(zone_idx: np.ndarray, dt: np.ndarray, zones: list[ZoneGeometry]) -> float:
    total = 0.0
    for i in range(len(dt)):
        j = zone_idx[i]
        if j != -1 and zones[j].quiet:
            total += dt[i]
    return total


def _restricted_zone_entries(zone_idx: np.ndarray, zones: list[ZoneGeometry]) -> int:
    entries = 0
    for i in range(1, len(zone_idx)):
        j = zone_idx[i]
        if j == -1 or not zones[j].restricted:
            continue
        if zone_idx[i - 1] != j:
            entries += 1
    return entries


def _door_open_series(events: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-door sorted (t_ns, is_open) event series keyed by authored door name."""
    door_events = events.filter((pl.col("kind") == "door") & (pl.col("field") == "state")).sort("t_ns")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if len(door_events) == 0:
        return result

    for _, group in door_events.group_by("entity"):
        g = group.sort("t_ns")
        name = entity_name(str(g["entity"][0]))
        times = g["t_ns"].to_numpy()
        is_open = np.array([v == "open" for v in g["current"].to_list()])
        result[name] = (times, is_open)
    return result


def _is_open_at(door_series: dict[str, tuple[np.ndarray, np.ndarray]], name: str, t_ns: int) -> bool:
    entry = door_series.get(name)
    if entry is None:
        return False

    times, is_open = entry
    idx = asof_position(times, t_ns)
    if idx < 0:
        return False
    return bool(is_open[idx])


class ComplianceMetricsCalculator(BaseMetricCalculator):
    """
    Compliance scoring against zone and door semantic annotations (SPEC_M1 M1.2).

    Zone membership is evaluated per odom sample against zone polygons loaded from
    the recorded world asset by name, not from the semantic snapshot (which only
    republishes the same literal YAML values), flattened into the single map frame
    the runtime renders via `World.compact_world`. Doorway blocking
    replays the same door-state transitions reconstructed from `semantic_snapshot`
    that `_time_waiting_at_doors` uses, seeded from the entity's first recorded
    snapshot value.
    Single-robot attribution assumption per SPEC_M1 M1.3. Returns schema-consistent
    `None` for every output key when the recorded world is not locally present.

    `doorway_blocking_time` is robot-attributed (ego pose only). Elevator etiquette metrics
    (`blocked_closing_door`, `boarded_before_exit`, stranded peds) live in
    `mechanism_metrics`, which uses the per-occupant semantic `members`.
    """

    NAME = "compliance_metrics"
    SERIES_KEY = ("t_ns",)
    SERIES_REQUIRED_TOPICS = ("robot_tracks",)
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_tracks"]
    PARAMS = ()
    passing_convention = "right"

    OUTPUTS = (
        SPEED_ZONE_VIOLATIONS,
        SPEED_ZONE_VIOLATION_SECONDS,
        QUIET_ZONE_DWELL_SECONDS,
        RESTRICTED_ZONE_ENTRIES,
        DOORWAY_BLOCKING_TIME,
        PASSING_RULE_COMPLIANCE,
    )
    PRIMARY_OUTPUTS = ["speed_zone_violations", "passing_rule_compliance"]

    STATIONARY_THRESHOLD = 0.05

    def __init__(self, robot_params: RobotParams) -> None:
        super().__init__(robot_params)
        self._world_cache: dict[str, tuple[list[ZoneGeometry], list[DoorGeometry]] | None] = {}

    def _load_world(self, world_name: str) -> tuple[list[ZoneGeometry], list[DoorGeometry]] | None:
        if world_name in self._world_cache:
            return self._world_cache[world_name]

        loaded = load_world(world_name)
        if loaded is None:
            self._world_cache[world_name] = None
            return None
        result = (extract_zone_geometry(loaded.flattened), extract_door_geometry(loaded.flattened))
        self._world_cache[world_name] = result
        return result

    def _doorway_blocking_time(
        self,
        episode: AlignedEpisodeBundle,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        speed: np.ndarray,
        t_ns: np.ndarray,
        doors: list[DoorGeometry],
    ) -> float:
        if not doors:
            return 0.0

        events = reconstruct_events(robot_env_snapshot(episode.semantic_snapshot, episode.robot_name))
        if len(events) == 0:
            return 0.0

        door_series = _door_open_series(events)
        n = len(pos_x)
        if n < 2:
            return None

        dt = np.diff(t_ns) / 1e9

        total = 0.0
        for i in range(n - 1):
            if speed[i] > self.STATIONARY_THRESHOLD:
                continue
            t = int(t_ns[i])
            for door in doors:
                dx = pos_x[i] - door.center_x
                dy = pos_y[i] - door.center_y
                if dx * dx + dy * dy <= door.radius**2 and _is_open_at(door_series, door.name, t):
                    total += dt[i]
                    break
        return total

    def _compute_passing_compliance(self, episode: AlignedEpisodeBundle) -> float | None:
        peds_df = self.native_ped_frame(episode)
        if peds_df is None or "x" not in peds_df.columns:
            return None

        pos_x, pos_y, yaw, t_odom = self.resolve_native_pose(episode)
        if len(pos_x) == 0:
            if episode.data is None or len(episode.data) == 0:
                return None
            pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)
            t_odom = episode.data["t_ns"].to_numpy().astype(np.int64) if "t_ns" in episode.data.columns else np.arange(len(pos_x), dtype=np.int64)

        if len(pos_x) < 2:
            return None

        peds_time_ns = peds_df["t_ns"].unique().sort().to_numpy()
        n_times = len(peds_time_ns)
        if n_times == 0:
            return None

        vx_full, vy_full = self.velocity_from_pose(pos_x, pos_y, t_odom)
        rvx = self.values_at_times(vx_full, t_odom, peds_time_ns)
        rvy = self.values_at_times(vy_full, t_odom, peds_time_ns)
        rpx, rpy, _ = self.pose_at_times(peds_time_ns, pos_x, pos_y, yaw, t_odom)

        robot_interp_lf = pl.LazyFrame(
            {
                "t_ns": peds_time_ns,
                "rx": rpx,
                "ry": rpy,
                "rvx": rvx,
                "rvy": rvy,
            }
        ).sort("t_ns")

        peds_lf = peds_df.lazy().sort(["ped_id", "t_ns"])
        dt_ped_s = pl.col("t_ns").diff().over("ped_id") * 1e-9
        dt_ped_safe = pl.when(dt_ped_s > 0).then(dt_ped_s).otherwise(1e-3)

        has_vx = "vx" in peds_df.columns
        has_vy = "vy" in peds_df.columns
        # A first sample has no previous step, so its velocity stays NULL: it is unknown, not zero.
        peds_with_vel = peds_lf.with_columns(
            vx=(pl.when(pl.col("vx").is_null() | pl.col("vx").is_nan()).then(pl.col("x").diff().over("ped_id") / dt_ped_safe).otherwise(pl.col("vx")) if has_vx else pl.col("x").diff().over("ped_id") / dt_ped_safe),
            vy=(pl.when(pl.col("vy").is_null() | pl.col("vy").is_nan()).then(pl.col("y").diff().over("ped_id") / dt_ped_safe).otherwise(pl.col("vy")) if has_vy else pl.col("y").diff().over("ped_id") / dt_ped_safe),
        )

        joined = peds_with_vel.join(robot_interp_lf, on="t_ns", how="inner")

        dx = pl.col("x") - pl.col("rx")
        dy = pl.col("y") - pl.col("ry")
        dist = (dx.pow(2) + dy.pow(2)).sqrt()
        speed_r = (pl.col("rvx").pow(2) + pl.col("rvy").pow(2)).sqrt()
        speed_p = (pl.col("vx").pow(2) + pl.col("vy").pow(2)).sqrt()
        denom = speed_r * speed_p
        cos_angle = pl.when(denom > 1e-6).then((pl.col("rvx") * pl.col("vx") + pl.col("rvy") * pl.col("vy")) / denom).otherwise(0.0)

        is_head_on = (dist <= HEAD_ON_CENTRE_DISTANCE_M) & (speed_r > 0.05) & (speed_p > 0.05) & (cos_angle < -0.3)
        lat_offset = pl.col("rvx") * dy - pl.col("rvy") * dx

        pairs_analyzed = (
            joined.with_columns(
                dist=dist,
                is_head_on=is_head_on,
                lat_offset=lat_offset,
            )
            .group_by("t_ns")
            .agg(
                head_on_found=pl.col("is_head_on").any(),
                curr_min_dist=pl.col("dist").filter(pl.col("is_head_on")).min(),
                curr_lat=pl.col("lat_offset").filter(pl.col("is_head_on")).sort_by(pl.col("dist").filter(pl.col("is_head_on"))).first(),
            )
            .sort("t_ns")
            .with_columns(
                head_on_found=pl.col("head_on_found").fill_null(False),
            )
            .with_columns(
                bout_id=pl.col("head_on_found").rle_id(),
            )
            .collect()
        )

        bouts = pairs_analyzed.filter(pl.col("head_on_found")).partition_by("bout_id", maintain_order=True)
        if not bouts:
            return None

        encounters_total = len(bouts)
        encounters_compliant = 0

        for bout in bouts:
            min_idx = bout["curr_min_dist"].arg_min()
            lat_at_min = bout["curr_lat"][min_idx]
            if self.passing_convention == "right" and lat_at_min >= 0.0:
                encounters_compliant += 1
            elif self.passing_convention == "left" and lat_at_min <= 0.0:
                encounters_compliant += 1

        return float(encounters_compliant) / encounters_total

    def missing_input_provenance(self, episode: AlignedEpisodeBundle) -> dict[str, str]:
        """Zone and door compliance needs the recorded world semantics; without it the values are unknown."""
        if episode.semantic_snapshot is None or episode.semantic_snapshot.height == 0:
            return {key: f"{key}:topic_missing:world_semantics" for key in _ZONE_OUTPUT_KEYS}
        return {}

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        result: dict[str, float | None] = {metric.name: None for metric in self.OUTPUTS}

        # 1. Passing Rule Compliance (Independent of world)
        result["passing_rule_compliance"] = self._compute_passing_compliance(episode)

        # 2. Zone/Door Metrics (Requires the world and its recorded semantic snapshot). The world file
        # alone is not enough: without the recorded world semantics the zone/door occupancy is unknown,
        # so the values stay null instead of collapsing to a fabricated zero.
        has_semantics = episode.semantic_snapshot is not None and episode.semantic_snapshot.height > 0
        if self.world is not None and episode.start_pos:
            loaded = self._load_world(self.world)
            if loaded is not None and has_semantics:
                zones, doors = loaded
                pos_x, pos_y, _yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)

                if episode.data is not None and len(episode.data) > 0 and "vel_linear" in episode.data.columns:
                    speed = np.abs(episode.data["vel_linear"].to_numpy())
                    t_ns = episode.data["t_ns"].to_numpy()
                    n = len(pos_x)
                    if n > 0 and len(speed) == n:
                        dt = np.diff(t_ns) / 1e9
                        zone_idx = zone_membership(pos_x, pos_y, zones)
                        violations, violation_seconds = _speed_zone_metrics(zone_idx, speed, dt, zones)
                        quiet_seconds = _quiet_zone_dwell(zone_idx, dt, zones)
                        restricted_entries = _restricted_zone_entries(zone_idx, zones)
                        doorway_blocking = self._doorway_blocking_time(episode, pos_x, pos_y, speed, t_ns, doors)

                        result.update(
                            {
                                "speed_zone_violations": violations,
                                "speed_zone_violation_seconds": violation_seconds,
                                "quiet_zone_dwell_seconds": quiet_seconds,
                                "restricted_zone_entries": restricted_entries,
                                "doorway_blocking_time": doorway_blocking,
                            }
                        )

        return result
