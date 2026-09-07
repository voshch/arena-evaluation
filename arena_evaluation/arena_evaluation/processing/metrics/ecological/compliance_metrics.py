from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ZoneGeometry:
    name: str
    polygon: object
    max_speed: float | None
    quiet: bool
    restricted: bool


@dataclasses.dataclass
class _DoorGeometry:
    name: str
    center_x: float
    center_y: float
    radius: float


def _extract_zone_geometry(flattened: typing.Any, require_annotation: bool = True) -> list[_ZoneGeometry]:
    import shapely

    zones: list[_ZoneGeometry] = []
    for zone in flattened.zones:
        if len(zone.corners) < 3:
            continue

        max_speed = None
        quiet = False
        restricted = False
        has_annotation = False
        for cfg in zone.semantics:
            if cfg.role == "state" and cfg.name == "max_speed":
                has_annotation = True
                if cfg.value is not None:
                    max_speed = float(cfg.value)
            elif cfg.role == "predicate" and cfg.name == "quiet":
                has_annotation = True
                quiet = bool(cfg.value)
            elif cfg.role == "predicate" and cfg.name == "restricted":
                has_annotation = True
                restricted = bool(cfg.value)

        if require_annotation and not has_annotation:
            continue

        polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.corners])
        zones.append(_ZoneGeometry(name=zone.name, polygon=polygon, max_speed=max_speed, quiet=quiet, restricted=restricted))
    return zones


def _extract_door_geometry(flattened: typing.Any) -> list[_DoorGeometry]:
    doors: list[_DoorGeometry] = []
    for door in flattened.all_doors:
        center_x = (door.start.x + door.end.x) / 2.0
        center_y = (door.start.y + door.end.y) / 2.0
        radius = max(door.activation_distance)
        doors.append(_DoorGeometry(name=door.name, center_x=center_x, center_y=center_y, radius=radius))
    return doors


def _offset_zones(zones: list[_ZoneGeometry], offset: tuple[float, float]) -> list[_ZoneGeometry]:
    """World-frame zones translated into the recorded map frame by the env packing offset."""
    if offset == (0.0, 0.0):
        return zones
    import shapely.affinity

    return [
        dataclasses.replace(zone, polygon=shapely.affinity.translate(zone.polygon, offset[0], offset[1]))
        for zone in zones
    ]


def _offset_doors(doors: list[_DoorGeometry], offset: tuple[float, float]) -> list[_DoorGeometry]:
    """World-frame door centers translated into the recorded map frame."""
    if offset == (0.0, 0.0):
        return doors
    return [
        dataclasses.replace(door, center_x=door.center_x + offset[0], center_y=door.center_y + offset[1])
        for door in doors
    ]


def _zone_membership(pos_x: np.ndarray, pos_y: np.ndarray, zones: list[_ZoneGeometry]) -> np.ndarray:
    import shapely

    idx = np.full(len(pos_x), -1, dtype=np.int64)
    for i in range(len(pos_x)):
        point = shapely.Point(pos_x[i], pos_y[i])
        for j, zone in enumerate(zones):
            if zone.polygon.covers(point):
                idx[i] = j
                break
    return idx


def _speed_zone_metrics(
    zone_idx: np.ndarray, speed: np.ndarray, dt: np.ndarray, zones: list[_ZoneGeometry]
) -> tuple[int, float]:
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


def _quiet_zone_dwell(zone_idx: np.ndarray, dt: np.ndarray, zones: list[_ZoneGeometry]) -> float:
    total = 0.0
    for i in range(len(dt)):
        j = zone_idx[i]
        if j != -1 and zones[j].quiet:
            total += dt[i]
    return total


def _restricted_zone_entries(zone_idx: np.ndarray, zones: list[_ZoneGeometry]) -> int:
    entries = 0
    for i in range(1, len(zone_idx)):
        j = zone_idx[i]
        if j == -1 or not zones[j].restricted:
            continue
        if zone_idx[i - 1] != j:
            entries += 1
    return entries


_RECONSTRUCTED_EVENTS_SCHEMA = {
    "time_ns": pl.Int64, "entity": pl.Utf8, "kind": pl.Utf8, "field": pl.Utf8,
    "previous": pl.Utf8, "current": pl.Utf8,
}


def _stringify_snapshot_value(row: dict) -> str:
    """Native snapshot value to the historic event wire convention: lowercase bool, str(float), raw string."""
    if row["field_kind"] == "predicate":
        return "true" if row["value_bool"] else "false"
    if row["field_kind"] == "continuous":
        return str(row["value_num"])
    return row["value_str"] if row["value_str"] is not None else ""


def _reconstruct_events(snapshot: pl.DataFrame | None) -> pl.DataFrame:
    """Event-equivalent (time_ns, entity, kind, field, previous, current) rows, derived by
    keeping only the change points of each entity+field's stepwise snapshot series. The first
    kept row per series is the seed value (from the episode's seed snapshot), `previous` empty.
    """
    empty = pl.DataFrame(schema=_RECONSTRUCTED_EVENTS_SCHEMA)
    if snapshot is None or len(snapshot) == 0:
        return empty

    rows = snapshot.filter(pl.col("field_kind") != "members").sort("time_ns")
    if len(rows) == 0:
        return empty

    out_time: list[int] = []
    out_entity: list[str] = []
    out_kind: list[str] = []
    out_field: list[str] = []
    out_previous: list[str] = []
    out_current: list[str] = []

    for (entity, kind, field), group in rows.group_by(["entity", "kind", "field"]):
        previous: str | None = None
        for row in group.sort("time_ns").iter_rows(named=True):
            current = _stringify_snapshot_value(row)
            if current != previous:
                out_time.append(row["time_ns"])
                out_entity.append(entity)
                out_kind.append(kind)
                out_field.append(field)
                out_previous.append(previous if previous is not None else "")
                out_current.append(current)
                previous = current

    if not out_time:
        return empty

    return pl.DataFrame({
        "time_ns": out_time,
        "entity": out_entity,
        "kind": out_kind,
        "field": out_field,
        "previous": out_previous,
        "current": out_current,
    }).sort("time_ns")


def _door_open_series(events: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-door sorted (time_ns, is_open) event series, name keyed by entity suffix."""
    door_events = events.filter((pl.col("kind") == "door") & (pl.col("field") == "state")).sort("time_ns")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if len(door_events) == 0:
        return result

    for _, group in door_events.group_by("entity"):
        g = group.sort("time_ns")
        name = str(g["entity"][0]).rsplit("/", 1)[-1]
        times = g["time_ns"].to_numpy()
        is_open = np.array([v == "open" for v in g["current"].to_list()])
        result[name] = (times, is_open)
    return result


def _is_open_at(door_series: dict[str, tuple[np.ndarray, np.ndarray]], name: str, t_ns: int) -> bool:
    entry = door_series.get(name)
    if entry is None:
        return False

    times, is_open = entry
    idx = int(np.searchsorted(times, t_ns, side="right")) - 1
    if idx < 0:
        return False
    return bool(is_open[idx])


class ComplianceMetricsCalculator(BaseMetricCalculator):
    """
    Compliance scoring against zone and door semantic annotations (SPEC_M1 M1.2).

    Zone membership is evaluated per odom sample against zone polygons loaded from
    the recorded world asset by name, not from the semantic snapshot (which only
    republishes the same literal YAML values), flattened into the single map frame
    the runtime renders via `WorldDescription.compact_world`. Doorway blocking
    replays the same door-state transitions reconstructed from `semantic_snapshot`
    that `_time_waiting_at_doors` uses, seeded from the entity's first recorded
    snapshot value.
    Single-robot attribution assumption per SPEC_M1 M1.3. Returns schema-consistent
    `None` for every output key when the recorded world is not locally present.

    Elevator etiquette metrics (`blocked_closing_door`, `boarded_before_exit`,
    `stranded_ped_seconds`) are deferred to M2: v1 `occupants` is a scalar with no
    per-occupant identity or position (SPEC_M1 M1.2).
    """

    NAME = "compliance_metrics"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = [("tf_gt", "odom"), "peds"]
    passing_convention = "right"

    UNITS = {
        "speed_zone_violations": "",
        "speed_zone_violation_seconds": "s",
        "quiet_zone_dwell_seconds": "s",
        "restricted_zone_entries": "",
        "doorway_blocking_time": "s",
        "passing_rule_compliance": "",
    }

    STATIONARY_THRESHOLD = 0.05

    world: str | None = None

    def __init__(self, robot_params: typing.Any) -> None:
        super().__init__(robot_params)
        self._world_cache: dict[str, tuple[list[_ZoneGeometry], list[_DoorGeometry]] | None] = {}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "speed_zone_violations",
            "speed_zone_violation_seconds",
            "quiet_zone_dwell_seconds",
            "restricted_zone_entries",
            "doorway_blocking_time",
            "passing_rule_compliance",
        ]

    def _load_world(self, world_name: str) -> tuple[list[_ZoneGeometry], list[_DoorGeometry]] | None:
        if world_name in self._world_cache:
            return self._world_cache[world_name]

        from arena_simulation_setup.tree.World import WorldIdentifier

        try:
            view = WorldIdentifier(world_name).resolve_sync()
            world = view.load()
        except FileNotFoundError as e:
            logger.warning("compliance_metrics: world '%s' not available locally: %s", world_name, e)
            self._world_cache[world_name] = None
            return None

        origins = view.level_origins()
        if origins is None:
            origins = {level_id: (0.0, 0.0) for level_id in world.levels}
        flattened = world.compact_world(origins)

        result = (_extract_zone_geometry(flattened), _extract_door_geometry(flattened))
        self._world_cache[world_name] = result
        return result

    def _doorway_blocking_time(
        self,
        episode: AlignedEpisodeBundle,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        speed: np.ndarray,
        time_ns: np.ndarray,
        doors: list[_DoorGeometry],
    ) -> float:
        if not doors:
            return 0.0

        events = _reconstruct_events(episode.semantic_snapshot)
        if len(events) == 0:
            return 0.0

        door_series = _door_open_series(events)
        n = len(pos_x)
        if n < 2:
            return 0.0

        dt = np.diff(time_ns) / 1e9

        total = 0.0
        for i in range(n - 1):
            if speed[i] > self.STATIONARY_THRESHOLD:
                continue
            t = int(time_ns[i])
            for door in doors:
                dx = pos_x[i] - door.center_x
                dy = pos_y[i] - door.center_y
                if dx * dx + dy * dy <= door.radius**2 and _is_open_at(door_series, door.name, t):
                    total += dt[i]
                    break
        return total

    def _compute_passing_compliance(self, episode: AlignedEpisodeBundle) -> float | None:
        peds_df = self.native_ped_frame(episode)
        if peds_df is None or "peds_positions" not in peds_df.columns:
            return None
        
        pos_x, pos_y, yaw, t_odom = self.resolve_native_pose(episode)
        if len(pos_x) == 0:
            return None
            
        peds_time_ns = peds_df["time_ns"].to_numpy()
        N = len(peds_time_ns)
        if N == 0:
            return None
            
        vx_full, vy_full = self.velocity_from_pose(pos_x, pos_y, t_odom)
        rvx = self.values_at_times(vx_full, t_odom, peds_time_ns)
        rvy = self.values_at_times(vy_full, t_odom, peds_time_ns)
        rpx, rpy, _ = self.pose_at_times(peds_time_ns, pos_x, pos_y, yaw, t_odom)
        dt = np.diff(peds_time_ns, prepend=peds_time_ns[0]) / 1e9
        
        peds_positions = peds_df["peds_positions"].to_list()
        num_peds_col = peds_df["num_pedestrians"].to_numpy() if "num_pedestrians" in peds_df.columns else None
        peds_twists_list = peds_df["peds_twists"].to_list() if "peds_twists" in peds_df.columns else None
        
        in_encounter = False
        min_dist = float('inf')
        lat_offset_at_min = 0.0
        
        encounters_total = 0
        encounters_compliant = 0
        
        prev_peds_arr: np.ndarray | None = None
        for i in range(N):
            peds_arr = self._parse_peds(peds_positions[i], num_peds_col[i] if num_peds_col is not None else None)
            if peds_arr.shape[0] == 0:
                prev_peds_arr = None
                continue

            ped_vels = None
            if peds_twists_list is not None and i < len(peds_twists_list):
                tw_raw = peds_twists_list[i]
                if tw_raw and len(tw_raw) > 0:
                    import ast
                    if isinstance(tw_raw, str):
                        try:
                            tw_raw = ast.literal_eval(tw_raw)
                        except Exception:
                            tw_raw = []
                    tw_arr = np.array(tw_raw, dtype=np.float64)
                    if tw_arr.size > 0 and len(tw_arr) % 3 == 0:
                        ped_vels = tw_arr.reshape(-1, 3)

            # Finite difference fallback if twists missing
            if ped_vels is None and prev_peds_arr is not None and prev_peds_arr.shape == peds_arr.shape:
                dt_i = max(float(dt[i]), 1e-3)
                ped_vels = (peds_arr - prev_peds_arr) / dt_i
            prev_peds_arr = peds_arr

            rx, ry = float(rpx[i]), float(rpy[i])
            vr_x, vr_y = float(rvx[i]), float(rvy[i])
            speed_r = np.hypot(vr_x, vr_y)
            
            head_on_found = False
            curr_min_dist = float('inf')
            curr_lat = 0.0
            
            for j in range(peds_arr.shape[0]):
                vp_x, vp_y = 0.0, 0.0
                if ped_vels is not None and j < ped_vels.shape[0]:
                    vp_x, vp_y = float(ped_vels[j, 0]), float(ped_vels[j, 1])
                speed_p = np.hypot(vp_x, vp_y)
                
                px, py = float(peds_arr[j, 0]), float(peds_arr[j, 1])
                dist = np.hypot(px - rx, py - ry)
                
                if dist <= 4.0 and speed_r > 0.05 and speed_p > 0.05:
                    cos_angle = (vr_x * vp_x + vr_y * vp_y) / (speed_r * speed_p)
                    # Opposing direction: approaching each other within 120 degrees
                    if cos_angle < -0.3:
                        head_on_found = True
                        if dist < curr_min_dist:
                            curr_min_dist = dist
                            # Positive when pedestrian is to the robot's left (robot passes on right)
                            curr_lat = vr_x * (py - ry) - vr_y * (px - rx)
            
            if head_on_found:
                if not in_encounter:
                    in_encounter = True
                    min_dist = float('inf')
                
                if curr_min_dist < min_dist:
                    min_dist = curr_min_dist
                    lat_offset_at_min = curr_lat
            else:
                if in_encounter:
                    in_encounter = False
                    encounters_total += 1
                    if self.passing_convention == "right" and lat_offset_at_min >= 0.0:
                        encounters_compliant += 1
                    elif self.passing_convention == "left" and lat_offset_at_min <= 0.0:
                        encounters_compliant += 1
                        
        if in_encounter:
            encounters_total += 1
            if self.passing_convention == "right" and lat_offset_at_min >= 0.0:
                encounters_compliant += 1
            elif self.passing_convention == "left" and lat_offset_at_min <= 0.0:
                encounters_compliant += 1
                
        if encounters_total == 0:
            return 1.0
        return float(encounters_compliant) / encounters_total


    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        result = {k: None for k in self.output_keys()}

        # 1. Passing Rule Compliance (Independent of world)
        result["passing_rule_compliance"] = self._compute_passing_compliance(episode)

        # 2. Zone/Door Metrics (Requires world)
        if self.world is not None and episode.start_pos:
            loaded = self._load_world(self.world)
            if loaded is not None:
                zones, doors = loaded
                pos_x, pos_y, _yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)

                if episode.data is not None and len(episode.data) > 0 and "vel_linear" in episode.data.columns:
                    speed = np.abs(episode.data["vel_linear"].to_numpy())
                    time_ns = episode.data["time_ns"].to_numpy()
                    n = len(pos_x)
                    if n > 0 and len(speed) == n:
                        dt = np.diff(time_ns) / 1e9
                        zone_idx = _zone_membership(pos_x, pos_y, zones)
                        violations, violation_seconds = _speed_zone_metrics(zone_idx, speed, dt, zones)
                        quiet_seconds = _quiet_zone_dwell(zone_idx, dt, zones)
                        restricted_entries = _restricted_zone_entries(zone_idx, zones)
                        doorway_blocking = self._doorway_blocking_time(episode, pos_x, pos_y, speed, time_ns, doors)

                        result.update({
                            "speed_zone_violations": violations,
                            "speed_zone_violation_seconds": violation_seconds,
                            "quiet_zone_dwell_seconds": quiet_seconds,
                            "restricted_zone_entries": restricted_entries,
                            "doorway_blocking_time": doorway_blocking,
                        })

        return result
