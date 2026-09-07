from __future__ import annotations

import bisect
import dataclasses
import logging
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    _DoorGeometry,
    _extract_door_geometry,
    _offset_doors,
    _offset_zones,
    _reconstruct_events,
    _zone_membership,
)
from arena_evaluation.processing.metrics.ecological.semantic_metrics import (
    _parse_bool,
    _parse_float,
)

from arena_evaluation.storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)

_CMD_VEL_COLUMNS = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")


@dataclasses.dataclass
class _CapZoneGeometry:
    name: str
    polygon: object


def _extract_occupancy_zone_geometry(flattened: typing.Any) -> list[_CapZoneGeometry]:
    """Zones annotated with the occupancy_cap preset (`cap`, `occupancy`, or `over_cap`)."""
    import shapely

    zones: list[_CapZoneGeometry] = []
    for zone in flattened.zones:
        if len(zone.corners) < 3:
            continue

        has_cap = any(
            (cfg.role == "state" and cfg.name in ("cap", "occupancy"))
            or (cfg.role == "predicate" and cfg.name == "over_cap")
            for cfg in zone.semantics
        )
        if not has_cap:
            continue

        polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.corners])
        zones.append(_CapZoneGeometry(name=zone.name, polygon=polygon))
    return zones


def _windows(
    events: pl.DataFrame,
    kind: str,
    field: str,
    truthy: typing.Callable[[str], bool],
    end_time_ns: int | None,
) -> dict[str, list[tuple[int, int]]]:
    """Per-entity true intervals for one field, seeded false before the first recorded row,
    held to `end_time_ns` when still true, keyed by the name suffix after the last `/`."""
    rows = events.filter((pl.col("kind") == kind) & (pl.col("field") == field)).sort("time_ns")
    windows: dict[str, list[tuple[int, int]]] = {}
    if len(rows) == 0:
        return windows

    for _entity, group in rows.group_by("entity"):
        g = group.sort("time_ns")
        name = str(g["entity"][0]).rsplit("/", 1)[-1]
        state = False
        start: int | None = None
        entity_windows: list[tuple[int, int]] = []

        for row in g.iter_rows(named=True):
            t = row["time_ns"]
            new_state = truthy(row["current"])
            if new_state and not state:
                start = t
            elif not new_state and state and start is not None:
                entity_windows.append((start, t))
                start = None
            state = new_state

        if state and start is not None and end_time_ns is not None and end_time_ns > start:
            entity_windows.append((start, end_time_ns))

        if entity_windows:
            windows[name] = entity_windows

    return windows


def _active_at(intervals: list[tuple[int, int]], t_ns: int) -> bool:
    return any(start <= t_ns < end for start, end in intervals)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _used_elevator_during_alarm(events: pl.DataFrame, end_time_ns: int | None) -> int:
    """Count of alarm (`schedule.active`) windows overlapped by nonzero elevator occupancy,
    the single-robot proxy since per-episode `members` identity is unavailable."""
    alarm_windows = _windows(events, "schedule", "active", _parse_bool, end_time_ns)
    occupancy_windows = _windows(events, "elevator", "occupants", lambda v: _parse_float(v) > 0.0, end_time_ns)

    all_alarm = [w for entity_windows in alarm_windows.values() for w in entity_windows]
    all_occupancy = [w for entity_windows in occupancy_windows.values() for w in entity_windows]

    return sum(
        1
        for a_start, a_end in all_alarm
        if any(_overlaps(a_start, a_end, o_start, o_end) for o_start, o_end in all_occupancy)
    )


def _entered_over_cap_zone(
    events: pl.DataFrame,
    zone_idx: np.ndarray,
    time_ns: np.ndarray,
    end_time_ns: int | None,
    zone_names: list[str],
) -> int:
    """Count of outside-to-inside transitions into an occupancy_cap zone while it is over cap."""
    over_cap_windows = _windows(events, "occupancy_cap", "over_cap", _parse_bool, end_time_ns)
    if not over_cap_windows:
        return 0

    count = 0
    for i in range(1, len(zone_idx)):
        j = zone_idx[i]
        if j == -1 or zone_idx[i - 1] == j:
            continue
        windows = over_cap_windows.get(zone_names[j], [])
        if _active_at(windows, int(time_ns[i])):
            count += 1
    return count


def _ran_red_signal(
    events: pl.DataFrame,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    time_ns: np.ndarray,
    end_time_ns: int | None,
    doors: list[_DoorGeometry],
) -> int:
    """Count of entries into the same-named door's activation radius while the signal is
    `stop`. Signals carry no geometry, the paired door's radius stands in for the segment."""
    stop_windows = _windows(events, "signal", "stop", _parse_bool, end_time_ns)
    door_by_name = {door.name: door for door in doors}

    count = 0
    for signal_name, windows in stop_windows.items():
        door = door_by_name.get(signal_name)
        if door is None:
            continue

        prev_inside = False
        for i in range(len(pos_x)):
            dx = pos_x[i] - door.center_x
            dy = pos_y[i] - door.center_y
            inside = dx * dx + dy * dy <= door.radius**2
            if inside and not prev_inside and _active_at(windows, int(time_ns[i])):
                count += 1
            prev_inside = inside

    return count


def _cmd_vel_change_times(data: pl.DataFrame, epsilon: float) -> list[int] | None:
    """Sorted sample times where the recorded `cmd_vel` changed by more than `epsilon`."""
    cmd_cols = [c for c in _CMD_VEL_COLUMNS if c in data.columns]
    if not cmd_cols:
        return None

    time_ns = data["time_ns"].to_numpy()
    if len(time_ns) < 2:
        return []

    values = np.stack([data[c].to_numpy() for c in cmd_cols], axis=1)
    diffs = np.abs(np.diff(values, axis=0))
    changed = np.any(diffs > epsilon, axis=1)
    return [int(time_ns[i + 1]) for i in range(len(changed)) if changed[i]]


def _replan_triggers(events: pl.DataFrame) -> list[int]:
    """Trigger times: door `open` false->true, gate `locked` true->false, any signal
    `state` change, any schedule `active` change (SPEC_M2 M2.C8)."""
    triggers: list[int] = []

    door_open = events.filter(
        (pl.col("kind") == "door")
        & (pl.col("field") == "open")
        & (pl.col("previous") == "false")
        & (pl.col("current") == "true")
    )
    triggers.extend(door_open["time_ns"].to_list())

    gate_unlock = events.filter(
        (pl.col("kind") == "gate")
        & (pl.col("field") == "locked")
        & (pl.col("previous") == "true")
        & (pl.col("current") == "false")
    )
    triggers.extend(gate_unlock["time_ns"].to_list())

    signal_state = events.filter((pl.col("kind") == "signal") & (pl.col("field") == "state"))
    triggers.extend(signal_state["time_ns"].to_list())

    schedule_active = events.filter((pl.col("kind") == "schedule") & (pl.col("field") == "active"))
    triggers.extend(schedule_active["time_ns"].to_list())

    return sorted(triggers)


def _latency_distribution(triggers: list[int], changes: list[int]) -> list[float]:
    """Seconds from each trigger to the next `cmd_vel` change, dropping triggers with none."""
    latencies: list[float] = []
    for t in triggers:
        idx = bisect.bisect_right(changes, t)
        if idx < len(changes):
            latencies.append((changes[idx] - t) / 1e9)
    return latencies


class RegimeMetricsCalculator(BaseMetricCalculator):
    """
    Regime-change metrics replayed from the semantic snapshot, reconstructed into a
    per-field change-point series (see `_reconstruct_events`).

    used_elevator_during_alarm: alarm windows overlapped by nonzero elevator occupancy
    (single-robot proxy, `members` is snapshot-only and not carried per episode).
    entered_over_cap_zone: trajectory entries into an over-cap zone via the M1 zone
    geometry path. ran_red_signal: entries into the same-named door's activation radius
    while the signal is `stop`. replan_latency_after_state_change: seconds from each
    state-change trigger to the next `cmd_vel` change (median, p95), `cmd_vel` only
    since `plan` is never joined into the aligned episode.
    Every key is `None` when its kind never appears or the world asset is not local.
    """

    NAME = "regime_metrics"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["odom"]

    UNITS = {
        "used_elevator_during_alarm": "",
        "ran_red_signal": "",
        "entered_over_cap_zone": "",
        "replan_latency_after_state_change_median": "s",
        "replan_latency_after_state_change_p95": "s",
    }

    CMD_CHANGE_EPSILON = 0.05

    world: str | None = None

    def __init__(self, robot_params: typing.Any) -> None:
        super().__init__(robot_params)
        self._world_cache: dict[str, tuple[list[_CapZoneGeometry], list[_DoorGeometry]] | None] = {}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "used_elevator_during_alarm",
            "ran_red_signal",
            "entered_over_cap_zone",
            "replan_latency_after_state_change_median",
            "replan_latency_after_state_change_p95",
        ]

    def _load_world(self, world_name: str) -> tuple[list[_CapZoneGeometry], list[_DoorGeometry]] | None:
        if world_name in self._world_cache:
            return self._world_cache[world_name]

        from arena_simulation_setup.tree.World import WorldIdentifier

        try:
            view = WorldIdentifier(world_name).resolve_sync()
            world = view.load()
        except FileNotFoundError as e:
            logger.warning("regime_metrics: world '%s' not available locally: %s", world_name, e)
            self._world_cache[world_name] = None
            return None

        origins = view.level_origins()
        if origins is None:
            origins = {level_id: (0.0, 0.0) for level_id in world.levels}
        flattened = world.compact_world(origins)

        result = (_extract_occupancy_zone_geometry(flattened), _extract_door_geometry(flattened))
        self._world_cache[world_name] = result
        return result

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())

        events = _reconstruct_events(episode.semantic_snapshot)
        has_events = len(events) > 0
        kinds = set(events["kind"].unique().to_list()) if has_events else set()

        pos_x, pos_y, _yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)

        if episode.data is None or "time_ns" not in episode.data.columns or len(episode.data) == 0:
            return results

        time_ns = episode.data["time_ns"].to_numpy()
        if len(time_ns) == 0:
            return results
        end_time_ns = int(time_ns.max())

        if has_events and {"schedule", "elevator"}.issubset(kinds):
            results["used_elevator_during_alarm"] = _used_elevator_during_alarm(events, end_time_ns)

        if has_events and (kinds & {"door", "gate", "signal", "schedule"}):
            cmd_changes = _cmd_vel_change_times(episode.data, self.CMD_CHANGE_EPSILON)
            if cmd_changes is not None:
                latencies = _latency_distribution(_replan_triggers(events), cmd_changes)
                if latencies:
                    results["replan_latency_after_state_change_median"] = float(np.median(latencies))
                    results["replan_latency_after_state_change_p95"] = float(np.percentile(latencies, 95))

        n = len(pos_x)
        if self.world is not None and episode.start_pos and n > 0 and len(time_ns) == n:
            loaded = self._load_world(self.world)
            if loaded is not None:
                zones, doors = loaded

                if has_events and "occupancy_cap" in kinds and zones:
                    zone_idx = _zone_membership(pos_x, pos_y, zones)
                    zone_names = [zone.name for zone in zones]
                    results["entered_over_cap_zone"] = _entered_over_cap_zone(
                        events, zone_idx, time_ns, end_time_ns, zone_names
                    )

                if has_events and "signal" in kinds:
                    results["ran_red_signal"] = _ran_red_signal(events, pos_x, pos_y, time_ns, end_time_ns, doors)

        return results
