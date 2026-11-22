from __future__ import annotations

import bisect
import dataclasses
import logging
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.caching import load_world
from arena_evaluation.processing.metrics.common.events import (
    reconstruct_events as _reconstruct_events,
)
from arena_evaluation.processing.metrics.common.parsing import (
    parse_bool as _parse_bool,
)
from arena_evaluation.processing.metrics.common.parsing import (
    parse_float as _parse_float,
)
from arena_evaluation.processing.metrics.common.zones import (
    DoorGeometry,
    extract_door_geometry,
    zone_membership,
)
from arena_evaluation.processing.metrics.ecological.mechanism_common import (
    RobotIdentity,
    SignalGeom,
    entity_name,
    episode_end_t_ns,
    mechanism_world_from_description,
    parse_members,
    robot_env_snapshot,
)
from arena_evaluation.storage.registry import MetricScope, NullReason, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

if typing.TYPE_CHECKING:
    import shapely
    from arena_simulation_setup.tree.World import Level

logger = logging.getLogger(__name__)

ENTERED_OVER_CAP_ZONE = define_metric(
    "entered_over_cap_zone",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Count of robot entries into an occupancy-cap zone while it is over cap; NULL when the occupancy-cap stream, world asset or robot track is absent.",
    null_reason=True,
)
RAN_RED_SIGNAL = define_metric(
    "ran_red_signal",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Count of robot entries into a stopped signal's guarded disc; NULL when the signal stream, world asset or robot track is absent.",
    null_reason=True,
)
REPLAN_LATENCY_AFTER_STATE_CHANGE_MEDIAN = define_metric(
    "replan_latency_after_state_change_median",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Median seconds from a trigger to the next cmd_vel change; unanswered triggers are excluded, replan_unanswered_triggers counts them; NULL without either stream.",
    null_reason=True,
)
REPLAN_LATENCY_AFTER_STATE_CHANGE_P95 = define_metric(
    "replan_latency_after_state_change_p95",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="P95 seconds from a trigger to the next cmd_vel change; unanswered triggers are excluded, replan_unanswered_triggers counts them; NULL without either stream.",
    null_reason=True,
)
REPLAN_UNANSWERED_TRIGGERS = define_metric(
    "replan_unanswered_triggers",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Count of replan triggers with no later cmd_vel change, excluded from the latency percentiles; NULL without the trigger or cmd_vel stream.",
    null_reason=True,
)
USED_ELEVATOR_DURING_ALARM_ANYONE = define_metric(
    "used_elevator_during_alarm_anyone",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    description="Alarm windows overlapped by elevator occupancy by anyone aboard.",
    null_reason=True,
)
USED_ELEVATOR_DURING_ALARM_ROBOT = define_metric(
    "used_elevator_during_alarm_robot",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Alarm windows overlapped by the robot itself as an elevator member (NULL when no elevator reports members).",
    null_reason=True,
)

_CMD_VEL_COLUMNS = (
    "linear_x",
    "linear_y",
    "linear_z",
    "angular_x",
    "angular_y",
    "angular_z",
)


@dataclasses.dataclass
class _CapZoneGeometry:
    name: str
    polygon: shapely.Polygon


def _extract_occupancy_zone_geometry(flattened: Level) -> list[_CapZoneGeometry]:
    """Zones annotated with the occupancy_cap preset (`cap`, `occupancy`, or `over_cap`)."""
    import shapely

    zones: list[_CapZoneGeometry] = []
    for zone in flattened.zones:
        if len(zone.polygon) < 3:
            continue

        has_cap = any((cfg.role == "state" and cfg.name in ("cap", "occupancy")) or (cfg.role == "predicate" and cfg.name == "over_cap") for cfg in zone.semantics)
        if not has_cap:
            continue

        polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.polygon])
        zones.append(_CapZoneGeometry(name=zone.name, polygon=polygon))
    return zones


def _windows(
    events: pl.DataFrame,
    kind: str,
    field: str,
    truthy: typing.Callable[[str], bool],
    end_t_ns: int | None,
) -> dict[str, list[tuple[int, int]]]:
    """Per-entity true intervals for one field, seeded false before the first recorded row,
    held to `end_t_ns` when still true, keyed by authored entity name (`entity_name`)."""
    rows = events.filter((pl.col("kind") == kind) & (pl.col("field") == field)).sort("t_ns")
    windows: dict[str, list[tuple[int, int]]] = {}
    if len(rows) == 0:
        return windows

    for _entity, group in rows.group_by("entity"):
        g = group.sort("t_ns")
        name = entity_name(str(g["entity"][0]))
        state = False
        start: int | None = None
        entity_windows: list[tuple[int, int]] = []

        for t_raw, current in zip(g["t_ns"].to_numpy(), g["current"].to_numpy(), strict=True):
            if current is None:
                continue
            t = int(t_raw)
            new_state = truthy(current)
            if new_state and not state:
                start = t
            elif not new_state and state and start is not None:
                entity_windows.append((start, t))
                start = None
            state = new_state

        if state and start is not None and end_t_ns is not None and end_t_ns > start:
            entity_windows.append((start, end_t_ns))

        if entity_windows:
            windows.setdefault(name, []).extend(entity_windows)
            windows[name].sort()

    return windows


def _active_at(intervals: list[tuple[int, int]], t_ns: int) -> bool:
    return any(start <= t_ns < end for start, end in intervals)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _used_elevator_during_alarm(events: pl.DataFrame, end_t_ns: int | None, identity: RobotIdentity | None = None) -> int | None:
    """Count of alarm (`schedule.active`) windows overlapped by elevator occupancy, by anyone aboard when `identity` is None, else by the robot's own `members` (None when no elevator reports members)."""
    if identity is not None and len(events.filter((pl.col("kind") == "elevator") & (pl.col("field") == "members"))) == 0:
        return None
    alarm_windows = _windows(events, "schedule", "active", _parse_bool, end_t_ns)
    if identity is None:
        occupancy_windows = _windows(events, "elevator", "occupants", lambda v: _parse_float(v) > 0.0, end_t_ns)
    else:
        occupancy_windows = _windows(
            events,
            "elevator",
            "members",
            lambda v: identity.any_of(parse_members(v)),
            end_t_ns,
        )

    all_alarm = [w for entity_windows in alarm_windows.values() for w in entity_windows]
    all_occupancy = [w for entity_windows in occupancy_windows.values() for w in entity_windows]

    return sum(1 for a_start, a_end in all_alarm if any(_overlaps(a_start, a_end, o_start, o_end) for o_start, o_end in all_occupancy))


def _entered_over_cap_zone(
    events: pl.DataFrame,
    zone_idx: np.ndarray,
    t_ns: np.ndarray,
    end_t_ns: int | None,
    zone_names: list[str],
) -> int:
    """Count of outside-to-inside transitions into an occupancy_cap zone while it is over cap."""
    over_cap_windows = _windows(events, "occupancy_cap", "over_cap", _parse_bool, end_t_ns)
    if not over_cap_windows:
        return 0

    count = 0
    for i in range(1, len(zone_idx)):
        j = zone_idx[i]
        if j == -1 or zone_idx[i - 1] == j:
            continue
        windows = over_cap_windows.get(zone_names[j], [])
        if _active_at(windows, int(t_ns[i])):
            count += 1
    return count


DEFAULT_SIGNAL_RADIUS = 1.0


def _signal_area(name: str, signals: dict[str, SignalGeom], doors: dict[str, DoorGeometry]) -> tuple[float, float, float] | None:
    """(x, y, radius) a signal guards, from its own params, else the same-named door."""
    signal = signals.get(name)
    door = doors.get(name)
    if signal is not None:
        radius = signal.radius if signal.radius is not None else (door.radius if door is not None else DEFAULT_SIGNAL_RADIUS)
        return signal.x, signal.y, radius
    if door is not None:
        return door.center_x, door.center_y, door.radius
    return None


def _ran_red_signal(
    events: pl.DataFrame,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    t_ns: np.ndarray,
    end_t_ns: int | None,
    doors: list[DoorGeometry],
    signals: list[SignalGeom] | None = None,
) -> int:
    """Count of robot entries into a signal's guarded disc while that signal is `stop`."""
    stop_windows = _windows(events, "signal", "stop", _parse_bool, end_t_ns)
    door_by_name = {door.name: door for door in doors}
    signal_by_name = {signal.name: signal for signal in signals or []}

    count = 0
    for signal_name, windows in stop_windows.items():
        area = _signal_area(signal_name, signal_by_name, door_by_name)
        if area is None:
            continue
        cx, cy, radius = area

        prev_inside = False
        for i in range(len(pos_x)):
            dx = pos_x[i] - cx
            dy = pos_y[i] - cy
            inside = dx * dx + dy * dy <= radius**2
            if inside and not prev_inside and _active_at(windows, int(t_ns[i])):
                count += 1
            prev_inside = inside

    return count


def _cmd_vel_change_times(data: pl.DataFrame, epsilon: float) -> list[int] | None:
    """Sorted sample times where the recorded `cmd_vel` changed by more than `epsilon`."""
    cmd_cols = [c for c in _CMD_VEL_COLUMNS if c in data.columns]
    if not cmd_cols:
        return None

    t_ns = data["t_ns"].to_numpy()
    if len(t_ns) < 2:
        return []

    values = np.stack([data[c].to_numpy() for c in cmd_cols], axis=1)
    diffs = np.abs(np.diff(values, axis=0))
    changed = np.any(diffs > epsilon, axis=1)
    return [int(t_ns[i + 1]) for i in range(len(changed)) if changed[i]]


def _replan_triggers(events: pl.DataFrame) -> list[int]:
    """Trigger times: door `open` false->true, gate `locked` true->false, any signal
    `state` change, any schedule `active` change (SPEC_M2 M2.C8)."""
    door_open = (pl.col("kind") == "door") & (pl.col("field") == "open") & (pl.col("previous") == "false") & (pl.col("current") == "true")
    gate_unlock = (pl.col("kind") == "gate") & (pl.col("field") == "locked") & (pl.col("previous") == "true") & (pl.col("current") == "false")
    signal_state = (pl.col("kind") == "signal") & (pl.col("field") == "state")
    schedule_active = (pl.col("kind") == "schedule") & (pl.col("field") == "active")
    matched = events.filter(door_open | gate_unlock | signal_state | schedule_active).sort("t_ns")
    return [int(t) for t in matched["t_ns"].to_numpy()]


def _latency_distribution(triggers: list[int], changes: list[int]) -> list[float]:
    """Seconds from each trigger to the next `cmd_vel` change, dropping triggers with none."""
    latencies: list[float] = []
    for t in triggers:
        idx = bisect.bisect_right(changes, t)
        if idx < len(changes):
            latencies.append((changes[idx] - t) / 1e9)
    return latencies


def _unanswered_trigger_count(triggers: list[int], changes: list[int]) -> int:
    """Count of triggers with no later `cmd_vel` change."""
    return sum(1 for t in triggers if bisect.bisect_right(changes, t) == len(changes))


class RegimeMetricsCalculator(BaseMetricCalculator):
    """
    Regime-change metrics replayed from the semantic snapshot, reconstructed into a
    per-field change-point series (see `_reconstruct_events`).

    used_elevator_during_alarm_anyone: alarm windows overlapped by nonzero elevator
    occupancy by anyone. used_elevator_during_alarm_robot: alarm windows overlapped by the
    robot itself being in an elevator's `members` (None when no elevator reports members).
    entered_over_cap_zone: trajectory entries into an over-cap zone via the M1 zone
    geometry path. ran_red_signal: robot entries into the signal's guarded disc while the
    signal is `stop`, the disc being the signal preset's `position`/`radius` params when
    authored, else the same-named door's activation radius. replan_latency_after_state_change: seconds from each
    state-change trigger to the next `cmd_vel` change (median, p95), `cmd_vel` only
    since `plan` is never joined into the aligned episode.
    Every key is `None` when its kind never appears or the world asset is not local.
    """

    NAME = "regime_metrics"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["robot_tracks"]

    OUTPUTS = (
        USED_ELEVATOR_DURING_ALARM_ANYONE,
        USED_ELEVATOR_DURING_ALARM_ROBOT,
        RAN_RED_SIGNAL,
        ENTERED_OVER_CAP_ZONE,
        REPLAN_LATENCY_AFTER_STATE_CHANGE_MEDIAN,
        REPLAN_LATENCY_AFTER_STATE_CHANGE_P95,
        REPLAN_UNANSWERED_TRIGGERS,
    )
    DIRECTIONS = {
        "ran_red_signal": Direction.LOWER,
        "entered_over_cap_zone": Direction.LOWER,
        "used_elevator_during_alarm_anyone": Direction.LOWER,
        "used_elevator_during_alarm_robot": Direction.LOWER,
        "replan_latency_after_state_change_median": Direction.LOWER,
        "replan_latency_after_state_change_p95": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = ["ran_red_signal", "entered_over_cap_zone"]

    CMD_CHANGE_EPSILON = 0.05

    def __init__(self, robot_params: RobotParams) -> None:
        super().__init__(robot_params)
        self._world_cache: dict[
            str,
            tuple[list[_CapZoneGeometry], list[DoorGeometry], list[SignalGeom]] | None,
        ] = {}

    def _load_world(self, world_name: str) -> tuple[list[_CapZoneGeometry], list[DoorGeometry], list[SignalGeom]] | None:
        if world_name in self._world_cache:
            return self._world_cache[world_name]

        loaded = load_world(world_name)
        if loaded is None:
            self._world_cache[world_name] = None
            return None
        signals = mechanism_world_from_description(loaded.world, loaded.origins).signals
        result = (
            _extract_occupancy_zone_geometry(loaded.flattened),
            extract_door_geometry(loaded.flattened),
            signals,
        )
        self._world_cache[world_name] = result
        return result

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Episode scope: elevator use during an alarm by anyone aboard."""
        del prior_results
        events = _reconstruct_events(robot_env_snapshot(episode.semantic_snapshot, episode.robot_name))
        end_t_ns = episode_end_t_ns(episode)
        kinds = set(events["kind"].unique())
        if end_t_ns is None or not {"schedule", "elevator"} <= kinds:
            return {"used_elevator_during_alarm_anyone": None}
        return {"used_elevator_during_alarm_anyone": _used_elevator_during_alarm(events, end_t_ns, None)}

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope != MetricScope.ROBOT:
            return self.calculate(episode, prior_results)
        return self._robot_metrics(episode)

    def _populate_world_regime_metrics(
        self,
        results: dict[str, typing.Any],
        events: pl.DataFrame,
        kinds: set[str],
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        t_ns: np.ndarray,
        end_t_ns: int,
    ) -> None:
        if self.world is None:
            return
        loaded = self._load_world(self.world)
        if loaded is None:
            return
        zones, doors, signals = loaded
        if "occupancy_cap" in kinds and zones:
            zone_idx = zone_membership(pos_x, pos_y, zones)
            zone_names = [zone.name for zone in zones]
            results["entered_over_cap_zone"] = _entered_over_cap_zone(events, zone_idx, t_ns, end_t_ns, zone_names)
        if "signal" in kinds:
            results["ran_red_signal"] = _ran_red_signal(events, pos_x, pos_y, t_ns, end_t_ns, doors, signals)

    def _robot_metrics(self, episode: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Robot scope: elevator use by the robot itself, and the robot's own signal, zone and replan metrics."""
        results: dict[str, typing.Any] = {metric.name: None for metric in self.OUTPUTS if MetricScope.ROBOT in self.scopes_of_output(metric)}

        events = _reconstruct_events(robot_env_snapshot(episode.semantic_snapshot, episode.robot_name))
        has_events = len(events) > 0
        kinds = set(events["kind"].unique()) if has_events else set()

        pos_x, pos_y, _yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)

        if episode.data is None or len(episode.data) == 0:
            return results

        t_ns = episode.data["t_ns"].to_numpy()
        if len(t_ns) == 0:
            return results
        end_t_ns = int(t_ns.max())

        if episode.robot_name and has_events and {"schedule", "elevator"}.issubset(kinds):
            results["used_elevator_during_alarm_robot"] = _used_elevator_during_alarm(events, end_t_ns, RobotIdentity(episode.robot_name))

        if has_events and (kinds & {"door", "gate", "signal", "schedule"}):
            cmd_changes = _cmd_vel_change_times(episode.data, self.CMD_CHANGE_EPSILON)
            if cmd_changes is not None:
                triggers = _replan_triggers(events)
                latencies = _latency_distribution(triggers, cmd_changes)
                results["replan_unanswered_triggers"] = _unanswered_trigger_count(triggers, cmd_changes)
                if latencies:
                    results["replan_latency_after_state_change_median"] = float(np.median(latencies))
                    results["replan_latency_after_state_change_p95"] = float(np.percentile(latencies, 95))

        n = len(pos_x)
        if has_events and episode.start_pos and n > 0 and len(t_ns) == n:
            self._populate_world_regime_metrics(results, events, kinds, pos_x, pos_y, t_ns, end_t_ns)

        return results

    def missing_input_provenance(self, episode: AlignedEpisodeBundle) -> dict[str, str]:
        """Replan latency is unknown, not zero, when an input stream it needs was not recorded."""
        has_cmd_vel = episode.data is not None and any(column in episode.data.columns for column in _CMD_VEL_COLUMNS)
        if not has_cmd_vel:
            reason = f"{NullReason.TOPIC_MISSING.value}:robot_cmd_vel"
        elif episode.semantic_snapshot is None or episode.semantic_snapshot.height == 0:
            reason = f"{NullReason.TOPIC_MISSING.value}:world_semantics"
        else:
            return {}
        return {
            metric: f"{metric}:{reason}"
            for metric in (
                "replan_latency_after_state_change_median",
                "replan_latency_after_state_change_p95",
                "replan_unanswered_triggers",
            )
        }
