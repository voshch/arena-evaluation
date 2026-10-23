from __future__ import annotations

import dataclasses
import typing

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import (
    DoorGeom,
    ElevatorGeom,
    MechanismWorld,
    PedTracks,
    RobotIdentity,
    RobotTrack,
    SemanticView,
    in_polygon,
    load_mechanism_world,
    parse_members,
    ped_tracks,
    robot_identity,
    robot_track,
    segment_distance,
    side_of,
    true_runs,
)
from arena_evaluation.processing.pose_segments import MAX_SPEED_MPS, MIN_JUMP_M
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

STATIONARY_SPEED = 0.05
PED_WAIT_SPEED = 0.1
DOORWAY_MARGIN_M = 0.3
ARRIVAL_WINDOW_S = 3.0
YIELD_WINDOW_S = 2.0
YIELD_RADIUS_M = 2.0


@dataclasses.dataclass
class _Ctx:
    track: RobotTrack
    view: SemanticView
    world: MechanismWorld | None
    identity: RobotIdentity | None
    peds: PedTracks | None
    robot_radius: float
    _memo: dict = dataclasses.field(default_factory=dict)

    def values(self, entity: str, field: str) -> list[str] | None:
        key = ("values", entity, field)
        if key not in self._memo:
            series = self.view.get(entity, field)
            self._memo[key] = series.sample(self.track.time_ns) if series is not None else None
        return self._memo[key]

    def members(self, entity: str) -> list[list[str]] | None:
        key = ("members", entity)
        if key not in self._memo:
            values = self.values(entity, "members")
            self._memo[key] = [parse_members(v) for v in values] if values is not None else None
        return self._memo[key]

    def others(self, entity: str, i: int) -> list[str]:
        """Non-robot members of `entity` at sample i."""
        members = self.members(entity)
        if members is None:
            return []
        if self.identity is None:
            return members[i]
        return [m for m in members[i] if not self.identity.matches(m)]

    def door_state(self, entity: str) -> list[str] | None:
        """Door `state` per sample, synthesised from the `open` predicate when `state` is absent."""
        state = self.values(entity, "state")
        if state is not None:
            return state
        is_open = self.values(entity, "open")
        if is_open is not None:
            return ["open" if v == "true" else "closed" for v in is_open]
        return None

    def _robot_member_mask(self, entity: str, kind: str) -> np.ndarray | None:
        if self.identity is None or not self.view.members_seen(kind):
            return None
        members = self.members(entity)
        if members is None:
            return None
        return np.array([self.identity.any_of(m) for m in members], dtype=bool)

    def robot_in_cabin(self, entity: str) -> np.ndarray | None:
        """Robot inside the elevator cabin: semantic members when recorded, else cabin footprint."""
        mask = self._robot_member_mask(entity, "elevator")
        if mask is not None:
            return mask
        geom = self.elevator(entity)
        if geom is None:
            return None
        return in_polygon(self.track.x, self.track.y, geom.polygon)

    def robot_near_door(self, entity: str) -> np.ndarray | None:
        """Robot within the door's activation radius: semantic members when recorded, else geometry."""
        mask = self._robot_member_mask(entity, "door")
        if mask is not None:
            return mask
        geom = self.door(entity)
        if geom is None:
            return None
        cx, cy = geom.center
        return np.hypot(self.track.x - cx, self.track.y - cy) <= geom.radius

    def door(self, entity: str) -> DoorGeom | None:
        return self.world.door(entity) if self.world is not None else None

    def elevator(self, entity: str) -> ElevatorGeom | None:
        return self.world.elevator(entity) if self.world is not None else None

    def doorway_band(self) -> float:
        return self.robot_radius + DOORWAY_MARGIN_M


def _is_jump(track: RobotTrack, i: int) -> bool:
    """Step i-1 -> i is a teleport (`pose_segments` rule: above MIN_JUMP_M and MAX_SPEED_MPS)."""
    dist = float(np.hypot(track.x[i] - track.x[i - 1], track.y[i] - track.y[i - 1]))
    dt = (int(track.time_ns[i]) - int(track.time_ns[i - 1])) / 1e9
    return dist > MIN_JUMP_M and (dt <= 0.0 or dist / dt > MAX_SPEED_MPS)


def in_call_area(x: np.ndarray, y: np.ndarray, geom: ElevatorGeom) -> np.ndarray:
    """Within the elevator's activation distance of its cabin door centre and outside the cabin."""
    cx, cy = geom.door_center
    return (np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) <= geom.activation_distance) & ~in_polygon(x, y, geom.polygon)


def _cabin_masks(ctx: _Ctx) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for entity in ctx.view.entities("elevator"):
        mask = ctx.robot_in_cabin(entity)
        if mask is not None:
            masks[entity] = mask
    return masks


def _boardings(ctx: _Ctx, masks: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Per element, sample indices where the robot enters the cabin on foot, ride arrivals excluded."""
    t = ctx.track.time_ns
    window_ns = int(ARRIVAL_WINDOW_S * 1e9)
    out: dict[str, list[int]] = {}
    for entity, mask in masks.items():
        indices: list[int] = []
        for i in range(1, len(mask)):
            if not mask[i] or mask[i - 1]:
                continue
            if _is_jump(ctx.track, i):
                continue
            lo = int(np.searchsorted(t, t[i] - window_ns, side="left"))
            if any(other is not mask and np.any(other[lo:i]) for other in masks.values()):
                continue
            indices.append(i)
        out[entity] = indices
    return out


def _visit_start(call: np.ndarray, b: int) -> int | None:
    """Start index of the call-area run that ends at boarding index b, None if b-1 is outside it."""
    if b < 1 or not call[b - 1]:
        return None
    s = b - 1
    while s > 0 and call[s - 1]:
        s -= 1
    return s


def _boarded_before_exit(ctx: _Ctx, boardings: dict[str, list[int]]) -> int | None:
    """Robot boardings while the cabin still reports `just_arrived` and holds another agent."""
    if not ctx.view.has_field("elevator", "just_arrived"):
        return None
    members_seen = ctx.view.members_seen("elevator")
    count = 0
    for entity, indices in boardings.items():
        arrived = ctx.values(entity, "just_arrived")
        if arrived is None:
            continue
        occupants = ctx.values(entity, "occupants")
        for b in indices:
            if arrived[b] != "true":
                continue
            if members_seen:
                busy = bool(ctx.others(entity, b))
            else:
                busy = occupants is not None and float(occupants[b]) > 0.0
            if busy:
                count += 1
    return count


def _elevator_wait_to_board_s(ctx: _Ctx, boardings: dict[str, list[int]]) -> float | None:
    """Sum over boardings of the time from entering the call area to entering the cabin."""
    total = 0.0
    boarded = False
    t = ctx.track.time_ns
    for entity, indices in boardings.items():
        geom = ctx.elevator(entity)
        if geom is None:
            continue
        call = in_call_area(ctx.track.x, ctx.track.y, geom)
        for b in indices:
            boarded = True
            s = _visit_start(call, b)
            if s is not None:
                total += (t[b] - t[s]) / 1e9
    return total if boarded else None


def _call_visits(ctx: _Ctx, entity: str, cabin: np.ndarray) -> list[tuple[int, int]]:
    """Robot call-area runs [s, e) of an elevator, dropping runs that start right after leaving the cabin."""
    geom = ctx.elevator(entity)
    if geom is None:
        return []
    call = in_call_area(ctx.track.x, ctx.track.y, geom)
    return [(s, e) for s, e in true_runs(call) if not (s > 0 and cabin[s - 1])]


def _run_end_time(track: RobotTrack, e: int) -> int:
    return int(track.time_ns[e]) if e < len(track) else int(track.time_ns[-1]) + 1


def _elevator_missed_cycles(ctx: _Ctx, masks: dict[str, np.ndarray]) -> int:
    """Complete cabin door open -> closed cycles while the robot waits in the call area."""
    missed = 0
    for entity, cabin in masks.items():
        series = ctx.view.get(entity, "cabin_door")
        if series is None:
            continue
        for s, e in _call_visits(ctx, entity, cabin):
            t0 = int(ctx.track.time_ns[s])
            seen_open = series.at(t0) == "open"
            for _t, value in series.changes_in(t0 + 1, _run_end_time(ctx.track, e)):
                if value == "open":
                    seen_open = True
                elif value == "closed" and seen_open:
                    missed += 1
                    seen_open = False
    return missed


def _peds_in_call_area(ctx: _Ctx, geom: ElevatorGeom, i: int) -> set[str]:
    if ctx.peds is None or i >= len(ctx.peds.xy) or len(ctx.peds.xy[i]) == 0:
        return set()
    xy = ctx.peds.xy[i]
    inside = in_call_area(xy[:, 0], xy[:, 1], geom)
    return {ctx.peds.keys[i][j] for j in np.flatnonzero(inside)}


def _elevator_unnecessary_calls(ctx: _Ctx, masks: dict[str, np.ndarray], boardings: dict[str, list[int]]) -> int:
    """Robot call-area visits that ended without boarding although the cabin door opened with no pedestrian waiting."""
    calls = 0
    t = ctx.track.time_ns
    for entity, cabin in masks.items():
        series = ctx.view.get(entity, "cabin_door")
        geom = ctx.elevator(entity)
        if series is None or geom is None:
            continue
        boarded_at = set(boardings.get(entity, []))
        for s, e in _call_visits(ctx, entity, cabin):
            if e in boarded_at:
                continue
            for t_change, value in series.changes_in(int(t[s]), _run_end_time(ctx.track, e)):
                if value not in ("opening", "open"):
                    continue
                i = min(int(np.searchsorted(t, t_change, side="right")) - 1, len(t) - 1)
                if not _peds_in_call_area(ctx, geom, max(i, 0)):
                    calls += 1
                    break
    return calls


def _entered_full_cabin(ctx: _Ctx, boardings: dict[str, list[int]]) -> int | None:
    """Boardings into a cabin that was already at capacity just before the robot entered."""
    has_info = ctx.view.has_field("elevator", "capacity") or ctx.view.has_field("elevator", "full") or (ctx.world is not None and any(g.capacity is not None for g in ctx.world.elevators))
    if not has_info:
        return None
    count = 0
    for entity, indices in boardings.items():
        capacity = ctx.values(entity, "capacity")
        full = ctx.values(entity, "full")
        live = ctx.values(entity, "live_occupants")
        geom = ctx.elevator(entity)
        for b in indices:
            k = b - 1
            cap: float | None = None
            if capacity is not None:
                cap = float(capacity[k])
            elif geom is not None and geom.capacity is not None:
                cap = float(geom.capacity)
            if cap is not None and cap > 0:
                occupancy = float(live[k]) if live is not None else float(len(ctx.others(entity, k)))
                is_full = occupancy >= cap
            elif cap is not None:
                is_full = False
            elif full is not None:
                is_full = full[k] == "true"
            else:
                continue
            if is_full:
                count += 1
    return count


def _cabin_door_blocking_s(ctx: _Ctx) -> float | None:
    """Seconds the robot stands still within the cabin doorway band while the cabin door is open or closing."""
    if ctx.world is None or not ctx.view.has_field("elevator", "cabin_door"):
        return None
    stationary = ctx.track.speed <= STATIONARY_SPEED
    blocking = np.zeros(len(ctx.track), dtype=bool)
    for entity in ctx.view.entities("elevator"):
        geom = ctx.elevator(entity)
        door = ctx.values(entity, "cabin_door")
        if geom is None or door is None:
            continue
        band = segment_distance(ctx.track.x, ctx.track.y, geom.door_start, geom.door_end) <= ctx.doorway_band()
        held = np.array([v in ("open", "closing") for v in door], dtype=bool)
        blocking |= band & held & stationary
    return float(np.sum(ctx.track.dt[blocking]))


def _elevator_queue_jumps(ctx: _Ctx, masks: dict[str, np.ndarray], boardings: dict[str, list[int]]) -> int | None:
    """Boardings that overtake a pedestrian who was already waiting in the call area."""
    if ctx.peds is None:
        return None
    jumps = 0
    for entity, indices in boardings.items():
        geom = ctx.elevator(entity)
        if geom is None:
            continue
        call = in_call_area(ctx.track.x, ctx.track.y, geom)
        for b in indices:
            s = _visit_start(call, b)
            if s is None:
                continue
            waiting = _peds_in_call_area(ctx, geom, s)
            for i in range(s + 1, b + 1):
                if not waiting:
                    break
                waiting &= _peds_in_call_area(ctx, geom, i)
            if waiting:
                jumps += 1
    return jumps


def _blocked_closing_door(ctx: _Ctx) -> tuple[int | None, float | None]:
    """(count, seconds) of door `blocked_close` episodes the robot is responsible for."""
    if not ctx.view.has_field("door", "blocked_close"):
        return None, None
    count = 0
    seconds = 0.0
    attributable = False
    for entity in ctx.view.entities("door"):
        blocked = ctx.values(entity, "blocked_close")
        if blocked is None:
            continue
        geom = ctx.door(entity)
        if geom is not None:
            robot = segment_distance(ctx.track.x, ctx.track.y, geom.start, geom.end) <= ctx.doorway_band()
        else:
            robot = ctx.robot_near_door(entity)
            if robot is None:
                continue
        attributable = True
        mask = np.array([v == "true" for v in blocked], dtype=bool)
        for s, e in true_runs(mask):
            if np.any(robot[s:e]):
                count += 1
        seconds += float(np.sum(ctx.track.dt[mask & robot]))
    if not attributable:
        return None, None
    return count, seconds


def _door_freeze_and_oscillations(ctx: _Ctx) -> tuple[float | None, int | None]:
    """door_freeze_s: seconds standing still in a steady door's activation radius. door_oscillations: re-openings per visit."""
    stationary = ctx.track.speed <= STATIONARY_SPEED
    frozen = np.zeros(len(ctx.track), dtype=bool)
    oscillations = 0
    resolvable = False
    for entity in ctx.view.entities("door"):
        near = ctx.robot_near_door(entity)
        state = ctx.door_state(entity)
        if near is None or state is None:
            continue
        resolvable = True
        steady = np.array([v in ("open", "closed") for v in state], dtype=bool)
        frozen |= near & steady & stationary
        series = ctx.view.get(entity, "state")
        if series is None:
            continue
        for s, e in true_runs(near):
            openings = sum(1 for _t, v in series.changes_in(int(ctx.track.time_ns[s]), _run_end_time(ctx.track, e)) if v == "opening")
            oscillations += max(openings - 1, 0)
    if not resolvable:
        return None, None
    return float(np.sum(ctx.track.dt[frozen])), oscillations


def _door_yield_violations(ctx: _Ctx) -> int | None:
    """Robot door crossings with a pedestrian approaching from the far side within `YIELD_RADIUS_M` in the `YIELD_WINDOW_S` before."""
    if ctx.world is None or ctx.peds is None:
        return None
    doors = [g for g in (ctx.door(e) for e in ctx.view.entities("door")) if g is not None]
    if not doors:
        return None
    t = ctx.track.time_ns
    window_ns = int(YIELD_WINDOW_S * 1e9)
    violations = 0
    for geom in doors:
        side = side_of(ctx.track.x, ctx.track.y, geom.start, geom.end)
        dist = segment_distance(ctx.track.x, ctx.track.y, geom.start, geom.end)
        cx, cy = geom.center
        for i in range(1, len(side)):
            if side[i] == 0 or side[i - 1] == 0 or side[i] == side[i - 1]:
                continue
            if min(dist[i], dist[i - 1]) > ctx.doorway_band():
                continue
            if _is_jump(ctx.track, i):
                continue
            lo = int(np.searchsorted(t, t[i] - window_ns, side="left"))
            for k in range(lo, i + 1):
                if k >= len(ctx.peds.xy) or len(ctx.peds.xy[k]) == 0:
                    continue
                xy = ctx.peds.xy[k]
                vel = ctx.peds.vel[k]
                far = side_of(xy[:, 0], xy[:, 1], geom.start, geom.end) == side[i]
                close = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= YIELD_RADIUS_M
                moving = np.hypot(vel[:, 0], vel[:, 1]) > PED_WAIT_SPEED
                approaching = (vel[:, 0] * (cx - xy[:, 0]) + vel[:, 1] * (cy - xy[:, 1])) > 0.0
                if np.any(far & close & moving & approaching):
                    violations += 1
                    break
    return violations


def _ped_mechanism_delay_seconds(ctx: _Ctx) -> float | None:
    """Pedestrian-seconds spent waiting (slower than `PED_WAIT_SPEED`) at a closed mechanism."""
    if ctx.world is None or ctx.peds is None:
        return None
    doors: list[tuple[DoorGeom, list[str]]] = []
    for entity in ctx.view.entities("door"):
        geom = ctx.door(entity)
        state = ctx.door_state(entity)
        if geom is not None and state is not None:
            doors.append((geom, state))
    elevators: list[tuple[ElevatorGeom, list[str]]] = []
    for entity in ctx.view.entities("elevator"):
        geom = ctx.elevator(entity)
        state = ctx.values(entity, "cabin_door")
        if geom is not None and state is not None:
            elevators.append((geom, state))
    if not doors and not elevators:
        return None

    total = 0.0
    for i in range(min(len(ctx.track), len(ctx.peds.xy))):
        xy = ctx.peds.xy[i]
        if len(xy) == 0 or ctx.track.dt[i] <= 0.0:
            continue
        vel = ctx.peds.vel[i]
        waiting = np.hypot(vel[:, 0], vel[:, 1]) < PED_WAIT_SPEED
        delayed = np.zeros(len(xy), dtype=bool)
        for geom, state in doors:
            if state[i] != "open":
                cx, cy = geom.center
                delayed |= np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= geom.radius
        for geom, state in elevators:
            if state[i] != "open":
                delayed |= in_call_area(xy[:, 0], xy[:, 1], geom)
        total += float(np.count_nonzero(delayed & waiting)) * float(ctx.track.dt[i])
    return total


def _alarm_egress_blocking_s(ctx: _Ctx) -> float | None:
    """Seconds the robot stands still inside an `egress` zone while any schedule is `active`."""
    if ctx.world is None or not ctx.world.egress_zones or not ctx.view.has_field("schedule", "active"):
        return None
    alarm = np.zeros(len(ctx.track), dtype=bool)
    for entity in ctx.view.entities("schedule"):
        active = ctx.values(entity, "active")
        if active is not None:
            alarm |= np.array([v == "true" for v in active], dtype=bool)
    egress = np.zeros(len(ctx.track), dtype=bool)
    for zone in ctx.world.egress_zones:
        egress |= in_polygon(ctx.track.x, ctx.track.y, zone.polygon)
    mask = alarm & egress & (ctx.track.speed <= STATIONARY_SPEED)
    return float(np.sum(ctx.track.dt[mask]))


class MechanismMetricsCalculator(BaseMetricCalculator):
    """Robot behaviour at world mechanisms (doors, elevators, alarms), replayed from the semantic snapshot."""

    NAME = "mechanism_metrics"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = [("tf_gt", "odom")]

    UNITS = {
        "boarded_before_exit": "",
        "blocked_closing_door": "",
        "blocked_closing_door_s": "s",
        "ped_mechanism_delay_seconds": "s",
        "elevator_wait_to_board_s": "s",
        "elevator_missed_cycles": "",
        "elevator_unnecessary_calls": "",
        "entered_full_cabin": "",
        "cabin_door_blocking_s": "s",
        "door_yield_violations": "",
        "elevator_queue_jumps": "",
        "alarm_egress_blocking_s": "s",
        "door_freeze_s": "s",
        "door_oscillations": "",
    }

    OUTPUT_DIRECTIONS = dict.fromkeys(UNITS, "lower")

    world: str | None = None

    def __init__(self, robot_params: RobotParams) -> None:
        super().__init__(robot_params)
        self._world_cache: dict[str, MechanismWorld | None] = {}

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS)

    def _mechanism_world(self) -> MechanismWorld | None:
        if self.world is None:
            return None
        if self.world not in self._world_cache:
            self._world_cache[self.world] = load_mechanism_world(self.world)
        return self._world_cache[self.world]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())

        track = robot_track(self, episode)
        if track is None or len(track) == 0:
            return results
        view = SemanticView.from_episode(episode)
        if not view.kinds:
            return results

        ctx = _Ctx(
            track=track,
            view=view,
            world=self._mechanism_world(),
            identity=robot_identity(episode),
            peds=ped_tracks(episode.data, track.time_ns),
            robot_radius=float(self.robot_params.robot_radius),
        )

        if view.has_kind("door"):
            results["blocked_closing_door"], results["blocked_closing_door_s"] = _blocked_closing_door(ctx)
            results["door_freeze_s"], results["door_oscillations"] = _door_freeze_and_oscillations(ctx)
            results["door_yield_violations"] = _door_yield_violations(ctx)

        if view.has_kind("door") or view.has_kind("elevator"):
            results["ped_mechanism_delay_seconds"] = _ped_mechanism_delay_seconds(ctx)

        if view.has_kind("elevator"):
            masks = _cabin_masks(ctx)
            if masks:
                boardings = _boardings(ctx, masks)
                results["boarded_before_exit"] = _boarded_before_exit(ctx, boardings)
                results["entered_full_cabin"] = _entered_full_cabin(ctx, boardings)
                if ctx.world is not None:
                    results["elevator_wait_to_board_s"] = _elevator_wait_to_board_s(ctx, boardings)
                    if view.has_field("elevator", "cabin_door"):
                        results["elevator_missed_cycles"] = _elevator_missed_cycles(ctx, masks)
                        results["elevator_unnecessary_calls"] = _elevator_unnecessary_calls(ctx, masks, boardings)
                    results["elevator_queue_jumps"] = _elevator_queue_jumps(ctx, masks, boardings)
            results["cabin_door_blocking_s"] = _cabin_door_blocking_s(ctx)

        if view.has_kind("schedule"):
            results["alarm_egress_blocking_s"] = _alarm_egress_blocking_s(ctx)

        return results
