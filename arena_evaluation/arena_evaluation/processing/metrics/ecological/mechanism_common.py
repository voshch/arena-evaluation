"""Shared naming, geometry and track helpers for the mechanism metrics."""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import typing

import numpy as np
import polars as pl

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared import Signal
    from arena_simulation_setup.tree.World import Level, World, Zone

    from arena_evaluation.processing.metrics.base import BaseMetricCalculator
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

logger = logging.getLogger(__name__)

_ENV_PREFIX = re.compile(r"^(env_\d+)/")
_ROBOT_DIR = re.compile(r"^(env_\d+)_(.+)$")

MEMBERS_SEPARATOR = ","


def _norm_path(name: str) -> str:
    """Slash-normalised path without leading, trailing or doubled slashes."""
    return "/".join(p for p in str(name).split("/") if p)


def entity_env(entity: str) -> str | None:
    """`env_<n>` prefix of a recorded entity or agent path, None when unprefixed."""
    match = _ENV_PREFIX.match(_norm_path(entity) + "/")
    return match.group(1) if match else None


def entity_name(entity: str) -> str:
    """Authored name of a recorded semantic entity: env prefix and level suffix dropped."""
    rest = _ENV_PREFIX.sub("", _norm_path(entity), count=1)
    return rest.split("/", 1)[0]


def entity_level(entity: str) -> str | None:
    """Level id suffix of a recorded semantic entity, None when the path carries none."""
    rest = _ENV_PREFIX.sub("", _norm_path(entity), count=1)
    parts = rest.split("/")
    return parts[1] if len(parts) > 1 else None


def parse_members(token: str | None) -> list[str]:
    """Member names from the `_reconstruct_events` members string (comma joined, sorted)."""
    if not token:
        return []
    return [m for m in token.split(MEMBERS_SEPARATOR) if m]


class RobotIdentity:
    """Matches semantic member names against one recorder robot name."""

    def __init__(self, robot_name: str) -> None:
        self.robot_name = _norm_path(robot_name)
        match = _ROBOT_DIR.match(self.robot_name)
        if match:
            self.env: str | None = match.group(1)
            self.base = match.group(2)
        else:
            self.env = entity_env(self.robot_name)
            self.base = _ENV_PREFIX.sub("", self.robot_name, count=1)
        aliases = {self.robot_name, self.base}
        if self.env is not None:
            aliases.add(f"{self.env}/{self.base}")
        self._aliases = aliases

    def matches(self, member: str) -> bool:
        name = _norm_path(member)
        if not name:
            return False
        if name in self._aliases or name.replace("/", "_") == self.robot_name:
            return True
        env = entity_env(name)
        if env is not None and self.env is not None and env != self.env:
            return False
        rest = _ENV_PREFIX.sub("", name, count=1)
        return rest == self.base or rest.split("/")[-1] == self.base.split("/")[-1]

    def any_of(self, members: typing.Iterable[str]) -> bool:
        return any(self.matches(m) for m in members)


def robot_identity(episode: AlignedEpisodeBundle) -> RobotIdentity | None:
    return RobotIdentity(episode.robot_name) if episode.robot_name else None


def robot_env_snapshot(snapshot: pl.DataFrame | pl.LazyFrame | None, robot_name: str | None) -> pl.DataFrame | None:
    """The snapshot restricted to the robot's env when it carries several envs, unchanged otherwise."""
    if snapshot is None:
        return None
    if isinstance(snapshot, pl.LazyFrame):
        snapshot = snapshot.collect()
    if len(snapshot) == 0 or "entity" not in snapshot.columns or not robot_name:
        return snapshot
    env = RobotIdentity(robot_name).env
    if env is None:
        return snapshot
    envs = snapshot.select(pl.col("entity").str.extract(r"^/*(env_\d+)/", 1).unique())["entity"].to_list()
    if env not in envs or len(envs) < 2:
        return snapshot
    return snapshot.filter(pl.col("entity").str.contains(rf"^/*{env}/"))


@dataclasses.dataclass
class FieldSeries:
    """Stepwise series of one entity field, holding the earliest (seed) value before it starts."""

    times: np.ndarray
    values: list[str]

    def at(self, t_ns: int) -> str:
        idx = int(np.searchsorted(self.times, t_ns, side="right")) - 1
        return self.values[max(idx, 0)]

    def sample(self, query_ns: np.ndarray) -> list[str]:
        idx = np.searchsorted(self.times, query_ns, side="right") - 1
        idx = np.clip(idx, 0, len(self.values) - 1)
        return [self.values[i] for i in idx]

    def changes_in(self, t0: int, t1: int) -> list[tuple[int, str]]:
        """(time, value) change points with t0 <= time < t1."""
        lo = int(np.searchsorted(self.times, t0, side="left"))
        hi = int(np.searchsorted(self.times, t1, side="left"))
        return [(int(self.times[i]), self.values[i]) for i in range(lo, hi)]


class SemanticView:
    """Robot-env scoped, per-entity stepwise view of an episode's semantic snapshot."""

    def __init__(self, events: pl.DataFrame) -> None:
        self.kinds: dict[str, str] = {}
        self.series: dict[tuple[str, str], FieldSeries] = {}
        if len(events) == 0:
            return
        for (entity, kind, field), group in events.group_by(["entity", "kind", "field"]):
            g = group.sort("time_ns")
            self.kinds.setdefault(entity, kind)
            self.series[(entity, field)] = FieldSeries(g["time_ns"].to_numpy().astype(np.int64), g["current"].to_list())

    @classmethod
    def from_episode(cls, episode: AlignedEpisodeBundle) -> SemanticView:
        from arena_evaluation.processing.metrics.ecological.compliance_metrics import _reconstruct_events

        return cls(_reconstruct_events(robot_env_snapshot(episode.semantic_snapshot, episode.robot_name)))

    def entities(self, kind: str) -> list[str]:
        return sorted(e for e, k in self.kinds.items() if k == kind)

    def has_kind(self, kind: str) -> bool:
        return kind in self.kinds.values()

    def has_field(self, kind: str, field: str) -> bool:
        return any((e, field) in self.series for e in self.entities(kind))

    def get(self, entity: str, field: str) -> FieldSeries | None:
        return self.series.get((entity, field))

    def members_seen(self, kind: str) -> bool:
        """True when any entity of `kind` reports a non-empty members list in the episode."""
        for entity in self.entities(kind):
            s = self.get(entity, "members")
            if s is not None and any(s.values):
                return True
        return False


@dataclasses.dataclass
class DoorGeom:
    name: str
    level: str
    start: tuple[float, float]
    end: tuple[float, float]
    radius: float
    kind: str = "sliding"
    swing: int = 1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0)

    def leaf(self, progress: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Closed-part segment of the slab at `progress` (0 closed, 1 open)."""
        (ax, ay), (bx, by) = self.start, self.end
        dx, dy = bx - ax, by - ay
        if self.kind == "hinged":
            angle = self.swing * progress * math.pi / 2.0
            return self.start, (ax + dx * math.cos(angle) - dy * math.sin(angle), ay + dx * math.sin(angle) + dy * math.cos(angle))
        return self.start, (ax + dx * (1.0 - progress), ay + dy * (1.0 - progress))


@dataclasses.dataclass
class ElevatorGeom:
    name: str
    level: str
    polygon: list[tuple[float, float]]
    door_start: tuple[float, float]
    door_end: tuple[float, float]
    activation_distance: float
    capacity: int | None = None
    destinations: list[str] = dataclasses.field(default_factory=list)

    @property
    def door_center(self) -> tuple[float, float]:
        return ((self.door_start[0] + self.door_end[0]) / 2.0, (self.door_start[1] + self.door_end[1]) / 2.0)


@dataclasses.dataclass
class SignalGeom:
    name: str
    level: str
    x: float
    y: float
    radius: float | None


@dataclasses.dataclass
class EgressZone:
    name: str
    level: str
    polygon: list[tuple[float, float]]


@dataclasses.dataclass
class MechanismWorld:
    """Mechanism geometry of a world in the recorded (level-origin shifted) map frame."""

    doors: list[DoorGeom] = dataclasses.field(default_factory=list)
    elevators: list[ElevatorGeom] = dataclasses.field(default_factory=list)
    signals: list[SignalGeom] = dataclasses.field(default_factory=list)
    egress_zones: list[EgressZone] = dataclasses.field(default_factory=list)

    @staticmethod
    def _pick[G: (DoorGeom, ElevatorGeom, SignalGeom)](items: list[G], name: str, level: str | None) -> G | None:
        candidates = [item for item in items if item.name == name]
        if level is not None:
            on_level = [item for item in candidates if item.level == level]
            if on_level:
                return on_level[0]
        return candidates[0] if candidates else None

    def door(self, entity: str) -> DoorGeom | None:
        return self._pick(self.doors, entity_name(entity), entity_level(entity))

    def elevator(self, entity: str) -> ElevatorGeom | None:
        return self._pick(self.elevators, entity_name(entity), entity_level(entity))

    def signal(self, entity: str) -> SignalGeom | None:
        return self._pick(self.signals, entity_name(entity), entity_level(entity))


def _cabin_door_segment(cx: float, cy: float, hw: float, hh: float, side: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if side == "-x":
        return (cx - hw, cy - hh), (cx - hw, cy + hh)
    if side == "+y":
        return (cx - hw, cy + hh), (cx + hw, cy + hh)
    if side == "-y":
        return (cx - hw, cy - hh), (cx + hw, cy - hh)
    return (cx + hw, cy - hh), (cx + hw, cy + hh)


def _signal_geometry(signal: Signal, world: World, origins: dict[str, tuple[float, float]]) -> SignalGeom | None:
    """Stop line from the signal params in the frame of the level whose zones cover it (else the first level), None when unplaced."""
    try:
        position = signal.stop_position
    except (TypeError, ValueError, IndexError):
        return None
    if position is None:
        return None
    x, y = position
    level_id = next(
        (lid for lid, level in world.levels.items() if any(in_polygon(np.array([x]), np.array([y]), [(c.x, c.y) for c in zone.polygon])[0] for zone in level.zones)),
        next(iter(world.levels), ""),
    )
    origin = origins.get(level_id, (0.0, 0.0))
    return SignalGeom(signal.name, level_id, x + origin[0], y + origin[1], signal.stop_radius)


def _is_egress(zone: Zone) -> bool:
    return any(cfg.role == "predicate" and cfg.name == "egress" and cfg.value is not False for cfg in zone.semantics)


def _level_geometry(out: MechanismWorld, level_id: str, level: Level) -> None:
    for door in level.all_doors:
        out.doors.append(DoorGeom(door.name, level_id, (door.start.x, door.start.y), (door.end.x, door.end.y), float(max(door.activation_distance)), door.kind, door.swing_direction))
    for elevator in level.all_elevators:
        cx, cy = elevator.position.x, elevator.position.y
        hw, hh = elevator.size[0] / 2.0, elevator.size[1] / 2.0
        door_start, door_end = _cabin_door_segment(cx, cy, hw, hh, elevator.door_side or "+x")
        out.elevators.append(
            ElevatorGeom(
                name=elevator.name,
                level=level_id,
                polygon=[(c.x, c.y) for c in elevator.cabin_corners()],
                door_start=door_start,
                door_end=door_end,
                activation_distance=float(elevator.activation_distance),
                capacity=elevator.capacity,
                destinations=list(elevator.destinations),
            )
        )
    for zone in level.zones:
        if len(zone.polygon) >= 3 and _is_egress(zone):
            out.egress_zones.append(EgressZone(zone.name, level_id, [(c.x, c.y) for c in zone.polygon]))


def mechanism_world_from_description(world: World, origins: dict[str, tuple[float, float]]) -> MechanismWorld:
    """Extract mechanism geometry per level, shifted by each level's origin like `compact_world`."""
    out = MechanismWorld()
    for level_id, level in world.levels.items():
        _level_geometry(out, level_id, level.shifted(*origins.get(level_id, (0.0, 0.0))))
    for signal in world.signals:
        geom = _signal_geometry(signal, world, origins)
        if geom is not None:
            out.signals.append(geom)
    return out


def load_mechanism_world(world_name: str) -> MechanismWorld | None:
    """Mechanism geometry of a locally resolvable world, None (with a warning) when absent."""
    from arena_simulation_setup.tree.World import WorldIdentifier

    try:
        view = WorldIdentifier(world_name).resolve_sync()
        world = view.load()
    except FileNotFoundError as e:
        logger.warning("mechanism metrics: world '%s' not available locally: %s", world_name, e)
        return None

    origins = view.level_origins()
    if origins is None:
        origins = {level_id: (0.0, 0.0) for level_id in world.levels}
    return mechanism_world_from_description(world, origins)


def segment_distance(px: np.ndarray, py: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """Euclidean distance of points to the segment a-b."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / length_sq, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def side_of(px: np.ndarray, py: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """Signed side of points relative to the directed line a->b (+1 left, -1 right, 0 on it)."""
    cross = (b[0] - a[0]) * (np.asarray(py, dtype=np.float64) - a[1]) - (b[1] - a[1]) * (np.asarray(px, dtype=np.float64) - a[0])
    return np.sign(cross)


def in_polygon(px: np.ndarray, py: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    """Vectorised even-odd point-in-polygon."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    n = len(polygon)
    if n < 3:
        return inside
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= crosses & (px < x_at)
        j = i
    return inside


@dataclasses.dataclass
class RobotTrack:
    """Map-frame robot samples on the episode odom axis; dt[i] is the hold until sample i+1 (s)."""

    time_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    dt: np.ndarray

    def __len__(self) -> int:
        return len(self.time_ns)


def hold_dt(time_ns: np.ndarray) -> np.ndarray:
    dt = np.zeros(len(time_ns))
    if len(time_ns) > 1:
        dt[:-1] = np.clip(np.diff(time_ns) / 1e9, 0.0, None)
    return dt


def robot_track(calc: BaseMetricCalculator, episode: AlignedEpisodeBundle) -> RobotTrack | None:
    """Resolved map-frame robot track, None when the episode has no usable pose or start pose."""
    if episode.data is None or len(episode.data) == 0 or not episode.start_pos:
        return None
    pos_x, pos_y, _yaw, _ox, _oy, _oyaw = calc.resolve_robot_pose(episode)
    data = episode.data
    if data is None or "time_ns" not in data.columns or len(pos_x) == 0 or len(data) != len(pos_x):
        return None
    time_ns = data["time_ns"].to_numpy().astype(np.int64)
    if "vel_linear" in data.columns:
        speed = np.abs(np.nan_to_num(data["vel_linear"].to_numpy().astype(np.float64), nan=0.0))
    else:
        speed = calc.speed_from_pose(pos_x, pos_y, time_ns)
    return RobotTrack(time_ns, np.asarray(pos_x, dtype=np.float64), np.asarray(pos_y, dtype=np.float64), speed, hold_dt(time_ns))


@dataclasses.dataclass
class PedTracks:
    """Per-sample pedestrian keys, positions and velocities on the episode odom axis."""

    keys: list[list[str]]
    xy: list[np.ndarray]
    vel: list[np.ndarray]
    named: bool

    def position_of(self, i: int, key: str) -> tuple[float, float] | None:
        try:
            j = self.keys[i].index(key)
        except ValueError:
            return None
        return float(self.xy[i][j, 0]), float(self.xy[i][j, 1])


def _flat_triples(raw: typing.Iterable[float] | None) -> np.ndarray:
    if raw is None:
        return np.empty((0, 3))
    arr = np.asarray(list(raw), dtype=np.float64)
    if arr.size == 0 or arr.size % 3 != 0:
        return np.empty((0, 3))
    return arr.reshape(-1, 3)


def ped_tracks(data: pl.DataFrame | None, time_ns: np.ndarray | None = None) -> PedTracks | None:
    """Pedestrian tracks from the aligned `peds_*` columns, None when no positions were recorded."""
    if data is None or "peds_positions" not in data.columns or len(data) == 0:
        return None
    from arena_evaluation.processing.metrics.base import BaseMetricCalculator

    positions = data["peds_positions"].to_list()
    hints = data["num_pedestrians"].to_list() if "num_pedestrians" in data.columns else None
    names = data["peds_names"].to_list() if "peds_names" in data.columns else None
    twists = data["peds_twists"].to_list() if "peds_twists" in data.columns else None
    if time_ns is None:
        time_ns = data["time_ns"].to_numpy()

    keys: list[list[str]] = []
    xy: list[np.ndarray] = []
    vel: list[np.ndarray] = []
    named = names is not None
    prev: dict[str, tuple[float, float]] = {}
    prev_t: int | None = None
    for i in range(len(positions)):
        parsed = BaseMetricCalculator._parse_peds(positions[i], hints[i] if hints is not None else None)
        pos = parsed[:, :2] if parsed.ndim == 2 and parsed.shape[1] >= 2 else np.empty((0, 2))
        row_names = list(names[i]) if names is not None and names[i] is not None else None
        if row_names is not None and len(row_names) == len(pos):
            row_keys = [_norm_path(n) for n in row_names]
        else:
            row_keys = [f"#{j}" for j in range(len(pos))]
            if row_names is not None and len(pos):
                named = False
        tw = _flat_triples(twists[i])[:, :2] if twists is not None else np.empty((0, 2))
        if len(tw) != len(pos):
            tw = np.zeros((len(pos), 2))
            if prev_t is not None:
                step = max((int(time_ns[i]) - prev_t) / 1e9, 1e-3)
                for j, key in enumerate(row_keys):
                    if key in prev:
                        tw[j] = (pos[j] - np.asarray(prev[key])) / step
        keys.append(row_keys)
        xy.append(pos)
        vel.append(tw)
        prev = {key: (float(pos[j, 0]), float(pos[j, 1])) for j, key in enumerate(row_keys)}
        prev_t = int(time_ns[i])
    return PedTracks(keys, xy, vel, named)


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index runs [start, end) where `mask` is true."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs
