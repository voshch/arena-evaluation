from __future__ import annotations

import dataclasses
import logging
import re
import typing
from collections import defaultdict

import numpy as np
import polars as pl
from arena_simulation_setup.shared.conditions import Atom, EpisodeCondition, parse_atom
from arena_simulation_setup.shared.judge import Sample, atom_holds, operator_verdict, values_equal

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    _extract_zone_geometry,
    _reconstruct_events,
    _ZoneGeometry,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

logger = logging.getLogger(__name__)

_ENV_PREFIX = re.compile(r"^env_\d+/")
_ROBOT_DIR_PREFIX = re.compile(r"^env_\d+_")

_world_cache: dict[str, list[_ZoneGeometry] | None] = {}


def _strip_env(name: str) -> str:
    """Drop the leading `env_<n>/` segment, keeping the rest incl. any level suffix."""
    return _ENV_PREFIX.sub("", name, count=1)


def bare_robot_name(robot_dir: str) -> str:
    """`env_0_jackal_0` -> `jackal_0`, the name phases and atoms use."""
    return _ROBOT_DIR_PREFIX.sub("", robot_dir, count=1)


def _values_equal(recorded: str, expected: str) -> bool:
    return values_equal(recorded, expected)


def _entity_roster(snapshot: pl.DataFrame | None) -> dict[str, str]:
    """Bare entity name -> recorded sim_path, for names resolving to a single sim_path, a world entity also by its name without the level suffix."""
    if snapshot is None or len(snapshot) == 0 or "entity" not in snapshot.columns:
        return {}
    paths: dict[str, set[str]] = defaultdict(set)
    for entity in snapshot["entity"].unique().to_list():
        paths[_strip_env(entity)].add(entity)
    unsuffixed: dict[str, set[str]] = defaultdict(set)
    for bare, recorded in paths.items():
        if "/" in bare:
            unsuffixed[bare.rsplit("/", 1)[0]] |= recorded
    roster = {name: next(iter(p)) for name, p in unsuffixed.items() if name not in paths and len(p) == 1}
    return roster | {bare: next(iter(p)) for bare, p in paths.items() if len(p) == 1}


def _ped_roster(data: pl.DataFrame) -> dict[str, str]:
    """Bare ped name -> recorded name, only for names that resolve uniquely on the axis."""
    if "peds_names" not in data.columns:
        return {}
    grouped: dict[str, set[str]] = defaultdict(set)
    for names in data["peds_names"].to_list():
        if not names:
            continue
        for name in names:
            grouped[_strip_env(name)].add(name)
    return {bare: next(iter(full)) for bare, full in grouped.items() if len(full) == 1}


def _entity_field_series(events: pl.DataFrame, entity: str, field: str) -> tuple[np.ndarray, list[str]]:
    """Sorted (event times, values) for one recorded entity and field."""
    rows = events.filter((pl.col("entity") == entity) & (pl.col("field") == field)).sort("time_ns")
    return rows["time_ns"].to_numpy(), rows["current"].to_list()


def _value_at(times: np.ndarray, values: list[str], t_ns: int) -> str:
    """Stepwise value at `t_ns`, holding the earliest known (seed) value before it."""
    idx = int(np.searchsorted(times, t_ns, side="right")) - 1
    if idx < 0:
        idx = 0
    return values[idx]


def _first_true(series: np.ndarray) -> int | None:
    idxs = np.flatnonzero(series)
    return int(idxs[0]) if len(idxs) else None


def load_zone_geometry(world_name: str) -> list[_ZoneGeometry] | None:
    """Zone polygons of a world in the flattened map frame, None when the world asset is not available locally."""
    if world_name in _world_cache:
        return _world_cache[world_name]

    from arena_simulation_setup.tree.World import WorldIdentifier

    try:
        view = WorldIdentifier(world_name).resolve_sync()
        world = view.load()
    except FileNotFoundError as e:
        logger.warning("condition_compliance: world '%s' not available locally: %s", world_name, e)
        _world_cache[world_name] = None
        return None

    origins = view.level_origins()
    if origins is None:
        origins = {level_id: (0.0, 0.0) for level_id in world.levels}
    flattened = world.compact_world(origins)

    zones = _extract_zone_geometry(flattened, require_annotation=False)
    _world_cache[world_name] = zones
    return zones


@dataclasses.dataclass
class _EvalContext:
    """Recorded inputs of one robot's episode on one time axis, the ego robot being `robot`."""

    events: pl.DataFrame
    time_ns: np.ndarray
    pos_x: np.ndarray
    pos_y: np.ndarray
    data: pl.DataFrame
    zones_by_name: dict[str, _ZoneGeometry]
    entity_roster: dict[str, str]
    ped_roster: dict[str, str]
    pose_valid: bool = True
    yaw: np.ndarray | None = None
    fleet: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = dataclasses.field(default_factory=dict)
    _fields: dict[tuple[str, str], tuple[np.ndarray, list[str]] | None] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _ped_rows: tuple[list, list] | None = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if "peds_names" in self.data.columns and "peds_positions" in self.data.columns:
            self._ped_rows = (self.data["peds_names"].to_list(), self.data["peds_positions"].to_list())

    def field_at(self, entity: str, field: str, t_ns: int) -> str | None:
        key = (entity, field)
        if key not in self._fields:
            recorded = self.entity_roster.get(entity)
            series = None if recorded is None else _entity_field_series(self.events, recorded, field)
            self._fields[key] = None if series is None or len(series[0]) == 0 else series
        series = self._fields[key]
        if series is None:
            return None
        return _value_at(series[0], series[1], t_ns)

    def peds_at(self, i: int) -> dict[str, tuple[float, float]]:
        if self._ped_rows is None:
            return {}
        names = self._ped_rows[0][i]
        positions = self._ped_rows[1][i]
        if not names or not positions:
            return {}
        peds: dict[str, tuple[float, float]] = {}
        for j, name in enumerate(names):
            if 3 * j + 1 >= len(positions):
                break
            peds[_strip_env(name)] = (positions[3 * j], positions[3 * j + 1])
        return peds

    def sample(self, i: int) -> Sample:
        robots: dict[str, tuple[float, float, float]] = {}
        if self.pose_valid:
            robots["robot"] = (float(self.pos_x[i]), float(self.pos_y[i]), float(self.yaw[i]) if self.yaw is not None else 0.0)
        for name, (xs, ys, yaws) in self.fleet.items():
            robots[name] = (float(xs[i]), float(ys[i]), float(yaws[i]))
        t_ns = int(self.time_ns[i])
        known = frozenset(self.ped_roster) | frozenset(self.fleet)
        return Sample(t=t_ns / 1e9, robots=robots, peds=self.peds_at(i), field=lambda entity, field: self.field_at(entity, field, t_ns), known=known)

    @property
    def polygons(self) -> dict[str, object]:
        return {name: zone.polygon for name, zone in self.zones_by_name.items()}


def _atom_series(atom: Atom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    """Boolean series of one atom over the context axis, (None, False) when any sample cannot resolve it."""
    polygons = ctx.polygons
    series = np.zeros(len(ctx.time_ns), dtype=bool)
    for i in range(len(ctx.time_ns)):
        value = atom_holds(atom, ctx.sample(i), polygons, "robot")
        if value is None:
            return None, False
        series[i] = value
    return series, True


def _entity_atom_series(atom: Atom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    return _atom_series(atom, ctx)


def _robot_zone_series(atom: Atom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    return _atom_series(atom, ctx)


def _ped_zone_series(atom: Atom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    return _atom_series(atom, ctx)


def _operator_verdict(
    op: str,
    p_series: np.ndarray | None,
    p_ok: bool,
    q_series: np.ndarray | None,
    q_ok: bool,
) -> bool | None:
    return operator_verdict(op, p_series, p_ok, q_series, q_ok)


def _clause_verdict(clause: dict, ctx: _EvalContext) -> bool | None:
    """Score one clause dict, returning UNKNOWN (None) on any malformed or unresolvable input."""
    try:
        cond = EpisodeCondition.parse(clause)
        p_atom = parse_atom(cond.p)
        q_atom = parse_atom(cond.q) if cond.q is not None else None
    except Exception:
        return None

    p_series, p_ok = _atom_series(p_atom, ctx)
    if q_atom is None:
        return _operator_verdict(cond.op, p_series, p_ok, None, False)
    q_series, q_ok = _atom_series(q_atom, ctx)
    return _operator_verdict(cond.op, p_series, p_ok, q_series, q_ok)


def _resample_fleet(fleet: dict[str, pl.DataFrame], time_ns: np.ndarray, ego: str | None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Other robots' judge poses held onto the ego axis, robots without a sample before an instant are left out."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, frame in fleet.items():
        if name == ego or frame is None or len(frame) == 0:
            continue
        frame = frame.sort("time_ns")
        times = frame["time_ns"].to_numpy()
        idx = np.searchsorted(times, time_ns, side="right") - 1
        if np.any(idx < 0):
            continue
        out[name] = (frame["pos_x"].to_numpy()[idx], frame["pos_y"].to_numpy()[idx], frame["yaw"].to_numpy()[idx])
    return out


class ConditionComplianceCalculator(BaseMetricCalculator):
    """
    Offline verdicts for the episode's `conditions` clause list (SPEC_M3 M3.3).

    Each clause is one of five operators (`always`, `never`, `eventually`, `before`,
    `never_during`) over atoms judged per odom sample by the shared judge
    (`arena_simulation_setup.shared.judge`), the same code the task generator runs
    online: `entity.field == value` from the stepwise `semantic_snapshot` series (seeded
    by the latest snapshot at-or-before the episode start), `<subject> in <zone>` and
    `<subject> within <r> of <subject>` on the ego pose, other robots' recorded judge
    poses and recorded pedestrians, against zone polygons loaded from the recorded world
    asset in the flattened multi-level frame. An atom is UNKNOWN when its zone, entity
    field, ped or robot is unresolvable on any sample, a clause is UNKNOWN when any of
    its atoms is, and `condition_success` is FALSE when any clause is FALSE, else UNKNOWN
    when any clause is UNKNOWN, else TRUE. An episode with no `conditions` and an absent
    world asset both report every key as None.
    """

    NAME = "condition_compliance"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["odom"]

    UNITS = {
        "condition_success": "",
        "clauses_total": "",
        "clauses_passed": "",
        "clauses_failed": "",
        "clauses_unknown": "",
    }

    world: str | None = None

    def __init__(self, robot_params: RobotParams) -> None:
        super().__init__(robot_params)
        self._world_cache = _world_cache

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "condition_success",
            "clauses_total",
            "clauses_passed",
            "clauses_failed",
            "clauses_unknown",
        ]

    def _load_world(self, world_name: str) -> list[_ZoneGeometry] | None:
        return load_zone_geometry(world_name)

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        empty = dict.fromkeys(self.output_keys())

        conditions = episode.conditions
        if not conditions:
            return empty
        if self.world is None:
            return empty

        zones = self._load_world(self.world)
        if zones is None:
            return empty

        pos_x, pos_y, yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)
        if episode.data is None or "time_ns" not in episode.data.columns or len(episode.data) == 0:
            return empty

        time_ns = episode.data["time_ns"].to_numpy()
        if len(time_ns) == 0 or len(pos_x) != len(time_ns):
            return empty

        ctx = _EvalContext(
            events=_reconstruct_events(episode.semantic_snapshot),
            time_ns=time_ns,
            pos_x=pos_x,
            pos_y=pos_y,
            data=episode.data,
            zones_by_name={zone.name: zone for zone in zones},
            entity_roster=_entity_roster(episode.semantic_snapshot),
            ped_roster=_ped_roster(episode.data),
            pose_valid=bool(episode.start_pos),
            yaw=yaw,
            fleet=_resample_fleet(episode.fleet, time_ns, bare_robot_name(episode.robot_name or "")),
        )

        verdicts = [_clause_verdict(clause, ctx) for clause in conditions]
        passed = sum(1 for v in verdicts if v is True)
        failed = sum(1 for v in verdicts if v is False)
        unknown = sum(1 for v in verdicts if v is None)

        if failed:
            success: float | None = 0.0
        elif unknown:
            success = None
        else:
            success = 1.0

        return {
            "condition_success": success,
            "clauses_total": len(verdicts),
            "clauses_passed": passed,
            "clauses_failed": failed,
            "clauses_unknown": unknown,
        }
