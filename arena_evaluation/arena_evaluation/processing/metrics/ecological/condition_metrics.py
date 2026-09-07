from __future__ import annotations

import dataclasses
import logging
import re
import typing
from collections import defaultdict

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    _ZoneGeometry,
    _extract_zone_geometry,
    _offset_zones,
    _reconstruct_events,
    _zone_membership,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from arena_simulation_setup.shared.conditions import EntityAtom, EpisodeCondition, MembershipAtom, parse_atom

logger = logging.getLogger(__name__)

_ENV_PREFIX = re.compile(r"^env_\d+/")
_FLOAT_TOLERANCE = 1e-6


def _strip_env(name: str) -> str:
    """Drop the leading `env_<n>/` segment, keeping the rest incl. any level suffix."""
    return _ENV_PREFIX.sub("", name, count=1)


def _try_float(token: str) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _values_equal(recorded: str, expected: str) -> bool:
    """Float compare within tolerance when both parse as float, else exact string."""
    rf = _try_float(recorded)
    ef = _try_float(expected)
    if rf is not None and ef is not None:
        return abs(rf - ef) <= _FLOAT_TOLERANCE
    return recorded == expected


def _entity_roster(snapshot: pl.DataFrame | None) -> dict[str, str]:
    """Bare entity name -> recorded sim_path, for names resolving to a single sim_path."""
    if snapshot is None or len(snapshot) == 0 or "entity" not in snapshot.columns:
        return {}
    paths: dict[str, set[str]] = defaultdict(set)
    for entity in snapshot["entity"].unique().to_list():
        paths[_strip_env(entity)].add(entity)
    return {bare: next(iter(p)) for bare, p in paths.items() if len(p) == 1}


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


@dataclasses.dataclass
class _EvalContext:
    events: pl.DataFrame
    time_ns: np.ndarray
    pos_x: np.ndarray
    pos_y: np.ndarray
    data: pl.DataFrame
    zones_by_name: dict[str, _ZoneGeometry]
    entity_roster: dict[str, str]
    ped_roster: dict[str, str]
    pose_valid: bool = True


def _entity_atom_series(atom: EntityAtom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    """Boolean `entity.field == value` series over the odom axis, plus a resolvable flag."""
    entity = ctx.entity_roster.get(atom.entity)
    if entity is None:
        return None, False

    times, values = _entity_field_series(ctx.events, entity, atom.field)
    if len(times) == 0:
        return None, False

    series = np.zeros(len(ctx.time_ns), dtype=bool)
    for i in range(len(ctx.time_ns)):
        series[i] = _values_equal(_value_at(times, values, int(ctx.time_ns[i])), atom.value)
    return series, True


def _robot_zone_series(atom: MembershipAtom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    """Boolean `robot in zone` series over the odom axis, plus a resolvable flag."""
    if not ctx.pose_valid:
        return None, False
    zone = ctx.zones_by_name.get(atom.zone)
    if zone is None:
        return None, False
    idx = _zone_membership(ctx.pos_x, ctx.pos_y, [zone])
    return idx >= 0, True


def _ped_zone_series(atom: MembershipAtom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    """Boolean `<ped> in zone` series, zero-order-held onto the odom axis by the aligner."""
    import shapely

    zone = ctx.zones_by_name.get(atom.zone)
    if zone is None:
        return None, False
    recorded = ctx.ped_roster.get(atom.subject)
    if recorded is None:
        return None, False

    positions = ctx.data["peds_positions"].to_list()
    names = ctx.data["peds_names"].to_list()
    series = np.zeros(len(ctx.time_ns), dtype=bool)
    for i in range(len(ctx.time_ns)):
        row_names = names[i]
        row_positions = positions[i]
        if not row_names or recorded not in row_names:
            continue
        j = list(row_names).index(recorded)
        if 3 * j + 1 >= len(row_positions):
            continue
        point = shapely.Point(row_positions[3 * j], row_positions[3 * j + 1])
        series[i] = zone.polygon.covers(point)
    return series, True


def _atom_series(atom: EntityAtom | MembershipAtom, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    if isinstance(atom, MembershipAtom):
        if atom.subject == "robot":
            return _robot_zone_series(atom, ctx)
        return _ped_zone_series(atom, ctx)
    return _entity_atom_series(atom, ctx)


def _operator_verdict(
    op: str,
    p_series: np.ndarray | None,
    p_ok: bool,
    q_series: np.ndarray | None,
    q_ok: bool,
) -> bool | None:
    """One clause verdict from its atom series, None when any used atom is unresolvable."""
    if op in ("always", "never", "eventually"):
        if not p_ok:
            return None
        if op == "always":
            return bool(np.all(p_series))
        if op == "never":
            return not bool(np.any(p_series))
        return bool(np.any(p_series))

    if not (p_ok and q_ok):
        return None
    if op == "before":
        first_p = _first_true(p_series)
        first_q = _first_true(q_series)
        return first_p is not None and (first_q is None or first_p < first_q)
    return not bool(np.any(p_series & q_series))


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


class ConditionComplianceCalculator(BaseMetricCalculator):
    """
    Offline verdicts for the episode's `conditions` clause list (SPEC_M3 M3.3).

    Each clause is one of five operators (`always`, `never`, `eventually`, `before`,
    `never_during`) over bare atoms, an `entity.field == value` test reconstructed as a
    stepwise series from the episode's `semantic_snapshot` rows (seeded by the latest
    snapshot at-or-before the episode start), or a `<subject> in <zone>` test
    on the ego odom pose or a recorded pedestrian, both against zone polygons loaded
    from the recorded world asset in the flattened multi-level frame. An atom is
    UNKNOWN when its zone, entity field, or ped is unresolvable, a clause is UNKNOWN
    when any of its atoms is, and `condition_success` is FALSE when any clause is FALSE,
    else UNKNOWN when any clause is UNKNOWN, else TRUE. An episode with no `conditions`
    and an absent world asset both report every key as None. Single-robot attribution
    per SPEC_M1 M1.3.
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

    def __init__(self, robot_params: typing.Any) -> None:
        super().__init__(robot_params)
        self._world_cache: dict[str, list[_ZoneGeometry] | None] = {}

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
        if world_name in self._world_cache:
            return self._world_cache[world_name]

        from arena_simulation_setup.tree.World import WorldIdentifier

        try:
            view = WorldIdentifier(world_name).resolve_sync()
            world = view.load()
        except FileNotFoundError as e:
            logger.warning("condition_compliance: world '%s' not available locally: %s", world_name, e)
            self._world_cache[world_name] = None
            return None

        origins = view.level_origins()
        if origins is None:
            origins = {level_id: (0.0, 0.0) for level_id in world.levels}
        flattened = world.compact_world(origins)

        zones = _extract_zone_geometry(flattened, require_annotation=False)
        self._world_cache[world_name] = zones
        return zones

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

        pos_x, pos_y, _yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)
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
