from __future__ import annotations

import re
import typing
from collections import defaultdict

import numpy as np
import polars as pl
from arena_simulation_setup.shared.conditions import EpisodeCondition, Node
from arena_simulation_setup.shared.judge import (
    NodeMonitor,
    WallGrid,
    operator_verdict,
    zone_polygons,
)

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, Direction
from arena_evaluation.processing.metrics.common.alignment import asof_index
from arena_evaluation.processing.metrics.common.caching import load_world, world_cache
from arena_evaluation.processing.metrics.common.context import (
    EvalContext as _EvalContext,
)
from arena_evaluation.processing.metrics.common.events import (
    reconstruct_events as _reconstruct_events,
)
from arena_evaluation.processing.metrics.common.resampling import (
    resample_fleet as _resample_fleet,
)
from arena_evaluation.processing.metrics.common.rosters import (
    entity_roster as _entity_roster,
)
from arena_evaluation.processing.metrics.common.rosters import (
    ped_roster as _ped_roster,
)
from arena_evaluation.processing.metrics.common.rosters import (
    strip_env,
)
from arena_evaluation.processing.metrics.common.zones import (
    ZoneGeometry,
    extract_zone_geometry,
)
from arena_evaluation.processing.pedestrian_frames import frames_of
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World import Level

CLAUSES_FAILED = define_metric(
    "clauses_failed",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Count of condition clauses whose verdict is FALSE; NULL when the conditions, world or robot samples were not recorded.",
    null_reason=True,
)
CLAUSES_PASSED = define_metric(
    "clauses_passed",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Count of condition clauses whose verdict is TRUE; NULL when the conditions, world or robot samples were not recorded.",
    null_reason=True,
)
CLAUSES_TOTAL = define_metric(
    "clauses_total",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Count of condition clauses evaluated for the episode; NULL when the conditions, world or robot samples were not recorded.",
    null_reason=True,
)
CLAUSES_UNKNOWN = define_metric(
    "clauses_unknown",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Count of condition clauses whose verdict is UNKNOWN; NULL when the conditions, world or robot samples were not recorded.",
    null_reason=True,
)
CONDITION_SUCCESS = define_metric(
    "condition_success",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    description="Episode verdict: TRUE when every clause passes, FALSE when any fails, NULL when undecided or its inputs were not recorded.",
    null_reason=True,
)

_ROBOT_DIR_PREFIX = re.compile(r"^env_\d+_")

_wall_cache: dict[str, WallGrid | None] = {}


def bare_robot_name(robot_dir: str) -> str:
    """`env_0_jackal_0` -> `jackal_0`, the name phases and atoms use."""
    return _ROBOT_DIR_PREFIX.sub("", robot_dir, count=1)


def judge_zone_geometry(flattened: Level) -> list[ZoneGeometry]:
    """Every polygon the online judge resolves by name on a flattened world: zones, doors and elevator cabins."""
    zones = extract_zone_geometry(flattened, require_annotation=False)
    known = {zone.name for zone in zones}
    zones.extend(ZoneGeometry(name=name, polygon=polygon, max_speed=None, quiet=False, restricted=False) for name, polygon in zone_polygons(flattened).items() if name not in known)
    return zones


def _flattened_level(world_name: str) -> Level | None:
    """The world compacted into the flattened map frame, None when the world asset is not available locally."""
    loaded = load_world(world_name)
    return None if loaded is None else loaded.flattened


def load_zone_geometry(world_name: str) -> list[ZoneGeometry] | None:
    """Zone, door and elevator polygons of a world in the flattened map frame, None when the world asset is not available locally."""
    if world_name in world_cache:
        return world_cache[world_name]
    flattened = _flattened_level(world_name)
    zones = None if flattened is None else judge_zone_geometry(flattened)
    world_cache[world_name] = zones
    return zones


def load_wall_grid(world_name: str) -> WallGrid | None:
    """Wall cells of a world in the flattened map frame, None when the world asset is not available locally."""
    if world_name in _wall_cache:
        return _wall_cache[world_name]
    flattened = _flattened_level(world_name)
    walls = None if flattened is None else WallGrid.of_level(flattened)
    _wall_cache[world_name] = walls
    return walls


def clips_on_axis(peds: pl.DataFrame | None, gestures: pl.DataFrame | None, t_ns: np.ndarray) -> list[dict[str, frozenset[str]]] | None:
    """Per instant, the gesture clips of every pedestrian in the frame held at it, None when the recording carries no pedestrian_gestures table."""
    if gestures is None or peds is None or len(peds) == 0:
        return None
    if not {"t_ns", "ped_id", "clip"} <= set(gestures.columns):
        return None
    if not {"t_ns", "ped_id"} <= set(peds.columns):
        return None
    playing: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for t, ped_id, clip in zip(
        gestures["t_ns"].to_numpy(),
        gestures["ped_id"].to_numpy(),
        gestures["clip"].to_numpy(),
        strict=True,
    ):
        playing[int(t)][str(ped_id)].add(str(clip))
    frames = frames_of(peds)
    times = np.array([int(frame["t_ns"][0]) for frame in frames], dtype=np.int64)
    rows: list[dict[str, frozenset[str]]] = []
    for k, frame in enumerate(frames):
        rows.append({strip_env(str(name)): frozenset(playing.get(int(times[k]), {}).get(str(ped_id), ())) for name, ped_id in zip(frame["ped_name"], frame["ped_id"], strict=True)})
    idx = asof_index(times, t_ns)
    return [rows[i] if i >= 0 else {} for i in idx]


def _node_series(node: Node, ctx: _EvalContext) -> tuple[np.ndarray | None, bool]:
    """Boolean series of one condition node stepped over the context axis in time order, (None, False) when any sample cannot resolve it."""
    polygons = ctx.polygons
    monitor = NodeMonitor(node, "robot")
    series = np.zeros(len(ctx.t_ns), dtype=bool)
    for i in range(len(ctx.t_ns)):
        value = monitor.step(ctx.sample(i), polygons)
        if value is None:
            return None, False
        series[i] = value
    return series, True


def _clause_verdict(clause: dict, ctx: _EvalContext) -> bool | None:
    """Score one clause dict, returning UNKNOWN (None) on any malformed or unresolvable input."""
    try:
        cond = EpisodeCondition.parse(clause)
    except (ValueError, TypeError, KeyError):
        # A malformed clause is an unresolvable condition, not a calculator failure.
        return None

    p_series, p_ok = _node_series(cond.p_node, ctx)
    if cond.q_node is None:
        return operator_verdict(cond.op, p_series, p_ok, None, False)
    q_series, q_ok = _node_series(cond.q_node, ctx)
    return operator_verdict(cond.op, p_series, p_ok, q_series, q_ok)


class ConditionComplianceCalculator(BaseMetricCalculator):
    """
    Offline verdicts for the episode's `conditions` clause list (SPEC_M3 M3.3).

    Each clause is one of five operators (`always`, `never`, `eventually`, `before`,
    `never_during`) over conditions, text atoms or structured nodes with holds, stepped in order over the odom samples by the shared judge
    (`arena_simulation_setup.shared.judge`), the same code the task generator runs
    online: entity fields and field families from the stepwise `semantic_snapshot` series (seeded
    by the latest snapshot at-or-before the episode start), `<subject> in <zone>`,
    `<subject> within <r> of <subject>` and `robot sees <ped> within <r>` on the ego
    pose, other robots' recorded judge poses and recorded pedestrians, against zone
    polygons and the wall grid loaded from the recorded world asset in the flattened
    multi-level frame, and `<ped> plays <clip>` on the recorded `ped_gestures` table.
    An atom is UNKNOWN when its zone, entity field, ped, robot, wall grid or gesture
    table is unresolvable on any sample, a clause is UNKNOWN when any of
    its atoms is, and `condition_success` is FALSE when any clause is FALSE, else UNKNOWN
    when any clause is UNKNOWN, else TRUE. An episode with no `conditions` and an absent
    world asset both report every key as None.
    """

    NAME = "condition_compliance"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["robot_tracks"]
    PARAMS = ()

    OUTPUTS = (
        CONDITION_SUCCESS,
        CLAUSES_TOTAL,
        CLAUSES_PASSED,
        CLAUSES_FAILED,
        CLAUSES_UNKNOWN,
    )
    DIRECTIONS = {
        "condition_success": Direction.HIGHER,
        "clauses_passed": Direction.HIGHER,
        "clauses_failed": Direction.LOWER,
        "clauses_unknown": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = ["condition_success", "clauses_failed"]

    def _load_world(self, world_name: str) -> list[ZoneGeometry] | None:
        return load_zone_geometry(world_name)

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        empty = {metric.name: None for metric in self.OUTPUTS}

        conditions = episode.conditions
        if not conditions:
            return empty
        if self.world is None:
            return empty

        zones = self._load_world(self.world)
        if zones is None:
            return empty

        pos_x, pos_y, yaw, _ox, _oy, _oyaw = self.resolve_robot_pose(episode)
        if episode.data is None or len(episode.data) == 0:
            return empty

        t_ns = episode.data["t_ns"].to_numpy()
        if len(t_ns) == 0 or len(pos_x) != len(t_ns):
            return empty

        peds_df = self.native_ped_frame(episode)
        peds_frames = frames_of(peds_df) if peds_df is not None and len(peds_df) > 0 else []
        ctx = _EvalContext(
            events=_reconstruct_events(episode.semantic_snapshot),
            t_ns=t_ns,
            pos_x=pos_x,
            pos_y=pos_y,
            data=episode.data,
            zones_by_name={zone.name: zone for zone in zones},
            entity_roster=_entity_roster(episode.semantic_snapshot),
            ped_roster=_ped_roster(peds_df),
            pose_valid=bool(episode.start_pos),
            yaw=yaw,
            fleet=_resample_fleet(episode.fleet, t_ns, episode.robot_name or ""),
            walls=load_wall_grid(self.world),
            clips=clips_on_axis(
                peds_df,
                episode.topics.get("pedestrian_gestures") if episode.topics else None,
                t_ns,
            ),
            peds_frames=peds_frames,
            peds_times=np.array([int(frame["t_ns"][0]) for frame in peds_frames], dtype=np.int64),
        )

        verdicts = [_clause_verdict(clause, ctx) for clause in conditions]
        passed = sum(1 for v in verdicts if v is True)
        failed = sum(1 for v in verdicts if v is False)
        unknown = sum(1 for v in verdicts if v is None)

        if failed:
            success: bool | None = False
        elif unknown:
            success = None
        else:
            success = True

        return {
            "condition_success": success,
            "clauses_total": len(verdicts),
            "clauses_passed": passed,
            "clauses_failed": failed,
            "clauses_unknown": unknown,
        }
