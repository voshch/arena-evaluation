"""Edge-case criticality and formation metrics measured from recorded trajectories."""

from __future__ import annotations

import re
import typing
from collections.abc import Iterable, Mapping

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.alignment import interpolate_values
from arena_evaluation.processing.pedestrian_frames import pedestrians_at
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import Metric, MetricScope, define_metric

from .criticality import SETTLE_S, Sample, formation_hold, score
from .provenance import Provenance, load_provenance, scenario_name

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SAMPLE_HZ = 10.0
PEDS_TOLERANCE_NS = 100_000_000
TIMELINE_KIND = "timeline"
STIMULUS_PREFIX = "stimulus."
FORM_NEED = "form"
BREAK_NEED = "break"

_ENV_PREFIX = re.compile(r"^/?env_\d+/")


def _metric(
    name: str,
    dtype: pl.DataType | type[pl.DataType],
    scope: MetricScope,
    unit: str,
    description: str,
) -> Metric:
    """An edge-case metric published at one scope, with its null always explained."""
    return define_metric(
        name,
        dtype,
        scopes=(scope,),
        unit=unit,
        description=description,
        null_reason=True,
    )


EDGE_CASE_COLLIDED = _metric(
    "edge_case_collided",
    pl.Boolean,
    MetricScope.ROBOT,
    "",
    "Whether the case's robot made contact with one of its scored agents.",
)
EDGE_CASE_COLLISION_AT = _metric(
    "edge_case_collision_at",
    pl.Float64,
    MetricScope.CONTACT,
    "s",
    "Seconds after the episode start of the first collision with a scored agent.",
)
EDGE_CASE_COLLISION_PED_CLOSING = _metric(
    "edge_case_collision_ped_closing",
    pl.Float64,
    MetricScope.CONTACT,
    "m/s",
    "Pedestrian closing speed along the line of sight at the first collision.",
)
EDGE_CASE_COLLISION_ROBOT_CLOSING = _metric(
    "edge_case_collision_robot_closing",
    pl.Float64,
    MetricScope.CONTACT,
    "m/s",
    "Robot closing speed along the line of sight at the first collision.",
)
EDGE_CASE_COLLISION_WITH = _metric(
    "edge_case_collision_with",
    pl.String,
    MetricScope.CONTACT,
    "",
    "Agent the robot first collided with.",
)
EDGE_CASE_FREEZE_S = _metric(
    "edge_case_freeze_s",
    pl.Float64,
    MetricScope.ROBOT,
    "s",
    "Seconds after the settling window the robot stood below the freeze speed.",
)
EDGE_CASE_ID = _metric(
    "edge_case_id",
    pl.String,
    MetricScope.EPISODE,
    "",
    "Identifier of the case the episode ran, from its provenance.",
)
EDGE_CASE_INTRUSION_S = _metric(
    "edge_case_intrusion_s",
    pl.Float64,
    MetricScope.ROBOT,
    "s",
    "Seconds a scored pedestrian was inside the personal-space edge clearance.",
)
EDGE_CASE_MIN_CLEARANCE_AT = _metric(
    "edge_case_min_clearance_at",
    pl.Float64,
    MetricScope.ROBOT,
    "s",
    "Seconds after the episode start of the closest approach to a scored agent.",
)
EDGE_CASE_MIN_CLEARANCE_M = _metric(
    "edge_case_min_clearance_m",
    pl.Float64,
    MetricScope.ROBOT,
    "m",
    "Closest surface clearance between the robot and a scored agent.",
)
EDGE_CASE_MIN_CLEARANCE_WITH = _metric(
    "edge_case_min_clearance_with",
    pl.String,
    MetricScope.ROBOT,
    "",
    "Agent the closest approach was measured against.",
)
EDGE_CASE_MIN_TTC_AT = _metric(
    "edge_case_min_ttc_at",
    pl.Float64,
    MetricScope.ROBOT,
    "s",
    "Seconds after the episode start of the shortest time to collision.",
)
EDGE_CASE_MIN_TTC_S = _metric(
    "edge_case_min_ttc_s",
    pl.Float64,
    MetricScope.ROBOT,
    "s",
    "Shortest constant-velocity time to collision with a scored agent.",
)
EDGE_CASE_MIN_TTC_WITH = _metric(
    "edge_case_min_ttc_with",
    pl.String,
    MetricScope.ROBOT,
    "",
    "Agent the shortest time to collision was measured against.",
)
EDGE_CASE_PET_S = _metric(
    "edge_case_pet_s",
    pl.Float64,
    MetricScope.ENCOUNTER,
    "s",
    "Post-encroachment time at the case's designed encounter point.",
)
EDGE_CASE_PET_WITH = _metric(
    "edge_case_pet_with",
    pl.String,
    MetricScope.ENCOUNTER,
    "",
    "Agent the post-encroachment time was measured with.",
)
EDGE_CASE_PLAYED = _metric(
    "edge_case_played",
    pl.Boolean,
    MetricScope.EPISODE,
    "",
    "Whether every encounter agent received a stimulus after the episode start.",
)
EDGE_CASE_SCOPE = _metric(
    "edge_case_scope",
    pl.String,
    MetricScope.EPISODE,
    "",
    "Population the case was scored against: its own agents, the unplayed set, or everyone.",
)
EDGE_CASE_TRIGGERED_AT = _metric(
    "edge_case_triggered_at",
    pl.Float64,
    MetricScope.EPISODE,
    "s",
    "Seconds after the episode start the first encounter stimulus fired.",
)

EDGE_CASE_FORMATION_ARRIVED = _metric(
    "edge_case_formation_arrived",
    pl.Int64,
    MetricScope.EPISODE,
    "",
    "Most formation members recorded inside the formation radius at one sample.",
)
EDGE_CASE_FORMATION_BREAK_FIRED_AT = _metric(
    "edge_case_formation_break_fired_at",
    pl.Float64,
    MetricScope.EPISODE,
    "s",
    "Seconds after the episode start the formation's break stimulus fired.",
)
EDGE_CASE_FORMATION_BROKE_AT = _metric(
    "edge_case_formation_broke_at",
    pl.Float64,
    MetricScope.EPISODE,
    "s",
    "Seconds after the episode start the held formation broke.",
)
EDGE_CASE_FORMATION_FORM_FIRED_AT = _metric(
    "edge_case_formation_form_fired_at",
    pl.Float64,
    MetricScope.EPISODE,
    "s",
    "Seconds after the episode start the formation's form stimulus fired.",
)
EDGE_CASE_FORMATION_FORMED_AT = _metric(
    "edge_case_formation_formed_at",
    pl.Float64,
    MetricScope.EPISODE,
    "s",
    "Seconds after the episode start every recorded member first stood inside the formation radius.",
)
EDGE_CASE_FORMATION_HELD_S = _metric(
    "edge_case_formation_held_s",
    pl.Float64,
    MetricScope.EPISODE,
    "s",
    "Seconds the recorded formation members held together.",
)
EDGE_CASE_FORMATION_SEEN = _metric(
    "edge_case_formation_seen",
    pl.Boolean,
    MetricScope.EPISODE,
    "",
    "Whether at least one formation member was recorded.",
)
EDGE_CASE_FORMATION_VARIANT = _metric(
    "edge_case_formation_variant",
    pl.String,
    MetricScope.EPISODE,
    "",
    "Formation variant the case's provenance names.",
)


def agent_name(recorded: str) -> str:
    """`/env_0/d_1/0` -> `d_1`, the name the scenario gives the pedestrian."""
    return _ENV_PREFIX.sub("", recorded, count=1).lstrip("/").split("/", 1)[0]


def episode_provenance(episode: AlignedEpisodeBundle, world: str | None) -> Provenance | None:
    """The provenance of the scenario the episode ran, None when it has none."""
    world = episode.map or world
    scenario = scenario_name(episode.run)
    if not world or not scenario:
        return None
    return load_provenance(world, scenario, episode.run_dir)


def _case_agent(episode: AlignedEpisodeBundle, actor_id: object) -> str | None:
    """The scenario name of the pedestrian the recorded id names, None when the episode records no such pedestrian."""
    if episode.peds is None or actor_id is None:
        return None
    names = episode.peds.filter(pl.col("ped_id").cast(pl.String) == str(actor_id)).get_column("ped_name").head(1).to_numpy()
    return agent_name(str(names[0])) if len(names) > 0 else None


def fired_stimuli(snapshot: pl.DataFrame | None, start_s: float | None) -> dict[str, dict[str, float]] | None:
    """Sim time in s each `stimulus.<agent>.<need>` of the `timeline` entity fired from `start_s` on, None without the entity."""
    if snapshot is None or len(snapshot) == 0:
        return None
    rows = snapshot.filter(pl.col("kind") == TIMELINE_KIND)
    if len(rows) == 0:
        return None
    fired: dict[str, dict[str, float]] = {}
    matched = rows.filter((pl.col("field_kind") == "discrete") & pl.col("field_name").str.starts_with(STIMULUS_PREFIX))
    for field, value in zip(
        matched.get_column("field_name").to_numpy(),
        matched.get_column("value_str").to_numpy(),
        strict=True,
    ):
        agent, _, need = field[len(STIMULUS_PREFIX) :].rpartition(".")
        try:
            at = float(value)
        except (TypeError, ValueError):
            continue
        if not agent or not need or (start_s is not None and at < start_s - PEDS_TOLERANCE_NS / 1e9):
            continue
        needs = fired.setdefault(agent, {})
        needs[need] = min(at, needs.get(need, at))
    return fired


def _first_fired(
    fired: Mapping[str, Mapping[str, float]],
    agents: Iterable[str],
    need: str | None,
    start_s: float,
) -> float | None:
    """Seconds after `start_s` the first of `agents` received `need` (any need when None), None when none did."""
    times = [at for agent in agents for name, at in fired.get(agent, {}).items() if need is None or name == need]
    return min(times) - start_s if times else None


def _pedestrian_radius_m(name: str, radii: Mapping[str, float], recorded_radius_m: float | None) -> float:
    """The scenario-pinned radius, else the pedestrian's recorded radius_m; neither is never replaced by a default."""
    pinned = radii.get(name)
    if pinned is not None:
        if not np.isfinite(pinned) or pinned <= 0.0:
            raise MetricInputError(f"edge_case needs a positive radius for pedestrian {name!r}; the scenario pins {pinned!r}")
        return float(pinned)
    if recorded_radius_m is None or not np.isfinite(recorded_radius_m) or recorded_radius_m <= 0.0:
        raise MetricInputError(f"edge_case needs the radius of pedestrian {name!r}: the scenario pins none and pedestrian_arena_peds records none")
    return float(recorded_radius_m)


def episode_samples(
    calc: BaseMetricCalculator,
    episode: AlignedEpisodeBundle,
    radii: Mapping[str, float],
) -> list[Sample]:
    """The robot on a SAMPLE_HZ grid with the pedestrians of a peds message at most 0.1 s old, by scenario name and `radii`."""
    pos_x, pos_y, _yaw, t_ns = calc.resolve_native_pose(episode)
    if len(t_ns) < 2:
        return []
    step_ns = int(1e9 / SAMPLE_HZ)
    grid = np.arange(int(t_ns[0]), int(t_ns[-1]) + 1, step_ns, dtype=np.int64)
    rx = interpolate_values(t_ns, pos_x, grid)
    ry = interpolate_values(t_ns, pos_y, grid)
    vx, vy = calc.velocity_from_pose(rx, ry, grid)
    peds_by_frame: dict[int, dict[str, tuple[tuple[float, float], tuple[float, float], float]]] = {}
    peds = calc.native_ped_frame(episode)
    if peds is not None and peds.height > 0:
        held = pedestrians_at(pl.Series("t_ns", grid), peds, PEDS_TOLERANCE_NS).with_columns(
            frame_ix=((pl.col("t_ns") - int(grid[0])) // step_ns).cast(pl.Int64),
            bare_name=(pl.col("ped_name").cast(pl.String).str.replace(_ENV_PREFIX.pattern, "").str.split("/").list.first()),
            recorded_radius_m=pl.col("radius_m").cast(pl.Float64),
        )
        # agent-contracts: ignore AC004 - builds Python Sample dicts for criticality scorer
        for row in held.iter_rows(named=True):
            name = str(row["bare_name"])
            peds_by_frame.setdefault(int(row["frame_ix"]), {})[name] = (
                (float(row["x"]), float(row["y"])),
                (float(row["vx"]), float(row["vy"])),
                _pedestrian_radius_m(name, radii, row["recorded_radius_m"]),
            )
    radius = calc.robot_params.robot_radius
    return [
        Sample(
            t=float(t) / 1e9,
            robot=(float(x), float(y)),
            robot_vel=(float(u), float(w)),
            robot_radius=radius,
            peds=peds_by_frame.get(i, {}),
        )
        for i, (t, x, y, u, w) in enumerate(zip(grid, rx, ry, vx, vy, strict=True))
    ]


class EdgeCaseCriticalityCalculator(BaseMetricCalculator):
    """Criticality of an edge-case episode against the case's own agents: clearance, TTC, PET, freezes and the collision outcome."""

    NAME = "edge_case_criticality"
    CATEGORY = "edge_case"
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_arena_peds"]

    OUTPUTS = (
        EDGE_CASE_COLLIDED,
        EDGE_CASE_COLLISION_AT,
        EDGE_CASE_COLLISION_PED_CLOSING,
        EDGE_CASE_COLLISION_ROBOT_CLOSING,
        EDGE_CASE_COLLISION_WITH,
        EDGE_CASE_FREEZE_S,
        EDGE_CASE_ID,
        EDGE_CASE_INTRUSION_S,
        EDGE_CASE_MIN_CLEARANCE_AT,
        EDGE_CASE_MIN_CLEARANCE_M,
        EDGE_CASE_MIN_CLEARANCE_WITH,
        EDGE_CASE_MIN_TTC_AT,
        EDGE_CASE_MIN_TTC_S,
        EDGE_CASE_MIN_TTC_WITH,
        EDGE_CASE_PET_S,
        EDGE_CASE_PET_WITH,
        EDGE_CASE_PLAYED,
        EDGE_CASE_SCOPE,
        EDGE_CASE_TRIGGERED_AT,
    )
    PRIMARY_OUTPUTS = [
        "edge_case_min_clearance_m",
        "edge_case_pet_s",
        "edge_case_freeze_s",
        "edge_case_collided",
    ]
    DIRECTIONS = {
        "edge_case_collided": Direction.LOWER,
        "edge_case_freeze_s": Direction.LOWER,
        "edge_case_intrusion_s": Direction.LOWER,
        "edge_case_min_clearance_m": Direction.HIGHER,
        "edge_case_min_ttc_s": Direction.HIGHER,
        "edge_case_pet_s": Direction.HIGHER,
    }
    CHILD_OUTPUTS = (
        "edge_case_steps",
        "edge_case_agents",
        "edge_case_missing",
        "edge_case_unplayed",
    )

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        return self._case_criticality(episode, focus=None)

    def _case_criticality(self, episode: AlignedEpisodeBundle, focus: str | None) -> dict[str, typing.Any]:
        """The case scored over the whole episode, or over the one case agent `focus` that a CONTACT or ENCOUNTER slice names."""
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        provenance = episode_provenance(episode, self.world)
        if provenance is None:
            return results
        samples = episode_samples(self, episode, provenance.radii)
        recorded = {name for s in samples for name in s.peds}
        seen = [name for name in provenance.agents if name in recorded]
        encountered = list(dict.fromkeys(e.agent for e in provenance.encounters))
        start_s = samples[0].t if samples else None
        fired = fired_stimuli(episode.semantic_snapshot, start_s)
        unplayed = [] if fired is None else [agent for agent in encountered if agent not in fired]
        scored = [name for name in seen if name not in unplayed]
        results["edge_case_id"] = provenance.id
        results["edge_case_steps"] = list(provenance.steps)
        results["edge_case_scope"] = "own" if scored else "unplayed" if seen else "all"
        results["edge_case_agents"] = scored
        results["edge_case_missing"] = [name for name in provenance.agents if name not in recorded]
        if fired is not None and encountered:
            results["edge_case_played"] = not unplayed
            results["edge_case_unplayed"] = unplayed
            if start_s is not None:
                results["edge_case_triggered_at"] = _first_fired(fired, encountered, None, start_s)
        if seen and not scored:
            return results
        if focus is not None and focus not in scored:
            return results
        encounters = [(e.agent, e.point) for e in provenance.encounters if e.agent not in unplayed and (focus is None or e.agent == focus)]
        result = score(
            samples,
            only=[focus] if focus is not None else (scored or None),
            encounters=encounters,
            settle_s=SETTLE_S + provenance.hold_s,
        )
        if not result.observed:
            return results
        results.update(
            {
                "edge_case_min_ttc_s": result.min_ttc_s,
                "edge_case_min_ttc_with": result.min_ttc_with or None,
                "edge_case_min_ttc_at": result.min_ttc_at,
                "edge_case_min_clearance_m": result.min_clearance_m,
                "edge_case_min_clearance_with": result.min_clearance_with or None,
                "edge_case_min_clearance_at": result.min_clearance_at,
                "edge_case_pet_s": result.pet_s,
                "edge_case_pet_with": result.pet_with or None,
                "edge_case_freeze_s": result.freeze_duration_s,
                "edge_case_intrusion_s": result.intrusion_time_s,
                "edge_case_collided": result.collided,
                "edge_case_collision_with": result.collision_with or None,
                "edge_case_collision_at": result.collision_at,
                "edge_case_collision_robot_closing": result.collision_robot_closing,
                "edge_case_collision_ped_closing": result.collision_ped_closing,
            }
        )
        return results

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """Each scope publishes its own keys: ROBOT the whole case, CONTACT and ENCOUNTER only the case agent their slice names."""
        if scope_slice is None or scope_slice.scope == MetricScope.EPISODE:
            return self.calculate(episode, prior_results)
        full = scope_slice.full_episode
        focus = None
        if scope_slice.scope != MetricScope.ROBOT:
            actor_key = "obstacle_id" if scope_slice.scope == MetricScope.CONTACT else "actor_2"
            focus = _case_agent(full, scope_slice.identity.get(actor_key))
            if focus is None:
                return dict.fromkeys(self.output_keys_for_scope(scope_slice.scope))
        scored = self._case_criticality(full, focus=focus)
        return {key: scored[key] for key in self.output_keys_for_scope(scope_slice.scope)}


class EdgeCaseFormationCalculator(BaseMetricCalculator):
    """Whether the formation a case's D step owns formed around its center, how long it held and when it broke."""

    NAME = "edge_case_formation"
    CATEGORY = "edge_case"
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_arena_peds"]

    OUTPUTS = (
        EDGE_CASE_FORMATION_ARRIVED,
        EDGE_CASE_FORMATION_BREAK_FIRED_AT,
        EDGE_CASE_FORMATION_BROKE_AT,
        EDGE_CASE_FORMATION_FORM_FIRED_AT,
        EDGE_CASE_FORMATION_FORMED_AT,
        EDGE_CASE_FORMATION_HELD_S,
        EDGE_CASE_FORMATION_SEEN,
        EDGE_CASE_FORMATION_VARIANT,
    )
    PRIMARY_OUTPUTS = ["edge_case_formation_held_s"]
    DIRECTIONS = {
        "edge_case_formation_arrived": Direction.HIGHER,
        "edge_case_formation_held_s": Direction.HIGHER,
    }
    CHILD_OUTPUTS = (
        "edge_case_formation_members",
        "edge_case_formation_absent",
    )

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        provenance = episode_provenance(episode, self.world)
        members = provenance.owned.get("D", ()) if provenance is not None else ()
        if provenance is None or provenance.formation is None or not members:
            return results
        formation = provenance.formation
        samples = episode_samples(self, episode, provenance.radii)
        start_s = samples[0].t if samples else None
        fired = fired_stimuli(episode.semantic_snapshot, start_s) or {}
        form_at = break_at = None
        if start_s is not None:
            form_at = _first_fired(fired, members, FORM_NEED, start_s)
            break_at = _first_fired(fired, members, BREAK_NEED, start_s)
        hold = formation_hold(
            samples,
            members,
            radius=formation.radius,
            center=formation.center,
            form_at=form_at,
        )
        results["edge_case_formation_variant"] = formation.variant or None
        results["edge_case_formation_members"] = list(hold.members)
        results["edge_case_formation_absent"] = list(hold.absent)
        results["edge_case_formation_seen"] = hold.seen
        results["edge_case_formation_formed_at"] = hold.formed_at
        results["edge_case_formation_held_s"] = hold.held_s
        results["edge_case_formation_broke_at"] = hold.broke_at
        results["edge_case_formation_arrived"] = hold.arrived
        results["edge_case_formation_form_fired_at"] = form_at
        results["edge_case_formation_break_fired_at"] = break_at
        return results

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE scope only: the case provenance and its agents are episode-level records."""
        if scope_slice is not None and scope_slice.scope != MetricScope.EPISODE:
            return dict.fromkeys(self.output_keys())
        return self.calculate(episode, prior_results)
