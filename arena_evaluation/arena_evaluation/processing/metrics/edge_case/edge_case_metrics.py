from __future__ import annotations

import re
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from .criticality import Sample, formation_hold, score
from .provenance import Provenance, load_provenance, scenario_name

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SAMPLE_HZ = 10.0
PED_RADIUS = 0.3

_ENV_PREFIX = re.compile(r"^/?env_\d+/")


def agent_name(recorded: str) -> str:
    """`/env_0/d_1/0` -> `d_1`, the name the scenario gives the pedestrian."""
    return _ENV_PREFIX.sub("", recorded, count=1).lstrip("/").split("/", 1)[0]


def episode_provenance(episode: AlignedEpisodeBundle, world: str | None) -> Provenance | None:
    """The provenance of the scenario the episode ran, None when it has none."""
    world = episode.map or world
    scenario = scenario_name(episode.run)
    if not world or not scenario:
        return None
    return load_provenance(world, scenario)


def episode_samples(calc: BaseMetricCalculator, episode: AlignedEpisodeBundle) -> list[Sample]:
    """The robot on a SAMPLE_HZ grid with the pedestrians of the latest peds message, keyed by scenario name."""
    pos_x, pos_y, _yaw, time_ns = calc.resolve_native_pose(episode)
    if len(time_ns) < 2:
        return []
    grid = np.arange(int(time_ns[0]), int(time_ns[-1]) + 1, int(1e9 / SAMPLE_HZ), dtype=np.int64)
    rx = np.interp(grid, time_ns, pos_x)
    ry = np.interp(grid, time_ns, pos_y)
    vx, vy = calc.velocity_from_pose(rx, ry, grid)
    rows: list[dict[str, tuple[tuple[float, float], tuple[float, float], float]]] = [{} for _ in grid]
    peds = calc.native_ped_frame(episode)
    if peds is not None and "peds_names" in peds.columns and "peds_positions" in peds.columns:
        twists = "peds_twists" in peds.columns
        frame = peds.select(["time_ns", "peds_names", "peds_positions", *(["peds_twists"] if twists else [])])
        held = pl.DataFrame({"time_ns": grid}).join_asof(frame, on="time_ns", strategy="backward")
        for i, row in enumerate(held.iter_rows(named=True)):
            names = row["peds_names"] or []
            if not names:
                continue
            positions = row["peds_positions"] or []
            velocities = (row["peds_twists"] if twists else None) or []
            stride = len(positions) // len(names)
            v_stride = len(velocities) // len(names)
            for k, name in enumerate(names):
                p = (float(positions[k * stride]), float(positions[k * stride + 1]))
                v = (float(velocities[k * v_stride]), float(velocities[k * v_stride + 1])) if v_stride >= 2 else (0.0, 0.0)
                rows[i][agent_name(str(name))] = (p, v, PED_RADIUS)
    radius = calc.robot_params.robot_radius
    return [Sample(t=float(t) / 1e9, robot=(float(x), float(y)), robot_vel=(float(u), float(w)), robot_radius=radius, peds=rows[i]) for i, (t, x, y, u, w) in enumerate(zip(grid, rx, ry, vx, vy, strict=True))]


class EdgeCaseCriticalityCalculator(BaseMetricCalculator):
    """Criticality of an edge-case episode against the case's own agents: clearance, TTC, PET, freezes and the collision outcome."""

    NAME = "edge_case_criticality"
    CATEGORY = "edge_case"
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = [("tf_gt", "odom"), "peds"]

    UNITS = {
        "edge_case_id": "",
        "edge_case_steps": "",
        "edge_case_scope": "",
        "edge_case_agents": "",
        "edge_case_missing": "",
        "edge_case_min_ttc_s": "s",
        "edge_case_min_ttc_with": "",
        "edge_case_min_ttc_at": "s",
        "edge_case_min_clearance_m": "m",
        "edge_case_min_clearance_with": "",
        "edge_case_min_clearance_at": "s",
        "edge_case_pet_s": "s",
        "edge_case_pet_with": "",
        "edge_case_freeze_s": "s",
        "edge_case_intrusion_s": "s",
        "edge_case_collided": "",
        "edge_case_collision_fault": "",
        "edge_case_collision_with": "",
        "edge_case_collision_at": "s",
        "edge_case_collision_robot_closing": "m/s",
        "edge_case_collision_ped_closing": "m/s",
    }

    PRIMARY_OUTPUTS = ["edge_case_min_clearance_m", "edge_case_pet_s", "edge_case_freeze_s", "edge_case_collided"]
    OUTPUT_DIRECTIONS = {
        "edge_case_min_ttc_s": "higher",
        "edge_case_min_clearance_m": "higher",
        "edge_case_pet_s": "higher",
        "edge_case_freeze_s": "lower",
        "edge_case_intrusion_s": "lower",
        "edge_case_collided": "lower",
    }

    world: str | None = None

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        provenance = episode_provenance(episode, self.world)
        if provenance is None:
            return results
        samples = episode_samples(self, episode)
        recorded = {name for s in samples for name in s.peds}
        seen = [name for name in provenance.agents if name in recorded]
        results["edge_case_id"] = provenance.id
        results["edge_case_steps"] = list(provenance.steps)
        results["edge_case_scope"] = "own" if seen else "all"
        results["edge_case_agents"] = seen
        results["edge_case_missing"] = [name for name in provenance.agents if name not in recorded]
        result = score(samples, only=seen or None, encounters=[(e.agent, e.point) for e in provenance.encounters])
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
                "edge_case_collision_fault": result.collision_fault or None,
                "edge_case_collision_with": result.collision_with or None,
                "edge_case_collision_at": result.collision_at,
                "edge_case_collision_robot_closing": result.collision_robot_closing,
                "edge_case_collision_ped_closing": result.collision_ped_closing,
            }
        )
        return results


class EdgeCaseFormationCalculator(BaseMetricCalculator):
    """Whether the formation a case's D step owns formed around its centre, how long it held and when it broke."""

    NAME = "edge_case_formation"
    CATEGORY = "edge_case"
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = [("tf_gt", "odom"), "peds"]

    UNITS = {
        "edge_case_formation_variant": "",
        "edge_case_formation_members": "",
        "edge_case_formation_seen": "",
        "edge_case_formation_formed_at": "s",
        "edge_case_formation_held_s": "s",
        "edge_case_formation_broke_at": "s",
        "edge_case_formation_arrived": "",
    }

    PRIMARY_OUTPUTS = ["edge_case_formation_held_s"]
    OUTPUT_DIRECTIONS = {"edge_case_formation_held_s": "higher", "edge_case_formation_arrived": "higher"}

    world: str | None = None

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        provenance = episode_provenance(episode, self.world)
        members = provenance.owned.get("D", ()) if provenance is not None else ()
        if provenance is None or provenance.formation is None or not members:
            return results
        formation = provenance.formation
        hold = formation_hold(episode_samples(self, episode), members, radius=formation.radius, centre=formation.centre)
        results["edge_case_formation_variant"] = formation.variant or None
        results["edge_case_formation_members"] = hold["members"]
        results["edge_case_formation_seen"] = hold["seen"]
        results["edge_case_formation_formed_at"] = hold["formed_at"]
        results["edge_case_formation_held_s"] = hold["held_s"]
        results["edge_case_formation_broke_at"] = hold["broke_at"]
        results["edge_case_formation_arrived"] = hold["arrived"]
        return results
