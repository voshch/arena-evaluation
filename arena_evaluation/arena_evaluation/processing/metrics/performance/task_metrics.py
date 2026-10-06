from __future__ import annotations

import logging
import typing

import attrs
import numpy as np
import polars as pl
from arena_simulation_setup.shared.conditions import EpisodeCondition
from arena_simulation_setup.shared.judge import ConditionMonitor, PhaseMonitor
from arena_simulation_setup.shared.task import GoToPhase, TaskPhase
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import _reconstruct_events
from arena_evaluation.processing.metrics.ecological.condition_metrics import (
    _entity_roster,
    _EvalContext,
    _ped_roster,
    _resample_fleet,
    _world_cache,
    bare_robot_name,
    load_zone_geometry,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

logger = logging.getLogger(__name__)


def _split_indices(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v]


def _robot_events(events: pl.DataFrame, robot: str) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Stepwise series of the robot entity's fields (phase, met, failed, dropped, violated), keyed by field."""
    rows = events.filter((pl.col("kind") == "robot") & (pl.col("entity").str.replace(r"^env_\d+/", "") == robot))
    series: dict[str, tuple[np.ndarray, list[str]]] = {}
    for (field,), group in rows.group_by(["field"]):
        g = group.sort("time_ns")
        series[str(field)] = (g["time_ns"].to_numpy(), g["current"].to_list())
    return series


def _last(series: dict[str, tuple[np.ndarray, list[str]]], field: str) -> str:
    entry = series.get(field)
    return entry[1][-1] if entry is not None and entry[1] else ""


def _active_phase_spans(series: dict[str, tuple[np.ndarray, list[str]]], end_ns: int) -> dict[int, tuple[int, int]]:
    """[start, end) recording interval of every phase that was active, from the `phase` field transitions."""
    entry = series.get("phase")
    if entry is None:
        return {}
    times, values = entry
    spans: dict[int, tuple[int, int]] = {}
    for i, value in enumerate(values):
        if value == "":
            continue
        start = int(times[i])
        end = int(times[i + 1]) if i + 1 < len(times) else end_ns
        index = int(value)
        first, _ = spans.get(index, (start, end))
        spans[index] = (min(first, start), end)
    return spans


class TaskReplayCalculator(BaseMetricCalculator):
    """Replays the robot's recorded phases with the shared judge and compares its verdicts to the recorded ones."""

    NAME = "task_replay"
    CATEGORY = "performance"
    REQUIRED_TOPICS = ["task_pose"]

    UNITS = {
        "phases_total": "",
        "phases_met": "",
        "phases_failed": "",
        "phases_dropped": "",
        "goal_condition_rate": "",
        "phase_outcomes": "",
        "unsafe": "",
        "judge_agrees": "",
        "judge_disagreements": "",
    }

    PRIMARY_OUTPUTS = ["goal_condition_rate", "unsafe", "judge_agrees"]
    OUTPUT_DIRECTIONS = {"goal_condition_rate": "higher", "unsafe": "lower"}

    world: str | None = None

    def __init__(self, robot_params: RobotParams) -> None:
        super().__init__(robot_params)
        self._world_cache = _world_cache

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS.keys())

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        empty = dict.fromkeys(self.output_keys())
        spec = episode.phases
        if not spec or not spec.get("phases"):
            return empty
        phases = [TaskPhase.parse(p) for p in spec["phases"]]
        robot = bare_robot_name(episode.robot_name or "")
        snapshot = episode.semantic_snapshot
        events = _reconstruct_events(snapshot)
        robot_series = _robot_events(events, robot)

        met = set(_split_indices(_last(robot_series, "met")))
        failed = set(_split_indices(_last(robot_series, "failed")))
        dropped = set(_split_indices(_last(robot_series, "dropped")))
        violated = {v for v in _last(robot_series, "violated").split(",") if v}

        results: dict[str, typing.Any] = dict(empty)
        results["phases_total"] = len(phases)
        results["phases_met"] = len(met)
        results["phases_failed"] = len(failed)
        results["phases_dropped"] = len(dropped)
        results["goal_condition_rate"] = len(met) / len(phases) if phases else None
        results["phase_outcomes"] = ["met" if i in met else "failed" if i in failed else "dropped" if i in dropped else "pending" for i in range(len(phases))]
        results["unsafe"] = len(violated)

        poses = episode.topics.get("task_pose") if episode.topics else None
        zones = load_zone_geometry(self.world) if self.world else None
        if poses is None or len(poses) == 0 or zones is None:
            return results

        poses = poses.sort("time_ns")
        time_ns = poses["time_ns"].to_numpy()
        ctx = _EvalContext(
            events=events,
            time_ns=time_ns,
            pos_x=poses["pos_x"].to_numpy(),
            pos_y=poses["pos_y"].to_numpy(),
            data=_peds_on_axis(episode.peds, time_ns),
            zones_by_name={zone.name: zone for zone in zones},
            entity_roster=_entity_roster(snapshot),
            ped_roster={},
            yaw=poses["yaw"].to_numpy(),
            fleet=_resample_fleet(episode.fleet, time_ns, robot),
        )
        ctx.ped_roster = _ped_roster(ctx.data)
        polygons = ctx.polygons
        spans = _active_phase_spans(robot_series, int(time_ns[-1]) + 1)

        disagreements: list[str] = []
        replay_violated: set[str] = set()
        scoped = [(c, ConditionMonitor(EpisodeCondition.parse({k: v for k, v in c.items() if k not in ("id", "from", "to")}), "robot")) for c in spec.get("conditions", [])]

        for index, phase in enumerate(phases):
            span = spans.get(index)
            if span is None:
                continue
            lo = int(np.searchsorted(time_ns, span[0], side="left"))
            hi = int(np.searchsorted(time_ns, span[1], side="left"))
            monitor = PhaseMonitor(phase, "robot")
            replay_met = False
            for i in range(lo, hi):
                sample = ctx.sample(i)
                if isinstance(phase, GoToPhase) and phase.hold and "robot" in sample.robots:
                    x, y, yaw = sample.robots["robot"]
                    phase = attrs.evolve(phase, pose=Pose(Position(x, y), Orientation.from_yaw(yaw)))
                    monitor.phase = phase
                closing = i == hi - 1 and index in met | failed
                action_done = None if isinstance(phase, GoToPhase) else closing
                signal = phase.signal if isinstance(phase, GoToPhase) and phase.signal and closing else None
                for cond, cmon in scoped:
                    if _covers(cond, index):
                        cmon.update(sample, polygons)
                if monitor.step(sample, polygons, action_done, signal):
                    replay_met = True
                    break
            for j, verdict in enumerate(monitor.finish()):
                if verdict is False:
                    replay_violated.add(f"{index}:{j}")
            closed = index in met or index in failed
            if monitor.misfire is not None and index in met:
                disagreements.append(f"phase {index} recorded as met but {monitor.misfire} in replay")
            elif closed and not replay_met and monitor.misfire is None:
                disagreements.append(f"phase {index} recorded as met but not met in replay")
            if index not in met | failed | dropped and replay_met:
                disagreements.append(f"phase {index} met in replay but still pending in the record")

        for cond, cmon in scoped:
            if cmon.finish() is False:
                replay_violated.add(str(cond.get("id", "")))
        if replay_violated != violated:
            disagreements.append(f"violated conditions differ: replay {sorted(replay_violated)} vs recorded {sorted(violated)}")

        results["judge_agrees"] = not disagreements
        results["judge_disagreements"] = disagreements
        return results


def _covers(cond: dict, index: int) -> bool:
    first = cond.get("from", 0) or 0
    last = cond.get("to")
    return index >= first and (last is None or index <= last)


def _peds_on_axis(peds: pl.DataFrame | None, time_ns: np.ndarray) -> pl.DataFrame:
    """Recorded ped names and positions held onto the judge pose axis."""
    if peds is None or len(peds) == 0 or "peds_names" not in peds.columns:
        return pl.DataFrame({"time_ns": time_ns})
    peds = peds.sort("time_ns")
    times = peds["time_ns"].to_numpy()
    idx = np.searchsorted(times, time_ns, side="right") - 1
    names = peds["peds_names"].to_list()
    positions = peds["peds_positions"].to_list()
    return pl.DataFrame(
        {
            "time_ns": time_ns,
            "peds_names": [names[i] if i >= 0 else [] for i in idx],
            "peds_positions": [positions[i] if i >= 0 else [] for i in idx],
        }
    )


__all__ = ["TaskReplayCalculator"]
