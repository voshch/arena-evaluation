from __future__ import annotations

import typing

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.condition_metrics import bare_robot_name
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

_ENDINGS = ("met", "failed", "dropped")


def _events(tasks: list[dict]) -> list[dict]:
    """Events carrying a task id, in time order (report order breaks ties)."""
    kept = [e for e in tasks if e.get("task")]
    return sorted(kept, key=lambda e: float(e.get("t") or 0.0))


def _span_s(episode: AlignedEpisodeBundle, last_event_s: float) -> float:
    """Episode length in seconds: the aligned data span, at least the last task event."""
    span = 0.0
    if episode.data is not None and "time_ns" in episode.data.columns and len(episode.data) > 1:
        time_ns = episode.data["time_ns"]
        span = float(time_ns[-1] - time_ns[0]) / 1e9
    return max(span, last_event_s)


class TaskEventMetricsCalculator(BaseMetricCalculator):
    """Task-level metrics of any task layer from the events of EpisodeRecord.tasks."""

    NAME = "task_events"
    CATEGORY = "performance"

    UNITS = {
        "tasks_total": "",
        "tasks_met": "",
        "tasks_failed": "",
        "task_reassignments": "",
        "task_success_rate": "",
        "makespan_s": "s",
        "busy_s": "s",
        "idle_s": "s",
        "task_ids": "",
        "task_outcomes": "",
        "task_durations_s": "s",
    }

    PRIMARY_OUTPUTS = ["task_success_rate", "makespan_s", "idle_s"]
    OUTPUT_DIRECTIONS = {"task_success_rate": "higher", "makespan_s": "lower", "idle_s": "lower", "tasks_failed": "lower", "task_reassignments": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(cls.UNITS.keys())

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        results: dict[str, typing.Any] = dict.fromkeys(self.output_keys())
        events = _events(episode.tasks)
        if not events:
            return results

        ids: list[str] = []
        first_start: dict[str, float] = {}
        final: dict[str, tuple[str, float]] = {}
        last_met: float | None = None
        reassignments = 0
        open_since: dict[tuple[str, str], float] = {}
        intervals: list[tuple[str, float, float | None]] = []
        for event in events:
            task, kind, robot, t = str(event["task"]), str(event.get("kind", "")), str(event.get("robot", "")), float(event.get("t") or 0.0)
            if task not in ids:
                ids.append(task)
            if kind == "reassigned":
                reassignments += 1
            elif kind == "started":
                first_start.setdefault(task, t)
                open_since[(task, robot)] = t
            elif kind in _ENDINGS:
                final[task] = (kind, t)
                if kind == "met":
                    last_met = t
                for key in [k for k in open_since if k[0] == task]:
                    intervals.append((key[1], open_since.pop(key), t))
        for (_, robot), since in open_since.items():
            intervals.append((robot, since, None))

        span = _span_s(episode, float(events[-1].get("t") or 0.0))
        total = len(ids)
        met = sum(1 for kind, _ in final.values() if kind == "met")
        starts = list(first_start.values())

        me = bare_robot_name(episode.robot_name or "")
        busy = sum(max((end if end is not None else span) - start, 0.0) for robot, start, end in intervals if bare_robot_name(robot) == me)

        results["tasks_total"] = total
        results["tasks_met"] = met
        results["tasks_failed"] = sum(1 for kind, _ in final.values() if kind == "failed")
        results["task_reassignments"] = reassignments
        results["task_success_rate"] = met / total
        results["makespan_s"] = last_met - min(starts) if last_met is not None and starts else None
        results["busy_s"] = busy
        results["idle_s"] = max(span - busy, 0.0)
        results["task_ids"] = ids
        results["task_outcomes"] = [final[task][0] if task in final else "pending" for task in ids]
        results["task_durations_s"] = [final[task][1] - first_start[task] if task in final and task in first_start else None for task in ids]
        return results
