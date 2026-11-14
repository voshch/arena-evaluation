import json

import polars as pl
import pytest
from arena_evaluation.processing.metrics.performance.task_event_metrics import TaskEventMetricsCalculator
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.pipeline import _record_tasks
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def event(task, kind, robot, t):
    return {"task": task, "kind": kind, "robot": robot, "t": t, "detail": {}}


def bundle(tasks, robot="env_0_r1", span_s=100.0):
    data = pl.DataFrame({"time_ns": [0, int(span_s * 1e9)]})
    return AlignedEpisodeBundle(episode_id=0, data=data, start_pos=[0.0, 0.0], goal_pos=[1.0, 0.0], robot_name=robot, tasks=tasks)


def run(tasks, robot="env_0_r1", span_s=100.0):
    return TaskEventMetricsCalculator(RobotParams()).calculate(bundle(tasks, robot, span_s), {})


FULL_RUN = [
    event("", "allocated", "", 0.0),
    event("a", "assigned", "r1", 0.0),
    event("b", "assigned", "r2", 0.0),
    event("c", "assigned", "r1", 0.0),
    event("a", "started", "r1", 5.0),
    event("b", "started", "r2", 10.0),
    event("a", "met", "r1", 25.0),
    event("c", "started", "r1", 25.0),
    event("b", "met", "r2", 40.0),
    event("c", "met", "r1", 60.0),
]


def test_full_run_counts_and_makespan():
    res = run(FULL_RUN)
    assert res["tasks_total"] == 3
    assert res["tasks_met"] == 3
    assert res["tasks_failed"] == 0
    assert res["task_reassignments"] == 0
    assert res["task_success_rate"] == 1.0
    assert res["makespan_s"] == 55.0
    assert res["task_ids"] == ["a", "b", "c"]
    assert res["task_outcomes"] == ["met", "met", "met"]
    assert res["task_durations_s"] == [20.0, 30.0, 35.0]


def test_full_run_busy_and_idle_per_robot():
    first = run(FULL_RUN, robot="env_0_r1")
    second = run(FULL_RUN, robot="env_0_r2")
    assert first["busy_s"] == 55.0
    assert first["idle_s"] == 45.0
    assert second["busy_s"] == 30.0
    assert second["idle_s"] == 70.0


def test_span_without_aligned_data_ends_at_the_last_event():
    episode = AlignedEpisodeBundle(episode_id=0, data=pl.DataFrame({"time_ns": [0]}), start_pos=[0.0, 0.0], goal_pos=[1.0, 0.0], robot_name="env_0_r1", tasks=FULL_RUN)
    res = TaskEventMetricsCalculator(RobotParams()).calculate(episode, {})
    assert res["busy_s"] == 55.0
    assert res["idle_s"] == 5.0


def test_failure_with_reassignment():
    tasks = [
        event("a", "assigned", "r1", 0.0),
        event("a", "started", "r1", 5.0),
        event("a", "failed", "r1", 15.0),
        event("a", "reassigned", "r2", 15.0),
        event("a", "started", "r2", 16.0),
        event("a", "met", "r2", 40.0),
    ]
    res = run(tasks)
    assert res["tasks_total"] == 1
    assert res["tasks_met"] == 1
    assert res["tasks_failed"] == 0
    assert res["task_reassignments"] == 1
    assert res["task_durations_s"] == [35.0]
    assert res["makespan_s"] == 35.0
    assert res["busy_s"] == 10.0
    assert run(tasks, robot="env_0_r2")["busy_s"] == 24.0


def test_final_failure_after_reassignment_counts_failed():
    tasks = [
        event("a", "started", "r1", 5.0),
        event("a", "failed", "r1", 15.0),
        event("a", "reassigned", "r2", 15.0),
        event("a", "started", "r2", 16.0),
        event("a", "failed", "r2", 30.0),
    ]
    res = run(tasks)
    assert res["tasks_failed"] == 1
    assert res["tasks_met"] == 0
    assert res["task_success_rate"] == 0.0
    assert res["makespan_s"] is None
    assert res["task_outcomes"] == ["failed"]


def test_dropped_task_is_neither_met_nor_failed():
    tasks = [
        event("a", "started", "r1", 2.0),
        event("a", "met", "r1", 12.0),
        event("b", "assigned", "r1", 0.0),
        event("b", "started", "r1", 12.0),
        event("b", "dropped", "r1", 20.0),
    ]
    res = run(tasks)
    assert res["tasks_total"] == 2
    assert res["tasks_met"] == 1
    assert res["tasks_failed"] == 0
    assert res["task_success_rate"] == 0.5
    assert res["task_ids"] == ["b", "a"]
    assert res["task_outcomes"] == ["dropped", "met"]
    assert res["task_durations_s"] == [8.0, 10.0]
    assert res["busy_s"] == 18.0


def test_unfinished_task_is_pending_and_busy_until_episode_end():
    tasks = [event("a", "started", "r1", 40.0)]
    res = run(tasks, span_s=100.0)
    assert res["tasks_total"] == 1
    assert res["task_outcomes"] == ["pending"]
    assert res["task_durations_s"] == [None]
    assert res["busy_s"] == 60.0
    assert res["idle_s"] == 40.0


def test_record_without_tasks_yields_no_metrics():
    res = run([])
    assert set(res) == set(TaskEventMetricsCalculator.output_keys())
    assert all(value is None for value in res.values())


def test_bundle_defaults_to_no_tasks():
    plain = AlignedEpisodeBundle(episode_id=0, data=pl.DataFrame({"time_ns": [0]}), start_pos=[], goal_pos=[])
    assert plain.tasks == []


def test_registry_registers_task_events():
    registry = MetricRegistry(RobotParams())
    assert isinstance(registry.calculators["task_events"], TaskEventMetricsCalculator)


def test_record_tasks_takes_last_nonempty_row():
    first = json.dumps([event("a", "started", "r1", 1.0)])
    last = json.dumps([event("a", "started", "r1", 1.0), event("a", "met", "r1", 4.0)])
    record = pl.DataFrame({"time_ns": [1, 2, 3], "tasks": [first, last, ""]})
    assert [e["kind"] for e in _record_tasks(record)] == ["started", "met"]


@pytest.mark.parametrize("record", [pl.DataFrame({"time_ns": [1, 2]}), pl.DataFrame({"time_ns": [1], "tasks": [""]}), pl.DataFrame({"time_ns": [1], "tasks": ["not json"]})])
def test_record_tasks_tolerates_old_recordings(record):
    assert _record_tasks(record) == []
