import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.judge")

import shapely

from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry
from arena_evaluation.processing.metrics.performance.task_metrics import TaskReplayCalculator, _active_phase_spans, _robot_events
from arena_evaluation.processing.metrics.ecological.compliance_metrics import _reconstruct_events
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

S = 1_000_000_000


def _square(name, x0, y0):
    return _ZoneGeometry(name=name, polygon=shapely.Polygon([(x0, y0), (x0, y0 + 4), (x0 + 4, y0 + 4), (x0 + 4, y0)]), max_speed=None, quiet=False, restricted=False)


def _calc(world="synthetic_task"):
    calc = TaskReplayCalculator(RobotParams(0.2, 0.0, 10.0))
    calc.world = world
    calc._world_cache[world] = [_square("kitchen", 0, 0), _square("hallway", 10, 0)]
    return calc


def _snapshot(rows):
    """Long-format snapshot rows (time_ns, entity, kind, field, value_str)."""
    return pl.DataFrame(
        {
            "time_ns": [r[0] for r in rows],
            "entity": [r[1] for r in rows],
            "kind": [r[2] for r in rows],
            "field": [r[3] for r in rows],
            "field_kind": ["discrete"] * len(rows),
            "value_str": [r[4] for r in rows],
            "value_num": [None] * len(rows),
            "value_bool": [None] * len(rows),
        }
    )


def _robot_rows(t, phase, met="", failed="", dropped="", violated=""):
    entity = "env_0/jackal_0"
    return [
        (t, entity, "robot", "phase", phase),
        (t, entity, "robot", "met", met),
        (t, entity, "robot", "failed", failed),
        (t, entity, "robot", "dropped", dropped),
        (t, entity, "robot", "violated", violated),
    ]


def _poses(points):
    """(time_s, x, y) -> task_pose frame."""
    return pl.DataFrame(
        {
            "time_ns": [int(t * S) for t, _, _ in points],
            "pos_x": [x for _, x, _ in points],
            "pos_y": [y for _, _, y in points],
            "yaw": [0.0] * len(points),
        }
    )


def _episode(phases, poses, snapshot, peds=None, fleet=None):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"time_ns": [0]}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[],
        robot_name="env_0_jackal_0",
        semantic_snapshot=snapshot,
        phases=phases,
        fleet=fleet or {},
        peds=peds,
        topics={"task_pose": poses},
    )


def test_robot_events_and_spans():
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(2 * S, "1", met="0") + _robot_rows(4 * S, "", met="0,1"))
    series = _robot_events(_reconstruct_events(snapshot), "jackal_0")
    assert series["met"][1] == ["", "0", "0,1"]
    assert _active_phase_spans(series, 5 * S) == {0: (0, 2 * S), 1: (2 * S, 4 * S)}


def test_replay_agrees_with_recorded_gotos():
    phases = {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}, {"goto": "hallway", "pose": [12.0, 2.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": []}
    poses = _poses([(0.0, 5.0, 5.0), (1.0, 1.0, 1.0), (2.0, 6.0, 1.0), (3.0, 12.0, 1.0)])
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(int(1.5 * S), "1", met="0") + _robot_rows(int(3.5 * S), "", met="0,1"))
    results = _calc().calculate(_episode(phases, poses, snapshot), {})
    assert results["phases_total"] == 2
    assert results["phases_met"] == 2
    assert results["goal_condition_rate"] == 1.0
    assert results["phase_outcomes"] == ["met", "met"]
    assert results["unsafe"] == 0
    assert results["judge_agrees"] is True
    assert results["judge_disagreements"] == []


def test_replay_disagrees_when_recorded_met_is_not_reached():
    phases = {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": []}
    poses = _poses([(0.0, 5.0, 5.0), (1.0, 4.0, 4.0)])
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(int(1.5 * S), "", met="0"))
    results = _calc().calculate(_episode(phases, poses, snapshot), {})
    assert results["judge_agrees"] is False
    assert results["judge_disagreements"] == ["phase 0 recorded as met but not met in replay"]


def test_replay_takes_gesture_outcome_from_the_record():
    phases = {"phases": [{"gesture": "wave"}, {"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": []}
    poses = _poses([(0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (2.0, 1.0, 1.0)])
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(int(1.5 * S), "1", failed="0") + _robot_rows(int(2.5 * S), "", met="1", failed="0"))
    results = _calc().calculate(_episode(phases, poses, snapshot), {})
    assert results["phase_outcomes"] == ["failed", "met"]
    assert results["phases_failed"] == 1
    assert results["judge_agrees"] is True


def test_replay_skips_dropped_phases_and_counts_them_as_not_met():
    phases = {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}, {"goto": [9.0, 9.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": []}
    poses = _poses([(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(int(0.5 * S), "1", met="0") + _robot_rows(int(0.7 * S), "", met="0", dropped="1"))
    results = _calc().calculate(_episode(phases, poses, snapshot), {})
    assert results["phase_outcomes"] == ["met", "dropped"]
    assert results["goal_condition_rate"] == 0.5
    assert results["judge_agrees"] is True


@pytest.mark.parametrize(
    "phase",
    [
        {"goto": "kitchen", "pose": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0, "until": "not alice in kitchen"},
        {"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0, "until": "not alice in kitchen"},
        {"tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0, "until": "not alice in kitchen"},
    ],
)
def test_replay_hold_phase_with_until_on_ped_leaving(phase):
    phases = {"phases": [phase], "conditions": []}
    poses = _poses([(0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (2.0, 1.0, 1.0)])
    peds = pl.DataFrame({"time_ns": [0, 2 * S], "peds_names": [["env_0/alice"], ["env_0/alice"]], "peds_positions": [[2.0, 2.0, 0.0], [12.0, 2.0, 0.0]]})
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(int(2.2 * S), "", met="0"))
    results = _calc().calculate(_episode(phases, poses, snapshot, peds=peds), {})
    assert results["judge_agrees"] is True
    early = _snapshot(_robot_rows(0, "0") + _robot_rows(int(1.2 * S), "", met="0"))
    assert _calc().calculate(_episode(phases, poses, early, peds=peds), {})["judge_agrees"] is False


def test_replay_scoped_condition_violation_matches_record():
    phases = {
        "phases": [{"goto": [12.0, 2.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0, "conditions": [{"op": "never", "p": "robot in hallway"}]}],
        "conditions": [{"id": "r0:0", "from": 0, "to": None, "op": "eventually", "p": "robot in kitchen"}],
    }
    poses = _poses([(0.0, 5.0, 5.0), (1.0, 12.0, 2.0)])
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(int(1.5 * S), "", met="0", violated="0:0,r0:0"))
    results = _calc().calculate(_episode(phases, poses, snapshot), {})
    assert results["unsafe"] == 2
    assert results["judge_agrees"] is True
    quiet = _snapshot(_robot_rows(0, "0") + _robot_rows(int(1.5 * S), "", met="0"))
    results = _calc().calculate(_episode(phases, poses, quiet), {})
    assert results["judge_agrees"] is False
    assert "violated conditions differ" in results["judge_disagreements"][0]


def test_replay_until_on_other_robot_phase_uses_fleet_and_entities():
    phases = {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0, "until": "robot_1.phase == 1"}], "conditions": []}
    poses = _poses([(0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (2.0, 1.0, 1.0)])
    other = [(0, "env_0/robot_1", "robot", "phase", "0"), (int(1.5 * S), "env_0/robot_1", "robot", "phase", "1")]
    snapshot = _snapshot(_robot_rows(0, "0") + other + _robot_rows(int(2.1 * S), "", met="0"))
    fleet = {"robot_1": _poses([(0.0, 20.0, 20.0), (2.0, 20.0, 20.0)])}
    results = _calc().calculate(_episode(phases, poses, snapshot, fleet=fleet), {})
    assert results["judge_agrees"] is True


def test_replay_empty_without_phases_or_poses():
    calc = _calc()
    assert calc.calculate(_episode(None, _poses([(0.0, 0.0, 0.0)]), _snapshot([])), {})["phases_total"] is None
    phases = {"phases": [{"goto": [1.0, 1.0, 0.0]}], "conditions": []}
    results = calc.calculate(_episode(phases, pl.DataFrame({"time_ns": [], "pos_x": [], "pos_y": [], "yaw": []}), _snapshot(_robot_rows(0, "0"))), {})
    assert results["phases_total"] == 1 and results["judge_agrees"] is None


def test_replay_judges_a_signal_phase_at_the_recorded_end():
    phases = {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0, "signal": "arrived"}], "conditions": []}
    poses = _poses([(0.0, 5.0, 5.0), (1.0, 1.0, 1.0), (2.0, 1.0, 1.0)])
    snapshot = _snapshot(_robot_rows(0, "0") + _robot_rows(int(1.5 * S), "", met="0"))
    assert _calc().calculate(_episode(phases, poses, snapshot), {})["judge_agrees"] is True
    misfired = _snapshot(_robot_rows(0, "0") + _robot_rows(int(0.5 * S), "", failed="0"))
    results = _calc().calculate(_episode(phases, poses, misfired), {})
    assert results["phase_outcomes"] == ["failed"]
    assert results["judge_agrees"] is True
    wrong = _snapshot(_robot_rows(0, "0") + _robot_rows(int(0.5 * S), "", met="0"))
    results = _calc().calculate(_episode(phases, poses, wrong), {})
    assert results["judge_disagreements"] == ["phase 0 recorded as met but signaled arrived 5.66 m from goal in replay"]
