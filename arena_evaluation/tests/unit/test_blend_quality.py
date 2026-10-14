from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from arena_evaluation.experiments import blend_quality as B


def test_transition_times_fire_on_any_change() -> None:
    a = frozenset({("body", "clip", "hold")})
    b = frozenset({("body", "clip", "release")})
    states = [(0.0, frozenset(), "walk"), (0.1, frozenset(), "walk"), (0.2, a, "idle"), (0.3, a, "idle"), (0.4, b, "idle"), (0.5, b, "walk")]
    assert B.transition_times(states) == [0.2, 0.4, 0.5]


def test_link_steps_are_rescaled_to_a_50ms_tick() -> None:
    def fk(angles: dict, _body: object):
        x = angles["x"]
        return {k: np.array([x, 0.0, 0.0]) for k in B.LINKS}, {}

    samples = [(0.0, {"x": 0.0}), (0.025, {"x": 0.1}), (0.075, {"x": 0.2})]
    t, step = B.link_steps(samples, fk, None)
    assert t.tolist() == pytest.approx([0.025, 0.075])
    assert step.tolist() == pytest.approx([0.2, 0.1])


def test_summary_counts_gate_violations() -> None:
    res = [{"events": 4, "events_over_gate": 1, "trans": np.array([0.1, 0.3]), "steady": np.array([0.01, 0.02])}]
    s = B.summarize(res)
    assert s["transitions"] == 4 and s["transition_pass_rate"] == pytest.approx(0.75)
    assert s["transition_ticks"]["over_gate"] == 1 and s["steady_ticks"]["over_gate"] == 0


def test_link_steps_need_two_samples() -> None:
    t, step = B.link_steps([(0.0, {"x": 0.0})], lambda _a, _b: ({k: np.zeros(3) for k in B.LINKS}, {}), None)
    assert t.tolist() == [] and step.tolist() == []


def test_summary_of_no_episodes_has_no_pass_rate() -> None:
    assert B.summarize([]) == {"episodes": 0, "transitions": 0, "transitions_over_gate": 0, "transition_pass_rate": None, "transition_ticks": {"ticks": 0}, "steady_ticks": {"ticks": 0}}


NS = "/arena/env_0"
SEC = 1_000_000_000
JOINT = "y_head"


def _bag(uri: pathlib.Path, transition: bool = True) -> pathlib.Path:
    rosbag2_py = pytest.importorskip("rosbag2_py")
    pytest.importorskip("arena_people_msgs.msg")
    from arena_people_msgs.msg import AnimationSlot, AnimationState, AnimationStates, Gesture, Pedestrian, Pedestrians
    from rclpy.serialization import serialize_message

    def peds(angle: float) -> Pedestrians:
        walker = Pedestrian(name="walker", id=1, animation_state=1)
        walker.pose.position.x, walker.pose.position.y = 1.0, 2.0
        walker.pose.orientation.z = walker.pose.orientation.w = math.sqrt(0.5)
        walker.twist.linear.y = 0.8
        point = Gesture(slot="arm", hand="r")
        point.at.x, point.at.y = 3.0, 4.0
        walker.gestures = [point]
        walker.joint_state.name = [JOINT]
        walker.joint_state.position = [angle]
        return Pedestrians(pedestrians=[walker, Pedestrian(name="idle", id=2)])

    def anim(phase: str) -> AnimationStates:
        state = AnimationState(id=1, name="walker", base="walk")
        state.slots = [AnimationSlot(slot="arm", animation="gesture:point", phase=phase)]
        return AnimationStates(peds=[state])

    messages = [("arena_peds", peds(0.0), 1.0), ("animation_states", anim("in"), 1.0), ("arena_peds", peds(0.3), 1.05)]
    if transition:
        messages.append(("animation_states", anim("hold"), 1.05))
    messages.append(("arena_peds", peds(0.3), 1.1))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name, cls in (("arena_peds", Pedestrians), ("animation_states", AnimationStates)):
        writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=f"{NS}/{name}", type=f"arena_people_msgs/msg/{cls.__name__}", serialization_format="cdr"))
    for name, msg, t in messages:
        writer.write(f"{NS}/{name}", serialize_message(msg), int(t * SEC))
    del writer
    return next(uri.glob("*.mcap"))


def test_episode_tracks_split_joints_animation_and_layer_inputs(tmp_path: pathlib.Path) -> None:
    joints, anim, inputs = B.episode_tracks(_bag(tmp_path / "bag"))
    assert [t for t, _ in joints[1]] == pytest.approx([1.0, 1.05, 1.1])
    assert [a[JOINT] for _, a in joints[1]] == pytest.approx([0.0, 0.3, 0.3])
    assert 2 not in joints
    t, state, speed, pose, channels = inputs[1][0]
    assert (t, state) == (1.0, 1)
    assert speed == pytest.approx(0.8)
    assert pose == pytest.approx((1.0, 2.0, math.pi / 2))
    assert channels == (("arm", (3.0, 4.0, 0.0), "", "r", False),)
    assert inputs[2][0][1:3] == (0, 0.0) and inputs[2][0][4] == ()
    assert [(round(t, 3), keys, base) for t, keys, base in anim[1]] == [(1.0, frozenset({("arm", "gesture:point", "in")}), "walk"), (1.05, frozenset({("arm", "gesture:point", "hold")}), "walk")]
    assert 2 not in anim


def _fk(angles: dict, _body: object) -> tuple[dict, dict]:
    return {k: np.array([angles[JOINT], 0.0, 0.0]) for k in B.LINKS}, {}


def test_analyze_episode_flags_a_snap_inside_a_transition_window(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("task_generator.simulators.human.gestures.qa")
    pytest.importorskip("task_generator.simulators.human.pointing.contract")
    result = B.analyze_episode(_bag(tmp_path / "bag"), _fk, None, source="recorded")
    assert (result["episode"], result["peds"], result["events"], result["events_over_gate"]) == ("bag", 1, 1, 1)
    assert result["trans"].tolist() == pytest.approx([0.3, 0.0])
    assert result["steady"].tolist() == []
    steady = B.analyze_episode(_bag(tmp_path / "flat", transition=False), _fk, None, source="recorded")
    assert (steady["events"], steady["events_over_gate"]) == (0, 0)
    assert steady["steady"].tolist() == pytest.approx([0.3, 0.0])
    assert steady["trans"].tolist() == []


def test_replay_runs_the_recorded_inputs_through_the_current_layer() -> None:
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("task_generator.simulators.human.animation_manager")
    inputs = {1: [(1.0, 0, 0.0, (1.0, 2.0, 0.0), ()), (1.05, 0, 0.0, (1.0, 2.0, 0.0), ())]}
    out = B.replay(inputs)
    assert [t for t, _ in out[1]] == [1.0, 1.05]
    assert all(set(B._joint_names()) <= set(angles) for _, angles in out[1])


def test_main_writes_the_summary_and_skips_unreadable_episodes(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("task_generator.simulators.human.pointing.skeleton")
    pytest.importorskip("task_generator.simulators.human.gestures")
    bench = tmp_path / "bench"
    (bench / "episodes").mkdir(parents=True)
    _bag(bench / "episodes" / "episode_000")
    (bench / "episodes" / "episode_001").mkdir()
    (bench / "episodes" / "episode_001" / "broken.mcap").write_bytes(b"")
    assert B.main(["--benchmark-dir", str(bench), "--source", "recorded", "--out", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "blend_quality_recorded.json").read_text())
    assert (summary["source"], summary["episodes"], summary["transitions"]) == ("recorded", 1, 1)
    assert summary["transition_ticks"]["ticks"] == 2
    assert "skip episode_001" in capsys.readouterr().err
