from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from arena_evaluation.experiments import blend_quality as B


def test_link_steps_need_two_samples() -> None:
    t, step = B.link_steps(
        [(0.0, {"x": 0.0})],
        lambda _a, _b: ({k: np.zeros(3) for k in B.LINKS}, {}),
        None,
    )
    assert t.tolist() == [] and step.tolist() == []


NS = "/arena/env_0"
SEC = 1_000_000_000
JOINT = "y_head"


def _bag(uri: pathlib.Path, transition: bool = True) -> pathlib.Path:
    rosbag2_py = pytest.importorskip("rosbag2_py")
    pytest.importorskip("arena_people_msgs.msg")
    from arena_people_msgs.msg import (
        AnimationSlot,
        AnimationState,
        AnimationStates,
        Gesture,
        Pedestrian,
        Pedestrians,
    )
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
        state.slots = [
            AnimationSlot(slot="arm", animation="gesture:point", phase=phase)
        ]
        return AnimationStates(peds=[state])

    messages = [
        ("arena_peds", peds(0.0), 1.0),
        ("animation_states", anim("in"), 1.0),
        ("arena_peds", peds(0.3), 1.05),
    ]
    if transition:
        messages.append(("animation_states", anim("hold"), 1.05))
    messages.append(("arena_peds", peds(0.3), 1.1))
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for name, cls in (
        ("arena_peds", Pedestrians),
        ("animation_states", AnimationStates),
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=0,
                name=f"{NS}/{name}",
                type=f"arena_people_msgs/msg/{cls.__name__}",
                serialization_format="cdr",
            )
        )
    for name, msg, t in messages:
        writer.write(f"{NS}/{name}", serialize_message(msg), int(t * SEC))
    del writer
    return next(uri.glob("*.mcap"))


def _fk(angles: dict, _body: object) -> tuple[dict, dict]:
    return {k: np.array([angles[JOINT], 0.0, 0.0]) for k in B.LINKS}, {}


def test_analyze_episode_flags_a_snap_inside_a_transition_window(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("arena_runtime.humans.gestures.qa")
    pytest.importorskip("arena_runtime.humans.pointing.contract")
    result = B.analyze_episode(_bag(tmp_path / "bag"), _fk, None, source="recorded")
    assert (
        result["episode"],
        result["peds"],
        result["events"],
        result["events_over_gate"],
    ) == ("bag", 1, 1, 1)
    assert result["trans"].tolist() == pytest.approx([0.3, 0.0])
    assert result["steady"].tolist() == []
    steady = B.analyze_episode(
        _bag(tmp_path / "flat", transition=False), _fk, None, source="recorded"
    )
    assert (steady["events"], steady["events_over_gate"]) == (0, 0)
    assert steady["steady"].tolist() == pytest.approx([0.3, 0.0])
    assert steady["trans"].tolist() == []


def test_replay_runs_the_recorded_inputs_through_the_current_layer() -> None:
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("arena_runtime.humans.animation_manager")
    inputs = {
        1: [(1.0, 0, 0.0, (1.0, 2.0, 0.0), ()), (1.05, 0, 0.0, (1.0, 2.0, 0.0), ())]
    }
    out = B.replay(inputs)
    assert [t for t, _ in out[1]] == [1.0, 1.05]
    assert all(set(B._joint_names()) <= set(angles) for _, angles in out[1])


def test_main_writes_the_summary_and_skips_unreadable_episodes(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("arena_runtime.humans.pointing.skeleton")
    pytest.importorskip("arena_runtime.humans.gestures")
    bench = tmp_path / "bench"
    (bench / "episodes").mkdir(parents=True)
    _bag(bench / "episodes" / "episode_000")
    (bench / "episodes" / "episode_001").mkdir()
    (bench / "episodes" / "episode_001" / "broken.mcap").write_bytes(b"")
    assert (
        B.main(
            [
                "--benchmark-dir",
                str(bench),
                "--source",
                "recorded",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "out" / "blend_quality_recorded.json").read_text())
    assert (summary["source"], summary["episodes"], summary["transitions"]) == (
        "recorded",
        1,
        1,
    )
    assert summary["transition_ticks"]["ticks"] == 2
    assert "skip episode_001" in capsys.readouterr().err


def test_ped_angles_read_the_body_pose_and_fall_back_to_the_recorded_joint_state():
    from types import SimpleNamespace

    from arena_people_msgs.msg import Pedestrian
    from arena_simulation_setup.shared.body import semantic_to_body
    from geometry_msgs.msg import Quaternion

    from arena_evaluation.experiments.blend_quality import ped_angles

    ped = Pedestrian(name="walker", id=1)
    ped.joint_state.name, ped.joint_state.position = ["l_elbow"], [0.4]
    assert ped_angles(ped) == {"l_elbow": 0.4}
    rotations = semantic_to_body({"l_knee": -0.7})
    ped.body.joints = list(rotations)
    ped.body.rotations = [
        Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        for q in rotations.values()
    ]
    assert ped_angles(ped)["l_knee"] == pytest.approx(-0.7)
    recorded = SimpleNamespace(joint_state=ped.joint_state)
    assert ped_angles(recorded) == {"l_elbow": 0.4}
