"""Round trip: interaction / animation state recorded through rosbag2 comes back as its own tables."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")
pytest.importorskip("arena_people_msgs.msg")

from arena_humansim_msgs.msg import AgentState, AgentStates, InteractionEvent, InteractionFrame
from arena_people_msgs.msg import AnimationSlot, AnimationState, AnimationStates, Gesture, Pedestrian, Pedestrians
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import MCAPReader

NS = "/arena/env_0"
SEC = 1_000_000_000


def _write(path: pathlib.Path, messages: list[tuple[str, object, int]]) -> pathlib.Path:
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    created: set[str] = set()
    for topic, msg, t in messages:
        if topic not in created:
            module = type(msg).__module__.split(".")[0]
            writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=topic, type=f"{module}/msg/{type(msg).__name__}", serialization_format="cdr"))
            created.add(topic)
        writer.write(topic, serialize_message(msg), t)
    del writer
    return next(path.glob("*.mcap"))


def _peds(x_a: float) -> Pedestrians:
    a = Pedestrian(name="hugger_1", id=1, animation_state=0, interaction_id=4, interaction_type=10)
    a.pose.position.x = x_a
    a.gestures = [Gesture(slot="body", clip="hug", render_pose_override=True)]
    b = Pedestrian(name="hugger_2", id=2, animation_state=1)
    b.pose.position.x = 9.0
    return Pedestrians(pedestrians=[a, b])


def _agents() -> AgentStates:
    a = AgentState(agent_id=1, kind=0, interaction_id=4, interaction_type=10)
    a.pose.x = 100.0
    robot = AgentState(agent_id=99, kind=1)
    return AgentStates(agents=[a, robot])


HUG = (4, 10, 1, "", [1, 2], [], True, True, 3.0)
QUEUE_USE = (7, 6, 0, "counter", [3], [5, 8], False, False, -1.0)


def _frame(stamp_s: int, rows: list[tuple], events: Sequence[int] = ()) -> InteractionFrame:
    msg = InteractionFrame()
    msg.header.stamp.sec = stamp_s
    offsets = [0]
    for iid, itype, outcome, object_id, participants, queue, arrived, holding, duration in rows:
        msg.interaction_id.append(iid)
        msg.interaction_type.append(itype)
        msg.outcome.append(outcome)
        msg.object_id.append(object_id)
        msg.arrived.append(arrived)
        msg.holding.append(holding)
        msg.duration.append(duration)
        msg.member_id.extend(participants + queue)
        msg.member_role.extend([0] * len(participants) + [1] * len(queue))
        offsets.append(len(msg.member_id))
    msg.member_offset = offsets
    msg.events = [InteractionEvent(interaction_id=4, interaction_type=10, event=e, participants=[1, 2]) for e in events]
    return msg


def _interactions(stamp_s: int, events: list[int]) -> InteractionFrame:
    return _frame(stamp_s, [HUG], events)


def _animation() -> AnimationStates:
    hugging = AnimationState(id=1, name="hugger_1", base="idle", base_phase=0.5)
    hugging.slots = [AnimationSlot(slot="body", kind="clip", channel="body", phase="hold", clip="hug", animation="gesture:clip:1:body", playhead=1.25, duration=5.0, weight=1.0, loop=True)]
    walking = AnimationState(id=2, name="hugger_2", base="walk")
    return AnimationStates(peds=[hugging, walking])


@pytest.fixture()
def bundle(tmp_path: pathlib.Path):
    mcap = _write(
        tmp_path / "bag",
        [
            (f"{NS}/arena_peds", _peds(8.0), 1 * SEC),
            (f"{NS}/task_generator_node/agent_states", _agents(), 1 * SEC),
            (f"{NS}/task_generator_node/interactions", _interactions(1, [0, 1]), 1 * SEC),
            (f"{NS}/animation_states", _animation(), 1 * SEC),
            (f"{NS}/arena_peds", _peds(8.5), 2 * SEC),
            (f"{NS}/task_generator_node/interactions", _interactions(2, [2]), 2 * SEC),
        ],
    )
    out = tmp_path / "topics"
    MCAPReader(mcap).read(out)
    return out / "env_0"


def _table(env_dir: pathlib.Path, name: str):
    import polars as pl

    return pl.read_parquet(env_dir / f"{name}.parquet").sort("time_ns")


def test_arena_peds_and_agent_states_land_in_separate_tables(bundle) -> None:
    peds = _table(bundle, "peds")
    assert peds.height == 2
    assert peds["peds_positions"][0][0] == 8.0
    assert peds["peds_ids"][0].to_list() == [1, 2]
    assert peds["peds_interaction_ids"][0].to_list() == [4, -1]
    assert peds["peds_interaction_types"][0].to_list() == [10, 0]
    assert peds["peds_animation_states"][0].to_list() == [0, 1]
    physics = _table(bundle, "peds_physics")
    assert physics.height == 1
    assert physics["peds_ids"][0].to_list() == [1]
    assert physics["peds_positions"][0][0] == 100.0


def test_gestures_come_from_the_rendered_side_only(bundle) -> None:
    gestures = _table(bundle, "ped_gestures")
    assert gestures.height == 2
    assert set(gestures["ped_id"]) == {1}
    assert gestures["clip"].to_list() == ["hug", "hug"]
    assert gestures["render_pose_override"].to_list() == [True, True]


def test_interaction_events_and_snapshots(bundle) -> None:
    events = _table(bundle, "interaction_events")
    assert events["event_name"].to_list() == ["ACTIVATED", "HOLD_ONSET", "RELEASED"]
    assert events["stamp_ns"].to_list() == [1 * SEC, 1 * SEC, 2 * SEC]
    assert events["participants"][0].to_list() == [1, 2]
    snapshots = _table(bundle, "interactions")
    assert snapshots.height == 2
    assert snapshots.columns == ["time_ns", "stamp_ns", "interaction_id", "interaction_type", "outcome", "object_id", "participants", "queue", "arrived", "holding", "duration"]
    assert snapshots["stamp_ns"].to_list() == [1 * SEC, 2 * SEC]
    assert snapshots["interaction_type"].to_list() == [10, 10]
    assert snapshots["participants"][0].to_list() == [1, 2]
    assert snapshots["queue"][0].to_list() == []
    assert snapshots["arrived"].to_list() == [True, True]
    assert snapshots["holding"].to_list() == [True, True]
    assert snapshots["duration"][0] == 3.0


def test_members_split_into_participants_and_queue_with_the_anchor_object(tmp_path: pathlib.Path) -> None:
    mcap = _write(tmp_path / "bag", [(f"{NS}/task_generator_node/interactions", _frame(1, [HUG, QUEUE_USE]), 1 * SEC)])
    MCAPReader(mcap).read(tmp_path / "topics")
    rows = _table(tmp_path / "topics" / "env_0", "interactions").sort("interaction_id")
    assert rows["interaction_id"].to_list() == [4, 7]
    assert rows["object_id"].to_list() == ["", "counter"]
    assert rows["outcome"].to_list() == [1, 0]
    assert rows["participants"].to_list() == [[1, 2], [3]]
    assert rows["queue"].to_list() == [[], [5, 8]]
    assert rows["arrived"].to_list() == [True, False]
    assert rows["duration"].to_list() == [3.0, -1.0]


def test_empty_frames_add_no_interaction_rows(tmp_path: pathlib.Path) -> None:
    topic = f"{NS}/task_generator_node/interactions"
    mcap = _write(tmp_path / "bag", [(topic, _frame(1, []), 1 * SEC), (topic, _frame(2, [HUG], [0]), 2 * SEC), (topic, _frame(3, []), 3 * SEC)])
    MCAPReader(mcap).read(tmp_path / "topics")
    rows = _table(tmp_path / "topics" / "env_0", "interactions")
    assert rows["time_ns"].to_list() == [2 * SEC]
    assert _table(tmp_path / "topics" / "env_0", "interaction_events")["event_name"].to_list() == ["ACTIVATED"]


def test_animation_states_one_row_per_slot_and_base_only_peds(bundle) -> None:
    rows = _table(bundle, "animation_states").sort("ped_id")
    assert rows["ped_id"].to_list() == [1, 2]
    hug = rows.row(0, named=True)
    assert (hug["slot"], hug["phase"], hug["clip"], hug["playhead"], hug["loop"]) == ("body", "hold", "hug", 1.25, True)
    walker = rows.row(1, named=True)
    assert (walker["base"], walker["slot"], walker["playhead"]) == ("walk", "", None)


def test_gesture_capable_peds_without_a_gesture_keep_an_empty_table(tmp_path: pathlib.Path) -> None:
    idle = Pedestrians(pedestrians=[Pedestrian(name="walker", id=3)])
    mcap = _write(tmp_path / "bag", [(f"{NS}/arena_peds", idle, 1 * SEC), (f"{NS}/arena_peds", idle, 2 * SEC)])
    (tmp_path / "topics" / "env_0_jackal").mkdir(parents=True)
    bundles = MCAPReader(mcap).read(tmp_path / "topics")
    gestures = _table(tmp_path / "topics" / "env_0", "ped_gestures")
    assert gestures.height == 0
    assert gestures.columns == ["time_ns", "ped_id", "slot", "clip", "hand", "at_x", "at_y", "at_z", "render_pose_override"]
    assert "ped_gestures" in bundles["env_0_jackal"].available()


def test_agent_states_without_gesture_support_write_no_gesture_table(tmp_path: pathlib.Path) -> None:
    mcap = _write(tmp_path / "bag", [(f"{NS}/task_generator_node/agent_states", _agents(), 1 * SEC)])
    MCAPReader(mcap).read(tmp_path / "topics")
    assert not (tmp_path / "topics" / "env_0" / "ped_gestures.parquet").exists()


def test_load_bundles_exposes_the_state_tables(tmp_path: pathlib.Path) -> None:
    mcap = _write(
        tmp_path / "bag",
        [
            (f"{NS}/task_generator_node/interactions", _interactions(1, [1]), 1 * SEC),
            (f"{NS}/animation_states", _animation(), 1 * SEC),
            (f"{NS}/arena_peds", _peds(8.0), 1 * SEC),
        ],
    )
    (tmp_path / "topics" / "env_0_jackal").mkdir(parents=True)
    bundles = MCAPReader(mcap).read(tmp_path / "topics")
    assert bundles, "a robot-less recording still yields the env bundle"
    rb = next(iter(bundles.values()))
    assert {"interactions", "interaction_events", "animation_states", "ped_gestures"} <= rb.available()
