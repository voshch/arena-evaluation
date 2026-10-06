import json

from arena_evaluation.processing.mcap_reader import MCAPReader


def test_map_poses_of_every_robot_move_by_the_env_offset() -> None:
    record = {
        "jackal": {"phases": [{"goto": [1.0, 2.0, 0.0]}, {"gesture": "wave"}], "conditions": [], "map_poses": [[56.0, 7.0, 0.5], None]},
        "burger": {"phases": [{"goto": "kitchen"}], "conditions": [], "map_poses": [[59.5, 4.0, 1.57]]},
    }
    shifted = json.loads(MCAPReader._shift_map_poses(json.dumps(record), 50.0, -1.0))
    assert shifted["jackal"]["map_poses"] == [[6.0, 8.0, 0.5], None]
    assert shifted["burger"]["map_poses"] == [[9.5, 5.0, 1.57]]
    assert shifted["jackal"]["phases"] == record["jackal"]["phases"]


def test_empty_phases_payload_passes_through() -> None:
    assert MCAPReader._shift_map_poses("", 50.0, -1.0) == ""
