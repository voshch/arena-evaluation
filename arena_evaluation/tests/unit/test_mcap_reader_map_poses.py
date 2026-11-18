import json
import pathlib

import polars as pl

from arena_evaluation.processing.mcap_reader import load_bundles


def _tf_row(t_ns: int, parent: str, child: str, tx: float, ty: float) -> dict:
    return {"t_ns": t_ns, "parent_frame": parent, "child_frame": child, "tx": tx, "ty": ty, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


def _phases_after_offset(tmp_path: pathlib.Path, phases_yaml: str) -> str:
    """The episode phases of env_0 after the map offset (50, -1) is applied."""
    topics = tmp_path / "topics"
    topics.mkdir()
    pl.DataFrame([_tf_row(0, "map", "env_0/map", 50.0, -1.0)]).write_parquet(topics / "tf_static.parquet")
    pl.DataFrame([{"t_ns": 0, "env_id": "env_0", "episode_id": 1, "phases_yaml": phases_yaml}]).write_parquet(topics / "episode_lifecycle.parquet")
    pl.DataFrame([{"t_ns": 0, "env_id": "env_0", "robot_id": "env_0_jackal", "odom_x": 0.0, "odom_y": 0.0, "odom_yaw": 0.0}]).write_parquet(topics / "robot_odom.parquet")
    return load_bundles(topics, {})["env_0_jackal"].episode_lifecycle.collect()["phases_yaml"][0]


def test_map_poses_of_every_robot_move_by_the_env_offset(tmp_path: pathlib.Path) -> None:
    record = {
        "jackal": {"phases": [{"goto": [1.0, 2.0, 0.0]}, {"gesture": "wave"}], "conditions": [], "map_poses": [[56.0, 7.0, 0.5], None]},
        "burger": {"phases": [{"goto": "kitchen"}], "conditions": [], "map_poses": [[59.5, 4.0, 1.57]]},
    }

    shifted = json.loads(_phases_after_offset(tmp_path, json.dumps(record)))

    assert shifted["jackal"]["map_poses"] == [[6.0, 8.0, 0.5], None]
    assert shifted["burger"]["map_poses"] == [[9.5, 5.0, 1.57]]
    assert shifted["jackal"]["phases"] == record["jackal"]["phases"]


def test_empty_phases_payload_passes_through(tmp_path: pathlib.Path) -> None:
    assert _phases_after_offset(tmp_path, "") == ""
