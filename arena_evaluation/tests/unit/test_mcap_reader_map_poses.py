import pathlib

import polars as pl

from arena_evaluation.processing.mcap_reader import align_to_environment, load_bundles


def _tf_row(t_ns: int, parent: str, child: str, tx: float, ty: float) -> dict:
    return {"t_ns": t_ns, "parent_frame": parent, "child_frame": child, "tx": tx, "ty": ty, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


def _phases_after_offset(tmp_path: pathlib.Path, phases_yaml: str) -> str:
    """The episode phases of env_0 after the map offset (50, -1) is applied."""
    topics = tmp_path / "topics"
    topics.mkdir()
    pl.DataFrame([_tf_row(0, "map", "env_0/map", 50.0, -1.0)]).write_parquet(topics / "tf_static.parquet")
    pl.DataFrame([{"t_ns": 0, "env_id": "env_0", "episode_id": 1, "phases_yaml": phases_yaml}]).write_parquet(topics / "episode_lifecycle.parquet")
    pl.DataFrame([{"t_ns": 0, "env_id": "env_0", "robot_id": "env_0_jackal", "odom_x": 0.0, "odom_y": 0.0, "odom_yaw": 0.0}]).write_parquet(topics / "robot_odom.parquet")
    align_to_environment(topics, {})
    return load_bundles(topics)["env_0_jackal"].episode_lifecycle.collect()["phases_yaml"][0]


