import pathlib

import polars as pl
import pytest
from mcap_ros2.writer import Writer

from arena_evaluation.processing.mcap_reader import MCAPReader

_PARAMETER_DEFS = """
================================================================================
MSG: rcl_interfaces/Parameter
string name
rcl_interfaces/ParameterValue value
================================================================================
MSG: rcl_interfaces/ParameterValue
uint8 type
bool bool_value
int64 integer_value
float64 double_value
string string_value
byte[] byte_array_value
bool[] bool_array_value
int64[] integer_array_value
float64[] double_array_value
string[] string_array_value
"""

_RECORD_FIELDS = """uint32 episode_id
uint8 outcome_state
string outcome_info
string goal_uuid
float32 goal_dist_start
float32 goal_dist_min
{extra}rcl_interfaces/Parameter[] robots_params
"""


def _write_records(path: pathlib.Path, with_tolerance: bool, tolerances: list[float]) -> None:
    fields = _RECORD_FIELDS.format(extra="float32 goal_tolerance\n" if with_tolerance else "")
    with path.open("wb") as f:
        writer = Writer(f)
        schema = writer.register_msgdef("task_generator_msgs/msg/EpisodeRecord", fields + _PARAMETER_DEFS)
        for i, tolerance in enumerate(tolerances):
            msg = {
                "episode_id": i,
                "outcome_state": 2,
                "outcome_info": "",
                "goal_uuid": "",
                "goal_dist_start": 5.0,
                "goal_dist_min": 1.0,
                "robots_params": [],
            }
            if with_tolerance:
                msg["goal_tolerance"] = tolerance
            writer.write_message("/env_0/task_generator_node/state/episode", schema, msg, log_time=i + 1, publish_time=i + 1)
        writer.finish()


def _records(tmp_path: pathlib.Path, with_tolerance: bool, tolerances: list[float]) -> pl.DataFrame:
    bag = tmp_path / "run.mcap"
    _write_records(bag, with_tolerance, tolerances)
    out = tmp_path / "topics"
    out.mkdir()
    MCAPReader(bag).read(out)
    return pl.read_parquet(out / "env_0" / "episode_record.parquet")


def test_goal_tolerance_read_from_episode_record(tmp_path: pathlib.Path) -> None:
    records = _records(tmp_path, with_tolerance=True, tolerances=[3.0, 0.5])
    assert records["goal_tolerance"].to_list() == pytest.approx([3.0, 0.5])


def test_recording_from_before_goal_tolerance_reads_none(tmp_path: pathlib.Path) -> None:
    records = _records(tmp_path, with_tolerance=False, tolerances=[0.0, 0.0])
    assert records["goal_tolerance"].to_list() == [None, None]
    assert records["episode_id"].to_list() == [0, 1]
