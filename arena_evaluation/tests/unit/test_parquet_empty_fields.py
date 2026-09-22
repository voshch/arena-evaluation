"""An episode with no doors must still write to Parquet."""

import pathlib

from arena_evaluation.processing.parquet_store import ParquetStore, _writable_rows


def test_empty_dict_becomes_none():
    assert _writable_rows([{"f": {}}]) == [{"f": None}]


def test_nested_empty_dict_becomes_none():
    rows = _writable_rows([{"wcaf": {"robot_x": 1.0, "door_states": {}}}])
    assert rows == [{"wcaf": {"robot_x": 1.0, "door_states": None}}]


def test_empty_dicts_inside_lists():
    assert _writable_rows([{"f": [{}, {"x": 1}]}]) == [{"f": [None, {"x": 1}]}]


def test_populated_values_are_untouched():
    row = {"a": 1, "b": "x", "c": [1.0, 2.0], "d": {"k": "v"}, "e": None}
    assert _writable_rows([row]) == [row]


def test_deep_nesting():
    rows = _writable_rows([{"a": {"b": {"c": {}, "d": [{}, {"x": 1}]}}}])
    assert rows == [{"a": {"b": {"c": None, "d": [None, {"x": 1}]}}}]


def test_no_doors_episode_writes_parquet(tmp_path: pathlib.Path):
    rows = [
        {
            "episode": 1,
            "worst_case_acoustic_frame": {
                "robot_x": 1.0,
                "robot_y": 2.0,
                "source_dba": 60.0,
                "pedestrians": [],
                "door_states": {},
            },
        }
    ]
    dest = tmp_path / "combined_metrics.parquet"
    ParquetStore.write_rows(rows, dest)
    back, _ = ParquetStore.read(dest)
    assert back["worst_case_acoustic_frame"][0]["door_states"] is None


def test_mixed_episodes_share_a_schema(tmp_path: pathlib.Path):
    rows = [
        {"episode": 1, "wcaf": {"door_states": {"world/d1": "open"}}},
        {"episode": 2, "wcaf": {"door_states": {}}},
    ]
    dest = tmp_path / "combined_metrics.parquet"
    ParquetStore.write_rows(rows, dest)
    back, _ = ParquetStore.read(dest)
    assert back.height == 2
