"""Manifest ``where`` clauses are compiled predicates, never spliced SQL."""

from __future__ import annotations

import pathlib

import pytest

from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotDataSpec,
    PlotSpec,
)
from star_benchmark import write_benchmark

_ROWS: list[dict[str, object]] = [
    {
        "episode_id": 0,
        "planner": "dwb",
        "success": True,
        "time_to_goal": 10.0,
        "is_reference": False,
        "crowd_condition": "visible",
        "robot_model": "jackal",
    },
    {
        "episode_id": 1,
        "planner": "dwb",
        "success": False,
        "time_to_goal": 20.0,
        "is_reference": False,
        "crowd_condition": "visible",
        "robot_model": "jackal",
    },
    {
        "episode_id": 2,
        "planner": "mppi",
        "success": True,
        "time_to_goal": 30.0,
        "is_reference": True,
        "crowd_condition": "blind",
        "robot_model": "jackal",
    },
    {
        "episode_id": 3,
        "planner": "mppi",
        "success": True,
        "time_to_goal": None,
        "is_reference": False,
        "crowd_condition": "blind",
        "robot_model": "jackal",
    },
]


def _resolver(tmp_path: pathlib.Path) -> DataResolver:
    write_benchmark(tmp_path, _ROWS)
    return DataResolver(benchmark_dir=tmp_path)


def _spec(where: str) -> PlotSpec:
    return PlotSpec(
        id="filtered",
        type="bar",
        data=PlotDataSpec(source="v_episode_summary", where=where),
        channels=PlotChannels(x="planner", y="time_to_goal"),
    )


def _episode_ids(resolver: DataResolver, where: str) -> list[int]:
    frame, _diagnostics = resolver.resolve(_spec(where))
    return sorted(frame["episode_id"].to_list())


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ("planner = 'dwb'", [0, 1]),
        ("planner == 'dwb'", [0, 1]),
        ("planner <> 'dwb'", [2, 3]),
        ("planner != 'dwb'", [2, 3]),
        ("planner = 'dwb' and success = true", [0]),
        ("planner = 'dwb' or success = true", [0, 1, 2, 3]),
        ("success = false or time_to_goal > 25", [1, 2]),
        ("(planner = 'dwb' or planner = 'mppi') and not success", [1]),
        ("planner in ('dwb', 'mppi')", [0, 1, 2, 3]),
        ("planner in ('dwb')", [0, 1]),
        ("planner in 'dwb'", [0, 1]),
        ("time_to_goal >= 30", [2]),
        ("time_to_goal < 20", [0]),
        ("'dwb' == planner", [0, 1]),
        ("20.0 > time_to_goal", [0]),
        ("time_to_goal is null", [3]),
        ("time_to_goal is not null", [0, 1, 2]),
        ("time_to_goal.is_null()", [3]),
        ("is_reference.is_not_null()", [0, 1, 2, 3]),
        ("not is_reference", [0, 1, 3]),
    ],
)
def test_where_predicate_filters_the_bound_table(
    tmp_path: pathlib.Path, where: str, expected: list[int]
) -> None:
    assert _episode_ids(_resolver(tmp_path), where) == expected


@pytest.mark.parametrize(
    "where",
    [
        "1 = 1; DROP TABLE fact_episodes",
        "planner = 'dwb' -- comment",
        "planner = 'dwb'; SELECT * FROM evidence_index",
        "__import__('os').system('touch pwned')",
        "planner = 'dwb' or eval('1')",
        "planner",
        "planner =",
        "planner = 'dwb' and",
    ],
)
def test_injection_and_malformed_predicates_are_rejected(
    tmp_path: pathlib.Path, where: str
) -> None:
    with pytest.raises(ValueError, match="unsupported where predicate") as error:
        _resolver(tmp_path).resolve(_spec(where))
    assert "Hint:" in str(error.value)


def test_an_unknown_column_names_the_bound_columns(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError) as error:
        _resolver(tmp_path).resolve(_spec("missing_column = 1"))
    message = str(error.value)
    assert "missing_column" in message
    assert "is not a column of the bound table" in message
    assert "planner" in message


@pytest.mark.parametrize(
    ("where", "reason"),
    [
        ("planner = 1", "does not fit column dtype"),
        ("success = 'yes'", "does not fit column dtype"),
        ("time_to_goal = null", "compare against null with is null"),
        ("planner is 3", "is / is not compares against null only"),
        ("planner in (1, 2)", "does not fit column dtype"),
        ("planner in 1", "does not fit column dtype"),
        ("success = true or planner = 'mppi' and", "does not parse"),
    ],
)
def test_predicates_type_check_the_column_dtype(
    tmp_path: pathlib.Path, where: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        _resolver(tmp_path).resolve(_spec(where))


def test_an_empty_predicate_is_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="empty where predicate"):
        _resolver(tmp_path).resolve(_spec("   "))
