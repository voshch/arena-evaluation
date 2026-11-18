"""Regression tests for the all-NULL metric and blank-image gates (H-004, L4-011, L4-003)."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from pydantic import ValidationError

from arena_evaluation.presentation.checks import (
    DiagnosticInspector,
    image_is_blank,
)
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _spec(**overrides: object) -> PlotSpec:
    payload: dict[str, object] = {
        "id": "p",
        "type": "table",
        "data": {"source": "v_episode_summary"},
        "channels": PlotChannels(x="planner", y="time_to_goal"),
    }
    payload.update(overrides)
    return PlotSpec.model_validate(payload)


def _all_null_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "time_to_goal": [None, None],
            "time_to_goal_null_reason": ["not_recorded", "not_recorded"],
        }
    )


def test_all_null_declared_metric_fails_with_its_null_reason() -> None:
    diagnostics = DiagnosticInspector.inspect(_all_null_frame(), _spec())

    assert diagnostics.status == "FAILED"
    assert any(
        "Declared metric 'time_to_goal' is 100% NULL" in issue
        and "not_recorded" in issue
        for issue in diagnostics.issues
    )


def test_all_null_declared_metric_fails_without_a_recorded_reason() -> None:
    frame = pl.DataFrame({"planner": ["dwb"], "time_to_goal": [None]})

    diagnostics = DiagnosticInspector.inspect(frame, _spec())

    assert diagnostics.status == "FAILED"
    assert any("100% NULL" in issue for issue in diagnostics.issues)


def test_allow_all_null_exempts_an_expected_empty_channel() -> None:
    spec = _spec(checks={"allow_all_null": ["time_to_goal"]})

    diagnostics = DiagnosticInspector.inspect(_all_null_frame(), spec)

    assert diagnostics.status == "PASSED"
    assert diagnostics.issues == []


def test_all_null_grouping_label_is_not_a_failed_metric() -> None:
    frame = pl.DataFrame(
        {
            "planner": pl.Series([None, None], dtype=pl.String),
            "time_to_goal": [1.0, 2.0],
        }
    )

    diagnostics = DiagnosticInspector.inspect(frame, _spec())

    assert diagnostics.status == "PASSED"


def test_declared_channel_missing_from_the_frame_is_a_failed_check() -> None:
    frame = pl.DataFrame({"planner": ["dwb"]})

    diagnostics = DiagnosticInspector.inspect(frame, _spec())

    assert diagnostics.status == "FAILED"
    assert any("time_to_goal" in issue for issue in diagnostics.issues)


def test_checks_reject_unknown_keys_instead_of_ignoring_them() -> None:
    with pytest.raises(ValidationError, match="min_row"):
        _spec(checks={"min_row": 3})


def test_min_rows_and_require_columns_still_work() -> None:
    spec = _spec(checks={"require_columns": ["n"], "min_rows": 5})
    frame = pl.DataFrame({"planner": ["dwb"], "time_to_goal": [1.0]})

    diagnostics = DiagnosticInspector.inspect(frame, spec)

    assert diagnostics.status == "FAILED"
    assert any("'n' not found" in issue for issue in diagnostics.issues)
    assert any("min_rows=5" in issue for issue in diagnostics.issues)


def test_zero_rows_frame_reports_only_the_row_issue() -> None:
    frame = pl.DataFrame(schema={"planner": pl.String, "time_to_goal": pl.Float64})

    diagnostics = DiagnosticInspector.inspect(frame, _spec())

    assert diagnostics.status == "FAILED"
    assert sum("100% NULL" in issue for issue in diagnostics.issues) == 0


def _write_image(path: pathlib.Path, pixels: list[list[int]]) -> pathlib.Path:
    from PIL import Image

    height = len(pixels)
    width = len(pixels[0])
    image = Image.new("L", (width, height))
    image.putdata([value for row in pixels for value in row])
    image.save(path)
    return path


def test_blank_image_is_detected(tmp_path: pathlib.Path) -> None:
    blank = _write_image(tmp_path / "blank.png", [[255] * 20 for _ in range(20)])

    assert image_is_blank(blank) is True


def test_rendered_image_with_marks_is_not_blank(tmp_path: pathlib.Path) -> None:
    pixels = [[255] * 20 for _ in range(20)]
    for index in range(20):
        pixels[index][index] = 0
    drawn = _write_image(tmp_path / "drawn.png", pixels)

    assert image_is_blank(drawn) is False
