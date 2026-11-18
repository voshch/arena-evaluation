"""Regression tests for the all-NULL metric and blank-image gates (H-004, L4-011, L4-003)."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.checks import (
    DiagnosticInspector,
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


def test_min_rows_and_require_columns_still_work() -> None:
    spec = _spec(checks={"require_columns": ["n"], "min_rows": 5})
    frame = pl.DataFrame({"planner": ["dwb"], "time_to_goal": [1.0]})

    diagnostics = DiagnosticInspector.inspect(frame, spec)

    assert diagnostics.status == "FAILED"
    assert any("'n' not found" in issue for issue in diagnostics.issues)
    assert any("min_rows=5" in issue for issue in diagnostics.issues)


def _write_image(path: pathlib.Path, pixels: list[list[int]]) -> pathlib.Path:
    from PIL import Image

    height = len(pixels)
    width = len(pixels[0])
    image = Image.new("L", (width, height))
    image.putdata([value for row in pixels for value in row])
    image.save(path)
    return path


