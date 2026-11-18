"""External-asset report mode: sibling bundle and PNG references instead of inline bytes."""

from __future__ import annotations

import pathlib

from star_benchmark import write_benchmark

from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotSpec,
    VizManifest,
)


def _manifest() -> VizManifest:
    return VizManifest(
        name="asset_mode",
        title="Asset Mode",
        plots=[
            PlotSpec(
                id="bar_success",
                type="bar",
                title="Success",
                data={
                    "computation": "group_summary",
                    "params": {"metrics": ["success"], "group_cols": ["planner"]},
                },
                channels=PlotChannels(x="planner", y="mean"),
            ),
            PlotSpec(
                id="box_time",
                type="box",
                title="Time",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="time_to_goal"),
            ),
        ],
    )


def _bench(tmp_path: pathlib.Path) -> pathlib.Path:
    bench = tmp_path / "bench"
    write_benchmark(
        bench,
        [
            {"episode_id": 0, "planner": "dwb", "success": True, "time_to_goal": 10.0},
            {"episode_id": 1, "planner": "teb", "success": False, "time_to_goal": 12.0},
        ],
    )
    return bench


def test_inline_is_the_default_and_external_is_smaller(tmp_path: pathlib.Path) -> None:
    bench = _bench(tmp_path)
    inline = ReportBuilder(bench, output_dir=tmp_path / "inline", manifest=_manifest())
    inline_path = inline.build()
    external = ReportBuilder(
        bench,
        output_dir=tmp_path / "external",
        manifest=_manifest(),
        external_assets=True,
    )
    external_path = external.build()

    assert inline_path is not None and external_path is not None
    inline_page = inline_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in inline_page
    assert "src='plotly.min.js'" not in inline_page
    assert not (inline_path.parent / "plotly.min.js").exists()
    assert external_path.stat().st_size < inline_path.stat().st_size
