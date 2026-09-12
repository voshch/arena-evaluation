"""Unit tests for the interactive HTML chart dispatcher (PlotlyRenderer)."""

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation import plot_types as pt_module
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.storage.schemas import PlotSpec

_ALL_PLOT_TYPES = [
    "violin",
    "box",
    "bar",
    "trajectory",
    "radar",
    "scatter",
    "histogram",
    "heatmap",
    "timeseries",
    "line",
    "table",
    "acoustic_field",
    "acoustic_field_animation",
]


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb"],
            "episode": [1, 2, 1],
            "success": [1.0, 0.5, 0.75],
        },
        schema={"planner": pl.Utf8, "episode": pl.Int64, "success": pl.Float64},
    )


def _spec(ptype: str, **options) -> PlotSpec:
    return PlotSpec(id=f"p_{ptype}", type=ptype, title=f"Title {ptype}", data_key="success", options=options)


# ── construction ───────────────────────────────────────────────────────────


def test_all_plot_types_registered():
    renderer = PlotlyRenderer()
    assert set(renderer.renderers) == set(_ALL_PLOT_TYPES)


def test_units_default_to_empty_dict():
    assert PlotlyRenderer().units == {}


def test_units_passed_through():
    renderer = PlotlyRenderer(units={"success": "%"})
    assert renderer.units == {"success": "%"}


# ── dispatch behaviour ─────────────────────────────────────────────────────


def test_render_bar_returns_html_fragment():
    html = PlotlyRenderer().render(_spec("bar"), _df())
    assert isinstance(html, str)
    assert html.startswith("<div")
    assert "plotly" in html.lower() or "data" in html


def test_render_unknown_type_returns_none():
    spec = PlotSpec(id="x", type="bogus_type", title="X", data_key="success")
    assert PlotlyRenderer().render(spec, _df()) is None


def test_render_none_result_is_passed_through(monkeypatch):
    monkeypatch.setattr(TrajectoryRenderer, "render_plotly", lambda self, df: None)
    spec = _spec("trajectory")
    assert PlotlyRenderer().render(spec, _df()) is None


def test_render_list_result_is_passed_through(monkeypatch):
    monkeypatch.setattr(TrajectoryRenderer, "render_plotly", lambda self, df: ["<div>a</div>", "<div>b</div>"])
    spec = _spec("trajectory")
    assert PlotlyRenderer().render(spec, _df()) == ["<div>a</div>", "<div>b</div>"]


def test_render_trajectory_disables_gifs_and_sets_run_dir(monkeypatch, tmp_path):
    seen = {}

    def _fake_render_plotly(self, df):
        seen["generate_gifs"] = getattr(self, "generate_gifs", "unset")
        seen["run_dir"] = self.run_dir
        seen["units"] = self.units
        return "<div>traj</div>"

    monkeypatch.setattr(TrajectoryRenderer, "render_plotly", _fake_render_plotly)
    spec = _spec("trajectory")
    out = PlotlyRenderer().render(spec, _df(), run_dir=tmp_path)
    assert out == "<div>traj</div>"
    assert seen["generate_gifs"] is False  # GIFs are seaborn-only
    assert seen["run_dir"] == tmp_path
    assert seen["units"] == {}


def test_render_non_trajectory_keeps_renderer_defaults(monkeypatch):
    captured = {}

    class _FakeScatter:
        __name__ = "ScatterRenderer"

        def __init__(self, spec, units=None):
            self.spec = spec
            self.units = units or {}
            self.run_dir = None
            self.generate_gifs = True

        def render_plotly(self, df):
            captured["generate_gifs"] = self.generate_gifs
            captured["units"] = self.units
            return "<div>scatter</div>"

    monkeypatch.setattr(pt_module, "ScatterRenderer", _FakeScatter)
    renderer = PlotlyRenderer(units={"success": "%"})
    out = renderer.render(_spec("scatter"), _df(), run_dir=pathlib.Path("/tmp"))
    assert out == "<div>scatter</div>"
    assert captured["generate_gifs"] is True  # untouched for non-trajectory
    assert captured["units"] == {"success": "%"}


def test_render_replaces_renderer_run_dir(monkeypatch, tmp_path):
    seen = {}

    def _fake_render_plotly(self, df):
        seen["run_dir"] = self.run_dir
        return "<div>ok</div>"

    monkeypatch.setattr(TrajectoryRenderer, "render_plotly", _fake_render_plotly)
    spec = _spec("trajectory")
    PlotlyRenderer().render(spec, _df(), run_dir=None)
    assert seen["run_dir"] is None
    PlotlyRenderer().render(spec, _df(), run_dir=tmp_path)
    assert seen["run_dir"] == tmp_path
