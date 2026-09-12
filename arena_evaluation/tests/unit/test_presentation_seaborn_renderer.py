"""Unit tests for the static PNG chart dispatcher (SeabornRenderer)."""

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation import plot_types as pt_module
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.presentation.seaborn_renderer import SeabornRenderer
from arena_evaluation.storage.schemas import PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_ALL_PLOT_TYPES = [
    "violin",
    "box",
    "bar",
    "trajectory",
    "radar",
    "scatter",
    "histogram",
    "heatmap",
    "line",
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


def _assert_png(path: pathlib.Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == _PNG_MAGIC


# ── construction ───────────────────────────────────────────────────────────


def test_all_plot_types_registered():
    renderer = SeabornRenderer()
    assert set(renderer.renderers) == set(_ALL_PLOT_TYPES)


def test_generate_gifs_default_false():
    assert SeabornRenderer().generate_gifs is False
    assert SeabornRenderer(generate_gifs=True).generate_gifs is True


def test_units_default_to_empty_dict():
    assert SeabornRenderer().units == {}
    assert SeabornRenderer(units={"success": "%"}).units == {"success": "%"}


# ── dispatch behaviour ─────────────────────────────────────────────────────


def test_render_bar_writes_png(tmp_path):
    out = tmp_path / "bar.png"
    SeabornRenderer().render(_spec("bar"), _df(), out)
    _assert_png(out)


def test_render_unknown_type_writes_nothing(tmp_path):
    out = tmp_path / "nothing.png"
    spec = PlotSpec(id="x", type="bogus_type", title="X", data_key="success")
    assert SeabornRenderer().render(spec, _df(), out) is None
    assert not out.exists()


def test_render_sets_run_dir_and_units(monkeypatch, tmp_path):
    seen = {}

    def _fake_render_seaborn(self, df, out_path):
        seen["run_dir"] = self.run_dir
        seen["units"] = self.units
        out_path.write_bytes(b"png")

    monkeypatch.setattr(TrajectoryRenderer, "render_seaborn", _fake_render_seaborn)
    out = tmp_path / "t.png"
    SeabornRenderer(units={"success": "%"}).render(_spec("trajectory"), _df(), out, run_dir=tmp_path)
    assert seen["run_dir"] == tmp_path
    assert seen["units"] == {"success": "%"}
    assert out.read_bytes() == b"png"


def test_render_trajectory_gif_flag_passthrough(monkeypatch, tmp_path):
    seen = {}

    def _fake_render_seaborn(self, df, out_path):
        seen["generate_gifs"] = getattr(self, "generate_gifs", "unset")
        out_path.write_bytes(b"png")

    monkeypatch.setattr(TrajectoryRenderer, "render_seaborn", _fake_render_seaborn)
    spec = _spec("trajectory")
    out = tmp_path / "t.png"
    SeabornRenderer(generate_gifs=True).render(spec, _df(), out)
    assert seen["generate_gifs"] is True
    SeabornRenderer(generate_gifs=False).render(spec, _df(), out)
    assert seen["generate_gifs"] is False


def test_render_passes_options_to_renderer(monkeypatch, tmp_path):
    """Dispatcher forwards the full spec; plot-type classes own their options."""
    captured = {}

    def _fake_render_seaborn(self, df, out_path):
        captured["title"] = self.spec.title
        captured["data_key"] = self.spec.data_key
        out_path.write_bytes(b"png")

    monkeypatch.setattr(pt_module.BarRenderer, "render_seaborn", _fake_render_seaborn)
    out = tmp_path / "b.png"
    SeabornRenderer().render(_spec("bar", note="hi"), _df(), out)
    assert captured == {"title": "Title bar", "data_key": "success"}
