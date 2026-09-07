"""Unit tests for the XY plot types: scatter, timeseries, trajectory."""

import json
import pathlib
import re

import matplotlib.animation
import pandas as pd
import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.scatter import ScatterRenderer
from arena_evaluation.presentation.plot_types.timeseries import TimeseriesRenderer
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.storage.schemas import PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ── helpers ────────────────────────────────────────────────────────────────


def _newplot_data(html: str) -> list:
    """Decode the trace-data array from a plotly ``to_html`` fragment.

    Plotly 6.x escapes forward slashes (``/`` -> ``\\/``) inside the embedded
    JSON, so literal substring checks against trace names are unreliable;
    decoding the JSON with ``json.JSONDecoder`` is the stable route.
    """
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', html)
    assert match, "plotly newPlot call not found in HTML"
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(html[match.end() :])
    return data


def _newplot_layout(html: str) -> dict:
    """Decode the layout object from a plotly ``to_html`` fragment."""
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', html)
    assert match, "plotly newPlot call not found in HTML"
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(html[match.end() :])
    layout, _ = decoder.raw_decode(html[match.end() + end :].lstrip()[1:].lstrip())
    return layout


def _mock_to_pandas(monkeypatch, pdf: pd.DataFrame) -> None:
    """Substitute ``to_pandas`` so the renderer operates on a frame whose list
    cells are plain python lists (as the trace-building code expects)."""
    monkeypatch.setattr(pl.DataFrame, "to_pandas", lambda self: pdf)


# ═══════════════════════════════════════════════════════════════════════════
# ScatterRenderer
# ═══════════════════════════════════════════════════════════════════════════


def _scatter_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb"],
            "episode": [1, 2, 1, 2],
            "time_to_goal": [15.0, 12.0, 20.0, 18.0],
            "path_length": [10.0, 9.5, 12.0, 11.0],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "time_to_goal": pl.Float64,
            "path_length": pl.Float64,
        },
    )


def _scatter_spec(y="path_length", differentiate="planner", filter=None, **options) -> PlotSpec:
    return PlotSpec(
        id="sc",
        type="scatter",
        title="T2G vs Length",
        data_key="time_to_goal",
        differentiate=differentiate,
        filter=filter,
        options={**options, "y": y},
    )


def test_scatter_plotly_happy_path_with_units():
    html = ScatterRenderer(_scatter_spec(), units={"time_to_goal": "s"}).render_plotly(_scatter_df())
    assert html is not None
    assert html.startswith("<div")
    assert "Time To Goal [s]" in html  # unit-suffixed axis title
    assert "Path Length" in html


def test_scatter_plotly_missing_y_option_returns_none():
    spec = PlotSpec(id="sc", type="scatter", title="T", data_key="time_to_goal")
    assert ScatterRenderer(spec).render_plotly(_scatter_df()) is None


def test_scatter_plotly_missing_columns_returns_none():
    assert ScatterRenderer(_scatter_spec(y="bogus")).render_plotly(_scatter_df()) is None
    spec = PlotSpec(id="sc", type="scatter", title="T", data_key="bogus", options={"y": "path_length"})
    assert ScatterRenderer(spec).render_plotly(_scatter_df()) is None


def test_scatter_plotly_missing_diff_col_returns_none():
    df = pl.DataFrame(
        {"time_to_goal": [1.0, 2.0], "path_length": [1.0, 2.0]},
        schema={"time_to_goal": pl.Float64, "path_length": pl.Float64},
    )
    assert ScatterRenderer(_scatter_spec()).render_plotly(df) is None


def test_scatter_plotly_empty_df_returns_none():
    assert ScatterRenderer(_scatter_spec()).render_plotly(pl.DataFrame()) is None


def test_scatter_plotly_filter_removes_all_rows_returns_none():
    spec = _scatter_spec(filter={"planner": ["nobody"]})
    assert ScatterRenderer(spec).render_plotly(_scatter_df()) is None


def test_scatter_plotly_list_x_column_is_exploded():
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "time_to_goal": [[15.0, 16.0], [20.0]],
            "path_length": [10.0, 12.0],
        },
        schema={
            "planner": pl.Utf8,
            "time_to_goal": pl.List(pl.Float64),
            "path_length": pl.Float64,
        },
    )
    html = ScatterRenderer(_scatter_spec()).render_plotly(df)
    assert html is not None and "dwb" in html


def test_scatter_plotly_list_y_column_is_exploded():
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "time_to_goal": [15.0, 20.0],
            "path_length": [[10.0, 11.0], [12.0]],
        },
        schema={
            "planner": pl.Utf8,
            "time_to_goal": pl.Float64,
            "path_length": pl.List(pl.Float64),
        },
    )
    html = ScatterRenderer(_scatter_spec()).render_plotly(df)
    assert html is not None and "dwb" in html


def test_scatter_plotly_both_list_columns_exploded():
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "time_to_goal": [[15.0, 16.0], [20.0]],
            "path_length": [[10.0, 11.0], [12.0]],
        },
        schema={
            "planner": pl.Utf8,
            "time_to_goal": pl.List(pl.Float64),
            "path_length": pl.List(pl.Float64),
        },
    )
    html = ScatterRenderer(_scatter_spec()).render_plotly(df)
    assert html is not None and "dwb" in html


def test_scatter_plotly_all_empty_lists_render_empty_figure():
    # polars explode of all-empty lists keeps rows (as nulls), so the post-
    # explode emptiness guard does not trigger; the figure is just empty.
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "time_to_goal": [[], []],
            "path_length": [1.0, 2.0],
        },
        schema={
            "planner": pl.Utf8,
            "time_to_goal": pl.List(pl.Float64),
            "path_length": pl.Float64,
        },
    )
    html = ScatterRenderer(_scatter_spec()).render_plotly(df)
    assert html is not None


def test_scatter_plotly_compound_label_diff():
    df = pl.DataFrame(
        {
            "local_planner": ["dwb", "dwb", "teb", "teb"],
            "stage": ["a", "b", "a", "b"],
            "time_to_goal": [15.0, 12.0, 20.0, 18.0],
            "path_length": [10.0, 9.5, 12.0, 11.0],
        },
        schema={
            "local_planner": pl.Utf8,
            "stage": pl.Utf8,
            "time_to_goal": pl.Float64,
            "path_length": pl.Float64,
        },
    )
    html = ScatterRenderer(_scatter_spec()).render_plotly(df)
    assert html is not None
    # Plotly's JSON encoder escapes "/" as / inside the embedded figure.
    assert "dwb \\u002f a" in html  # compound __label__ values colour the traces


def test_scatter_seaborn_happy_path(tmp_path):
    out = tmp_path / "scatter.png"
    ScatterRenderer(_scatter_spec()).render_seaborn(_scatter_df(), out)
    assert out.exists() and out.read_bytes()[:8] == _PNG_MAGIC


def test_scatter_seaborn_missing_y_writes_nothing(tmp_path):
    out = tmp_path / "scatter.png"
    spec = PlotSpec(id="sc", type="scatter", title="T", data_key="time_to_goal")
    ScatterRenderer(spec).render_seaborn(_scatter_df(), out)
    assert not out.exists()


def test_scatter_seaborn_missing_diff_writes_nothing(tmp_path):
    out = tmp_path / "scatter.png"
    df = pl.DataFrame({"time_to_goal": [1.0], "path_length": [1.0]})
    ScatterRenderer(_scatter_spec()).render_seaborn(df, out)
    assert not out.exists()


def test_scatter_seaborn_list_explode(tmp_path):
    out = tmp_path / "scatter.png"
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "time_to_goal": [[15.0, 16.0], [20.0]],
            "path_length": [10.0, 12.0],
        },
        schema={
            "planner": pl.Utf8,
            "time_to_goal": pl.List(pl.Float64),
            "path_length": pl.Float64,
        },
    )
    ScatterRenderer(_scatter_spec()).render_seaborn(df, out)
    assert out.exists() and out.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# TimeseriesRenderer
# ═══════════════════════════════════════════════════════════════════════════


def _ts_df() -> pl.DataFrame:
    """Polars List-typed frame, as produced by the real pipeline."""
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb"],
            "episode": [1, 2, 1],
            "timeseries_time_s": [[0.0, 1.0, 2.0], [0.0, 0.5, 1.0], [0.0, 1.0, 2.0]],
            "path_efficiency": [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.6, 0.8, 0.9]],
            "collision_amount": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "timeseries_time_s": pl.List(pl.Float64),
            "path_efficiency": pl.List(pl.Float64),
            "collision_amount": pl.List(pl.Float64),
        },
    )


def _ts_obj_df() -> pl.DataFrame:
    """Same data with Object-typed list columns so that ``to_pandas`` keeps
    the values as real Python lists (the renderer's isinstance checks pass)."""
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb"],
            "episode": [1, 2, 1],
            "timeseries_time_s": [[0.0, 1.0, 2.0], [0.0, 0.5, 1.0], [0.0, 1.0, 2.0]],
            "path_efficiency": [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.6, 0.8, 0.9]],
            "collision_amount": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "timeseries_time_s": pl.Object,
            "path_efficiency": pl.Object,
            "collision_amount": pl.Object,
        },
    )


def _ts_spec(**options) -> PlotSpec:
    base = dict(id="ts", type="timeseries", title="Power over Time", data_key="timeseries_time_s", differentiate="planner")
    base["options"] = {"metrics": ["path_efficiency", "collision_amount"], **options}
    return PlotSpec(**base)


def test_timeseries_plotly_happy_path_with_units():
    html = TimeseriesRenderer(_ts_spec(), units={"timeseries_time_s": "s"}).render_plotly(_ts_obj_df())
    assert html is not None
    assert html.startswith("<div")
    assert "Time S [s]" in html  # timeseries_ prefix stripped + unit
    assert "dwb - Ep 1" in html  # per-episode legend groups


def test_timeseries_plotly_metrics_fall_back_to_data_key():
    spec = PlotSpec(id="ts", type="timeseries", title="T", data_key="path_efficiency")
    html = TimeseriesRenderer(spec).render_plotly(_ts_obj_df())
    assert html is not None and "dwb - Ep 1" in html


def test_timeseries_plotly_no_metrics_and_no_data_key_returns_none():
    spec = PlotSpec(id="ts", type="timeseries", title="T", data_key="")
    assert TimeseriesRenderer(spec).render_plotly(_ts_df()) is None


def test_timeseries_plotly_metrics_missing_from_df_returns_none():
    spec = _ts_spec(metrics=["bogus_metric"])
    assert TimeseriesRenderer(spec).render_plotly(_ts_df()) is None


def test_timeseries_plotly_custom_x_column():
    df = _ts_obj_df().rename({"timeseries_time_s": "t_ns"})
    html = TimeseriesRenderer(_ts_spec(x="t_ns")).render_plotly(df)
    assert html is not None and "dwb - Ep 1" in html


def test_timeseries_plotly_x_column_missing_returns_none():
    df = _ts_df().drop("timeseries_time_s")
    assert TimeseriesRenderer(_ts_spec()).render_plotly(df) is None


def test_timeseries_plotly_empty_df_returns_none():
    assert TimeseriesRenderer(_ts_spec()).render_plotly(pl.DataFrame()) is None


def test_timeseries_plotly_no_episode_column_uses_unknown():
    df = _ts_obj_df().drop("episode")
    html = TimeseriesRenderer(_ts_spec()).render_plotly(df)
    assert html is not None and "dwb - Ep unknown" in html


def test_timeseries_plotly_missing_diff_col_uses_unknown_planner():
    df = pl.DataFrame(
        {
            "episode": [1],
            "timeseries_time_s": [[0.0, 1.0]],
            "path_efficiency": [[0.5, 0.6]],
        },
        schema={
            "episode": pl.Int64,
            "timeseries_time_s": pl.Object,
            "path_efficiency": pl.Object,
        },
    )
    spec = PlotSpec(id="ts", type="timeseries", title="T", data_key="timeseries_time_s", differentiate=None)
    html = TimeseriesRenderer(spec).render_plotly(df)
    assert html is not None and "unknown - Ep 1" in html


def test_timeseries_plotly_scalar_x_data_skips_trace():
    df = pl.DataFrame(
        {
            "planner": ["dwb"],
            "episode": [1],
            "timeseries_time_s": [0.0],
            "path_efficiency": [[0.5, 0.6]],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "timeseries_time_s": pl.Float64,
            "path_efficiency": pl.Object,
        },
    )
    html = TimeseriesRenderer(_ts_spec()).render_plotly(df)
    assert html is not None
    assert "dwb - Ep 1" not in html  # every trace was skipped -> empty figure


def test_timeseries_plotly_none_metric_skips_trace():
    df = pl.DataFrame(
        {
            "planner": ["dwb"],
            "episode": [1],
            "timeseries_time_s": [[0.0, 1.0]],
            "path_efficiency": [None],
            "collision_amount": [[0.0, 0.0]],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "timeseries_time_s": pl.Object,
            "path_efficiency": pl.Object,
            "collision_amount": pl.Object,
        },
    )
    html = TimeseriesRenderer(_ts_spec()).render_plotly(df)
    assert html is not None and "dwb - Ep 1" in html


def test_timeseries_plotly_polars_list_columns_render_successfully():
    """Verify that Polars List columns (which convert to NumPy arrays) render correctly."""
    html = TimeseriesRenderer(_ts_spec()).render_plotly(_ts_df())
    assert html is not None
    assert "dwb - Ep 1" in html


def test_timeseries_seaborn_is_pass_through(tmp_path):
    out = tmp_path / "ts.png"
    assert TimeseriesRenderer(_ts_spec()).render_seaborn(_ts_df(), out) is None
    assert not out.exists()


# ═══════════════════════════════════════════════════════════════════════════
# TrajectoryRenderer
# ═══════════════════════════════════════════════════════════════════════════


def _flat_traj_df() -> pl.DataFrame:
    """Single non-nested path per row."""
    return pl.DataFrame(
        {
            "planner": ["dwb"],
            "episode": [1],
            "result": ["SUCCESS"],
            "path": [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "result": pl.Utf8,
            "path": pl.List(pl.List(pl.Float64)),
        },
    )


def _nested_traj_df() -> pl.DataFrame:
    """Nested paths: sub-path lists (multi-agent), jumps, collisions."""
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb"],
            "episode": [1, 2, 1],
            "result": ["SUCCESS", "COLLISION", "SUCCESS"],
            "path": [
                [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]],
                [[[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]]],
                [[[0.0, 0.0], [0.02, 0.0], [5.0, 5.0]], [[0.0, 2.0], [1.0, 3.0]]],
            ],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "result": pl.Utf8,
            "path": pl.List(pl.List(pl.List(pl.Float64))),
        },
    )


def _traj_spec(group_by=None, differentiate="planner", filter=None, **options) -> PlotSpec:
    return PlotSpec(
        id="tr",
        type="trajectory",
        title="Paths",
        data_key="path",
        group_by=group_by,
        differentiate=differentiate,
        filter=filter,
        options={"show_map": False, **options},
    )


def _tiny_png(tmp_path: pathlib.Path) -> pathlib.Path:
    from PIL import Image

    p = tmp_path / "map.png"
    Image.new("RGB", (8, 8), "white").save(p)
    return p


def _map_meta(png_path: pathlib.Path) -> dict:
    return {"png_path": str(png_path), "resolution": 0.1, "width": 8, "height": 8}


# ── trajectory: plotly ─────────────────────────────────────────────────────


def test_trajectory_plotly_single_agent_with_markers():
    html = TrajectoryRenderer(_traj_spec()).render_plotly(_flat_traj_df())
    assert html is not None and html.startswith("<div")
    assert "Paths" in html
    assert "dwb" in html
    assert "dwb Starts" in html and "dwb Goals" in html


def test_trajectory_plotly_multi_agent_and_collision():
    html = TrajectoryRenderer(_traj_spec()).render_plotly(_nested_traj_df())
    assert html is not None
    # teb's row carries two sub-paths -> multi-agent traces for teb.
    assert "teb - Agent 0" in html
    assert "teb - Agent 1" in html
    assert "dwb Starts" in html
    assert "dwb Goals" in html
    assert "dwb Collisions" in html  # COLLISION row


def test_trajectory_plotly_overlay_markers_disabled():
    html = TrajectoryRenderer(_traj_spec(overlay_markers=False)).render_plotly(_nested_traj_df())
    assert html is not None
    assert "Starts" not in html and "Goals" not in html


def test_trajectory_plotly_missing_diff_col_uses_unknown():
    df = _flat_traj_df().drop("planner")
    spec = _traj_spec(differentiate=None)
    html = TrajectoryRenderer(spec).render_plotly(df)
    assert html is not None and "unknown" in html


def test_trajectory_plotly_missing_data_key_returns_none():
    df = _flat_traj_df().drop("path")
    assert TrajectoryRenderer(_traj_spec()).render_plotly(df) is None


def test_trajectory_plotly_empty_df_returns_none():
    assert TrajectoryRenderer(_traj_spec()).render_plotly(pl.DataFrame()) is None


def test_trajectory_plotly_map_overlay_embeds_png(monkeypatch, tmp_path):
    png = _tiny_png(tmp_path)
    monkeypatch.setattr(TrajectoryRenderer, "_load_map_image", lambda self, m, run_dir=None: _map_meta(png))
    df = _flat_traj_df().with_columns(pl.lit("map_empty").alias("map"))
    spec = PlotSpec(id="tr", type="trajectory", title="Paths", data_key="path", options={"show_map": True})
    html = TrajectoryRenderer(spec).render_plotly(df)
    assert html is not None
    # Plotly's JSON encoder escapes "/" -> image\/png;base64 inside the div.
    assert "image\\u002fpng;base64," in html  # layout image embedded


def test_trajectory_plotly_map_overlay_failure_warns(monkeypatch, tmp_path, capsys):
    meta = {"png_path": str(tmp_path / "missing.png"), "resolution": 0.1, "width": 8, "height": 8}
    monkeypatch.setattr(TrajectoryRenderer, "_load_map_image", lambda self, m, run_dir=None: meta)
    df = _flat_traj_df().with_columns(pl.lit("map_empty").alias("map"))
    spec = PlotSpec(id="tr", type="trajectory", title="Paths", data_key="path", options={"show_map": True})
    html = TrajectoryRenderer(spec).render_plotly(df)
    assert html is not None
    assert "Failed to overlay map" in capsys.readouterr().out


def test_trajectory_plotly_show_map_false_does_not_load_map(monkeypatch):
    calls = []
    monkeypatch.setattr(
        TrajectoryRenderer,
        "_load_map_image",
        lambda self, m, run_dir=None: calls.append(m) or None,
    )
    df = _flat_traj_df().with_columns(pl.lit("map_empty").alias("map"))
    TrajectoryRenderer(_traj_spec(show_map=False)).render_plotly(df)
    assert calls == []


def test_trajectory_plotly_group_by_string_returns_list():
    spec = _traj_spec(group_by="episode")
    htmls = TrajectoryRenderer(spec).render_plotly(_nested_traj_df())
    assert isinstance(htmls, list) and len(htmls) == 2  # episodes 1 and 2
    assert all(h.startswith("<div") for h in htmls)
    assert any("episode: 1" in h for h in htmls)
    assert any("episode: 2" in h for h in htmls)


def test_trajectory_plotly_group_by_two_cols_suffix():
    spec = _traj_spec(group_by=["episode", "result"])
    htmls = TrajectoryRenderer(spec).render_plotly(_nested_traj_df())
    assert isinstance(htmls, list) and len(htmls) == 2
    assert any("episode: 1 | result: SUCCESS" in h for h in htmls)
    assert any("episode: 2 | result: COLLISION" in h for h in htmls)


def test_trajectory_plotly_group_by_invalid_cols_falls_back_to_single():
    spec = _traj_spec(group_by=["bogus"])
    html = TrajectoryRenderer(spec).render_plotly(_nested_traj_df())
    assert isinstance(html, str) and html.startswith("<div")


def test_trajectory_plotly_group_by_map_col(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        TrajectoryRenderer,
        "_load_map_image",
        lambda self, m, run_dir=None: captured.setdefault("maps", []).append(m) or None,
    )
    df = _flat_traj_df().with_columns(pl.lit("map_empty").alias("map"))
    spec = _traj_spec(group_by=["map"], show_map=True)
    htmls = TrajectoryRenderer(spec).render_plotly(df)
    assert isinstance(htmls, list) and len(htmls) == 1
    assert captured["maps"] == ["map_empty"]


def test_trajectory_plotly_no_group_by_uses_frame_map_col(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        TrajectoryRenderer,
        "_load_map_image",
        lambda self, m, run_dir=None: captured.setdefault("maps", []).append(m) or None,
    )
    df = _flat_traj_df().with_columns(pl.lit("map_empty").alias("map"))
    html = TrajectoryRenderer(_traj_spec(show_map=True)).render_plotly(df)
    assert isinstance(html, str)
    assert captured["maps"] == ["map_empty"]


def test_trajectory_plotly_skips_empty_paths():
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb", "mppi"],
            "path": [
                [],
                [],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
        },
        schema={
            "planner": pl.Utf8,
            "path": pl.List(pl.List(pl.Float64)),
        },
    )
    spec = _traj_spec()
    html = TrajectoryRenderer(spec).render_plotly(df)
    assert html is not None
    assert "mppi" in html  # only the non-empty path is drawn


def test_trajectory_plotly_null_path_crashes_currently():
    """Suspected source bug (documented): a polars-null path becomes NaN in
    pandas, so the ``path is None`` guard misses it and ``len(NaN)`` raises
    TypeError. Pins current behaviour; should be removed once the renderer
    guards against NaN paths."""
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "path": [
                None,
                [[0.0, 0.0], [1.0, 1.0]],
            ],
        },
        schema={
            "planner": pl.Utf8,
            "path": pl.List(pl.List(pl.Float64)),
        },
    )
    with pytest.raises(TypeError):
        TrajectoryRenderer(_traj_spec()).render_plotly(df)


def test_trajectory_plotly_skips_malformed_paths():
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "path": [
                [1, 2, 3],  # not point lists -> except -> continue
                [[0.0, 0.0], [1.0, 1.0]],
            ],
        },
        schema={"planner": pl.Utf8, "path": pl.Object},
    )
    html = TrajectoryRenderer(_traj_spec()).render_plotly(df)
    assert html is not None
    assert "teb" in html  # malformed dwb row skipped, teb still drawn


def test_trajectory_plotly_skips_short_points_subpath():
    df = pl.DataFrame(
        {
            "planner": ["dwb"],
            "path": [[[[0.0], [1.0]]]],  # points with a single coordinate
        },
        schema={"planner": pl.Utf8, "path": pl.List(pl.List(pl.List(pl.Float64)))},
    )
    html = TrajectoryRenderer(_traj_spec()).render_plotly(df)
    assert html is not None


def test_trajectory_plotly_none_subpath_inside_nested_path():
    df = pl.DataFrame(
        {
            "planner": ["dwb"],
            "path": [[[[0.0, 0.0], [1.0, 1.0]], None]],
        },
        schema={"planner": pl.Utf8, "path": pl.List(pl.List(pl.List(pl.Float64)))},
    )
    html = TrajectoryRenderer(_traj_spec()).render_plotly(df)
    assert html is not None and "dwb" in html


def test_trajectory_plotly_all_nan_path_skips_markers():
    df = pl.DataFrame(
        {
            "planner": ["dwb"],
            "path": [[[float("nan"), 0.0], [float("nan"), 1.0]]],
        },
        schema={"planner": pl.Utf8, "path": pl.List(pl.List(pl.Float64))},
    )
    html = TrajectoryRenderer(_traj_spec()).render_plotly(df)
    assert html is not None


# ── trajectory: seaborn ────────────────────────────────────────────────────


def test_trajectory_seaborn_single_agent_png(tmp_path):
    out = tmp_path / "traj.png"
    TrajectoryRenderer(_traj_spec()).render_seaborn(_flat_traj_df(), out)
    assert out.exists() and out.read_bytes()[:8] == _PNG_MAGIC


def test_trajectory_seaborn_missing_data_key_writes_nothing(tmp_path):
    out = tmp_path / "traj.png"
    df = _flat_traj_df().drop("path")
    TrajectoryRenderer(_traj_spec()).render_seaborn(df, out)
    assert not out.exists()


def test_trajectory_seaborn_group_by_writes_suffixed_files(tmp_path):
    out = tmp_path / "traj.png"
    spec = _traj_spec(group_by=["episode"])
    TrajectoryRenderer(spec).render_seaborn(_nested_traj_df(), out)
    assert (tmp_path / "traj_1.png").exists()
    assert (tmp_path / "traj_2.png").exists()
    assert (tmp_path / "traj_1.png").stat().st_size > 0


def test_trajectory_seaborn_group_by_two_cols_suffix(tmp_path):
    out = tmp_path / "traj.png"
    spec = _traj_spec(group_by=["episode", "result"])
    TrajectoryRenderer(spec).render_seaborn(_nested_traj_df(), out)
    assert (tmp_path / "traj_1_SUCCESS.png").exists()
    assert (tmp_path / "traj_2_COLLISION.png").exists()


def test_trajectory_seaborn_group_by_invalid_writes_single_file(tmp_path):
    out = tmp_path / "traj.png"
    spec = _traj_spec(group_by=["bogus"])
    TrajectoryRenderer(spec).render_seaborn(_flat_traj_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_trajectory_seaborn_multi_agent_with_true_start_goal(tmp_path):
    df = _nested_traj_df().with_columns(
        [
            pl.Series("start", [[0.0, 0.0], [0.5, 0.5], [0.0, 0.0]]),
            pl.Series("goal", [[2.0, 2.0], [1.0, 1.0], [3.0, 3.0]]),
        ]
    )
    out = tmp_path / "traj.png"
    TrajectoryRenderer(_traj_spec()).render_seaborn(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_trajectory_seaborn_map_imshow(monkeypatch, tmp_path):
    png = _tiny_png(tmp_path)
    monkeypatch.setattr(TrajectoryRenderer, "_load_map_image", lambda self, m, run_dir=None: _map_meta(png))
    df = _flat_traj_df().with_columns(pl.lit("map_empty").alias("map"))
    out = tmp_path / "traj.png"
    spec = PlotSpec(id="tr", type="trajectory", title="Paths", data_key="path", options={"show_map": True})
    TrajectoryRenderer(spec).render_seaborn(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_trajectory_seaborn_map_imread_failure_warns(monkeypatch, tmp_path, capsys):
    bad_file = tmp_path / "bad.png"
    bad_file.write_text("not an image")
    meta = {"png_path": str(bad_file), "resolution": 0.1, "width": 8, "height": 8}
    monkeypatch.setattr(TrajectoryRenderer, "_load_map_image", lambda self, m, run_dir=None: meta)
    df = _flat_traj_df().with_columns(pl.lit("map_empty").alias("map"))
    out = tmp_path / "traj.png"
    spec = PlotSpec(id="tr", type="trajectory", title="Paths", data_key="path", options={"show_map": True})
    TrajectoryRenderer(spec).render_seaborn(df, out)
    assert "Failed to overlay map" in capsys.readouterr().out
    assert out.exists()


def test_trajectory_seaborn_gif_generated(tmp_path):
    renderer = TrajectoryRenderer(_traj_spec())
    renderer.generate_gifs = True
    out = tmp_path / "traj.png"
    renderer.render_seaborn(_nested_traj_df(), out)
    gif = out.with_suffix(".gif")
    assert gif.exists() and gif.stat().st_size > 0
    assert gif.read_bytes()[:6] == b"GIF89a"


def test_trajectory_seaborn_gif_save_failure_warns(monkeypatch, tmp_path, capsys):
    class _FailingAnimation:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, *args, **kwargs):
            raise RuntimeError("writer failed")

    monkeypatch.setattr(matplotlib.animation, "FuncAnimation", _FailingAnimation)
    renderer = TrajectoryRenderer(_traj_spec())
    renderer.generate_gifs = True
    out = tmp_path / "traj.png"
    renderer.render_seaborn(_nested_traj_df(), out)
    assert "Failed to save GIF" in capsys.readouterr().out
    assert not out.with_suffix(".gif").exists()


def test_trajectory_seaborn_no_lines_skips_gif(tmp_path):
    df = pl.DataFrame(
        {"planner": ["dwb"], "path": [[]]},
        schema={"planner": pl.Utf8, "path": pl.List(pl.List(pl.Float64))},
    )
    renderer = TrajectoryRenderer(_traj_spec())
    renderer.generate_gifs = True
    out = tmp_path / "traj.png"
    renderer.render_seaborn(df, out)
    assert not out.with_suffix(".gif").exists()


def test_trajectory_seaborn_null_path_crashes_currently(tmp_path):
    """Same documented source bug as the plotly variant: NaN path from a
    polars null reaches ``len()`` and raises TypeError."""
    df = pl.DataFrame(
        {"planner": ["dwb"], "path": [None]},
        schema={"planner": pl.Utf8, "path": pl.List(pl.List(pl.Float64))},
    )
    out = tmp_path / "traj.png"
    with pytest.raises(TypeError):
        TrajectoryRenderer(_traj_spec()).render_seaborn(df, out)


def test_trajectory_seaborn_empty_pdf_writes_nothing(tmp_path):
    out = tmp_path / "traj.png"
    spec = _traj_spec(filter={"planner": ["nobody"]})
    TrajectoryRenderer(spec).render_seaborn(_flat_traj_df(), out)
    assert not out.exists()
