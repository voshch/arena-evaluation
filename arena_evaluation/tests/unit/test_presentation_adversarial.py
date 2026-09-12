import pytest
import polars as pl
import pathlib
import tempfile
import os
from unittest.mock import patch, MagicMock

from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.seaborn_renderer import SeabornRenderer
from arena_evaluation.presentation.viz_manifest import VizManifest
from arena_evaluation.storage.schemas import PlotSpec


@pytest.fixture
def temp_benchmark_dir():
    with tempfile.TemporaryDirectory() as td:
        yield pathlib.Path(td)


def test_report_builder_empty_dataframe(temp_benchmark_dir):
    """Fuzzer: Completely empty DataFrame should not crash."""
    df = pl.DataFrame()
    builder = ReportBuilder(temp_benchmark_dir)

    with patch("arena_evaluation.processing.parquet_store.ParquetStore.read", return_value=(df, {})):
        manifest = VizManifest(plots=[PlotSpec(id="test_bar", type="bar", title="Empty", data_key="non_existent")])
        with patch("arena_evaluation.presentation.viz_manifest.VizManifest.load", return_value=manifest):
            builder.build()


def test_report_builder_missing_columns_and_nans(temp_benchmark_dir):
    """Fuzzer: Dataframe missing critical columns or full of nulls."""
    df = pl.DataFrame(
        {
            "planner": ["A", "B", "C"],
            "some_data": [None, float('nan'), 1],  # Mixing None and NaN
        }
    )
    builder = ReportBuilder(temp_benchmark_dir)
    with patch("arena_evaluation.processing.parquet_store.ParquetStore.read", return_value=(df, {})):
        manifest = VizManifest(
            plots=[
                PlotSpec(id="test1", type="bar", title="T", data_key="some_data", differentiate="planner"),
                PlotSpec(id="test2", type="violin", title="T", data_key="missing_col", differentiate="planner"),
            ]
        )
        with patch("arena_evaluation.presentation.viz_manifest.VizManifest.load", return_value=manifest):
            builder.build()


def test_renderers_empty_df():
    """Fuzzer: Directly feed empty dataframes to the renderers."""
    df = pl.DataFrame()
    plotly_renderer = PlotlyRenderer()
    seaborn_renderer = SeabornRenderer()

    plot_types = ["violin", "box", "bar", "trajectory", "radar", "scatter", "histogram", "heatmap", "timeseries", "line", "table"]
    for ptype in plot_types:
        spec = PlotSpec(id=f"test_{ptype}", type=ptype, title="T", data_key="k")
        try:
            plotly_renderer.render(spec, df)
        except Exception as e:
            pytest.fail(f"Plotly renderer crashed on empty dataframe for {ptype}: {e}")

        with tempfile.TemporaryDirectory() as td:
            png_path = pathlib.Path(td) / f"{ptype}.png"
            try:
                seaborn_renderer.render(spec, df, png_path)
            except Exception as e:
                # Some seaborn plots might legitimately crash on empty, but we're fuzzing for robustness
                pytest.fail(f"Seaborn renderer crashed on empty dataframe for {ptype}: {e}")


def test_huge_dataframe(temp_benchmark_dir):
    """Fuzzer: 100k rows DataFrame simulation."""
    df = pl.DataFrame({"planner": ["A"] * 50000 + ["B"] * 50000, "val": list(range(100000)), "success": [True, False] * 50000, "time_to_goal": [1.0] * 100000})
    builder = ReportBuilder(temp_benchmark_dir)
    with patch("arena_evaluation.processing.parquet_store.ParquetStore.read", return_value=(df, {})):
        manifest = VizManifest(plots=[PlotSpec(id="huge_box", type="box", title="Huge", data_key="val", differentiate="planner")])
        with patch("arena_evaluation.presentation.viz_manifest.VizManifest.load", return_value=manifest):
            builder.build()


def test_conflicting_dimensions(temp_benchmark_dir):
    """Fuzzer: Differentiate column exists but contains conflicting types or doesn't exist."""
    df = pl.DataFrame({"planner": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
    builder = ReportBuilder(temp_benchmark_dir)
    with patch("arena_evaluation.processing.parquet_store.ParquetStore.read", return_value=(df, {})):
        manifest = VizManifest(
            plots=[
                # differentiate column doesn't exist
                PlotSpec(id="test_missing_diff", type="bar", title="T", data_key="val", differentiate="non_existent_diff"),
                # group_by column doesn't exist
                PlotSpec(id="test_missing_group", type="bar", title="T", data_key="val", group_by=["missing_group"]),
            ]
        )
        with patch("arena_evaluation.presentation.viz_manifest.VizManifest.load", return_value=manifest):
            builder.build()


def test_seaborn_corrupt_palette(temp_benchmark_dir):
    """Fuzzer: Corrupt palette provided by color_utils."""
    df = pl.DataFrame({"planner": ["A", "B"], "val": [1, 2]})
    seaborn_renderer = SeabornRenderer()
    spec = PlotSpec(id="test", type="box", title="T", data_key="val", differentiate="planner")

    with patch("arena_evaluation.presentation.color_utils.get_color_palette", return_value={"A": "invalid_color_code_xyz", "B": None}):
        png_path = temp_benchmark_dir / "test.png"
        try:
            seaborn_renderer.render(spec, df, png_path)
        except Exception as e:
            pytest.fail(f"Seaborn renderer crashed with corrupt palette: {e}")


def test_malformed_combined_metrics_types(temp_benchmark_dir):
    """Fuzzer: Data keys point to unexpected types (e.g., string instead of numeric)."""
    df = pl.DataFrame(
        {
            "planner": ["A", "B"],
            "val": ["string1", "string2"],  # Should be numeric for violin
            "path": [{"x": 1}, {"y": 2}],  # Malformed trajectory
        }
    )

    plotly_renderer = PlotlyRenderer()

    # These should gracefully handle or fail without throwing unhandled exceptions to the main loop
    spec_violin = PlotSpec(id="test_violin", type="violin", title="T", data_key="val", differentiate="planner")
    plotly_renderer.render(spec_violin, df)

    spec_traj = PlotSpec(id="test_traj", type="trajectory", title="T", data_key="path", differentiate="planner")
    plotly_renderer.render(spec_traj, df)
