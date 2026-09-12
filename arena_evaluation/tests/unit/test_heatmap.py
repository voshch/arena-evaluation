import polars as pl
from arena_evaluation.storage.schemas import PlotSpec
from arena_evaluation.presentation.plot_types.heatmap import HeatmapRenderer


def test_heatmap_correlation():
    df = pl.DataFrame(
        {
            "success": [1.0, 0.5, 0.8],
            "time_to_goal": [15.0, 20.0, 18.0],
            "path_length": [10.0, 12.0, 11.0],
            "collision_amount": [0.0, 1.0, 0.0],
        }
    )

    spec = PlotSpec(
        id="correlation_matrix",
        type="heatmap",
        title="Metrics Correlation Matrix",
        data_key="*",
        layout_group="overview",
    )

    renderer = HeatmapRenderer(spec)

    # This should return the plotly HTML string without raising any AttributeErrors
    html = renderer.render_plotly(df)
    assert html is not None
    assert "Correlation" in html


def test_heatmap_seaborn(tmp_path):
    import pathlib

    df = pl.DataFrame(
        {
            "success": [1.0, 0.5, 0.8],
            "time_to_goal": [15.0, 20.0, 18.0],
            "path_length": [10.0, 12.0, 11.0],
            "collision_amount": [0.0, 1.0, 0.0],
        }
    )

    spec = PlotSpec(
        id="correlation_matrix",
        type="heatmap",
        title="Metrics Correlation Matrix",
        data_key="*",
        layout_group="overview",
    )

    renderer = HeatmapRenderer(spec)
    out_path = tmp_path / "correlation_matrix.png"

    renderer.render_seaborn(df, out_path)
    assert out_path.exists()
