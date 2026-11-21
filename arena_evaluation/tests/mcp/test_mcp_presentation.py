"""Integration and unit tests for MCP Presentation Gateway (Layer 5)."""

from __future__ import annotations

import os

import pathlib
from typing import Final

import polars as pl
import pytest

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError

from arena_evaluation.mcp.presentation import (
    evaluation_plot_declarations,
    evaluation_read_plot_declaration,
    evaluation_render_plot,
    evaluation_save_plot_declaration,
)
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES

DIM_EPISODES_SCHEMA = DIMENSION_TABLES["dim_episodes"].polars_schema()
DIM_ROBOTS_SCHEMA = DIMENSION_TABLES["dim_robots"].polars_schema()
DIM_PEDESTRIANS_SCHEMA = DIMENSION_TABLES["dim_pedestrians"].polars_schema()
DIM_WORLD_ENTITIES_SCHEMA = DIMENSION_TABLES["dim_world_entities"].polars_schema()
FACT_EPISODES_SCHEMA = FACT_TABLES["episodes"].polars_schema()
FACT_ROBOTS_SCHEMA = {
    **FACT_TABLES["robots"].polars_schema(),
    "success": pl.Boolean,
    "time_to_goal": pl.Float64,
    "path_efficiency": pl.Float64,
    "energy_total_wh": pl.Float64,
    "collision_amount_wall": pl.Int64,
    "collision_amount_static": pl.Int64,
}
FACT_PEDESTRIANS_SCHEMA = FACT_TABLES["pedestrians"].polars_schema()
FACT_ENCOUNTERS_SCHEMA = FACT_TABLES["encounters"].polars_schema()
FACT_CONTACTS_SCHEMA = FACT_TABLES["contacts"].polars_schema()
FACT_COLLISIONS_SCHEMA = FACT_TABLES["collisions"].polars_schema()


def _write_tables(benchmark_dir: pathlib.Path, tables: dict[str, pl.DataFrame]) -> None:
    """Write dimension and per-episode fact tables in canonical Star Schema layout."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for name, table in DIMENSION_TABLES.items():
        frame = tables.get(name, pl.DataFrame(schema=table.polars_schema()))
        frame.write_parquet(benchmark_dir / table.relative_path())
    ep_ids: set[int] = set()
    for name in FACT_TABLES:
        fact_name = f"fact_{name}"
        if (
            fact_name in tables
            and tables[fact_name].height > 0
            and "episode_id" in tables[fact_name].columns
        ):
            ep_ids.update(int(x) for x in tables[fact_name]["episode_id"].to_list())
    if not ep_ids and "dim_episodes" in tables and tables["dim_episodes"].height > 0:
        ep_ids.update(int(x) for x in tables["dim_episodes"]["episode_id"].to_list())
    for ep_id in sorted(ep_ids):
        ep_dir = benchmark_dir / "episodes" / f"episode_{ep_id:03d}"
        (ep_dir / "metrics").mkdir(parents=True, exist_ok=True)
        for name, table in FACT_TABLES.items():
            fact_name = f"fact_{name}"
            frame = tables.get(fact_name, pl.DataFrame(schema=table.polars_schema()))
            if frame.height > 0 and "episode_id" in frame.columns:
                sub = frame.filter(pl.col("episode_id") == ep_id)
            else:
                sub = pl.DataFrame(schema=frame.schema)
            sub.write_parquet(ep_dir / table.relative_path())


_PNG_MAGIC: Final[bytes] = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def mcp_star_tables() -> dict[str, pl.DataFrame]:
    """Build a synthetic dict[str, pl.DataFrame] fixture with episodes, encounters, and contacts."""
    run_id = "run-mcp-pres"

    dim_episodes = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "benchmark_id": run_id,
                "suite_name": "socio_acoustic",
                "contest_name": "nav2_full",
                "stage": "stage_1",
                "world": "hospital_1",
                "scenario": "hallway_encounter",
                "planner": "dwb-standard",
                "inter_planner": "none",
                "dispatcher": "cbs",
                "crowd_condition": "visible",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 42,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 10.0,
                "outcome_state": 2,
                "outcome_info": "finished",
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "benchmark_id": run_id,
                "suite_name": "socio_acoustic",
                "contest_name": "nav2_full",
                "stage": "stage_1",
                "world": "hospital_1",
                "scenario": "hallway_encounter",
                "planner": "mppi",
                "inter_planner": "none",
                "dispatcher": "cbs",
                "crowd_condition": "blind",
                "ped_policy_default": "sfm",
                "crowd_size": 2,
                "seed": 43,
                "is_reference": False,
                "reference_type": None,
                "duration_s": 11.5,
                "outcome_state": 2,
                "outcome_info": "finished",
            },
        ],
        schema=DIMENSION_TABLES["dim_episodes"].polars_schema(),
    )

    fact_episodes = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "robot_id": "robot_0",
                "success": True,
                "collision": False,
                "timeout": False,
                "time_to_goal": 10.0,
                "path_length_m": 8.0,
                "path_efficiency": 0.88,
                "mean_speed_mps": 0.8,
                "mean_accel_mps2": 0.3,
                "mean_jerk_mps3": 0.9,
                "energy_total_j": 400.0,
                "energy_wh": 0.111,
                "acoustic_energy_proxy_db": 60.0,
                "near_miss_count": 1,
                "contact_count": 0,
                "min_pedestrian_clearance_m": 1.0,
                "mean_pedestrian_clearance_m": 1.6,
                "time_under_0_5m_s": 0.0,
                "time_under_1_0m_s": 1.0,
                "time_under_2_0m_s": 4.0,
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "robot_id": "robot_0",
                "success": False,
                "collision": True,
                "timeout": False,
                "time_to_goal": 11.5,
                "path_length_m": 7.6,
                "path_efficiency": 0.74,
                "mean_speed_mps": 0.66,
                "mean_accel_mps2": 0.4,
                "mean_jerk_mps3": 1.1,
                "energy_total_j": 500.0,
                "energy_wh": 0.138,
                "acoustic_energy_proxy_db": 66.0,
                "near_miss_count": 2,
                "contact_count": 1,
                "min_pedestrian_clearance_m": 0.15,
                "mean_pedestrian_clearance_m": 0.9,
                "time_under_0_5m_s": 1.8,
                "time_under_1_0m_s": 3.2,
                "time_under_2_0m_s": 5.5,
            },
        ],
        schema=FACT_EPISODES_SCHEMA,
    )

    fact_encounters = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 0,
                "robot_id": "robot_0",
                "encounter_id": "enc_0_0",
                "actor2_id": "ped_1",
                "approach_type": "head_on",
                "start_t_ns": 1_000_000_000,
                "end_t_ns": 3_500_000_000,
                "duration_s": 2.5,
                "min_clearance_m": 1.0,
                "mean_clearance_m": 1.4,
                "relative_speed_at_min_mps": 1.3,
                "closing_speed_max_mps": 1.5,
                "reference_energy_j": 70.0,
                "energy_surge_ratio": 1.43,
                "detour_time_s": 0.4,
            },
            {
                "run_id": run_id,
                "episode_id": 1,
                "robot_id": "robot_0",
                "encounter_id": "enc_1_0",
                "actor2_id": "ped_2",
                "approach_type": "crossing",
                "start_t_ns": 2_000_000_000,
                "end_t_ns": 5_500_000_000,
                "duration_s": 3.5,
                "min_clearance_m": 0.15,
                "mean_clearance_m": 0.7,
                "relative_speed_at_min_mps": 1.1,
                "closing_speed_max_mps": 1.7,
                "reference_energy_j": 85.0,
                "energy_surge_ratio": 2.23,
                "detour_time_s": 1.5,
            },
        ],
        schema=FACT_ENCOUNTERS_SCHEMA,
    )

    fact_contacts = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "episode_id": 1,
                "robot_id": "robot_0",
                "contact_id": "cnt_1_0",
                "encounter_id": "enc_1_0",
                "actor2_id": "ped_2",
                "timestamp_ns": 4_000_000_000,
                "relative_speed_mps": 1.05,
                "robot_speed_mps": 0.75,
                "actor2_speed_mps": 0.35,
                "robot_closing_share": 0.72,
                "blame_confidence": 0.82,
                "impact_speed_mps": 1.05,
            }
        ],
        schema=FACT_CONTACTS_SCHEMA,
    )

    return dict(
        dim_episodes=dim_episodes,
        dim_robots=pl.DataFrame([], schema=DIM_ROBOTS_SCHEMA),
        dim_pedestrians=pl.DataFrame([], schema=DIM_PEDESTRIANS_SCHEMA),
        dim_world_entities=pl.DataFrame([], schema=DIM_WORLD_ENTITIES_SCHEMA),
        fact_episodes=fact_episodes,
        fact_robots=pl.DataFrame([], schema=FACT_ROBOTS_SCHEMA),
        fact_pedestrians=pl.DataFrame([], schema=FACT_PEDESTRIANS_SCHEMA),
        fact_encounters=fact_encounters,
        fact_contacts=fact_contacts,
    )


def test_mcp_evaluation_plot_declarations() -> None:
    declarations = evaluation_plot_declarations()["declarations"]
    assert isinstance(declarations, list)
    assert len(declarations) >= 2

    ids = {d["id"] for d in declarations}
    assert "scorecard_table" in ids
    assert "scorecard_table" in ids
    assert "encounter_hierarchy_treemap" in ids

    for item in declarations:
        assert "title" in item
        assert "type" in item
        assert "encodings" in item
        assert "options" in item


def test_mcp_evaluation_read_plot_declaration() -> None:
    res = evaluation_read_plot_declaration(name="scorecard_table")
    assert res["name"] == "scorecard_table"
    assert isinstance(res["yaml"], str)
    assert isinstance(res["spec"], dict)
    assert res["spec"]["type"] == "table"

    with pytest.raises(ToolError, match="not found"):
        evaluation_read_plot_declaration(name="unknown_non_existent_recipe")


def test_mcp_evaluation_save_plot_declaration(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "data" / "benchmarks" / "run-pres-save"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")

    new_recipe_yaml = 'id: mcp_custom_density_bar\ntitle: "Encounter Density Profile Across Planners"\ntype: bar\ndata:\n  stored_query: near_miss_encounter_spikes\nchannels:\n  x: approach_type\n  y: mean_energy_surge_ratio\n  color: planner\noptions:\n  barmode: group\n'

    save_res = evaluation_save_plot_declaration(
        name="mcp_custom_density_bar",
        yaml_text=new_recipe_yaml,
        run_id="run-pres-save",
    )
    assert save_res["status"] == "saved"
    assert save_res["name"] == "mcp_custom_density_bar"
    assert save_res["id"] == "mcp_custom_density_bar"
    assert save_res["type"] == "bar"

    # Verify readable via read tool
    read_res = evaluation_read_plot_declaration(
        "mcp_custom_density_bar", run_id="run-pres-save"
    )
    assert read_res["spec"]["id"] == "mcp_custom_density_bar"
    assert read_res["spec"]["type"] == "bar"

    # Error case: Invalid YAML
    with pytest.raises(ToolError, match="not valid YAML"):
        evaluation_save_plot_declaration(name="invalid_yaml", yaml_text="foo: [bar: 1")


def test_mcp_evaluation_render_plot_recipe(
    tmp_path: pathlib.Path,
    mcp_star_tables: dict[str, pl.DataFrame],
) -> None:
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-001"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(run_dir, mcp_star_tables)

    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")

    # Render recipe to PNG
    res_png = evaluation_render_plot(
        run_id="run-pres-001",
        declaration="scorecard_table",
        output_format="png",
    )
    from mcp.types import ImageContent

    assert any(
        isinstance(c, ImageContent) and c.mime_type == "image/png"
        for c in res_png.content
    )
    sc_png = res_png.structured_content
    assert sc_png is not None
    assert sc_png["format"] == "png"
    assert sc_png["spec_id"] == "scorecard_table"
    assert "diagnostics" in sc_png
    png_path = pathlib.Path(sc_png["plot_path"])
    assert "analysis" in png_path.parts
    assert png_path.exists() and png_path.stat().st_size > 0
    assert png_path.read_bytes()[:8] == _PNG_MAGIC

    # Render recipe to HTML (Plotly standalone)
    res_html = evaluation_render_plot(
        run_id="run-pres-001",
        declaration="scorecard_table",
        output_format="html",
    )
    sc_html = res_html.structured_content
    assert sc_html is not None and sc_html["format"] == "html"
    html_path = pathlib.Path(sc_html["plot_path"])
    assert html_path.exists() and html_path.stat().st_size > 0
    html_text = html_path.read_text(encoding="utf-8")
    assert "plotly" in html_text.lower()
    assert '<script src="https://cdn.plot.ly' not in html_text
    assert len(html_text) > 100_000
    assert "Planner Scorecard" in html_text

    # Render recipe to SVG (Vector)
    res_svg = evaluation_render_plot(
        run_id="run-pres-001",
        declaration="scorecard_table",
        output_format="svg",
    )
    sc_svg = res_svg.structured_content
    assert sc_svg is not None and sc_svg["format"] == "svg"
    svg_path = pathlib.Path(sc_svg["plot_path"])
    assert svg_path.exists() and svg_path.stat().st_size > 0
    assert "<svg" in svg_path.read_text(encoding="utf-8")


def test_mcp_evaluation_render_plot_inline_yaml_and_overrides(
    tmp_path: pathlib.Path,
    mcp_star_tables: dict[str, pl.DataFrame],
) -> None:
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-002"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(run_dir, mcp_star_tables)

    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")

    inline_spec = """
    id: inline_scorecard_bar
    title: "Scorecard Mean Time to Goal"
    type: bar
    data:
      stored_query: planner_scorecard_overview
    channels:
      x: planner
      y: mean_time_to_goal
    """

    res = evaluation_render_plot(
        run_id="run-pres-002",
        spec_yaml=inline_spec,
        overrides={"title": "Custom Overridden Scorecard Title"},
        output_format="png",
    )
    sc = res.structured_content
    assert sc is not None and sc["spec_id"] == "inline_scorecard_bar"
    out_file = pathlib.Path(sc["plot_path"])
    assert out_file.exists() and out_file.stat().st_size > 0


def test_mcp_evaluation_render_plot_error_handling(
    tmp_path: pathlib.Path,
    mcp_star_tables: dict[str, pl.DataFrame],
) -> None:
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-003"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(run_dir, mcp_star_tables)
    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")

    # Error: Neither declaration nor spec_yaml
    with pytest.raises(
        ToolError, match="Must provide either 'declaration' or 'spec_yaml'"
    ):
        evaluation_render_plot(run_id="run-pres-003")

    # Error: Both declaration and spec_yaml
    with pytest.raises(
        ToolError, match="Cannot provide both 'declaration' and 'spec_yaml'"
    ):
        evaluation_render_plot(
            run_id="run-pres-003", declaration="foo", spec_yaml="bar"
        )

    # Error: Non-existent declaration
    with pytest.raises(
        ToolError, match="Plot declaration 'non_existent_rec' not found"
    ):
        evaluation_render_plot(run_id="run-pres-003", declaration="non_existent_rec")


def test_mcp_evaluation_render_plot_zero_row_diagnostic(
    tmp_path: pathlib.Path,
    mcp_star_tables: dict[str, pl.DataFrame],
) -> None:
    """Rendering a plot whose query returns 0 rows raises a clear ToolError diagnostic."""
    benchmarks_dir = tmp_path / "data" / "benchmarks"
    run_dir = benchmarks_dir / "run-pres-empty"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(run_dir, mcp_star_tables)
    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")

    spec_with_empty_filter = """
    id: empty_filter_plot
    title: "Empty Filter Plot"
    type: bar
    data:
      source: v_episode_summary
      where: "planner = 'non_existent_planner'"
    channels:
      x: planner
      y: mean_time_to_goal
    """

    with pytest.raises(ToolError) as exc_info:
        evaluation_render_plot(
            run_id="run-pres-empty",
            spec_yaml=spec_with_empty_filter,
            output_format="png",
        )

    err = exc_info.value
    assert "returned 0 rows; cannot render plot with an empty dataset" in str(err)
    assert (
        "Verify filter conditions or run 'evaluation_query' to check row counts."
        in (err.hint or "")
    )
