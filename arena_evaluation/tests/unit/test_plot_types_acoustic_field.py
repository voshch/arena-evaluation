"""Unit tests for the acoustic-field renderers."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
import pytest

import arena_evaluation.presentation.plot_types.acoustic_field as af_mod
from arena_evaluation.presentation.checks import PlotDiagnostics
from arena_evaluation.presentation.plot_types.acoustic_field import (
    AcousticFieldAnimationRenderer,
    AcousticFieldRenderer,
)
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec
from arena_evaluation.storage.exceptions import MetricInputError

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_TEST_WORLD_YAML = """format: arena-world/2
levels:
  - id: L1
    zones:
      - name: floor
        polygon:
          - [0.0, 0.0]
          - [4.0, 0.0]
          - [4.0, 4.0]
          - [0.0, 4.0]
    walls:
      - path: [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
      - path: [[2.0, 0.0], [2.0, 4.0]]
    doors:
      - name: d1
        kind: sliding
        span: [[2.0, 1.5], [2.0, 2.5]]
        width: 0.2
"""


def _write_test_world(run_dir: pathlib.Path, world_name: str = "af_test_map") -> None:
    world_dir = run_dir / "worlds" / world_name
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / "world.yaml").write_text(_TEST_WORLD_YAML, encoding="utf-8")


def _af_spec(
    spec_id: str = "af_test",
    facet_row: str | None = "planner",
    facet_col: str | None = "stage",
    **options: object,
) -> PlotSpec:
    return PlotSpec(
        id=spec_id,
        type="acoustic_field",
        title="Acoustic Field",
        channels=PlotChannels(facet_row=facet_row, facet_col=facet_col),
        options=dict(options),
    )


def _new_renderer(spec: PlotSpec, run_dir: pathlib.Path) -> AcousticFieldRenderer:
    renderer = AcousticFieldRenderer(spec)
    renderer.run_dir = run_dir
    return renderer


def _af_df(**overrides: object) -> pl.DataFrame:
    rows: dict[str, object] = {
        "planner": ["dwb", "teb"],
        "stage": ["hall", "hall"],
        "episode_id": [2, 1],
        "world": ["af_test_map", "af_test_map"],
        "ped_max_exposure_dba": [82.0, 105.0],
        "worst_case_acoustic_t_ns": [1_000_000_000, 1_000_000_000],
        "robot_x": [1.0, 1.0],
        "robot_y": [2.0, 2.0],
        "source_dba": [60.0, 68.0],
    }
    rows.update(overrides)
    return pl.DataFrame(rows)


def _make_episode(
    tmp_path: pathlib.Path,
    *,
    odom: bool = True,
    tracks: bool = False,
    acoustics: bool = True,
    peds: bool = True,
    collision: bool = True,
) -> pathlib.Path:
    bench = tmp_path / "bench"
    topics = bench / "episodes" / "episode_001" / "topics"
    topics.mkdir(parents=True, exist_ok=True)
    if tracks:
        pl.DataFrame(
            {
                "t_ns": [0, 1_000_000_000, 2_000_000_000],
                "env_id": ["env_0"] * 3,
                "robot_id": ["robot_ns"] * 3,
                "x": [1.0, 1.5, 2.0],
                "y": [2.0, 2.0, 2.0],
                "z": [0.0] * 3,
                "yaw": [0.0] * 3,
                "vx": [0.5] * 3,
                "vy": [0.0] * 3,
                "vz": [0.0] * 3,
                "omega_z": [0.0] * 3,
                "speed_mps": [0.5] * 3,
                "pose_source": ["gt"] * 3,
            }
        ).write_parquet(topics / "robot_tracks.parquet")
    if odom:
        pl.DataFrame(
            {
                "t_ns": [0, 1_000_000_000, 2_000_000_000],
                "env_id": ["env_0"] * 3,
                "robot_id": ["robot_ns"] * 3,
                "odom_x": [1.0, 1.5, 2.0],
                "odom_y": [2.0, 2.0, 2.0],
                "odom_yaw": [0.0] * 3,
                "odom_vx": [0.5] * 3,
                "odom_wz": [0.0] * 3,
                "drift_error_m": [0.0] * 3,
            }
        ).write_parquet(topics / "robot_odom.parquet")
    if acoustics:
        pl.DataFrame(
            {
                "t_ns": [0, 1_000_000_000, 2_000_000_000],
                "env_id": ["env_0"] * 3,
                "robot_id": ["robot_ns"] * 3,
                "total_level_af_dba": [60.0, 75.0, 62.0],
                "total_level_zf_db": [60.0, 75.0, 62.0],
                "baseline_level_dba": [50.0] * 3,
                "drivetrain_level_dba": [55.0] * 3,
                "operating_state": ["moving", "collision", "moving"],
                "calibration_status": ["ok"] * 3,
                "uncertainty_dba": [1.0] * 3,
                "validity_flags": [0] * 3,
            }
        ).write_parquet(topics / "robot_acoustics.parquet")
    if peds:
        pl.DataFrame(
            {
                "t_ns": [0, 1_000_000_000, 2_000_000_000],
                "env_id": ["env_0"] * 3,
                "ped_id": ["ped_1"] * 3,
                "ped_name": ["ped_1"] * 3,
                "x": [3.0, 3.1, 3.2],
                "y": [2.0, 2.0, 2.0],
                "z": [0.0] * 3,
                "yaw": [0.0] * 3,
                "vx": [0.1] * 3,
                "vy": [0.0] * 3,
                "vz": [0.0] * 3,
                "speed_mps": [0.1] * 3,
                "radius_m": [0.3] * 3,
                "desired_speed": [0.1] * 3,
                "policy": ["sfm"] * 3,
                "agent_type": ["adult"] * 3,
                "animation_state": [0] * 3,
                "interaction_id": [-1] * 3,
                "interaction_type": [0] * 3,
                "gait_phase": [0.0] * 3,
                "gait_cadence": [0.0] * 3,
            }
        ).write_parquet(topics / "pedestrian_arena_peds.parquet")
    if collision:
        pl.DataFrame(
            {
                "t_ns": [0, 1_000_000_000, 2_000_000_000],
                "env_id": ["env_0"] * 3,
                "robot_id": ["robot_ns"] * 3,
                "event_count": [0, 1, 0],
                "wall_count": [0, 0, 0],
                "static_count": [0, 0, 0],
                "ped_count": [0, 1, 0],
            }
        ).write_parquet(topics / "robot_collision_events.parquet")
    return bench


def test_grid_extent_starts_half_a_source_cell_below_the_origin() -> None:
    assert af_mod._cell_extent(
        origin_m=1.0, cells=10, cell_m=0.05, source_res_m=0.05
    ) == pytest.approx((0.975, 1.475))


def test_downsampled_grid_still_starts_half_a_source_cell_below_the_origin() -> None:
    assert af_mod._cell_extent(
        origin_m=0.0, cells=5, cell_m=0.1, source_res_m=0.05
    ) == pytest.approx((-0.025, 0.475))


def test_contour_centres_sit_in_the_middle_of_each_cell() -> None:
    assert af_mod._cell_centres(lower_m=0.975, cells=4, cell_m=0.05) == pytest.approx(
        [1.0, 1.05, 1.1, 1.15]
    )


def test_pick_worst_row_happy_with_prejoined_pose(tmp_path: pathlib.Path) -> None:
    renderer = _new_renderer(_af_spec(), tmp_path)
    worst = renderer._pick_worst_row(_af_df())
    assert worst is not None
    assert worst["robot_x"] == 1.0 and worst["robot_y"] == 2.0
    assert worst["source_dba"] == 68.0
    assert worst["ped_max_exposure_dba"] == 105.0
    assert worst["planner"] == "teb"
    assert worst["stage"] == "hall"


def test_pick_worst_row_resolves_pose_from_catalog_telemetry(
    tmp_path: pathlib.Path,
) -> None:
    bench = _make_episode(tmp_path, tracks=True)
    df = _af_df().drop(["robot_x", "robot_y", "source_dba"])
    renderer = _new_renderer(_af_spec(), bench)
    worst = renderer._pick_worst_row(df)
    assert worst is not None
    assert worst["robot_x"] == pytest.approx(1.5)
    assert worst["robot_y"] == pytest.approx(2.0)
    assert worst["source_dba"] == pytest.approx(75.0)
    assert len(worst["pedestrians"]) == 1


def test_pick_worst_row_missing_column_or_empty_returns_none(
    tmp_path: pathlib.Path,
) -> None:
    renderer = _new_renderer(_af_spec(), tmp_path)
    assert renderer._pick_worst_row(_af_df().drop("ped_max_exposure_dba")) is None
    assert renderer._pick_worst_row(_af_df().slice(0, 0)) is None


def test_prepared_df_excludes_reference_runs_by_default(
    tmp_path: pathlib.Path,
) -> None:
    df = _af_df().with_columns(pl.Series("is_reference", [True, False]))
    out = _new_renderer(_af_spec(), tmp_path)._prepared_df(df)
    assert out["planner"].to_list() == ["teb"]
    kept = _new_renderer(_af_spec(include_reference=True), tmp_path)._prepared_df(df)
    assert kept.height == 2


def test_facet_axes_and_filter_facet(tmp_path: pathlib.Path) -> None:
    renderer = _new_renderer(_af_spec(), tmp_path)
    row_col, col_col, row_vals, col_vals = renderer._facet_axes(_af_df())
    assert (row_col, col_col) == ("planner", "stage")
    assert row_vals == ["dwb", "teb"]
    assert col_vals == ["hall"]
    filtered = renderer._filter_facet(_af_df(), row_col, col_col, "dwb", "hall")
    assert filtered["planner"].to_list() == ["dwb"]


def test_parse_pedestrian_positions_variants(tmp_path: pathlib.Path) -> None:
    renderer = _new_renderer(_af_spec(), tmp_path)
    assert renderer._parse_pedestrian_positions([[0.0, 1.0], [2.0, 3.0]]) == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]
    assert renderer._parse_pedestrian_positions([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) == [
        (0.0, 1.0),
        (3.0, 4.0),
    ]
    assert renderer._parse_pedestrian_positions(
        [{"x": 1.0, "y": 2.0}, {"x": float("nan"), "y": 3.0}]
    ) == [(1.0, 2.0)]
    assert renderer._parse_pedestrian_positions(json.dumps([[1.0, 2.0]])) == [
        (1.0, 2.0)
    ]
    assert renderer._parse_pedestrian_positions("not json") == []
    assert renderer._parse_pedestrian_positions(None) == []


def test_load_episode_data_reads_catalog_telemetry(tmp_path: pathlib.Path) -> None:
    bench = _make_episode(tmp_path, odom=True, tracks=False)
    ep_df = AcousticFieldRenderer._load_episode_data(bench, "episode_001")
    assert ep_df is not None
    assert ep_df.height == 3
    assert {"t_ns", "pos_x_gt", "pos_y_gt", "source_dba", "has_collision"} <= set(
        ep_df.columns
    )
    assert ep_df["has_collision"].to_list() == [False, True, False]


def test_load_episode_data_missing_pose_returns_none(tmp_path: pathlib.Path) -> None:
    bench = _make_episode(tmp_path, odom=False, tracks=False)
    assert AcousticFieldRenderer._load_episode_data(bench, "episode_001") is None


def test_render_plotly_single_and_grid_write_pngs(tmp_path: pathlib.Path) -> None:
    _write_test_world(tmp_path)
    single_html = _new_renderer(
        _af_spec(spec_id="af_single", mode="single"), tmp_path
    ).render_plotly(_af_df())
    assert isinstance(single_html, str)
    assert "plots/af_single.png" in single_html
    single_png = tmp_path / "plots" / "af_single.png"
    assert single_png.read_bytes().startswith(_PNG_MAGIC)

    grid_html = _new_renderer(
        _af_spec(spec_id="af_grid", mode="grid"), tmp_path
    ).render_plotly(_af_df())
    assert isinstance(grid_html, str)
    assert "plots/af_grid_dwb_hall.png" in grid_html
    assert "plots/af_grid_teb_hall.png" in grid_html
    assert (
        (tmp_path / "plots" / "af_grid_dwb_hall.png")
        .read_bytes()
        .startswith(_PNG_MAGIC)
    )


def test_render_static_writes_png_with_trajectory_overlay(
    tmp_path: pathlib.Path,
) -> None:
    bench = _make_episode(tmp_path, tracks=True)
    _write_test_world(bench)
    out_png = tmp_path / "static.png"
    renderer = _new_renderer(_af_spec(overlay_trajectories=True), bench)
    renderer.render_static(_af_df().drop(["robot_x", "robot_y", "source_dba"]), out_png)
    assert out_png.read_bytes().startswith(_PNG_MAGIC)


def test_missing_world_raises_metric_input_error_and_records_notice(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(MetricInputError, match="missing world map"):
        AcousticFieldRenderer._load_grid_and_meta("nonexistent_world", run_dir=tmp_path)

    renderer = _new_renderer(_af_spec(), tmp_path)
    diag = PlotDiagnostics()
    renderer.diagnostics = diag
    assert renderer.render_plotly(_af_df()) == ""
    assert any("missing world map" in issue for issue in diag.issues)


def test_unknown_option_rejected_by_validate_frame(tmp_path: pathlib.Path) -> None:
    renderer = _new_renderer(_af_spec(bogus_option=1), tmp_path)
    with pytest.raises(ValueError, match="unknown option"):
        renderer.validate_frame(_af_df())


def test_animation_renderer_renders_gif_and_frames(tmp_path: pathlib.Path) -> None:
    bench = _make_episode(tmp_path, tracks=True)
    _write_test_world(bench)
    spec_gif = PlotSpec(
        id="anim_gif",
        type="acoustic_field_animation",
        title="Anim",
        options={"format": "gif", "fps": 4, "max_frames": 2, "downsample": 2},
    )
    renderer = AcousticFieldAnimationRenderer(spec_gif)
    renderer.run_dir = bench
    out_gif = tmp_path / "out.gif"
    renderer.render_static(_af_df(), out_gif)
    assert out_gif.is_file()
    assert out_gif.stat().st_size > 0
    html = renderer.render_plotly(_af_df())
    assert isinstance(html, str) and '<img src="plots/anim_gif.gif"' in html

    spec_frames = PlotSpec(
        id="anim_frames",
        type="acoustic_field_animation",
        title="Frames",
        options={"format": "frames", "fps": 4, "max_frames": 2, "downsample": 2},
    )
    renderer_frames = AcousticFieldAnimationRenderer(spec_frames)
    renderer_frames.run_dir = bench
    frames_out = tmp_path / "frames_out"
    renderer_frames.render_static(_af_df(), frames_out)
    assert (frames_out / "frame_0000.png").is_file()
