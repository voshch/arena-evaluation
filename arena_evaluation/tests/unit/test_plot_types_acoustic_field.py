"""Unit tests for the acoustic-field renderers."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest
from matplotlib.animation import FFMpegWriter
from star_benchmark import write_benchmark

import arena_evaluation.presentation.plot_types.acoustic_field as af_mod
from arena_evaluation.presentation.checks import PlotDiagnostics
from arena_evaluation.presentation.plot_types.acoustic_field import (
    AcousticFieldAnimationRenderer,
    AcousticFieldRenderer,
)
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotSpec,
    VizManifest,
)
from arena_evaluation.processing.acoustics.episode_series import load_aligned_episode
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
    acoustics_times_ns: list[int] | None = None,
    acoustics_levels: list[float] | None = None,
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
        times = acoustics_times_ns or [0, 1_000_000_000, 2_000_000_000]
        levels = acoustics_levels or [60.0, 75.0, 62.0]
        pl.DataFrame(
            {
                "t_ns": times,
                "env_id": ["env_0"] * len(times),
                "robot_id": ["robot_ns"] * len(times),
                "total_level_af_dba": levels,
                "total_level_zf_db": levels,
                "baseline_level_dba": [50.0] * len(times),
                "drivetrain_level_dba": [55.0] * len(times),
                "operating_state": ["moving"] * len(times),
                "calibration_status": ["ok"] * len(times),
                "uncertainty_dba": [1.0] * len(times),
                "validity_flags": [0] * len(times),
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
    worst = renderer.pick_worst_row(_af_df())
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
    worst = renderer.pick_worst_row(df)
    assert worst is not None
    assert worst["robot_x"] == pytest.approx(1.5)
    assert worst["robot_y"] == pytest.approx(2.0)
    assert worst["source_dba"] == pytest.approx(75.0)
    assert len(worst["pedestrians"]) == 1


def test_pick_worst_row_missing_column_or_empty_returns_none(
    tmp_path: pathlib.Path,
) -> None:
    renderer = _new_renderer(_af_spec(), tmp_path)
    assert renderer.pick_worst_row(_af_df().drop("ped_max_exposure_dba")) is None
    assert renderer.pick_worst_row(_af_df().slice(0, 0)) is None


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


def test_parse_pedestrian_positions_keeps_the_pair_list_encoding(
    tmp_path: pathlib.Path,
) -> None:
    renderer = _new_renderer(_af_spec(), tmp_path)
    assert renderer.parse_pedestrian_positions([(0.0, 1.0), (2.0, 3.0)]) == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]
    assert renderer.parse_pedestrian_positions([(1.0, 2.0), (float("nan"), 3.0)]) == [
        (1.0, 2.0)
    ]
    assert renderer.parse_pedestrian_positions([0.0, 1.0, 2.0, 3.0]) == []
    assert renderer.parse_pedestrian_positions([{"x": 1.0, "y": 2.0}]) == []
    assert renderer.parse_pedestrian_positions(None) == []


def test_load_episode_data_reads_aligned_catalog_telemetry(
    tmp_path: pathlib.Path,
) -> None:
    bench = _make_episode(tmp_path, tracks=True)
    ep_df = AcousticFieldRenderer.load_episode_data(bench, "episode_001")
    assert ep_df is not None
    assert ep_df.height == 3
    assert {
        "t_ns",
        "pos_x_gt",
        "pos_y_gt",
        "source_dba",
        "has_collision",
        "peds_positions",
    } <= set(ep_df.columns)
    assert ep_df["has_collision"].to_list() == [False, True, False]


def test_load_episode_data_never_substitutes_odometry(tmp_path: pathlib.Path) -> None:
    bench = _make_episode(tmp_path, odom=True, tracks=False)
    assert AcousticFieldRenderer.load_episode_data(bench, "episode_001") is None


def test_aligned_episode_drops_samples_beyond_the_alignment_tolerance(
    tmp_path: pathlib.Path,
) -> None:
    bench = _make_episode(
        tmp_path,
        tracks=True,
        acoustics_times_ns=[0],
        acoustics_levels=[61.0],
    )
    frame = load_aligned_episode(bench, "episode_001")
    assert frame is not None
    levels = {
        int(row["t_ns"]): row["source_dba"]
        for row in frame.select(["t_ns", "source_dba"]).iter_rows(named=True)
    }
    assert levels[0] == pytest.approx(61.0)
    assert levels[1_000_000_000] is None
    assert levels[2_000_000_000] is None


def test_aligned_episode_has_no_ground_truth_pose_without_robot_tracks(
    tmp_path: pathlib.Path,
) -> None:
    bench = _make_episode(
        tmp_path,
        odom=True,
        tracks=False,
        acoustics=False,
        peds=False,
        collision=False,
    )
    assert load_aligned_episode(bench, "episode_001") is None


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
        AcousticFieldRenderer.load_grid_and_meta("nonexistent_world", run_dir=tmp_path)

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


def test_facet_grid_renders_every_cell_and_static_renders_the_worst_frame(
    tmp_path: pathlib.Path,
) -> None:
    """Grid mode writes one PNG per (planner, stage) facet; static mode renders the worst row."""
    _write_test_world(tmp_path)
    bench = _make_episode(tmp_path, tracks=True)
    _write_test_world(bench)
    faceted = pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb"],
            "stage": ["hall", "door", "hall", "door"],
            "episode_id": [1, 2, 3, 4],
            "world": ["af_test_map"] * 4,
            "ped_max_exposure_dba": [70.0, 74.0, 90.0, 88.0],
            "worst_case_acoustic_t_ns": [1_000_000_000] * 4,
            "robot_x": [1.0] * 4,
            "robot_y": [2.0] * 4,
            "source_dba": [60.0] * 4,
        }
    )

    grid_html = _new_renderer(
        _af_spec(spec_id="af_facets", mode="grid"), tmp_path
    ).render_plotly(faceted)
    assert isinstance(grid_html, str)
    for planner in ("dwb", "teb"):
        for stage in ("hall", "door"):
            png = tmp_path / "plots" / f"af_facets_{planner}_{stage}.png"
            assert png.is_file(), png
            assert png.read_bytes().startswith(_PNG_MAGIC), png

    static_out = tmp_path / "facets_static.png"
    _new_renderer(_af_spec(spec_id="af_facets_static"), bench).render_static(
        faceted, static_out
    )
    assert static_out.read_bytes().startswith(_PNG_MAGIC)


@pytest.mark.skipif(not FFMpegWriter.isAvailable(), reason="ffmpeg is not on PATH")
def test_animation_renderer_writes_an_mp4_when_ffmpeg_is_available(
    tmp_path: pathlib.Path,
) -> None:
    """`format: mp4` produces the movie beside the static PNG and the HTML embeds it."""
    bench = _make_episode(tmp_path, tracks=True)
    _write_test_world(bench)
    spec = PlotSpec(
        id="anim_mp4",
        type="acoustic_field_animation",
        title="Mp4",
        options={"format": "mp4", "fps": 4, "max_frames": 2, "downsample": 2},
    )
    renderer = AcousticFieldAnimationRenderer(spec)
    renderer.run_dir = bench
    static_out = tmp_path / "anim_mp4.png"
    renderer.render_static(_af_df(), static_out)
    assert static_out.read_bytes().startswith(_PNG_MAGIC)
    movie = tmp_path / "anim_mp4.mp4"
    assert movie.is_file() and movie.stat().st_size > 0
    assert movie.read_bytes()[4:8] == b"ftyp"  # ISO base media file signature
    assert '<video src="plots/anim_mp4.mp4"' in renderer.render_plotly(_af_df())


@pytest.mark.skipif(
    FFMpegWriter.isAvailable(),
    reason="ffmpeg is installed, the refusal cannot be observed here",
)
def test_animation_renderer_refuses_mp4_without_ffmpeg(
    tmp_path: pathlib.Path,
) -> None:
    """Requesting mp4 without ffmpeg raises the typed error instead of a backend traceback."""
    bench = _make_episode(tmp_path, tracks=True)
    _write_test_world(bench)
    spec = PlotSpec(
        id="anim_nomp4",
        type="acoustic_field_animation",
        title="No mp4",
        options={"format": "mp4", "fps": 4, "max_frames": 2, "downsample": 2},
    )
    renderer = AcousticFieldAnimationRenderer(spec)
    renderer.run_dir = bench
    with pytest.raises(MetricInputError, match="ffmpeg"):
        renderer.render_static(_af_df(), tmp_path / "anim_nomp4.png")


def test_pick_worst_row_refuses_to_invent_a_source_level(
    tmp_path: pathlib.Path,
) -> None:
    bench = _make_episode(tmp_path, tracks=True, acoustics=False)
    renderer = _new_renderer(_af_spec(), bench)
    df = _af_df().drop(["robot_x", "robot_y", "source_dba"])
    with pytest.raises(MetricInputError, match="refusing to invent a source level"):
        renderer.pick_worst_row(df)


def test_report_builder_writes_the_static_png_for_an_animation_plot(
    tmp_path: pathlib.Path,
) -> None:
    bench = tmp_path / "bench"
    write_benchmark(
        bench,
        [
            {
                "episode_id": 1,
                "planner": "dwb",
                "stage": "hall",
                "world": "af_test_map",
                "success": True,
                "l_aeq_t": 70.0,
                "ped_id": "ped_1",
                "ped_name": "ped_1",
                "ped_max_exposure_dba": 88.0,
                "ped_leq_exposure_dba": 80.0,
                "worst_case_acoustic_t_ns": 1_000_000_000,
            }
        ],
    )
    _make_episode(tmp_path, tracks=True)
    query = (
        "SELECT s.world, s.planner, s.stage, s.episode_id, "
        "max(p.ped_max_exposure_dba) AS ped_max_exposure_dba, "
        "max(p.worst_case_acoustic_t_ns) AS worst_case_acoustic_t_ns "
        "FROM v_episode_summary s JOIN fact_pedestrians p "
        "ON s.run_id = p.run_id AND s.episode_id = p.episode_id "
        "AND s.env_id = p.env_id "
        "GROUP BY s.world, s.planner, s.stage, s.episode_id"
    )
    manifest = VizManifest(
        name="af_report",
        title="Acoustic Field Report",
        plots=[
            PlotSpec(
                id="acoustic_anim",
                type="acoustic_field_animation",
                title="Acoustic animation",
                data={"query": query},
                options={
                    "format": "gif",
                    "fps": 4,
                    "max_frames": 2,
                    "downsample": 2,
                },
            )
        ],
    )
    builder = ReportBuilder(bench, manifest=manifest)
    _write_test_world(builder.output_dir, "af_test_map")
    builder.build()

    static_png = builder.plots_dir / "acoustic_anim.png"
    assert static_png.read_bytes().startswith(_PNG_MAGIC)
    assert (builder.plots_dir / "acoustic_anim.gif").is_file()
    assert builder.build_checks.failed_count == 0


def test_field_timeseries_skips_frames_without_a_recorded_source(
    tmp_path: pathlib.Path,
) -> None:
    grid = np.zeros((4, 4), dtype=np.uint8)
    df = pl.DataFrame(
        {
            "t_ns": [0, 1],
            "pos_x_gt": [1.0, 1.0],
            "pos_y_gt": [1.0, 1.0],
            "source_dba": [70.0, None],
        }
    )
    frames = _new_renderer(_af_spec(), tmp_path).compute_field_timeseries(
        df, grid, 0.1, 0.0, 0.0, {}
    )
    assert frames[0] is not None
    assert frames[1] is None


def test_animation_collision_flags_come_from_recorded_collision_events(
    tmp_path: pathlib.Path,
) -> None:
    field = (np.zeros((2, 2)), 0.1, (2, 2), frozenset(), 70.0)
    renderer = _new_renderer(_af_spec(), tmp_path)
    collided = pl.DataFrame(
        {
            "t_ns": [0],
            "pos_x_gt": [1.0],
            "pos_y_gt": [1.0],
            "source_dba": [70.0],
            "has_collision": [True],
        }
    )
    quiet = pl.DataFrame(
        {
            "t_ns": [0],
            "pos_x_gt": [1.0],
            "pos_y_gt": [1.0],
            "source_dba": [120.0],
            "has_collision": [False],
        }
    )
    assert renderer._animation_metadata(
        collided, [(0, field)], 1, 120
    ).collision_per_frame == [True]
    assert renderer._animation_metadata(
        quiet, [(0, field)], 1, 120
    ).collision_per_frame == [False]
