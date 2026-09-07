"""Unit tests for radar charts and the acoustic-field renderers."""

import json
import pathlib

import matplotlib.animation
import numpy as np
import polars as pl
import pytest

import arena_evaluation.presentation.plot_types.acoustic_field as af_mod
from arena_evaluation.presentation.plot_types.acoustic_field import (
    AcousticFieldAnimationRenderer,
    AcousticFieldRenderer,
)
from arena_evaluation.presentation.plot_types.radar import RadarRenderer
from arena_evaluation.storage.schemas import PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ═══════════════════════════════════════════════════════════════════════════
# RadarRenderer
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_METRICS = [
    "path_efficiency",
    "time_to_goal",
    "collision_amount",
    "roughness_mean",
    "jerk_mean",
]


def _radar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "success": [0.8, 1.0],
            "path_efficiency": [0.5, 0.9],
            "collision_amount": [1.0, 2.0],
            "time_to_goal": [10.0, 20.0],
            "roughness_mean": [3.0, 4.0],
        },
        schema={
            "planner": pl.Utf8,
            "success": pl.Float64,
            "path_efficiency": pl.Float64,
            "collision_amount": pl.Float64,
            "time_to_goal": pl.Float64,
            "roughness_mean": pl.Float64,
        },
    )


def _radar_spec(**options) -> PlotSpec:
    return PlotSpec(id="rad", type="radar", title="Radar Overview", data_key="*", differentiate="planner", options=options)


def test_radar_plotly_happy_path():
    html = RadarRenderer(_radar_spec()).render_plotly(_radar_df())
    assert html is not None
    assert html.startswith("<div")
    assert "scatterpolar" in html


def test_radar_plotly_custom_metrics():
    spec = _radar_spec(metrics=["success", "path_efficiency", "collision_amount"])
    html = RadarRenderer(spec).render_plotly(_radar_df())
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_fewer_than_three_metrics_returns_none():
    spec = _radar_spec(metrics=["success", "path_efficiency"])
    assert RadarRenderer(spec).render_plotly(_radar_df()) is None


def test_radar_plotly_null_metrics_are_filtered_out():
    df = _radar_df().with_columns(pl.lit(None).alias("bogus_a"), pl.lit(None).alias("bogus_b"))
    spec = _radar_spec(metrics=["success", "bogus_a", "bogus_b"])
    assert RadarRenderer(spec).render_plotly(df) is None


def test_radar_plotly_missing_diff_col_returns_none():
    df = _radar_df().drop("planner")
    assert RadarRenderer(_radar_spec()).render_plotly(df) is None


def test_radar_plotly_all_null_diff_col_renders_null_group():
    # polars group_by keeps null keys as their own group, so a fully-null
    # differentiate column still aggregates and renders (trace named None).
    df = _radar_df().with_columns(pl.lit(None).alias("planner"))
    html = RadarRenderer(_radar_spec()).render_plotly(df)
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_all_zero_metric_normalizes_to_one():
    df = _radar_df().with_columns(pl.lit(0.0).alias("success"))
    spec = _radar_spec(metrics=["success", "path_efficiency", "collision_amount"])
    html = RadarRenderer(spec).render_plotly(df)
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_positive_metric_scales_by_max():
    # success values [0.8, 1.0] -> [0.8, 1.0] (val / max)
    spec = _radar_spec(metrics=["success", "path_efficiency", "collision_amount"])
    html = RadarRenderer(spec).render_plotly(_radar_df())
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_negative_metric_with_positive_min_inverts():
    # collision_amount [1.0, 2.0] -> [1.0, 0.5] (min / val)
    spec = _radar_spec(metrics=["success", "path_efficiency", "collision_amount"])
    html = RadarRenderer(spec).render_plotly(_radar_df())
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_metric_crossing_zero_uses_complement():
    df = _radar_df().with_columns(pl.Series("collision_amount", [-1.0, 1.0]))
    spec = _radar_spec(metrics=["success", "path_efficiency", "collision_amount"])
    html = RadarRenderer(spec).render_plotly(df)
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_log_scale_option():
    spec = _radar_spec(use_log_scale=True)
    html = RadarRenderer(spec).render_plotly(_radar_df())
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_single_planner_row():
    df = _radar_df().slice(0, 1)
    html = RadarRenderer(_radar_spec()).render_plotly(df)
    assert html is not None and "dwb" in html


def test_radar_plotly_units_in_labels():
    html = RadarRenderer(_radar_spec(), units={"collision_amount": "n"}).render_plotly(_radar_df())
    assert html is not None
    assert "Collision Amount [n]" in html


def test_radar_seaborn_happy_path(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec()).render_seaborn(_radar_df(), out)
    assert out.exists() and out.read_bytes()[:8] == _PNG_MAGIC


def test_radar_seaborn_missing_diff_writes_nothing(tmp_path):
    out = tmp_path / "radar.png"
    df = _radar_df().drop("planner")
    RadarRenderer(_radar_spec()).render_seaborn(df, out)
    assert not out.exists()


def test_radar_seaborn_too_few_metrics_writes_nothing(tmp_path):
    out = tmp_path / "radar.png"
    spec = _radar_spec(metrics=["success", "path_efficiency"])
    RadarRenderer(spec).render_seaborn(_radar_df(), out)
    assert not out.exists()


def test_radar_seaborn_log_scale(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec(use_log_scale=True)).render_seaborn(_radar_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_radar_seaborn_metric_crossing_zero(tmp_path):
    out = tmp_path / "radar.png"
    df = _radar_df().with_columns(pl.Series("collision_amount", [-1.0, 1.0]))
    RadarRenderer(_radar_spec()).render_seaborn(df, out)
    assert out.exists() and out.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════
# AcousticFieldRenderer — helpers
# ═══════════════════════════════════════════════════════════════════════════


def _grid(n: int = 20) -> np.ndarray:
    grid = np.zeros((n, n), dtype=np.uint8)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[:, n // 2] = 1
    return grid


def _meta(n: int = 20) -> dict:
    return {"png_path": "synthetic", "resolution": 0.1, "origin": [0.0, 0.0, 0.0], "width": n, "height": n}


def _door_mask(n: int = 20) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[5:7, n // 2] = True
    return mask


def _patch_grid(monkeypatch, n: int = 20) -> np.ndarray:
    grid = _grid(n)
    monkeypatch.setattr(
        af_mod.AcousticFieldRenderer,
        "_load_grid_and_meta",
        staticmethod(lambda map_name, run_dir=None: (grid, _meta(n))),
    )
    return grid


def _patch_doors(monkeypatch, doors=None):
    monkeypatch.setattr(af_mod, "door_segments", lambda *a, **k: doors or {})


def _af_df(**overrides) -> pl.DataFrame:
    rows = {
        "planner": ["dwb", "teb"],
        "stage": ["hall", "hall"],
        "episode": [2, 1],  # worst-exposure row (teb) is episode 1 -> episode_001
        "map": ["af_test_map", "af_test_map"],
        "ped_max_exposure_dba": [82.0, 105.0],
        "worst_case_acoustic_frame": [
            {"robot_x": 1.0, "robot_y": 1.0, "source_dba": 60.0, "pedestrians": [[0.5, 0.5]], "door_states": {"world/d1": "closed"}},
            {"robot_x": 2.0, "robot_y": 2.0, "source_dba": 60.0, "pedestrians": [], "door_states": {"world/d1": "open"}},
        ],
    }
    rows.update(overrides)
    schema = {
        "planner": pl.Utf8,
        "stage": pl.Utf8,
        "episode": pl.Int64,
        "map": pl.Utf8,
        "ped_max_exposure_dba": pl.Float64,
        "worst_case_acoustic_frame": pl.Object,
    }
    return pl.DataFrame(rows, schema=schema)


def _af_spec(spec_id: str = "af_test", differentiate="planner", group_by=None, **options) -> PlotSpec:
    if group_by is None:
        group_by = ["stage"]
    return PlotSpec(id=spec_id, type="acoustic_field", title="Acoustic Field", data_key="ped_max_exposure_dba", differentiate=differentiate, group_by=group_by, options=options)


def _new_renderer(spec: PlotSpec, run_dir: pathlib.Path) -> AcousticFieldRenderer:
    renderer = AcousticFieldRenderer(spec)
    renderer.run_dir = run_dir
    return renderer


# ── _pick_worst_row ────────────────────────────────────────────────────────


def _worst_frame_df(**overrides) -> pl.DataFrame:
    rows = {
        "planner": ["dwb", "teb"],
        "stage": ["hall", "hall"],
        "ped_max_exposure_dba": [82.0, 105.0],
        # the worst-exposure row (teb) carries the usable frame
        "worst_case_acoustic_frame": [None, {"robot_x": 1.0, "robot_y": 1.0}],
    }
    rows.update(overrides)
    return pl.DataFrame(
        rows,
        schema={
            "planner": pl.Utf8,
            "stage": pl.Utf8,
            "ped_max_exposure_dba": pl.Float64,
            "worst_case_acoustic_frame": pl.Object,
        },
    )


def test_pick_worst_row_happy():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    worst = renderer._pick_worst_row(_worst_frame_df())
    assert worst is not None
    assert worst["robot_x"] == 1.0 and worst["robot_y"] == 1.0
    assert worst["source_dba"] == 60.0
    assert worst["pedestrians"] == []
    assert worst["ped_max_exposure_dba"] == 105.0
    assert worst["planner"] == "teb"
    assert worst["stage"] == "hall"


def test_pick_worst_row_missing_column_returns_none():
    df = _worst_frame_df().drop("ped_max_exposure_dba")
    assert _new_renderer(_af_spec(), pathlib.Path("."))._pick_worst_row(df) is None


def test_pick_worst_row_empty_df_returns_none():
    df = _worst_frame_df().slice(0, 0)
    assert _new_renderer(_af_spec(), pathlib.Path("."))._pick_worst_row(df) is None


def test_pick_worst_row_null_frame_returns_none():
    df = _worst_frame_df()
    df = df.with_columns(pl.lit(None).alias("worst_case_acoustic_frame"))
    assert _new_renderer(_af_spec(), pathlib.Path("."))._pick_worst_row(df) is None


def test_pick_worst_row_frame_without_robot_x_returns_none():
    df = _worst_frame_df()
    df = df.with_columns(pl.lit({"robot_y": 5.0}).alias("worst_case_acoustic_frame"))
    assert _new_renderer(_af_spec(), pathlib.Path("."))._pick_worst_row(df) is None


def test_pick_worst_row_json_string_frame():
    df = _worst_frame_df()
    df = df.with_columns(
        pl.Series(
            "worst_case_acoustic_frame",
            [
                None,
                json.dumps({"robot_x": 3.0, "robot_y": 4.0, "source_dba": 70.0}),
            ],
        )
    )
    worst = _new_renderer(_af_spec(), pathlib.Path("."))._pick_worst_row(df)
    assert worst is not None
    assert worst["robot_x"] == 3.0
    assert worst["source_dba"] == 70.0


def test_pick_worst_row_bad_json_returns_none():
    df = _worst_frame_df()
    df = df.with_columns(pl.Series("worst_case_acoustic_frame", ["not json", None]))
    assert _new_renderer(_af_spec(), pathlib.Path("."))._pick_worst_row(df) is None


def test_pick_worst_row_planner_fallback_to_local_planner():
    df = _worst_frame_df()
    df = df.drop("planner").with_columns(pl.Series("local_planner", ["dwb", "teb"]))
    worst = _new_renderer(_af_spec(), pathlib.Path("."))._pick_worst_row(df)
    assert worst["planner"] == "teb"


# ── _prepared_df ───────────────────────────────────────────────────────────


def test_prepared_df_no_reference_column_unchanged():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    out = renderer._prepared_df(_af_df())
    assert out.height == 2


def test_prepared_df_excludes_reference_runs_by_default():
    df = _af_df().with_columns(pl.Series("is_reference", [True, False]))
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    out = renderer._prepared_df(df)
    assert out["planner"].to_list() == ["teb"]


def test_prepared_df_keeps_reference_when_requested():
    df = _af_df().with_columns(pl.Series("is_reference", [True, False]))
    renderer = _new_renderer(_af_spec(include_reference=True), pathlib.Path("."))
    out = renderer._prepared_df(df)
    assert out.height == 2


def test_prepared_df_all_reference_rows_removed():
    df = _af_df().with_columns(pl.Series("is_reference", [True, True]))
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer._prepared_df(df).height == 0


# ── _group_values / _filter_group ──────────────────────────────────────────


def test_group_values_with_diff_and_group():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    diff_col, group_cols, row_values, col_values = renderer._group_values(_af_df())
    assert diff_col == "planner"
    assert group_cols == ["stage"]
    assert row_values == ["dwb", "teb"]
    assert col_values == [("hall",)]


def test_group_values_no_diff_uses_single_row():
    spec = _af_spec(differentiate=None)
    renderer = _new_renderer(spec, pathlib.Path("."))
    diff_col, group_cols, row_values, col_values = renderer._group_values(_af_df())
    assert diff_col is None
    assert row_values == [""]
    assert col_values == [("hall",)]


def test_group_values_no_group_cols_uses_single_col():
    spec = _af_spec(group_by=[])
    renderer = _new_renderer(spec, pathlib.Path("."))
    _, _, row_values, col_values = renderer._group_values(_af_df())
    assert col_values == [()]


def test_group_values_missing_group_col_dropped():
    spec = _af_spec(group_by=["bogus", "stage"])
    renderer = _new_renderer(spec, pathlib.Path("."))
    _, group_cols, _, col_values = renderer._group_values(_af_df())
    assert group_cols == ["stage"]


def test_filter_group_single_and_multi_columns():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    out = renderer._filter_group(_af_df(), "planner", ["stage"], "dwb", ("hall",))
    assert out["planner"].to_list() == ["dwb"]
    out = renderer._filter_group(_af_df(), None, ["stage"], "", ("hall",))
    assert out.height == 2
    out = renderer._filter_group(_af_df(), "planner", [], "teb", ())
    assert out["planner"].to_list() == ["teb"]


# ── _parse_pedestrian_positions ────────────────────────────────────────────


def test_parse_pedestrian_positions_nested_and_flat():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer._parse_pedestrian_positions([[0.0, 1.0], [2.0, 3.0]]) == [(0.0, 1.0), (2.0, 3.0)]
    assert renderer._parse_pedestrian_positions([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) == [(0.0, 1.0), (3.0, 4.0)]


def test_parse_pedestrian_positions_skips_nan_and_short_entries():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer._parse_pedestrian_positions([[float("nan"), 1.0], [2.0, 3.0]]) == [(2.0, 3.0)]
    assert renderer._parse_pedestrian_positions([[1.0]]) == []
    # Flat triplet scan: leading NaN triplet dropped; trailing 2-tuple incomplete.
    assert renderer._parse_pedestrian_positions([float("nan"), 1.0, 2.0, 3.0]) == []


def test_parse_pedestrian_positions_json_string_and_empty():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer._parse_pedestrian_positions(json.dumps([[1.0, 2.0]])) == [(1.0, 2.0)]
    assert renderer._parse_pedestrian_positions("not json") == []
    assert renderer._parse_pedestrian_positions([]) == []
    assert renderer._parse_pedestrian_positions(None) == []


# ── _load_episode_data ─────────────────────────────────────────────────────


def _make_episode(tmp_path: pathlib.Path, odom=True, tf_gt=False, acoustics=True, acoustic_alt=False, peds=True, collision=True) -> pathlib.Path:
    bench = tmp_path / "bench"
    topics = bench / "episodes" / "episode_001" / "topics"
    robot = topics / "robot_ns"
    robot.mkdir(parents=True, exist_ok=True)
    if odom:
        pl.DataFrame({"time_ns": [0, 1, 2], "pos_x": [1.0, 1.5, 2.0], "pos_y": [1.0, 1.0, 1.0]}, schema={"time_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64}).write_parquet(robot / "odom.parquet")
    if tf_gt:
        pl.DataFrame({"time_ns": [0, 1, 2], "pos_x_gt": [1.0, 1.5, 2.0], "pos_y_gt": [1.0, 1.0, 1.0]}, schema={"time_ns": pl.Int64, "pos_x_gt": pl.Float64, "pos_y_gt": pl.Float64}).write_parquet(robot / "tf_gt.parquet")
    if acoustics:
        pl.DataFrame({"time_ns": [0, 2], "total_level_af_dba": [60.0, 62.0]}, schema={"time_ns": pl.Int64, "total_level_af_dba": pl.Float64}).write_parquet(robot / "acoustics.parquet")
    if acoustic_alt:
        pl.DataFrame({"time_ns": [0, 1, 2], "total_level_af_dba": [60.0, 61.0, 62.0]}, schema={"time_ns": pl.Int64, "total_level_af_dba": pl.Float64}).write_parquet(robot / "acoustic.parquet")
    if peds:
        pl.DataFrame({"time_ns": [0, 1, 2], "peds_positions": [[[0.0, 0.0]], [[1.0, 1.0]], []]}, schema={"time_ns": pl.Int64, "peds_positions": pl.List(pl.List(pl.Float64))}).write_parquet(topics / "peds.parquet")
    if collision:
        pl.DataFrame({"time_ns": [0, 1, 2], "collision_event": [False, True, False]}, schema={"time_ns": pl.Int64, "collision_event": pl.Boolean}).write_parquet(robot / "collision_events.parquet")
    return bench


def test_load_episode_data_missing_topics_returns_none(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    assert AcousticFieldRenderer._load_episode_data(bench, "episode_001") is None


def test_load_episode_data_no_robot_dirs_returns_none(tmp_path):
    bench = _make_episode(tmp_path, odom=False)
    assert AcousticFieldRenderer._load_episode_data(bench, "episode_001") is None


def test_load_episode_data_full_bundle(tmp_path):
    bench = _make_episode(tmp_path, tf_gt=True)
    df = AcousticFieldRenderer._load_episode_data(bench, "episode_001")
    assert df is not None
    columns = set(df.columns)
    assert {"time_ns", "pos_x_gt", "pos_y_gt", "source_dba", "peds_positions", "has_collision"} <= columns
    # acoustics sampled at [0, 2]; join_asof strategy="forward" picks the
    # first right timestamp >= the left one -> 62.0 fills time_ns=1.
    assert df["source_dba"].to_list() == [60.0, 62.0, 62.0]


def test_load_episode_data_odom_fallback_and_acoustic_alt(tmp_path):
    bench = _make_episode(tmp_path, acoustics=False, acoustic_alt=True, peds=False, collision=False)
    df = AcousticFieldRenderer._load_episode_data(bench, "episode_001")
    assert df is not None
    assert "pos_x_gt" in df.columns and "source_dba" in df.columns
    assert "peds_positions" not in df.columns
    assert "has_collision" not in df.columns


def test_load_episode_data_no_robot_position_frames_returns_none(tmp_path):
    # Robot dir with an unrelated parquet only -> no odom/tf_gt frame selected.
    bench = _make_episode(tmp_path, odom=False, tf_gt=False)
    assert AcousticFieldRenderer._load_episode_data(bench, "episode_001") is None


# ── compute_field_timeseries ───────────────────────────────────────────────


def _anim_df(n: int = 4, **overrides) -> pl.DataFrame:
    rows = {
        "time_ns": list(range(n)),
        "pos_x_gt": [1.0] * n,
        "pos_y_gt": [1.0] * n,
        "source_dba": [60.0] * n,
        "peds_positions": [[[0.0, 0.0]], [], [[1.0, 1.0]], []][:n],
        "has_collision": [False, True, False, False][:n],
    }
    rows.update(overrides)
    schema = {
        "time_ns": pl.Int64,
        "pos_x_gt": pl.Float64,
        "pos_y_gt": pl.Float64,
        "source_dba": pl.Float64,
        "peds_positions": pl.List(pl.List(pl.Float64)),
        "has_collision": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


@pytest.mark.slow
def test_compute_field_timeseries_happy_path():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    df = _anim_df()
    fields = renderer.compute_field_timeseries(df, _grid(12), 0.1, 0.0, 0.0, {}, downsample=2, stride=1, max_frames=3)
    assert len(fields) == 3
    field, res, (h, w), open_set, src = fields[0]
    assert field.shape == (6, 6)  # 12x12 downsampled by 2
    assert res == 0.2
    assert src == 60.0


@pytest.mark.slow
def test_compute_field_timeseries_stride_and_nan_position():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    df = _anim_df().with_columns(pl.Series("pos_y_gt", [1.0, 1.0, float("nan"), 1.0]))
    fields = renderer.compute_field_timeseries(df, _grid(12), 0.1, 0.0, 0.0, {}, downsample=1, stride=2, max_frames=10)
    assert len(fields) == 2
    assert fields[0] is not None  # index 0 is a valid frame
    assert fields[1] is None  # index 2 has a NaN position -> skipped


@pytest.mark.slow
def test_compute_field_timeseries_collision_boost_and_null_source():
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    # Frame 0: normal source (60 dBA)
    # Frame 1: published collision impact (100 dBA)
    # Frame 2: ongoing driving source (60 dBA)
    # Frame 3: null source (falls back to floor 30 dBA)
    df = _anim_df().with_columns(
        pl.Series("source_dba", [60.0, 100.0, 60.0, None]),
    )
    fields = renderer.compute_field_timeseries(df, _grid(12), 0.1, 0.0, 0.0, {}, stride=1, max_frames=4)
    assert fields[0][4] == 60.0  # normal driving
    assert fields[1][4] == 100.0  # collision impact from telemetry
    assert fields[2][4] == 60.0  # subsequent driving
    assert fields[3][4] == af_mod._FIELD_VMIN_DBA  # null source falls back to floor


@pytest.mark.slow
def test_compute_field_timeseries_with_door_timeline_caches_pixel_tl(monkeypatch):
    built = []

    def _fake_build_tl(grid, doors, open_doors=None, wall_tl_db=47.0):
        built.append(frozenset(open_doors or set()))
        return np.zeros(grid.shape, dtype=np.float32)

    monkeypatch.setattr(af_mod, "build_pixel_tl", _fake_build_tl)

    class _Timeline:
        def __init__(self):
            self._state = 0

        def open_doors_at(self, time_ns):
            return frozenset(["d1"]) if time_ns < 2 else frozenset()

    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    fields = renderer.compute_field_timeseries(_anim_df(), _grid(12), 0.1, 0.0, 0.0, {}, state_timeline=_Timeline(), stride=1, max_frames=4)
    assert len(fields) == 4
    assert fields[0][3] == frozenset(["d1"])
    assert built == [frozenset(["d1"]), frozenset()]  # per-unique-open-set only


def test_compute_field_timeseries_no_solver_returns_empty(monkeypatch):
    monkeypatch.setattr(af_mod, "compute_attenuations", None)
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer.compute_field_timeseries(_anim_df(), _grid(12), 0.1, 0.0, 0.0, {}) == []


# ── render_plotly (single + grid mode) ─────────────────────────────────────


def test_af_plotly_no_solver_returns_empty(monkeypatch):
    monkeypatch.setattr(af_mod, "compute_attenuations", None)
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer.render_plotly(_af_df()) == ""


def test_af_plotly_empty_after_reference_filter(monkeypatch):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    df = _af_df().with_columns(pl.Series("is_reference", [True, True]))
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer.render_plotly(df) == ""


def test_af_plotly_no_map_column(monkeypatch):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer.render_plotly(_af_df().drop("map")) == ""


def test_af_plotly_map_meta_missing(monkeypatch):
    monkeypatch.setattr(af_mod.AcousticFieldRenderer, "_load_grid_and_meta", staticmethod(lambda map_name, run_dir=None: None))
    _patch_doors(monkeypatch)
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer.render_plotly(_af_df()) == ""


@pytest.mark.slow
def test_af_plotly_grid_mode_renders_cells(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    renderer = _new_renderer(_af_spec(), tmp_path)
    html = renderer.render_plotly(_af_df())
    assert 'grid-template-columns:repeat(1,1fr)' in html
    assert html.count('<img src="plots/') == 2
    assert (tmp_path / "plots" / "af_test_dwb_hall.png").exists()
    assert (tmp_path / "plots" / "af_test_teb_hall.png").exists()
    assert "82 dBA" in html and "105 dBA" in html


@pytest.mark.slow
def test_af_plotly_grid_mode_with_doors(monkeypatch, tmp_path):
    _patch_grid(monkeypatch, n=16)
    _patch_doors(monkeypatch, {"world/d1": (_door_mask(16), 25.0)})
    renderer = _new_renderer(_af_spec(), tmp_path)
    html = renderer.render_plotly(_af_df())
    assert html.count('<img src="plots/') == 2
    assert (tmp_path / "plots" / "af_test_dwb_hall.png").exists()


def test_af_plotly_doors_with_downsample_succeeds(monkeypatch, tmp_path):
    """With downsample > 1 and doors present, grid and doors are downsampled
    together and rendering succeeds."""
    _patch_grid(monkeypatch, n=16)
    _patch_doors(monkeypatch, {"world/d1": (_door_mask(16), 25.0)})
    renderer = _new_renderer(_af_spec(downsample=2), tmp_path)
    html = renderer.render_plotly(_af_df())
    assert '<img src="plots/' in html


@pytest.mark.slow
def test_af_plotly_grid_mode_no_differentiate(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    spec = _af_spec(differentiate=None)
    renderer = _new_renderer(spec, tmp_path)
    html = renderer.render_plotly(_af_df())
    assert html.count('<img src="plots/') == 1  # single row group


@pytest.mark.slow
def test_af_plotly_grid_mode_group_by_two_cols(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    spec = _af_spec(group_by=["stage", "episode"])
    renderer = _new_renderer(spec, tmp_path)
    html = renderer.render_plotly(_af_df())
    assert html.count('<img src="plots/') == 2
    assert (tmp_path / "plots" / "af_test_dwb_hall_2.png").exists()
    assert (tmp_path / "plots" / "af_test_teb_hall_1.png").exists()


def test_af_plotly_grid_mode_no_worst_entries(monkeypatch):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    df = _af_df().drop("worst_case_acoustic_frame").with_columns(pl.lit(None).alias("worst_case_acoustic_frame"))
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    assert renderer.render_plotly(df) == ""


@pytest.mark.slow
def test_af_plotly_single_mode(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch, {"world/d1": (_door_mask(20), 25.0)})
    renderer = _new_renderer(_af_spec(mode="single"), tmp_path)
    html = renderer.render_plotly(_af_df())
    assert '<img src="plots/af_test.png"' in html
    assert (tmp_path / "plots" / "af_test.png").exists()
    assert "105 dBA" in html


@pytest.mark.slow
def test_af_plotly_downsample_with_doors(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch, {"world/d1": (_door_mask(20), 25.0)})
    renderer = _new_renderer(_af_spec(mode="single", downsample=2), tmp_path)
    html = renderer.render_plotly(_af_df())
    assert '<img src="plots/af_test.png"' in html
    assert (tmp_path / "plots" / "af_test.png").exists()


def test_af_plotly_single_mode_no_worst_frame(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    df = _af_df().drop("ped_max_exposure_dba")
    renderer = _new_renderer(_af_spec(mode="single"), tmp_path)
    assert renderer.render_plotly(df) == ""


@pytest.mark.slow
def test_af_plotly_single_mode_json_string_frames(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    df = _af_df().with_columns(
        pl.Series(
            "worst_case_acoustic_frame",
            [
                json.dumps({"robot_x": 1.0, "robot_y": 1.0, "source_dba": 60.0}),
                json.dumps({"robot_x": 2.0, "robot_y": 2.0, "source_dba": 60.0}),
            ],
        )
    )
    renderer = _new_renderer(_af_spec(mode="single"), tmp_path)
    html = renderer.render_plotly(df)
    assert '<img src="plots/af_test.png"' in html


def test_af_plotly_cell_render_failure_returns_empty(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    monkeypatch.setattr(af_mod.AcousticFieldRenderer, "_render_cell_png", staticmethod(lambda *a, **k: False))
    renderer = _new_renderer(_af_spec(), tmp_path)
    assert renderer.render_plotly(_af_df()) == ""


# ── render_seaborn (AcousticFieldRenderer) ─────────────────────────────────


def test_af_seaborn_no_solver_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(af_mod, "compute_attenuations", None)
    out = tmp_path / "af.png"
    renderer = _new_renderer(_af_spec(), tmp_path)
    renderer.render_seaborn(_af_df(), out)
    assert not out.exists()


def test_af_seaborn_empty_after_filter_writes_nothing(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    df = _af_df().with_columns(pl.Series("is_reference", [True, True]))
    out = tmp_path / "af.png"
    renderer = _new_renderer(_af_spec(), tmp_path)
    renderer.render_seaborn(df, out)
    assert not out.exists()


def test_af_seaborn_no_map_writes_nothing(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    out = tmp_path / "af.png"
    renderer = _new_renderer(_af_spec(), tmp_path)
    renderer.render_seaborn(_af_df().drop("map"), out)
    assert not out.exists()


def test_af_seaborn_meta_missing_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(af_mod.AcousticFieldRenderer, "_load_grid_and_meta", staticmethod(lambda map_name, run_dir=None: None))
    out = tmp_path / "af.png"
    renderer = _new_renderer(_af_spec(), tmp_path)
    renderer.render_seaborn(_af_df(), out)
    assert not out.exists()


def test_af_seaborn_no_worst_frame_writes_nothing(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    out = tmp_path / "af.png"
    renderer = _new_renderer(_af_spec(), tmp_path)
    renderer.render_seaborn(_af_df().drop("ped_max_exposure_dba"), out)
    assert not out.exists()


@pytest.mark.slow
def test_af_seaborn_happy_path_with_doors(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch, {"world/d1": (_door_mask(20), 25.0)})
    out = tmp_path / "af.png"
    renderer = _new_renderer(_af_spec(), tmp_path)
    renderer.render_seaborn(_af_df(), out)
    assert out.exists() and out.read_bytes()[:8] == _PNG_MAGIC


# ── render_animation ───────────────────────────────────────────────────────


@pytest.mark.slow
def test_render_animation_gif(tmp_path):
    renderer = _new_renderer(_af_spec(), tmp_path)
    out = tmp_path / "anim.gif"
    result = renderer.render_animation(
        _anim_df(),
        _grid(12),
        0.1,
        0.0,
        0.0,
        {},
        out_path=out,
        downsample=2,
        stride=1,
        max_frames=4,
        fps=10,
        robot_trail=3,
    )
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:6] == b"GIF89a"


@pytest.mark.slow
def test_render_animation_frames_format_exports_pngs(tmp_path):
    renderer = _new_renderer(_af_spec(), tmp_path)
    out = tmp_path / "anim.png"
    result = renderer.render_animation(
        _anim_df(2),
        _grid(12),
        0.1,
        0.0,
        0.0,
        {},
        out_path=out,
        downsample=2,
        max_frames=2,
        fmt="frames",
    )
    frames_dir = out.with_suffix("")
    assert result == frames_dir
    frames = sorted(frames_dir.glob("frame_*.png"))
    assert len(frames) == 2


@pytest.mark.slow
def test_render_animation_unknown_format_falls_back_but_cannot_save(tmp_path, capsys):
    """Suspected source bug (documented): the unknown-format fallback warns
    and retries as GIF, but the pillow writer rejects the unknown file
    extension, so the animation is never written (returns None)."""
    renderer = _new_renderer(_af_spec(), tmp_path)
    out = tmp_path / "anim.webm"
    result = renderer.render_animation(
        _anim_df(2),
        _grid(12),
        0.1,
        0.0,
        0.0,
        {},
        out_path=out,
        downsample=2,
        max_frames=2,
        fmt="webm",
    )
    assert result is None
    err = capsys.readouterr().err
    assert "Unknown format" in err
    assert "Failed to save animation" in err


def test_render_animation_no_valid_frames_returns_none(tmp_path):
    renderer = _new_renderer(_af_spec(), tmp_path)
    df = _anim_df(3).with_columns(pl.Series("pos_x_gt", [float("nan")] * 3))
    out = tmp_path / "anim.gif"
    assert renderer.render_animation(df, _grid(12), 0.1, 0.0, 0.0, {}, out_path=out) is None
    assert not out.exists()


# ── AcousticFieldAnimationRenderer ─────────────────────────────────────────


def _anim_spec(spec_id: str = "af_anim", **options) -> PlotSpec:
    return PlotSpec(id=spec_id, type="acoustic_field_animation", title="Acoustic Animation", data_key="ped_max_exposure_dba", options=options)


def _patch_animation(monkeypatch, seen: dict):
    def _fake_render_animation(self, df, grid, resolution, ox, oy, doors, state_timeline=None, out_path=None, downsample=1, stride=1, max_frames=120, fps=10, dpi=150, vmin=None, vmax=None, robot_trail=0, show_doors=True, fmt="gif", overlay_trajectories=False):
        seen.update({"out_path": out_path, "fmt": fmt, "fps": fps, "vmin": vmin, "downsample": downsample, "stride": stride, "max_frames": max_frames, "state_timeline": state_timeline, "vmax": vmax})
        return out_path

    monkeypatch.setattr(af_mod.AcousticFieldRenderer, "render_animation", _fake_render_animation)


def test_animation_renderer_seaborn_orchestrates(monkeypatch, tmp_path):
    bench = _make_episode(tmp_path, tf_gt=True)
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    seen = {}
    _patch_animation(monkeypatch, seen)
    out = tmp_path / "anim.png"
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    renderer.run_dir = bench
    renderer.render_seaborn(_af_df(), out)
    assert seen["out_path"] == tmp_path / "anim.gif"
    assert seen["fmt"] == "gif"
    assert seen["fps"] == 10
    assert seen["downsample"] == 2  # animation default
    assert seen["vmax"] is None  # auto-computed from field data
    assert seen["state_timeline"] is None


def test_animation_renderer_seaborn_finds_episode_in_parent_dir(monkeypatch, tmp_path):
    # run_dir is a subdirectory; the benchmark episodes live in its parent.
    import shutil

    bench = _make_episode(tmp_path, tf_gt=True)  # tmp_path/bench/episodes/...
    shutil.move(bench / "episodes", tmp_path / "episodes")
    run_dir = tmp_path / "out_dir"
    run_dir.mkdir()
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    seen = {}
    _patch_animation(monkeypatch, seen)
    out = run_dir / "anim.png"
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    renderer.run_dir = run_dir
    renderer.render_seaborn(_af_df(), out)
    assert seen["out_path"] == run_dir / "anim.gif"


def test_animation_renderer_seaborn_no_solver(monkeypatch, tmp_path):
    monkeypatch.setattr(af_mod, "compute_attenuations", None)
    out = tmp_path / "anim.png"
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    renderer.run_dir = tmp_path
    renderer.render_seaborn(_af_df(), out)
    assert not out.exists()


def test_animation_renderer_seaborn_empty_df(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    df = _af_df().with_columns(pl.Series("is_reference", [True, True]))
    out = tmp_path / "anim.png"
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    renderer.run_dir = tmp_path
    renderer.render_seaborn(df, out)
    assert not out.exists()


def test_animation_renderer_seaborn_no_episode_column(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    out = tmp_path / "anim.png"
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    renderer.run_dir = tmp_path
    renderer.render_seaborn(_af_df().drop("episode"), out)
    assert not out.exists()


def test_animation_renderer_seaborn_no_episode_data(monkeypatch, tmp_path):
    _patch_grid(monkeypatch)
    monkeypatch.setattr(af_mod.AcousticFieldRenderer, "_load_episode_data", staticmethod(lambda *a, **k: None))
    out = tmp_path / "anim.png"
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    renderer.run_dir = tmp_path
    renderer.render_seaborn(_af_df(), out)
    assert not out.exists()


def test_animation_renderer_plotly_empty_df_returns_empty_string():
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    assert renderer.render_plotly(pl.DataFrame()) == ""


def test_animation_renderer_plotly_no_solver_returns_empty(monkeypatch):
    monkeypatch.setattr(af_mod, "compute_attenuations", None)
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    assert renderer.render_plotly(_af_df()) == ""


def test_animation_renderer_plotly_frames_format_returns_empty():
    renderer = AcousticFieldAnimationRenderer(_anim_spec(format="frames"))
    assert renderer.render_plotly(_af_df()) == ""


def test_animation_renderer_plotly_embeds_gif():
    renderer = AcousticFieldAnimationRenderer(_anim_spec())
    html = renderer.render_plotly(_af_df())
    assert '<img src="plots/af_anim.gif"' in html
    assert "Acoustic Animation" in html


def test_animation_renderer_plotly_mp4_embeds_gif_reference():
    """options.format=mp4 still embeds a .gif src (ext maps mp4->gif) —
    documented source quirk."""
    renderer = AcousticFieldAnimationRenderer(_anim_spec(format="mp4"))
    html = renderer.render_plotly(_af_df())
    assert '<img src="plots/af_anim.gif"' in html


def test_animation_renderer_seaborn_characterization_null_peds(monkeypatch, tmp_path):
    bench = _make_episode(tmp_path, tf_gt=True)
    _patch_grid(monkeypatch)
    _patch_doors(monkeypatch)
    seen = {}
    _patch_animation(monkeypatch, seen)
    out = tmp_path / "anim.png"
    # Null ped_max_exposure_dba (characterization run without pedestrians)
    df = _af_df().with_columns(pl.lit(None, dtype=pl.Float64).alias("ped_max_exposure_dba"))
    renderer = AcousticFieldAnimationRenderer(_anim_spec(data_key="timeseries_char_dba"))
    renderer.run_dir = bench
    renderer.render_seaborn(df, out)
    assert seen.get("out_path") == tmp_path / "anim.gif"


def test_extract_trajectory_data_robot_and_peds():
    """Verify _extract_trajectory_data extracts robot path, start, goal, and ped tracks with 2 waypoints."""
    import polars as pl

    df = pl.DataFrame(
        {
            "time_ns": [0, 100000000, 200000000],
            "pos_x_gt": [1.0, 2.0, 3.0],
            "pos_y_gt": [0.5, 0.5, 0.5],
            "start": [[1.0, 0.5]] * 3,
            "goal": [[3.0, 0.5]] * 3,
            "peds_positions": [
                [{"id": 10, "x": 5.0, "y": 2.0}, {"id": 20, "x": 8.0, "y": 4.0}],
                [{"id": 10, "x": 5.5, "y": 2.0}, {"id": 20, "x": 7.5, "y": 4.0}],
                [{"id": 10, "x": 6.0, "y": 2.0}, {"id": 20, "x": 7.0, "y": 4.0}],
            ],
        }
    )
    traj_data = AcousticFieldRenderer._extract_trajectory_data(df)
    assert traj_data["robot_path"] is not None
    rx, ry = traj_data["robot_path"]
    assert len(rx) == 3 and rx[0] == 1.0 and rx[-1] == 3.0
    assert traj_data["robot_start"] == (1.0, 0.5)
    assert traj_data["robot_goal"] == (3.0, 0.5)

    # Pedestrian tracks are keyed by slot index within each frame
    ped_trajs = traj_data["ped_trajectories"]
    assert 0 in ped_trajs and len(ped_trajs[0]) == 3
    assert 1 in ped_trajs and len(ped_trajs[1]) == 3

    # Check 2 waypoints per ped (WP1: start, WP2: goal/max displacement)
    ped_wps = traj_data["ped_waypoints"]
    assert ped_wps[0] == ((5.0, 2.0), (6.0, 2.0))
    assert ped_wps[1] == ((8.0, 4.0), (7.0, 4.0))


def test_draw_trajectories_overlay_artists():
    """Verify _draw_trajectories_overlay draws lines and scatter artists on matplotlib axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj_data = {
        "robot_path": (np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.0])),
        "robot_start": (0.0, 0.0),
        "robot_goal": (2.0, 1.0),
        "ped_trajectories": {1: [(5.0, 5.0), (5.0, 6.0), (5.0, 7.0)]},
        "ped_waypoints": {1: ((5.0, 5.0), (5.0, 7.0))},
    }

    fig, ax = plt.subplots()
    AcousticFieldRenderer._draw_trajectories_overlay(ax, traj_data)
    # Check lines (robot track, ped track) and collections (start, goal, wp1, wp2)
    assert len(ax.lines) >= 2
    assert len(ax.collections) >= 3
    plt.close(fig)


@pytest.mark.slow
def test_render_cell_png_with_overlay_trajectories(tmp_path):
    """Verify _render_cell_png successfully overlays trajectories when requested."""
    renderer = _new_renderer(_af_spec(), pathlib.Path("."))
    grid = _grid(10)
    out_path = tmp_path / "cell_with_trajectories.png"
    traj_data = {
        "robot_path": (np.array([1.0, 1.5, 2.0]), np.array([1.0, 1.0, 1.0])),
        "robot_start": (1.0, 1.0),
        "robot_goal": (2.0, 1.0),
        "collision_pos": (1.5, 1.0),
        "ped_trajectories": {0: [(3.0, 3.0), (3.0, 3.5)]},
        "ped_waypoints": {0: ((3.0, 3.0), (3.0, 3.5))},
    }
    ok = renderer._render_cell_png(
        grid=grid,
        resolution=0.1,
        ox=0.0,
        oy=0.0,
        rx_m=1.0,
        ry_m=1.0,
        source_dba=60.0,
        peds=[(3.0, 3.0)],
        title="Test Trajectory Overlay",
        out_path=out_path,
        overlay_trajectories=True,
        trajectory_data=traj_data,
    )
    assert ok is True
    assert out_path.exists() and out_path.stat().st_size > 0


def test_extract_trajectory_data_flat_float_list():
    """Verify _extract_trajectory_data handles flat [x0,y0,z0,x1,y1,z1,...] list format."""
    import polars as pl

    df = pl.DataFrame(
        {
            "time_ns": [0, 100000000],
            "pos_x_gt": [0.0, 1.0],
            "pos_y_gt": [0.0, 1.0],
            "has_collision": [False, True],
            "peds_positions": [
                [1.0, 2.0, 0.0, 4.0, 5.0, 0.0],
                [1.5, 2.5, 0.0, 4.2, 5.2, 0.0],
            ],
        }
    )
    traj_data = AcousticFieldRenderer._extract_trajectory_data(df)
    assert 0 in traj_data["ped_trajectories"]
    assert 1 in traj_data["ped_trajectories"]
    assert traj_data["ped_trajectories"][0][0] == (1.0, 2.0)
    assert traj_data["ped_trajectories"][1][0] == (4.0, 5.0)
    assert traj_data["collision_pos"] == (1.0, 1.0)


def test_extract_trajectory_data_disk_fallback(tmp_path):
    """Verify _extract_trajectory_data falls back to disk topics/peds.parquet when df has no peds."""
    import polars as pl

    ep_dir = tmp_path / "episodes" / "episode_005" / "topics"
    ep_dir.mkdir(parents=True)
    peds_df = pl.DataFrame(
        {
            "time_ns": [0, 100000000, 200000000],
            "peds_positions": [
                [2.0, 3.0, 0.0],
                [2.5, 3.5, 0.0],
                [3.0, 4.0, 0.0],
            ],
        }
    )
    peds_df.write_parquet(ep_dir / "peds.parquet")

    # Pass empty df or df without peds_positions
    df_empty = pl.DataFrame({"pos_x_gt": [0.0], "pos_y_gt": [0.0]})
    traj_data = AcousticFieldRenderer._extract_trajectory_data(df_empty, run_dir=tmp_path, episode_id=5)
    assert 0 in traj_data["ped_trajectories"]
    assert len(traj_data["ped_trajectories"][0]) == 3
    assert traj_data["ped_trajectories"][0][0] == (2.0, 3.0)
    assert traj_data["ped_trajectories"][0][-1] == (3.0, 4.0)
