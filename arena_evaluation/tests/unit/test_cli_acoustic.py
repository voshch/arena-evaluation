"""Tests for the acoustic subcommands registered into the main CLI.

Covers :mod:`arena_evaluation.cli_acoustic`. Episode resolution and table
rendering run against real Polars frames written to ``tmp_path``. The heavy
renderer/parquet plumbing (``AcousticFieldRenderer``, ``door_segments``,
``DoorStateTimeline``, ``ParquetStore``) is replaced with in-memory stubs so
no plotting backend or ROS graph is required; parquet files are still real.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import polars as pl
import pytest

from arena_evaluation.cli_acoustic import (
    _acoustic_animate,
    _acoustic_list,
    _acoustic_snapshot,
    _handle_acoustic,
    _resolve_episode,
    setup_acoustic_subparsers,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_metrics(benchmark_dir: pathlib.Path, df: pl.DataFrame, name: str = "metrics.parquet") -> pathlib.Path:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    path = benchmark_dir / name
    df.write_parquet(path)
    return path


def _metrics_df(**overrides: object) -> pl.DataFrame:
    data = {
        "episode": [1, 2, 3],
        "ped_max_exposure_dba": [70.0, 83.4, None],
        "ped_leq_exposure_dba": [55.1, 60.2, None],
    }
    data.update(overrides)
    return pl.DataFrame(data)


def _acoustic_argv(*argv: str) -> argparse.Namespace:
    """Parse realistic acoustic argv through the real subparser wiring."""
    parser = argparse.ArgumentParser(prog="evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    setup_acoustic_subparsers(sub)
    return parser.parse_args(["acoustic", *argv])


# -- stub installation for the renderer/parquet plumbing --------------------

# Renderer calls are recorded module-level: the class is defined at module
# scope, so its methods close over the module namespace, not a test helper.
_RENDER_CALLS: list[dict] = []


class _FakeTimeline:
    def __init__(self, doors: list[int]):
        self._doors = doors

    def open_doors_at(self, time_ns: int) -> frozenset[int]:  # noqa: ARG001
        return frozenset(self._doors)

    @classmethod
    def from_semantic_frame(cls, df) -> _FakeTimeline:  # noqa: ARG001
        return cls([3, 4])


class _FakeRenderer:
    load_grid_result = ({"grid": True}, {"resolution": 0.1, "origin": (1.0, 2.0)})
    episode_df: pl.DataFrame | None = pl.DataFrame(
        {
            "pos_x_gt": [1.0, 2.0, 3.0],
            "pos_y_gt": [1.0, 2.0, 3.0],
            "source_dba": [50.0, 60.0, 90.0],
            "time_ns": [100, 200, 300],
            "peds_positions": [[], [], []],
        }
    )
    render_result: str | None = "rendered.gif"
    cell_render_ok = True
    peds: list = []

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass

    @classmethod
    def _load_grid_and_meta(cls, map_name, run_dir=None):  # noqa: ARG001
        return cls.load_grid_result

    @classmethod
    def _load_episode_data(cls, run_dir, episode_id):  # noqa: ARG001
        return cls.episode_df

    @classmethod
    def _parse_pedestrian_positions(cls, peds_pos):  # noqa: ARG001
        return cls.peds

    def render_animation(self, *args, **kwargs) -> str | None:
        _RENDER_CALLS.append(kwargs)
        return self.render_result

    def _render_cell_png(self, *args, **kwargs) -> bool:
        return self.cell_render_ok

    @classmethod
    def _extract_trajectory_data(cls, df=None, run_dir=None, episode_id=None):  # noqa: ARG003
        return None


def _install_stubs(monkeypatch, *, doors: list[int] | None = None) -> types.SimpleNamespace:
    """Swap heavy acoustic deps for stubs via sys.modules (imports are lazy)."""
    calls = types.SimpleNamespace(
        render_animation=[],
        render_cell=[],
        build_pixel_tl=[],
        door_segments=[],
        timeline=[],
    )

    class _FakeParquetStore:
        @staticmethod
        def read(source: pathlib.Path):
            return pl.read_parquet(source), None

    parquet_mod = types.ModuleType("arena_evaluation.processing.parquet_store")
    parquet_mod.ParquetStore = _FakeParquetStore

    field_mod = types.ModuleType("arena_evaluation.presentation.plot_types.acoustic_field")
    field_mod.AcousticFieldRenderer = _FakeRenderer

    door_state_mod = types.ModuleType("arena_evaluation.processing.acoustics.door_state")
    door_state_mod.DoorStateTimeline = _FakeTimeline

    door_map_mod = types.ModuleType("arena_evaluation.processing.acoustics.door_map")

    def _door_segments(map_name, grid, resolution, origin, run_dir=None):  # noqa: ARG001
        calls.door_segments.append(map_name)
        return doors if doors is not None else [1, 2, 3]

    def _build_pixel_tl(grid, doors_, open_doors=None):  # noqa: ARG001
        calls.build_pixel_tl.append(open_doors)
        return "pixel_tl"

    door_map_mod.door_segments = _door_segments
    door_map_mod.build_pixel_tl = _build_pixel_tl

    parents = [
        "arena_evaluation.presentation",
        "arena_evaluation.presentation.plot_types",
        "arena_evaluation.processing",
        "arena_evaluation.processing.acoustics",
    ]
    for name in parents:
        parent = types.ModuleType(name)
        parent.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, parent)
    monkeypatch.setitem(sys.modules, "arena_evaluation.processing.parquet_store", parquet_mod)
    monkeypatch.setitem(sys.modules, "arena_evaluation.presentation.plot_types.acoustic_field", field_mod)
    monkeypatch.setitem(sys.modules, "arena_evaluation.processing.acoustics.door_state", door_state_mod)
    monkeypatch.setitem(sys.modules, "arena_evaluation.processing.acoustics.door_map", door_map_mod)
    return calls


# ---------------------------------------------------------------------------
# setup_acoustic_subparsers
# ---------------------------------------------------------------------------


def test_acoustic_subparsers_list_defaults():
    args = _acoustic_argv("list", "--benchmark-dir", "/tmp/b")
    assert args.acoustic_command == "list"
    assert args.benchmark_dir == pathlib.Path("/tmp/b")


def test_acoustic_subparsers_animate_defaults():
    args = _acoustic_argv("animate", "--benchmark-dir", "/tmp/b")
    assert args.acoustic_command == "animate"
    assert args.episode == "worst"
    assert args.fps == 10
    assert args.max_frames == 120
    assert args.stride == 1
    assert args.downsample == 2
    assert args.format == "gif"
    assert args.dpi == 150
    assert args.vmin == 20.0
    assert args.vmax is None
    assert args.robot_trail == 0
    assert args.no_door_overlay is False


def test_acoustic_subparsers_snapshot_defaults():
    args = _acoustic_argv("snapshot", "--benchmark-dir", "/tmp/b")
    assert args.acoustic_command == "snapshot"
    assert args.frame is None
    assert args.output is None


def test_acoustic_subparsers_animate_custom():
    args = _acoustic_argv(
        "animate",
        "--benchmark-dir",
        "/tmp/b",
        "--episode",
        "episode_005",
        "--fps",
        "5",
        "--format",
        "mp4",
        "--max-frames",
        "60",
        "--stride",
        "2",
        "--downsample",
        "4",
        "--vmin",
        "35.0",
        "--vmax",
        "95.0",
        "--robot-trail",
        "3",
        "--no-door-overlay",
        "--output",
        "/tmp/out.mp4",
    )
    assert args.episode == "episode_005"
    assert args.fps == 5 and args.format == "mp4"
    assert args.max_frames == 60 and args.stride == 2 and args.downsample == 4
    assert args.vmin == 35.0 and args.vmax == 95.0
    assert args.robot_trail == 3 and args.no_door_overlay is True
    assert args.output == pathlib.Path("/tmp/out.mp4")


def test_acoustic_subparsers_invalid_format_rejected():
    with pytest.raises(SystemExit):
        _acoustic_argv("animate", "--benchmark-dir", "/tmp/b", "--format", "webp")


def test_acoustic_subparsers_missing_subcommand_rejected():
    with pytest.raises(SystemExit):
        _acoustic_argv("--benchmark-dir", "/tmp/b")


def test_acoustic_subparsers_missing_benchmark_dir_rejected():
    with pytest.raises(SystemExit):
        _acoustic_argv("list")


# ---------------------------------------------------------------------------
# _resolve_episode
# ---------------------------------------------------------------------------


def test_resolve_episode_explicit_found():
    df = _metrics_df()
    assert _resolve_episode(df, "episode_002") == "episode_002"


def test_resolve_episode_explicit_not_found(capsys):
    df = _metrics_df()
    assert _resolve_episode(df, "episode_999") is None
    assert "Error: episode_999 not found in metrics." in capsys.readouterr().out


def test_resolve_episode_explicit_without_episode_column():
    df = pl.DataFrame({"ped_max_exposure_dba": [70.0]})
    assert _resolve_episode(df, "episode_001") == "episode_001"


def test_resolve_episode_worst(capsys):
    df = _metrics_df()
    assert _resolve_episode(df, "worst") == "episode_002"
    out = capsys.readouterr().out
    assert "Using worst episode: episode_002 (max=83.4 dBA)" in out


def test_resolve_episode_worst_without_metric_column(capsys):
    df = pl.DataFrame({"episode": [1]})
    assert _resolve_episode(df, "worst") is None
    assert "ped_max_exposure_dba not found" in capsys.readouterr().out


def test_resolve_episode_worst_all_null_crashes(capsys):
    """Suspected bug: 'worst' on a frame with all-null ped_max crashes with
    polars OutOfBoundsError (empty filter, .row(0)) instead of a graceful
    message. Pins current behavior."""
    df = pl.DataFrame({"episode": [1, 2], "ped_max_exposure_dba": [None, None]})
    with pytest.raises(pl.exceptions.OutOfBoundsError):
        _resolve_episode(df, "worst")


def test_resolve_episode_loudest_source(capsys):
    df = pl.DataFrame(
        {
            "episode": [1, 2, 3, 4],
            "ped_max_exposure_dba": [70.0, 80.0, 90.0, 60.0],
            "worst_case_acoustic_frame": [
                {"source_dba": 55.0},
                None,
                {"source_dba": 88.5},
                {"source_dba": 30.0},  # lower than current best: not selected
            ],
        }
    )
    assert _resolve_episode(df, "loudest-source") == "episode_003"
    assert "Using loudest-source episode: episode_003 (source=88.5 dBA)" in capsys.readouterr().out


def test_resolve_episode_loudest_source_json_string(capsys):
    df = pl.DataFrame(
        {
            "episode": [1, 2],
            "ped_max_exposure_dba": [70.0, 80.0],
            "worst_case_acoustic_frame": ['{"source_dba": 40.0}', '{"source_dba": 72.0}'],
        }
    )
    assert _resolve_episode(df, "loudest-source") == "episode_002"
    assert "source=72.0 dBA" in capsys.readouterr().out


def test_resolve_episode_loudest_source_invalid_json_skipped(capsys):
    df = pl.DataFrame(
        {
            "episode": [1, 2],
            "ped_max_exposure_dba": [70.0, 80.0],
            "worst_case_acoustic_frame": ["{not json", '{"source_dba": 66.0}'],
        }
    )
    assert _resolve_episode(df, "loudest-source") == "episode_002"


def test_resolve_episode_loudest_source_missing_column(capsys):
    df = _metrics_df()
    assert _resolve_episode(df, "loudest-source") is None
    assert "worst_case_acoustic_frame not in metrics." in capsys.readouterr().out


def test_resolve_episode_max_total(capsys):
    ts = pl.Series(
        "timeseries_acoustic_exposure_dba",
        [[[1.0, 2.0], [3.0]], [[5.0], [6.0]], None, [[1.0]]],
        dtype=pl.List(pl.List(pl.Float64)),
    )
    # ped_max_exposure_dba must be present: the source gates all non-explicit
    # specifiers on it before dispatching to max-total.
    df = pl.DataFrame(
        {
            "episode": [1, 2, 3, 4],
            "ped_max_exposure_dba": [70.0, 80.0, 90.0, 60.0],
            "timeseries_acoustic_exposure_dba": ts,  # totals 6, 11, skip, 1
        }
    )
    assert _resolve_episode(df, "max-total") == "episode_002"
    assert "Using max-total episode: episode_002 (total=11.0)" in capsys.readouterr().out


def test_resolve_episode_max_total_missing_column(capsys):
    df = _metrics_df()
    assert _resolve_episode(df, "max-total") is None
    assert "timeseries_acoustic_exposure_dba not in metrics." in capsys.readouterr().out


def test_resolve_episode_unknown_spec(capsys):
    df = _metrics_df()
    assert _resolve_episode(df, "quietest") is None
    assert "Error: unknown episode specifier 'quietest'." in capsys.readouterr().out


def test_resolve_episode_non_numeric_suffix_raises_value_error():
    """Suspected bug: 'episode_abc' crashes with ValueError from int() instead
    of a graceful error message."""
    df = _metrics_df()
    with pytest.raises(ValueError):
        _resolve_episode(df, "episode_abc")


# ---------------------------------------------------------------------------
# _acoustic_list
# ---------------------------------------------------------------------------


def test_acoustic_list_no_metric_columns(capsys):
    """No episode column and no acoustic columns -> cols stays empty."""
    df = pl.DataFrame({"some_other_metric": [1, 2]})
    _acoustic_list(df)
    assert "No acoustic metric columns found in metrics file." in capsys.readouterr().out


def test_acoustic_list_prints_sorted_table(capsys):
    df = _metrics_df()
    _acoustic_list(df)
    out = capsys.readouterr().out
    assert "EPISODE" in out and "MAX_DBA" in out and "LEQ_DBA" in out
    # episode column prints as a plain int; polars sort(descending=True) puts
    # nulls FIRST, so episode 3 (null max) appears before 2 and 1.
    eps = [l.split()[0] for l in out.splitlines() if l.split() and l.split()[0] in ("1", "2", "3")]
    assert eps == ["3", "2", "1"]
    assert "83.4" in out and "70.0" in out
    assert "N/A" in out  # null ped_max renders as N/A


def test_acoustic_list_without_leq_column(capsys):
    df = pl.DataFrame({"episode": [1], "ped_max_exposure_dba": [70.0]})
    _acoustic_list(df)
    out = capsys.readouterr().out
    assert "70.0" in out
    assert "N/A" in out  # missing leq column -> N/A


def test_acoustic_list_total_exposure_column(capsys):
    ts = pl.Series(
        "timeseries_acoustic_exposure_dba",
        [[[1.0, 2.0], [3.0]], None],
        dtype=pl.List(pl.List(pl.Float64)),
    )
    df = pl.DataFrame({"episode": [1, 2], "ped_max_exposure_dba": [70.0, 80.0], "ped_leq_exposure_dba": [55.0, 60.0], "timeseries_acoustic_exposure_dba": ts})
    _acoustic_list(df)
    out = capsys.readouterr().out
    assert "TOTAL_EXP" in out
    assert "6.0" in out  # 1+2+3 summed
    assert "N/A" in out  # null timeseries


def test_acoustic_list_episode_placeholder_without_episode_col(capsys):
    df = pl.DataFrame({"ped_max_exposure_dba": [70.0], "ped_leq_exposure_dba": [55.0]})
    _acoustic_list(df)
    assert "?" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _handle_acoustic dispatch
# ---------------------------------------------------------------------------


def test_handle_acoustic_missing_dir(capsys):
    args = _acoustic_argv("list", "--benchmark-dir", "/nonexistent/dir")
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(args)
    assert exc.value.code == 1
    assert "Error: benchmark directory does not exist: /nonexistent/dir" in capsys.readouterr().out


def test_handle_acoustic_no_metrics_file(tmp_path: pathlib.Path, capsys):
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    args = _acoustic_argv("list", "--benchmark-dir", str(bench_dir))
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(args)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "no metrics.parquet or combined_metrics.parquet found" in out
    assert "Run 'evaluation process --benchmark-dir ...' first." in out


def test_handle_acoustic_list_dispatch(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df())
    args = _acoustic_argv("list", "--benchmark-dir", str(bench_dir))
    _handle_acoustic(args)
    assert "EPISODE" in capsys.readouterr().out


def test_handle_acoustic_prefers_combined_metrics(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(
        bench_dir,
        _metrics_df(
            episode=[9],
            ped_max_exposure_dba=[70.0],
            ped_leq_exposure_dba=[55.0],
        ),
        name="metrics.parquet",
    )
    _write_metrics(
        bench_dir,
        _metrics_df(
            episode=[1, 2],
            ped_max_exposure_dba=[70.0, 83.4],
            ped_leq_exposure_dba=[55.1, 60.2],
        ),
        name="combined_metrics.parquet",
    )
    args = _acoustic_argv("list", "--benchmark-dir", str(bench_dir))
    _handle_acoustic(args)
    out = capsys.readouterr().out
    # the combined file wins: episodes 1 and 2, not 9
    assert "9" not in [l.split()[0] for l in out.splitlines() if l.split() and l.split()[0] in ("1", "2", "9")]
    assert "83.4" in out


def test_handle_acoustic_animate_resolution_failure(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df())
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir), "--episode", "bogus")
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(args)
    assert exc.value.code == 1
    assert "unknown episode specifier" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _acoustic_animate
# ---------------------------------------------------------------------------


def test_animate_success_default_output(tmp_path: pathlib.Path, capsys, monkeypatch):
    calls = _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
    _acoustic_animate(_metrics_df(map="map1"), args)
    out = capsys.readouterr().out
    assert "Rendering animation for episode_002 (3 data frames)..." in out
    assert "Animation saved to: rendered.gif" in out
    assert calls.door_segments == ["map1"]
    assert _RENDER_CALLS
    # default output path: plots/{episode}_acoustic.gif
    assert _RENDER_CALLS[-1]["out_path"] == bench_dir / "plots" / "episode_002_acoustic.gif"


def test_animate_success_frames_format(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir), "--format", "frames")
    _acoustic_animate(_metrics_df(map="map1"), args)
    # no extension for frames format
    assert _RENDER_CALLS[-1]["out_path"] == bench_dir / "plots" / "episode_002_acoustic_frames"
    capsys.readouterr().out


def test_animate_success_output_override(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    out_path = tmp_path / "custom" / "my.gif"
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir), "--output", str(out_path))
    _acoustic_animate(_metrics_df(map="map1"), args)
    assert _RENDER_CALLS[-1]["out_path"] == out_path
    assert "Animation saved to: rendered.gif" in capsys.readouterr().out


def test_animate_missing_map_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df())  # no "map" column
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
    with pytest.raises(SystemExit) as exc:
        _acoustic_animate(_metrics_df(), args)
    assert exc.value.code == 1
    assert "Error: could not determine map name from metrics." in capsys.readouterr().out


def test_animate_grid_load_failure_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    _FakeRenderer.load_grid_result = None
    try:
        args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
        with pytest.raises(SystemExit) as exc:
            _acoustic_animate(_metrics_df(map="map1"), args)
        assert exc.value.code == 1
        assert "Error: could not load map 'map1'." in capsys.readouterr().out
    finally:
        _FakeRenderer.load_grid_result = ({"grid": True}, {"resolution": 0.1, "origin": (1.0, 2.0)})


def test_animate_missing_episode_data_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    _FakeRenderer.episode_df = None
    try:
        args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
        with pytest.raises(SystemExit) as exc:
            _acoustic_animate(_metrics_df(map="map1"), args)
        assert exc.value.code == 1
        assert "Error: no topic data for episode_002. Run 'evaluation extract' first." in capsys.readouterr().out
    finally:
        _FakeRenderer.episode_df = pl.DataFrame(
            {
                "pos_x_gt": [1.0, 2.0, 3.0],
                "pos_y_gt": [1.0, 2.0, 3.0],
                "source_dba": [50.0, 60.0, 90.0],
                "time_ns": [100, 200, 300],
                "peds_positions": [[], [], []],
            }
        )


def test_animate_render_failure_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    _FakeRenderer.render_result = None
    try:
        args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
        with pytest.raises(SystemExit) as exc:
            _acoustic_animate(_metrics_df(map="map1"), args)
        assert exc.value.code == 1
        assert "Animation generation failed." in capsys.readouterr().out
    finally:
        _FakeRenderer.render_result = "rendered.gif"


def test_animate_uses_combined_metrics(capsys, monkeypatch, tmp_path: pathlib.Path):
    """combined_metrics.parquet is preferred; the fallback reassignment is skipped."""
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"), name="combined_metrics.parquet")
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
    _acoustic_animate(_metrics_df(map="map1"), args)
    assert "Rendering animation for episode_002" in capsys.readouterr().out


def test_animate_no_metrics_files_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()  # no parquet at all
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
    with pytest.raises(SystemExit) as exc:
        _acoustic_animate(_metrics_df(map="map1"), args)
    assert exc.value.code == 1
    assert "Error: could not determine map name from metrics." in capsys.readouterr().out


def test_animate_with_semantic_timeline(capsys, monkeypatch, tmp_path: pathlib.Path):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    sem = bench_dir / "episodes" / "episode_002" / "topics"
    sem.mkdir(parents=True)
    pl.DataFrame({"t": [1]}).write_parquet(sem / "semantic_snapshot.parquet")
    args = _acoustic_argv("animate", "--benchmark-dir", str(bench_dir))
    _acoustic_animate(_metrics_df(map="map1"), args)
    out = capsys.readouterr().out
    assert "doors: 3 found, timeline: present" in out
    assert _RENDER_CALLS[-1]["state_timeline"] is not None


# ---------------------------------------------------------------------------
# _acoustic_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_success_explicit_frame(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir), "--frame", "1")
    _acoustic_snapshot(_metrics_df(map="map1"), args)
    out = capsys.readouterr().out
    assert "Rendering snapshot: frame 1" in out
    assert "Snapshot saved to:" in out
    assert str(bench_dir / "plots" / "episode_002_acoustic_snapshot.png") in out
    assert (bench_dir / "plots").is_dir()  # parent dir created


def test_snapshot_auto_best_frame(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
    _acoustic_snapshot(_metrics_df(map="map1"), args)
    out = capsys.readouterr().out
    assert "Using frame 2 (source=90.0 dBA). Use --frame N to pick a specific frame." in out
    assert "Rendering snapshot: frame 2" in out


def test_snapshot_out_of_range_frame_falls_back(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    # non-monotonic sources exercise the "lower than best" branch
    _FakeRenderer.episode_df = pl.DataFrame(
        {
            "pos_x_gt": [1.0, 2.0, 3.0],
            "pos_y_gt": [1.0, 2.0, 3.0],
            "source_dba": [50.0, 90.0, 60.0],
            "time_ns": [100, 200, 300],
            "peds_positions": [[], [], []],
        }
    )
    try:
        args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir), "--frame", "99")
        _acoustic_snapshot(_metrics_df(map="map1"), args)
        assert "Using frame 1 (source=90.0 dBA). Use --frame N to pick a specific frame." in capsys.readouterr().out
    finally:
        _FakeRenderer.episode_df = pl.DataFrame(
            {
                "pos_x_gt": [1.0, 2.0, 3.0],
                "pos_y_gt": [1.0, 2.0, 3.0],
                "source_dba": [50.0, 60.0, 90.0],
                "time_ns": [100, 200, 300],
                "peds_positions": [[], [], []],
            }
        )


def test_snapshot_uses_combined_metrics(capsys, monkeypatch, tmp_path: pathlib.Path):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"), name="combined_metrics.parquet")
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
    _acoustic_snapshot(_metrics_df(map="map1"), args)
    assert "Snapshot saved to:" in capsys.readouterr().out


def test_snapshot_no_metrics_files_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
    with pytest.raises(SystemExit) as exc:
        _acoustic_snapshot(_metrics_df(map="map1"), args)
    assert exc.value.code == 1
    assert "Error: could not determine map name." in capsys.readouterr().out


def test_snapshot_grid_load_failure_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    _FakeRenderer.load_grid_result = None
    try:
        args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
        with pytest.raises(SystemExit) as exc:
            _acoustic_snapshot(_metrics_df(map="map1"), args)
        assert exc.value.code == 1
        assert "Error: could not load map 'map1'." in capsys.readouterr().out
    finally:
        _FakeRenderer.load_grid_result = ({"grid": True}, {"resolution": 0.1, "origin": (1.0, 2.0)})


def test_handle_acoustic_snapshot_resolution_failure(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir), "--episode", "bogus")
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(args)
    assert exc.value.code == 1
    assert "unknown episode specifier" in capsys.readouterr().out


def test_snapshot_render_failure_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    _FakeRenderer.cell_render_ok = False
    try:
        args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
        with pytest.raises(SystemExit) as exc:
            _acoustic_snapshot(_metrics_df(map="map1"), args)
        assert exc.value.code == 1
        assert "Snapshot generation failed." in capsys.readouterr().out
    finally:
        _FakeRenderer.cell_render_ok = True


def test_snapshot_with_doors_and_timeline(tmp_path: pathlib.Path, capsys, monkeypatch):
    calls = _install_stubs(monkeypatch, doors=[1, 2])
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    # semantic snapshot parquet triggers the timeline branch
    sem = bench_dir / "episodes" / "episode_002" / "topics"
    sem.mkdir(parents=True)
    pl.DataFrame({"t": [1]}).write_parquet(sem / "semantic_snapshot.parquet")
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
    _acoustic_snapshot(_metrics_df(map="map1"), args)
    out = capsys.readouterr().out
    assert "doors: 2 found" not in out  # animate-only message
    assert "2 doors open" in out
    assert calls.build_pixel_tl  # pixel_tl built because doors exist


def test_snapshot_without_doors_skips_pixel_tl(tmp_path: pathlib.Path, capsys, monkeypatch):
    calls = _install_stubs(monkeypatch, doors=[])
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
    _acoustic_snapshot(_metrics_df(map="map1"), args)
    assert not calls.build_pixel_tl
    capsys.readouterr().out


def test_snapshot_missing_map_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df())
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
    with pytest.raises(SystemExit) as exc:
        _acoustic_snapshot(_metrics_df(), args)
    assert exc.value.code == 1
    assert "Error: could not determine map name." in capsys.readouterr().out


def test_snapshot_missing_episode_data_exits(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    _FakeRenderer.episode_df = None
    try:
        args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
        with pytest.raises(SystemExit) as exc:
            _acoustic_snapshot(_metrics_df(map="map1"), args)
        assert exc.value.code == 1
        assert "Error: no topic data for episode_002." in capsys.readouterr().out
    finally:
        _FakeRenderer.episode_df = pl.DataFrame(
            {
                "pos_x_gt": [1.0, 2.0, 3.0],
                "pos_y_gt": [1.0, 2.0, 3.0],
                "source_dba": [50.0, 60.0, 90.0],
                "time_ns": [100, 200, 300],
                "peds_positions": [[], [], []],
            }
        )


def test_snapshot_output_override(tmp_path: pathlib.Path, capsys, monkeypatch):
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    out_path = tmp_path / "custom" / "snap.png"
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir), "--output", str(out_path))
    _acoustic_snapshot(_metrics_df(map="map1"), args)
    assert str(out_path) in capsys.readouterr().out
    assert out_path.parent.is_dir()


def test_snapshot_via_handle_acoustic(tmp_path: pathlib.Path, capsys, monkeypatch):
    """Full dispatch: _handle_acoustic -> snapshot, including the map-column
    reload from the metrics parquet on disk."""
    _install_stubs(monkeypatch)
    bench_dir = tmp_path / "bench"
    _write_metrics(bench_dir, _metrics_df(map="map1"))
    args = _acoustic_argv("snapshot", "--benchmark-dir", str(bench_dir))
    _handle_acoustic(args)
    out = capsys.readouterr().out
    assert "Snapshot saved to:" in out
