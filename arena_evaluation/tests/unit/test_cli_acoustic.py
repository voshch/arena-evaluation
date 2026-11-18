"""Unit tests for the acoustic field CLI over the canonical Star Schema episode summary."""

from __future__ import annotations

import argparse
import pathlib

import polars as pl
import pytest

from arena_evaluation.cli_acoustic import (
    _acoustic_animate,
    _acoustic_list,
    _acoustic_snapshot,
    _episode_of_max_level,
    _handle_acoustic,
    _load_episode_summary,
    _loudest_source_episode,
    _map_for_episode,
    _resolve_episode,
    setup_acoustic_subparsers,
)
from star_benchmark import write_benchmark

_WORLD_YAML = """format: arena-world/2
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


def _rows() -> list[dict[str, object]]:
    return [
        {
            "episode_id": 1,
            "planner": "dwb",
            "stage": "s1",
            "world": "map1",
            "success": True,
            "l_aeq_t": 54.5,
        },
        {
            "episode_id": 2,
            "planner": "dwb",
            "stage": "s1",
            "world": "map1",
            "success": False,
            "l_aeq_t": 72.4,
        },
        {
            "episode_id": 3,
            "planner": "dwb",
            "stage": "s1",
            "world": "map2",
            "success": True,
            "l_aeq_t": None,
        },
    ]


def _benchmark(
    tmp_path: pathlib.Path, rows: list[dict[str, object]] | None = None
) -> pathlib.Path:
    return write_benchmark(tmp_path / "bench", rows or _rows())


def _write_world(bench_dir: pathlib.Path, world_name: str = "map1") -> None:
    world_dir = bench_dir / "worlds" / world_name
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / "world.yaml").write_text(_WORLD_YAML, encoding="utf-8")


def _write_episode_telemetry(
    bench_dir: pathlib.Path,
    ep_num: int,
    *,
    source_levels: list[float] | None = None,
    include_acoustics: bool = True,
) -> None:
    topics = bench_dir / "episodes" / f"episode_{ep_num:03d}" / "topics"
    topics.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "env_id": ["env_0", "env_0"],
            "robot_id": ["robot_ns", "robot_ns"],
            "x": [1.0, 1.5],
            "y": [2.0, 2.0],
            "z": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "vx": [0.5, 0.5],
            "vy": [0.0, 0.0],
            "vz": [0.0, 0.0],
            "omega_z": [0.0, 0.0],
            "speed_mps": [0.5, 0.5],
            "pose_source": ["gt", "gt"],
        }
    ).write_parquet(topics / "robot_tracks.parquet")
    if include_acoustics:
        levels = source_levels or [60.0, 68.0]
        pl.DataFrame(
            {
                "t_ns": [0, 1_000_000_000],
                "env_id": ["env_0", "env_0"],
                "robot_id": ["robot_ns", "robot_ns"],
                "total_level_af_dba": levels,
                "total_level_zf_db": levels,
                "baseline_level_dba": [50.0, 50.0],
                "drivetrain_level_dba": [55.0, 65.0],
                "operating_state": ["moving", "moving"],
                "calibration_status": ["ok", "ok"],
                "uncertainty_dba": [1.0, 1.0],
                "validity_flags": [0, 0],
            }
        ).write_parquet(topics / "robot_acoustics.parquet")
    pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "env_id": ["env_0", "env_0"],
            "ped_id": ["ped_1", "ped_1"],
            "ped_name": ["ped_1", "ped_1"],
            "x": [3.0, 3.0],
            "y": [2.0, 2.0],
            "z": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "vx": [0.0, 0.0],
            "vy": [0.0, 0.0],
            "vz": [0.0, 0.0],
            "speed_mps": [0.0, 0.0],
            "radius_m": [0.3, 0.3],
            "desired_speed": [0.0, 0.0],
            "policy": ["sfm", "sfm"],
            "agent_type": ["adult", "adult"],
            "animation_state": [0, 0],
            "interaction_id": [-1, -1],
            "interaction_type": [0, 0],
            "gait_phase": [0.0, 0.0],
            "gait_cadence": [0.0, 0.0],
        }
    ).write_parquet(topics / "pedestrian_arena_peds.parquet")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup_acoustic_subparsers(subparsers)
    return parser


def _acoustic_argv(
    command: str, benchmark_dir: pathlib.Path, *extra: str
) -> argparse.Namespace:
    return _arg_parser().parse_args(
        ["acoustic", command, "--benchmark-dir", str(benchmark_dir), *extra]
    )


def test_load_episode_summary_reads_the_canonical_view(tmp_path: pathlib.Path) -> None:
    df = _load_episode_summary(_benchmark(tmp_path))
    assert df["episode_id"].to_list() == [1, 2, 3]
    assert "l_aeq_t" in df.columns


def test_handle_acoustic_requires_star_schema_tables(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(_acoustic_argv("list", bench_dir))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "no Star Schema episode summary found" in out
    assert "Run 'evaluation process --benchmark-dir ...' first." in out


def test_handle_acoustic_missing_dir(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _handle_acoustic(_acoustic_argv("list", pathlib.Path("/nonexistent/dir")))
    assert exc.value.code == 1
    assert (
        "Error: benchmark directory does not exist: /nonexistent/dir"
        in capsys.readouterr().out
    )


def test_handle_acoustic_list_dispatch(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _handle_acoustic(_acoustic_argv("list", _benchmark(tmp_path)))
    assert "EPISODE" in capsys.readouterr().out


def test_map_for_episode_names_the_recorded_world() -> None:
    summary = pl.DataFrame(
        {"episode_id": [1, 2], "world": ["map1", "map2"]},
        schema={"episode_id": pl.Int64, "world": pl.String},
    )
    assert _map_for_episode(summary, "episode_001") == "map1"
    assert _map_for_episode(summary, "episode_002") == "map2"
    assert _map_for_episode(summary, "episode_009") is None


def test_episode_of_max_level_picks_the_loudest_episode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = pl.DataFrame(
        {"episode_id": [1, 2, 3], "l_aeq_t": [54.5, 72.4, None]},
        schema={"episode_id": pl.Int64, "l_aeq_t": pl.Float64},
    )
    assert _episode_of_max_level(summary, "worst") == "episode_002"
    assert "l_aeq_t=72.40" in capsys.readouterr().out


def test_episode_of_max_level_without_measurements(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = pl.DataFrame(
        {"episode_id": [1], "l_aeq_t": [None]},
        schema={"episode_id": pl.Int64, "l_aeq_t": pl.Float64},
    )
    assert _episode_of_max_level(summary, "worst") is None
    assert "has no values" in capsys.readouterr().out


def test_resolve_episode_accepts_a_recorded_episode(tmp_path: pathlib.Path) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "episode_002", tmp_path / "bench") == "episode_002"


def test_resolve_episode_rejects_an_absent_episode(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "episode_007", tmp_path / "bench") is None
    assert "not found in the episode summary" in capsys.readouterr().out


def test_resolve_episode_rejects_an_unknown_specifier(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "bogus", tmp_path / "bench") is None
    assert "unknown episode specifier 'bogus'" in capsys.readouterr().out


def test_resolve_episode_defaults_to_the_worst_level(tmp_path: pathlib.Path) -> None:
    summary = _load_episode_summary(_benchmark(tmp_path))
    assert _resolve_episode(summary, "worst", tmp_path / "bench") == "episode_002"
    assert _resolve_episode(summary, "max-total", tmp_path / "bench") == "episode_002"


def test_loudest_source_episode_picks_peak_source_from_telemetry(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench_dir = _benchmark(tmp_path)
    _write_episode_telemetry(bench_dir, 1, source_levels=[60.0, 91.5])
    _write_episode_telemetry(bench_dir, 2, source_levels=[65.0, 74.0])
    summary = _load_episode_summary(bench_dir)
    assert _loudest_source_episode(summary, bench_dir) == "episode_001"
    assert "source=91.5 dBA" in capsys.readouterr().out


def test_acoustic_snapshot_renders_png_from_catalog_and_world(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = _benchmark(tmp_path)
    _write_world(bench_dir, "map1")
    _write_episode_telemetry(bench_dir, 2, source_levels=[65.0, 78.0])
    out_png = tmp_path / "snapshot.png"
    summary = _load_episode_summary(bench_dir)
    args = _acoustic_argv(
        "snapshot", bench_dir, "--episode", "episode_002", "--output", str(out_png)
    )
    _acoustic_snapshot(summary, args)
    assert out_png.is_file()
    assert out_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_acoustic_snapshot_exits_when_source_dba_missing(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench_dir = _benchmark(tmp_path)
    _write_world(bench_dir, "map1")
    _write_episode_telemetry(bench_dir, 2, include_acoustics=False)
    summary = _load_episode_summary(bench_dir)
    args = _acoustic_argv("snapshot", bench_dir, "--episode", "episode_002")
    with pytest.raises(SystemExit) as exc:
        _acoustic_snapshot(summary, args)
    assert exc.value.code == 1
    assert "missing pose or source_dba" in capsys.readouterr().out


def test_acoustic_animate_renders_gif_from_catalog_and_world(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = _benchmark(tmp_path)
    _write_world(bench_dir, "map1")
    _write_episode_telemetry(bench_dir, 2, source_levels=[65.0, 78.0])
    out_gif = tmp_path / "anim.gif"
    summary = _load_episode_summary(bench_dir)
    args = _acoustic_argv(
        "animate",
        bench_dir,
        "--episode",
        "episode_002",
        "--fps",
        "4",
        "--max-frames",
        "2",
        "--output",
        str(out_gif),
    )
    _acoustic_animate(summary, args)
    assert out_gif.is_file()
    assert out_gif.stat().st_size > 0


def test_acoustic_list_prints_level_descending(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = pl.DataFrame(
        {
            "episode_id": [1, 2, 3],
            "l_aeq_t": [54.5, 72.4, None],
            "success": [True, False, True],
        },
        schema={
            "episode_id": pl.Int64,
            "l_aeq_t": pl.Float64,
            "success": pl.Boolean,
        },
    )
    _acoustic_list(summary)
    out = capsys.readouterr().out
    assert "EPISODE" in out and "L_AEQ_T_DBA" in out
    episodes = [
        line.split()[0] for line in out.splitlines() if line.startswith("episode_")
    ]
    assert episodes == ["episode_002", "episode_001"]
    assert "72.40" in out and "54.50" in out


def test_acoustic_list_without_measurements_prints_only_the_header(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = pl.DataFrame(
        {"episode_id": [1], "l_aeq_t": [None], "success": [True]},
        schema={
            "episode_id": pl.Int64,
            "l_aeq_t": pl.Float64,
            "success": pl.Boolean,
        },
    )
    _acoustic_list(summary)
    out = capsys.readouterr().out
    assert "EPISODE" in out
    assert not [line for line in out.splitlines() if line.startswith("episode_")]


def test_acoustic_flags_keep_their_names() -> None:
    args = _arg_parser().parse_args(
        [
            "acoustic",
            "animate",
            "--benchmark-dir",
            "/b",
            "--episode",
            "worst",
            "--fps",
            "5",
            "--max-frames",
            "10",
            "--format",
            "mp4",
            "--no-door-overlay",
        ]
    )
    assert (
        args.acoustic_command,
        args.episode,
        args.fps,
        args.max_frames,
        args.format,
        args.no_door_overlay,
    ) == ("animate", "worst", 5, 10, "mp4", True)
    snapshot = _arg_parser().parse_args(
        ["acoustic", "snapshot", "--benchmark-dir", "/b", "--frame", "3"]
    )
    assert (snapshot.acoustic_command, snapshot.frame) == ("snapshot", 3)
