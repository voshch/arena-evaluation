import pathlib
import tempfile
import polars as pl
import numpy as np
from arena_evaluation.processing.metrics.naturalness.trajectory import TrajectoryMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams, RunDescriptor
from arena_evaluation.processing.parquet_store import TopicParquetStore, ParquetStore
from arena_evaluation.storage.folder_manager import FolderManager


def test_trajectory_metrics_basic_and_irregularity():
    pos_x = np.array([0.0, 1.0, 2.0])
    pos_y = np.array([0.0, 0.0, 0.0])
    yaw = np.array([0.0, 0.0, 0.0])

    data = pl.DataFrame({"ts_iso": ["a", "b", "c"], "pos_x_gt": pos_x, "pos_y_gt": pos_y, "yaw_gt": yaw})

    bundle = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0.0, 0.0], goal_pos=[2.0, 0.0])

    calc = TrajectoryMetricsCalculator(RobotParams(0.5, 0.1, 10.0))
    res = calc.calculate(bundle, {})

    assert res["path_irregularity"] is not None
    assert abs(res["path_irregularity"]) < 1e-5  # straight line -> 0 irregularity


def test_topological_complexity():
    pos_x = np.array([0.0, 1.0, 2.0])
    pos_y = np.array([0.0, 0.0, 0.0])
    yaw = np.array([0.0, 0.0, 0.0])

    data = pl.DataFrame({"time_ns": [0, 1000, 2000], "ts_iso": ["a", "b", "c"], "pos_x_gt": pos_x, "pos_y_gt": pos_y, "yaw_gt": yaw})

    peds = pl.DataFrame({"time_ns": [0, 1000, 2000], "ts_iso": ["a", "b", "c"], "ped_id": [1, 1, 1], "pos_x": [1.0, 1.0, 1.0], "pos_y": [1.0, 1.0, 1.0]})

    bundle = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0.0, 0.0], goal_pos=[2.0, 0.0], peds=peds)

    calc = TrajectoryMetricsCalculator(RobotParams(0.5, 0.1, 10.0))
    res = calc.calculate(bundle, {})

    assert res["topological_complexity"] is not None
    assert res["ade"] is None
    assert res["fde"] is None
    assert res["adtw"] is None


def test_reference_path_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = pathlib.Path(tmpdir)
        fm = FolderManager(data_root)

        benchmark_id = "test_bm"
        planner = "dwa"
        stage = "stage1"

        main_run_dir = fm.run_dir(benchmark_id, planner, stage)
        ref_run_dir = fm.run_dir(benchmark_id, f"{planner}_unobstructed_robot", stage)

        # Save reference run topic parquet
        ref_topics_dir = fm.extracted_topics_path(ref_run_dir)
        ref_data = pl.DataFrame({"time_ns": [0, 1000, 2000, 3000, 4000, 5000], "pos_x": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0], "pos_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

        from arena_evaluation.storage.schemas import TopicBundle

        ref_bundle = TopicBundle(odom=ref_data)
        TopicParquetStore.write({"robot": ref_bundle}, ref_topics_dir)

        # Create test episode bundle referencing main run
        data = pl.DataFrame({"time_ns": [0, 1000, 2000, 3000, 4000, 5000], "ts_iso": ["a", "b", "c", "d", "e", "f"], "pos_x_gt": np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0]), "pos_y_gt": np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), "yaw_gt": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])})

        desc = RunDescriptor(run_dir=str(main_run_dir), benchmark_id=benchmark_id, planner=planner, stage=stage)

        ep_bundle = AlignedEpisodeBundle(
            episode_id=0,  # fallback episode id matching single episode in split
            data=data,
            start_pos=[0.0, 0.0],
            goal_pos=[2.0, 1.0],
            robot_name="robot",
            run=desc,
            folder_manager=fm,
        )

        calc = TrajectoryMetricsCalculator(RobotParams(0.5, 0.1, 10.0))
        res = calc.calculate(ep_bundle, {})

        assert res["fde"] is not None
        assert abs(res["fde"] - 1.0) < 1e-4  # y difference at goal: 1.0 - 0.0 = 1.0


def test_empty_pose_handling():
    data = pl.DataFrame({"ts_iso": [], "pos_x": [], "pos_y": [], "yaw": []})
    bundle = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0.0, 0.0], goal_pos=[2.0, 0.0])
    calc = TrajectoryMetricsCalculator(RobotParams(0.5, 0.1, 10.0))
    res = calc.calculate(bundle, {})
    assert all(v is None for v in res.values())
