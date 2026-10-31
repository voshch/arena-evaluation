import pathlib
import shutil

import polars as pl

from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "sample_benchmark"


def test_process_benchmark_writes_the_metrics_of_a_recorded_episode(tmp_path: pathlib.Path):
    shutil.copytree(FIXTURE, tmp_path / "sample_benchmark")
    fm = FolderManager(data_root=tmp_path)

    written = ProcessingPipeline(fm, workers=1).process_benchmark("sample_benchmark")

    assert written == fm.combined_metrics_path("sample_benchmark")
    metrics = pl.read_parquet(written)
    assert metrics.height == 1
    row = metrics.row(0, named=True)
    assert row["planner"] == "dwb"
    assert row["map"] == "barn_0"
    assert row["path_length"] > 0.0
