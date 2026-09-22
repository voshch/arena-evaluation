import pytest
import pathlib
import os

from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager


@pytest.mark.skip(reason="Requires valid sample MCAP data to run end-to-end")
def test_end_to_end_pipeline():
    fixtures_dir = pathlib.Path(__file__).parent.parent / "fixtures" / "sample_benchmark"
    fm = FolderManager(data_root=fixtures_dir)

    pipeline = ProcessingPipeline(fm)

    pipeline.process_benchmark("sample_benchmark")

    assert fm.combined_metrics_path("sample_benchmark").exists()
