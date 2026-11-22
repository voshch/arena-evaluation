import pathlib
import shutil

import polars as pl

from arena_evaluation.ingestion.migrate import migrate_run
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.registry.facts import FactEpisodes

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "sample_benchmark"


def test_process_benchmark_writes_the_star_schema_of_a_recorded_episode(
    tmp_path: pathlib.Path,
):
    shutil.copytree(FIXTURE, tmp_path / "sample_benchmark")
    migrate_run(tmp_path / "sample_benchmark")
    fm = FolderManager(data_root=tmp_path)

    written = ProcessingPipeline(fm, workers=1).process_benchmark("sample_benchmark")

    assert written == fm.benchmark_dir("sample_benchmark")
    assert (written / "dim_episodes.parquet").is_file()
    facts = FactEpisodes.scan(written / "episodes" / "episode_000").collect()
    assert facts.height == 1
    dimensions = pl.read_parquet(written / "dim_episodes.parquet")
    assert dimensions.height == 1
    episode = dimensions.row(0, named=True)
    assert episode["planner"] == "dwb"
    assert episode["world"] == "barn_0"
