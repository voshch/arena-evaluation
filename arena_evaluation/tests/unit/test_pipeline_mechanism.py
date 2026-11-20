import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _metric_world  # noqa: E402
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata  # noqa: E402


def _ep(map_name="unknown"):
    return EpisodeDescriptor(
        episode_dir="/nowhere",
        benchmark_id="b",
        episode_id=1,
        planner="p",
        stage="s",
        map=map_name,
    )


def test_metric_world_prefers_metadata_map():
    meta = RunMetadata.model_construct(
        benchmark_id="b", planner="p", map="mechanisms_elevator", stage="s"
    )
    assert _metric_world(meta, _ep("other")) == "mechanisms_elevator"
    assert _metric_world(None, _ep("hospital_1")) == "hospital_1"
    assert _metric_world(None, _ep()) is None
