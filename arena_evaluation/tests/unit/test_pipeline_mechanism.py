import numpy as np
import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _align_bundle, _metric_world
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata, TopicBundle


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


def test_align_bundle_keeps_the_whole_native_track_through_a_teleport():
    x = [0.0, 0.5, 1.0, 101.0, 101.5, 102.0, 102.5]
    frame = pl.DataFrame(
        {
            "t_ns": (np.arange(7) * 1_000_000_000).tolist(),
            "x": x,
            "y": [0.0] * 7,
            "yaw": [0.0] * 7,
        }
    )

    # D3: the aligned frame is the native /tf track for the whole episode window.
    aligned = _align_bundle(TopicBundle(robot_tracks=frame))
    assert aligned is not None
    assert len(aligned) == 7
