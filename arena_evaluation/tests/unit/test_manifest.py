import json
from arena_evaluation.storage.schemas import RunMetadata
import datetime


def test_run_metadata_serialization():
    dt = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta = RunMetadata(benchmark_id="test_bench", planner="teb", robot_model=["jackal"], map="map1", stage="stage1", episodes_requested=10, suite_name="suite1", contest_name="contest1", inter_planner="", agent_name="", recording_started_at=dt, python_version="3.10", ros_distro="humble")

    dumped = meta.model_dump(exclude_none=True)
    assert dumped["benchmark_id"] == "test_bench"
    assert "recording_ended_at" not in dumped

    reloaded = RunMetadata.model_validate(dumped)
    assert reloaded.planner == "teb"
    assert reloaded.robot_model == ["jackal"]
    assert reloaded.recording_ended_at is None
