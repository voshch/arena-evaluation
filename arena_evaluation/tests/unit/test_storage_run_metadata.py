"""MetadataWriter.read refuses metadata that still needs the migration to modes and params."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from arena_evaluation.storage.exceptions import UnmigratedMetadataError
from arena_evaluation.storage.manifest import MetadataWriter

_METADATA: dict[str, object] = {
    "benchmark_id": "bench",
    "planner": "dwa",
    "map": "map_empty",
    "stage": "stage_1",
    "recording_started_at": "2026-11-16T00:00:00",
    "python_version": "3.12",
    "ros_distro": "jazzy",
    "modes": {"robots": ["jackal"], "obstacles": ["static"]},
    "params": {"task.jackal.speed": 1.0},
}


def _write_metadata(tmp_path: pathlib.Path, data: dict[str, object]) -> pathlib.Path:
    path = tmp_path / "episode_000.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_metadata_with_a_tm_field_is_refused(tmp_path: pathlib.Path):
    path = _write_metadata(tmp_path, {**_METADATA, "tm_robots": "jackal"})

    with pytest.raises(UnmigratedMetadataError, match="tm_robots"):
        MetadataWriter.read(path)


@pytest.mark.parametrize("missing", ["modes", "params"])
def test_metadata_without_a_required_field_is_refused(
    tmp_path: pathlib.Path, missing: str
):
    data = {key: value for key, value in _METADATA.items() if key != missing}
    path = _write_metadata(tmp_path, data)

    with pytest.raises(UnmigratedMetadataError, match=missing):
        MetadataWriter.read(path)
