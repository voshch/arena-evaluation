"""Merged benchmark manifest and multi-run lazy composition without copying episodes."""

from __future__ import annotations

import dataclasses
import datetime
import pathlib

import yaml

from arena_evaluation.storage.atomic import atomic_write_text
from arena_evaluation.storage.star_schema import (
    MERGE_MANIFEST_FILENAME,
    has_star_schema,
)


@dataclasses.dataclass(frozen=True, slots=True)
class MergeSource:
    """One source benchmark referenced by a merged benchmark manifest."""

    run_id: str
    run_dir: str
    labels: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class MergeManifest:
    """Manifest describing a merged benchmark composed of multiple source runs."""

    name: str
    sources: tuple[MergeSource, ...]
    created_at: str


def load_merge_manifest(benchmark_dir: pathlib.Path) -> MergeManifest | None:
    """Load ``merge.yaml`` from ``benchmark_dir`` when present, otherwise return None."""
    manifest_path = benchmark_dir / MERGE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid merge manifest in {manifest_path}")
    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"Merge manifest {manifest_path} has no sources")
    sources = tuple(
        MergeSource(
            run_id=str(entry["run_id"]),
            run_dir=str(entry["run_dir"]),
            labels={str(k): str(v) for k, v in (entry.get("labels") or {}).items()},
        )
        for entry in raw_sources
    )
    return MergeManifest(
        name=str(raw.get("name") or benchmark_dir.name),
        sources=sources,
        created_at=str(raw.get("created_at") or ""),
    )


def create_merged_benchmark(
    benchmarks_dir: pathlib.Path,
    *,
    run_ids: list[str],
    name: str,
    labels: dict[str, dict[str, str]] | None = None,
) -> pathlib.Path:
    """Create a merged benchmark directory referencing ``run_ids`` without copying episodes."""
    if not run_ids:
        raise ValueError("run_ids must contain at least one run")
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"Invalid merged benchmark name {name!r}")
    target = benchmarks_dir / name
    if target.exists():
        raise FileExistsError(f"Benchmark {name!r} already exists at {target}")

    label_map = labels or {}
    sources: list[dict[str, object]] = []
    for run_id in run_ids:
        source_dir = benchmarks_dir / run_id
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source run {run_id!r} does not exist")
        if not has_star_schema(source_dir):
            raise ValueError(f"Source run {run_id!r} has no Star Schema tables")
        sources.append(
            {
                "run_id": run_id,
                "run_dir": str(source_dir.resolve()),
                "labels": {
                    str(k): str(v) for k, v in label_map.get(run_id, {}).items()
                },
            }
        )

    target.mkdir(parents=True, exist_ok=False)
    payload = {
        "name": name,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "sources": sources,
    }
    atomic_write_text(
        target / MERGE_MANIFEST_FILENAME,
        yaml.safe_dump(payload, sort_keys=False),
    )
    return target
