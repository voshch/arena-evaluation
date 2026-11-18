"""Merged benchmark manifest and multi-run lazy composition without copying episodes."""

from __future__ import annotations

import dataclasses
import datetime
import pathlib

import yaml

from arena_evaluation.storage.atomic import atomic_write_text
from arena_evaluation.storage.star_schema import MERGE_MANIFEST_FILENAME, has_star_schema


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


def _labels(raw: object) -> dict[str, str]:
    """Coerce a label mapping to the string pairs the manifest round-trips."""
    return {str(key): str(value) for key, value in (raw or {}).items()}


def _manifest_source(entry: dict[str, object], benchmarks_root: pathlib.Path) -> MergeSource:
    """Resolve one manifest entry against the benchmarks root and validate its directory."""
    run_id = str(entry["run_id"])
    stored = pathlib.Path(str(entry["run_dir"]))
    resolved = stored if stored.is_absolute() else benchmarks_root / stored
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Merged benchmark references source run {run_id!r} at {resolved}, which does not "
            "exist; fix the merge manifest or restore the run"
        )
    return MergeSource(run_id, str(resolved), _labels(entry.get("labels")))


def load_merge_manifest(benchmark_dir: pathlib.Path) -> MergeManifest | None:
    """Load ``merge.yaml`` when present, validating every source and resolving relative run dirs."""
    manifest_path = benchmark_dir / MERGE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid merge manifest in {manifest_path}")
    entries = raw.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Merge manifest {manifest_path} has no sources")
    benchmarks_root = benchmark_dir.resolve().parent
    return MergeManifest(
        str(raw.get("name") or benchmark_dir.name),
        tuple(_manifest_source(entry, benchmarks_root) for entry in entries),
        str(raw.get("created_at") or ""),
    )


def _source_record(
    benchmarks_dir: pathlib.Path, run_id: str, labels: dict[str, dict[str, str]]
) -> dict[str, object]:
    """Validate one source run and return its root-relative manifest entry."""
    source_dir = benchmarks_dir / run_id
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source run {run_id!r} does not exist")
    if not has_star_schema(source_dir):
        raise ValueError(f"Source run {run_id!r} has no Star Schema tables")
    return {
        "run_id": run_id,
        "run_dir": source_dir.relative_to(benchmarks_dir).as_posix(),
        "labels": _labels(labels.get(run_id, {})),
    }


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
    sources = [_source_record(benchmarks_dir, run_id, label_map) for run_id in run_ids]
    target.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        target / MERGE_MANIFEST_FILENAME,
        yaml.safe_dump(
            {
                "name": name,
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                "sources": sources,
            },
            sort_keys=False,
        ),
    )
    return target
