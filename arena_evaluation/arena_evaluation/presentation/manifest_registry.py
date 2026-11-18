"""Report manifest resolution supporting names, file paths, and inline YAML."""

from __future__ import annotations

import pathlib

import yaml

from arena_evaluation.storage.config_layout import (
    bundle_path,
    run_analysis_root,
)

from .viz_manifest import VizManifest

__all__ = [
    "ManifestNotFoundError",
    "available_manifests",
    "find_manifest_file",
    "is_inline",
    "resolve_manifest",
]


def is_inline(ref: str) -> bool:
    stripped = ref.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def find_manifest_file(
    stem: str, benchmark_dir: pathlib.Path | None = None
) -> pathlib.Path | None:
    """Resolve a report name: run-local analysis/reports/<name>/report.yaml first, then asset chain."""
    if benchmark_dir is not None:
        local = bundle_path(run_analysis_root(benchmark_dir), "report", stem)
        if local.is_file():
            return local

    from arena_evaluation.benchmark.tree import ManifestIdentifier

    try:
        return ManifestIdentifier(name=stem).resolve_source_sync().path
    except FileNotFoundError:
        return None


def available_manifests() -> list[str]:
    """Sorted stems of all bundled manifests."""
    from arena_evaluation.benchmark.tree import ManifestIdentifier

    return sorted({m.shortname for m in ManifestIdentifier.listall()})


class ManifestNotFoundError(FileNotFoundError):
    """Raised when a named manifest cannot be resolved anywhere."""

    def __init__(self, name: str, message: str | None = None) -> None:
        self.name = name
        available = ", ".join(available_manifests()) or "(none bundled)"
        super().__init__(
            message
            or f"Report manifest '{name}' not found. Available: {available}. Pass a name, a path to a YAML file, or inline {{...}} YAML."
        )


def resolve_manifest(
    ref: str | None,
    benchmark_dir: pathlib.Path | None = None,
) -> VizManifest:
    """Resolve a manifest reference to a VizManifest instance."""
    if ref is None:
        return resolve_manifest("standard", benchmark_dir)

    ref = ref.strip()
    if is_inline(ref):
        data = yaml.safe_load(ref)
        if not isinstance(data, dict):
            raise ValueError(
                f"Inline manifest must be a YAML mapping, got {type(data).__name__}"
            )
        return VizManifest.model_validate(data)

    p = pathlib.Path(ref)
    if p.exists():
        return VizManifest.load(p)

    stem = ref.removesuffix(".yaml")
    p = find_manifest_file(stem, benchmark_dir)
    if p is None:
        raise ManifestNotFoundError(ref)
    obj = VizManifest.load(p)
    if obj.name is None:
        obj = obj.model_copy(update={"name": stem})
    return obj
