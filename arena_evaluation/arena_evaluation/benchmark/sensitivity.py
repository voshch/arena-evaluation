"""Sensitivity rerun facade of the runner layer."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import polars as pl

__all__ = ["rerun_sensitivity"]


def rerun_sensitivity(
    benchmark_dir: pathlib.Path,
    *,
    episode_ids: list[int] | None = None,
    detector_params: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """Run the in-memory detector-level sensitivity rerun of ``benchmark_dir``."""
    from arena_evaluation.processing.sensitivity import rerun_sensitivity as _rerun

    return _rerun(
        benchmark_dir,
        episode_ids=episode_ids,
        detector_params=None if detector_params is None else dict(detector_params),
    )
