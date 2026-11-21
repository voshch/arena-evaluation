"""Tests verifying alignment helpers sit at a level both L1 pipeline and L2 metrics may import."""

from __future__ import annotations

import ast
import pathlib

import arena_evaluation.processing.pedestrian_frames as pedestrian_frames

METRIC_PACKAGE: str = "arena_evaluation.processing.metrics"


def test_l1_pedestrian_frames_do_not_import_the_l2_metric_package() -> None:
    tree = ast.parse(
        pathlib.Path(pedestrian_frames.__file__).read_text(encoding="utf-8")
    )
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not [module for module in imported if module.startswith(METRIC_PACKAGE)]


def test_the_alignment_helpers_import_from_the_processing_level_module() -> None:
    from arena_evaluation.processing.alignment import asof_join, interpolate_values

    assert callable(asof_join)
    assert callable(interpolate_values)
