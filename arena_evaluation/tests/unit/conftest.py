"""Ensure metric column definitions are registered when unit test modules run standalone."""

from __future__ import annotations

from arena_evaluation.processing.metrics.registry import load_metric_definitions

load_metric_definitions()
