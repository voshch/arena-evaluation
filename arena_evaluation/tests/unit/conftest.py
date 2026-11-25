"""Ensure metric column definitions are registered when unit test modules run standalone."""

from __future__ import annotations

import logging

import pytest

from arena_evaluation.processing.metrics.registry import load_metric_definitions

load_metric_definitions()


@pytest.fixture(scope="session", autouse=True)
def _restore_arena_logger_propagation() -> None:
    """Undo the ROS ``launch`` logger class so Arena log records reach ``caplog``."""
    logging.setLoggerClass(logging.Logger)
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.split(".")[0] in {
            "arena_evaluation",
            "arena_simulation_setup",
            "arena_runtime",
            "arena_bringup",
        }:
            logger.propagate = True
