"""Ensure metric column definitions are registered when unit test modules run standalone."""

from __future__ import annotations

import logging

import pytest

from arena_evaluation.processing.metrics.registry import load_metric_definitions

load_metric_definitions()


@pytest.fixture(scope="session", autouse=True)
def _restore_arena_logger_propagation() -> None:
    """Undo the ROS ``launch`` logger class, which would hide Arena log records from ``caplog``.

    ``arena_rclpy_mixins`` imports ``launch``, and ``launch.logging`` installs a logger class whose
    instances set ``propagate = False``. Every Arena logger created after that import (as the
    collected test modules are) would then never reach the root handler pytest captures. Restore the
    stock class and let the already-created Arena loggers propagate again; this is test infrastructure
    only and does not change where the application logs.
    """
    logging.setLoggerClass(logging.Logger)
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.split(".")[0] in {
            "arena_evaluation",
            "arena_simulation_setup",
            "arena_runtime",
            "arena_bringup",
        }:
            logger.propagate = True
