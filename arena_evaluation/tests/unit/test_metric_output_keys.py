from __future__ import annotations

import inspect

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.registry import MetricRegistry

_PACKAGE = "arena_evaluation.processing.metrics."


def _shipped_calculators() -> list[type[BaseMetricCalculator]]:
    """Every concrete calculator class of the package, found by walking the subclass tree."""
    MetricRegistry.discover_calculators_cls()
    found: list[type[BaseMetricCalculator]] = []
    pending = list(BaseMetricCalculator.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if cls.__module__.startswith(_PACKAGE) and cls.NAME and not inspect.isabstract(cls):
            found.append(cls)
    return found


