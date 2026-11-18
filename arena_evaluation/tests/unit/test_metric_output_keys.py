from __future__ import annotations

import collections
import inspect

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.registry import MetricScope, defined_metrics

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


def test_calculator_names_are_unique() -> None:
    names = collections.Counter(cls.NAME for cls in _shipped_calculators())

    assert [name for name, count in names.items() if count > 1] == []


def test_output_keys_are_unique_within_each_scope() -> None:
    """The registry merges calculators of one scope into one dict, so a shared key silently overwrites."""
    owners: dict[tuple[MetricScope, str], list[str]] = collections.defaultdict(list)
    for cls in _shipped_calculators():
        for scope in cls.effective_scopes():
            for key in cls.output_keys():
                owners[(scope, key)].append(cls.NAME)

    duplicated = {f"{scope}:{key}": names for (scope, key), names in owners.items() if len(names) > 1}
    assert duplicated == {}


def test_every_shipped_output_is_a_declared_metric() -> None:
    declared = {metric.name for metric in defined_metrics()}
    for cls in _shipped_calculators():
        assert {metric.name for metric in cls.OUTPUTS} <= declared, cls.NAME


def test_every_declared_metric_has_a_shipped_producer() -> None:
    produced = {metric.name for cls in _shipped_calculators() for metric in cls.OUTPUTS}
    declared = {metric.name for metric in defined_metrics()}

    assert declared - produced == set()


def test_directions_and_primary_outputs_reference_real_outputs() -> None:
    for cls in _shipped_calculators():
        keys = set(cls.output_keys())
        assert set(cls.DIRECTIONS) <= keys, cls.NAME
        assert set(cls.PRIMARY_OUTPUTS) <= keys, cls.NAME


def test_dependencies_resolve_to_known_calculators() -> None:
    known = {cls.NAME for cls in _shipped_calculators()}
    for cls in _shipped_calculators():
        assert set(cls.DEPENDS_ON) <= known, cls.NAME
