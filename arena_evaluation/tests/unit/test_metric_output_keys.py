import collections

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.registry import MetricRegistry

_PACKAGE = "arena_evaluation.processing.metrics."


def _shipped_calculators():
    """Every registered calculator in the package, ignoring test-local subclasses."""
    MetricRegistry.discover_calculators_cls()
    return [cls for cls in BaseMetricCalculator.__subclasses__() if cls.__module__.startswith(_PACKAGE) and cls.NAME]


def test_calculator_names_are_unique():
    names = collections.Counter(cls.NAME for cls in _shipped_calculators())
    assert [n for n, c in names.items() if c > 1] == []


def test_output_keys_are_globally_unique():
    """The registry merges every calculator into one dict, so a shared key silently
    overwrites, and a skipped calculator nulls the other one's value."""
    owners = collections.defaultdict(list)
    for cls in _shipped_calculators():
        for key in cls.output_keys():
            owners[key].append(cls.NAME)
    duplicated = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
    assert duplicated == {}


def test_units_and_directions_reference_real_outputs():
    for cls in _shipped_calculators():
        keys = set(cls.output_keys())
        assert set(cls.UNITS) <= keys, cls.NAME
        assert set(cls.OUTPUT_DIRECTIONS) <= keys, cls.NAME
        assert set(cls.PRIMARY_OUTPUTS) <= keys, cls.NAME


def test_dependencies_resolve_to_known_calculators():
    known = {cls.NAME for cls in _shipped_calculators()}
    for cls in _shipped_calculators():
        assert set(cls.DEPENDS_ON) <= known, cls.NAME
