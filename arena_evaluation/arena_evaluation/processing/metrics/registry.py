from __future__ import annotations

import functools
import importlib
import inspect
import pkgutil
import re
import types
import typing
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    NULL_REASON_SUFFIX,
    BaseMetricCalculator,
    ScopeExecutionOutcome,
    ScopeSlice,
)
from arena_evaluation.storage.exceptions import CircularDependencyError, MetricInputError, SchemaViolationError
from arena_evaluation.storage.registry import MetricScope, defined_metrics
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

_METRICS_PACKAGE: typing.Final[str] = "arena_evaluation.processing.metrics."
_SERIES_STEM_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _series_table_stem(class_name: str) -> str:
    """Snake-case calculator class name without its Calculator suffix, the 1NF series table stem."""
    return _SERIES_STEM_BOUNDARY.sub("_", class_name.removesuffix("Calculator")).lower()


def _leaf_calculator_classes(root: type[BaseMetricCalculator]) -> list[type[BaseMetricCalculator]]:
    """Concrete calculator subclasses of the metrics package; test-local classes excluded."""
    leaves: list[type[BaseMetricCalculator]] = []
    for sub in root.__subclasses__():
        if not inspect.isabstract(sub) and "<locals>" not in sub.__qualname__ and sub.__module__.startswith(_METRICS_PACKAGE):
            leaves.append(sub)
        leaves.extend(_leaf_calculator_classes(sub))
    return leaves


def scope_producers(calculators: Iterable[type[BaseMetricCalculator] | BaseMetricCalculator]) -> dict[tuple[str, MetricScope], str]:
    """The calculator publishing each metric at each scope; two calculators publishing one pair are refused."""
    producers: dict[tuple[str, MetricScope], str] = {}
    for calc in calculators:
        for metric in calc.OUTPUTS:
            for scope in calc.scopes_of_output(metric):
                owner = producers.setdefault((metric.name, scope), calc.NAME)
                if owner != calc.NAME:
                    raise SchemaViolationError(f"{metric.name!r} at {scope} is produced by both {owner} and {calc.NAME}")
    return producers


def _topic_row_count(table: pl.DataFrame | pl.LazyFrame | None) -> int:
    """Row count of a materialized or lazy topic table, 0 when None."""
    if table is None:
        return 0
    if isinstance(table, pl.LazyFrame):
        return int(table.select(pl.len()).collect().item())
    return len(table)


def _check_declared_keys(calc: BaseMetricCalculator, scope: MetricScope, produced: Mapping[str, object]) -> None:
    """Refuse keys no scope of the calculator declares, and declared keys of this scope it did not produce."""
    undeclared = sorted(set(produced) - set(calc.output_keys()))
    unproduced = sorted(set(calc.output_keys_for_scope(scope)) - set(produced))
    if undeclared:
        raise SchemaViolationError(f"{calc.NAME} produced undeclared keys: {undeclared}")
    if unproduced:
        raise SchemaViolationError(f"{calc.NAME} did not produce its declared keys at {scope}: {unproduced}")


def _mark_unavailable(keys: tuple[str, ...], reason: str, *, published: dict[str, typing.Any], provenance: dict[str, str]) -> None:
    """Publish this scope's keys as NULL with their reason; they shadow an earlier scope's value of the same name."""
    for key in keys:
        published[key] = None
        provenance[key] = f"{key}:{reason}"


def _is_nan(value: object) -> bool:
    """True for a float or numpy float NaN, the calculators' undefined value."""
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


def _is_defined(value: object) -> bool:
    """A published value is defined when it is neither None nor NaN."""
    return value is not None and not _is_nan(value)


def _published(results: Mapping[str, object]) -> dict[str, typing.Any]:
    """Outcome values with NaN replaced by None: an undefined value leaves the registry as NULL, never as a number."""
    return {key: None if _is_nan(value) else value for key, value in results.items()}


class MetricRegistry:
    """Discovers, validates, and executes metric calculators across all 5 evaluation scopes."""

    def __init__(
        self,
        robot_params: RobotParams,
        world: str | None = None,
        metrics_config: dict[str, typing.Any] | None = None,
        calculators: Sequence[type[BaseMetricCalculator]] | None = None,
    ):
        """Register the given calculator classes, or every shipped calculator when none are given."""
        self.robot_params = robot_params
        self.world = world
        self.metrics_config = dict(metrics_config or {})
        self.calculators: dict[str, BaseMetricCalculator] = {}
        self.calculators_by_scope: dict[MetricScope, dict[str, BaseMetricCalculator]] = {scope: {} for scope in MetricScope}
        self.execution_stages_by_scope: dict[MetricScope, list[list[str]]] = {scope: [] for scope in MetricScope}
        self.execution_stages: list[list[str]] = []
        if calculators is None:
            calculators = shipped_calculators()
        self._register_calculators(calculators)
        self._check_single_producer()
        for name, calc in self.calculators.items():
            for scope in calc.effective_scopes():
                if name not in self.calculators_by_scope[scope]:
                    self.calculators_by_scope[scope][name] = calc

        self.execution_stages_by_scope = {scope: self._compute_execution_order(scope) for scope in MetricScope}
        self.execution_stages = self.execution_stages_by_scope[MetricScope.EPISODE]

    @classmethod
    def discover_calculators_cls(cls) -> None:
        """Recursively discover all BaseMetricCalculator subclasses in the metrics package."""
        import arena_evaluation.processing.metrics as metrics_pkg

        def iter_namespace(pkg: types.ModuleType) -> typing.Iterator[pkgutil.ModuleInfo]:
            return pkgutil.iter_modules(pkg.__path__, pkg.__name__ + ".")

        for _, name, _ in iter_namespace(metrics_pkg):
            importlib.import_module(name)

        # Recursive discovery for subpackages
        for _, name, ispkg in iter_namespace(metrics_pkg):
            if ispkg:
                sub_pkg = importlib.import_module(name)
                for _, sub_name, _ in iter_namespace(sub_pkg):
                    importlib.import_module(sub_name)

    def _register_calculators(self, calculators: Iterable[type[BaseMetricCalculator]]) -> None:
        """Register each calculator class, indexed by name and by scope."""
        for calc_cls in calculators:
            if not calc_cls.NAME:
                raise ValueError(f"Calculator {calc_cls.__name__} must define a NAME")
            if calc_cls.NAME in self.calculators:
                raise ValueError(f"Duplicate calculator NAME: {calc_cls.NAME}")
            self.register(calc_cls)

    def _check_single_producer(self) -> None:
        """Refuse two calculators publishing the same metric at the same scope."""
        scope_producers(self.calculators.values())

    def register(self, calculator: type[BaseMetricCalculator] | BaseMetricCalculator) -> None:
        """Register a calculator class or instance across all of its effective scopes."""
        calc = calculator(self.robot_params) if isinstance(calculator, type) else calculator
        calc.world = self.world
        calc.metrics_config = self.metrics_config
        self.calculators[calc.NAME] = calc
        for scope in calc.effective_scopes():
            self.calculators_by_scope[scope][calc.NAME] = calc
            self.execution_stages_by_scope[scope] = self._compute_execution_order(scope)
        self.execution_stages = self.execution_stages_by_scope[MetricScope.EPISODE]

    def series_calculators(self) -> list[tuple[str, BaseMetricCalculator]]:
        """Registered calculators publishing a 1NF series table, with their table stem, ordered by stem."""
        return sorted(
            ((calc.SERIES_TABLE_STEM or _series_table_stem(type(calc).__name__), calc) for calc in self.calculators.values() if type(calc).series_table is not BaseMetricCalculator.series_table),
            key=lambda item: item[0],
        )

    @classmethod
    def get_all_units(cls) -> dict[str, str]:
        """Unit of every defined metric, read from the shared metric store."""
        cls.discover_calculators_cls()
        return {metric.name: metric.unit for metric in defined_metrics()}

    def _compute_execution_order(self, scope: MetricScope = MetricScope.EPISODE) -> list[list[str]]:
        """Compute topological sort of calculators in `scope` using Kahn's algorithm."""
        scope_calcs = self.calculators_by_scope[scope]
        in_degree = {name: 0 for name in scope_calcs}
        adj_list = defaultdict(list)

        for name, calc in scope_calcs.items():
            for dep in calc.DEPENDS_ON:
                if dep in scope_calcs:
                    adj_list[dep].append(name)
                    in_degree[name] += 1

        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        stages = []
        processed_count = 0

        while queue:
            stage_size = len(queue)
            current_stage = []

            for _ in range(stage_size):
                u = queue.popleft()
                current_stage.append(u)
                processed_count += 1

                for v in adj_list[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

            stages.append(current_stage)

        if processed_count != len(scope_calcs):
            raise CircularDependencyError(f"Circular dependency detected among metric calculators in scope '{scope}'")

        return stages

    def get_execution_order_for_scope(self, scope: MetricScope) -> list[list[str]]:
        """Return topological execution stages of calculators registered for `scope`."""
        return self.execution_stages_by_scope[scope]

    def list_metrics(self) -> list[dict[str, typing.Any]]:
        """Returns metadata about all registered metrics."""
        return [
            {
                "name": name,
                "category": calc.CATEGORY,
                "scopes": sorted(s.value for s in calc.effective_scopes()),
                "requires_pedsim": calc.REQUIRES_PEDSIM,
                "depends_on": calc.DEPENDS_ON,
                "required_topics": calc.REQUIRED_TOPICS,
                "outputs": calc.output_keys(),
                "primary_outputs": list(calc.PRIMARY_OUTPUTS),
                "output_directions": {name: str(direction) for name, direction in calc.DIRECTIONS.items()},
            }
            for name, calc in self.calculators.items()
        ]

    def execution_order(self) -> list[list[str]]:
        """Returns the ordered stages of calculation."""
        return self.execution_stages

    def run_scope(
        self,
        scope: MetricScope,
        episode: AlignedEpisodeBundle,
        *,
        available_topics: set[str] | None = None,
        prior_results: dict[str, typing.Any] | None = None,
        pedsim_available: bool = True,
        progress_callback: typing.Callable[[str, int, int], None] | None = None,
        scope_slice: ScopeSlice | None = None,
    ) -> ScopeExecutionOutcome:
        """Executes all calculators registered for `scope` in topological order."""
        slice_episode = scope_slice.window_bundle if scope_slice is not None else episode
        full_episode = scope_slice.full_episode if scope_slice is not None else episode
        full_topics = full_episode.topics if isinstance(full_episode.topics, Mapping) else {}
        slice_topics = slice_episode.topics if isinstance(slice_episode.topics, Mapping) else {}

        if available_topics is not None:
            present_topics = {topic for topic in available_topics if full_topics.get(topic, True) is not None}
        else:
            present_topics = {name for name, table in full_topics.items() if table is not None}

        prior = dict(prior_results or {})
        published: dict[str, typing.Any] = {}
        provenance: dict[str, str] = {}
        children: dict[str, typing.Any] = {}
        stages = self.execution_stages_by_scope[scope]
        total_calcs = sum(len(stage) for stage in stages)
        calc_idx = 0

        for stage in stages:
            for calc_name in stage:
                calc_idx += 1
                calc = self.calculators_by_scope[scope][calc_name]
                scope_keys = calc.output_keys_for_scope(scope)

                missing_topic: str | None = None
                empty_slice_topic = False
                for req in calc.required_topics_for_scope(scope):
                    if isinstance(req, str):
                        if req not in present_topics:
                            missing_topic = req
                            break
                        if req in slice_topics and _topic_row_count(slice_topics[req]) == 0:
                            empty_slice_topic = True
                    else:
                        if not any(t in present_topics for t in req):
                            missing_topic = req[0] if req else "unknown"
                            break
                        if all(t in slice_topics for t in req) and all(_topic_row_count(slice_topics[t]) == 0 for t in req):
                            empty_slice_topic = True

                if missing_topic is not None:
                    _mark_unavailable(scope_keys, f"topic_missing:{missing_topic}", published=published, provenance=provenance)
                    continue

                if empty_slice_topic:
                    _mark_unavailable(scope_keys, "no_samples", published=published, provenance=provenance)
                    continue

                missing_series = next(
                    (stem for stem in calc.SERIES_DEPENDS_ON if (frame := (full_episode.series or {}).get(stem)) is None or frame.height == 0),
                    None,
                )
                if missing_series is not None:
                    _mark_unavailable(scope_keys, f"not_recorded:{missing_series}", published=published, provenance=provenance)
                    continue

                if calc.REQUIRES_PEDSIM and not pedsim_available:
                    _mark_unavailable(scope_keys, "pedsim_unavailable", published=published, provenance=provenance)
                    continue

                if progress_callback is not None:
                    try:
                        progress_callback(calc_name, calc_idx, total_calcs)
                    except Exception:
                        pass

                try:
                    calc_out = dict(calc.calculate_for_scope(slice_episode, {**prior, **published}, scope_slice))
                    children.update({key: calc_out.pop(key) for key in calc.CHILD_OUTPUTS if key in calc_out})
                    null_reasons: dict[str, str] = {}
                    for key in calc.output_keys():
                        companion = calc_out.pop(f"{key}{NULL_REASON_SUFFIX}", None)
                        if isinstance(companion, str) and companion:
                            null_reasons[key] = companion
                    _check_declared_keys(calc, scope, calc_out)
                    own = {key: calc_out[key] for key in scope_keys}
                    published.update(own)
                    missing_inputs = calc.missing_input_provenance(slice_episode)
                    for key in scope_keys:
                        if _is_defined(own[key]):
                            provenance.pop(key, None)
                        elif key not in provenance:
                            provenance[key] = missing_inputs.get(key, f"{key}:{null_reasons.get(key, 'no_samples')}")

                except (MetricInputError, SchemaViolationError):
                    raise
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).warning(f"Calculator {calc_name} failed on episode {slice_episode.episode_id}: {exc}")
                    _mark_unavailable(scope_keys, f"exception:{type(exc).__name__}", published=published, provenance=provenance)

        return ScopeExecutionOutcome(scope=scope, results=_published(published), provenance=provenance, children=children)

    def run(
        self,
        episode: AlignedEpisodeBundle,
        *,
        available_topics: set[str] | None = None,
        pedsim_available: bool = True,
        progress_callback: typing.Callable[[str, int, int], None] | None = None,
    ) -> dict[str, typing.Any]:
        """Executes EPISODE scope calculators in topological order."""
        outcome = self.run_scope(
            MetricScope.EPISODE,
            episode,
            available_topics=available_topics,
            pedsim_available=pedsim_available,
            progress_callback=progress_callback,
        )
        return outcome.results


def shipped_calculators() -> list[type[BaseMetricCalculator]]:
    """Every concrete calculator class of the metrics package, found by importing the package's modules."""
    MetricRegistry.discover_calculators_cls()
    return _leaf_calculator_classes(BaseMetricCalculator)


@functools.cache
def load_metric_definitions() -> None:
    """Import every calculator module once, so the shared metric store, and the fact schemas built from it, is the same in every process."""
    MetricRegistry.discover_calculators_cls()
