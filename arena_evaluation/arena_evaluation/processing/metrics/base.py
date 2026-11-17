from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np
import polars as pl

from ...storage.registry import Metric, MetricScope
from .common.alignment import TRACK_ALIGN_TOLERANCE_NS, asof_values, interpolate_values

if typing.TYPE_CHECKING:
    from ...storage.schemas import AlignedEpisodeBundle, RobotParams

# A calculator may return `<output>_null_reason` next to a null output; the registry turns that
# companion string into the output's provenance code instead of the generic no_samples.
NULL_REASON_SUFFIX: typing.Final[str] = "_null_reason"


class Direction(StrEnum):
    """Whether a higher or a lower value of a metric is the better one."""

    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeSlice:
    """Immutable spatiotemporal slice for one evaluation scope entity or window."""

    scope: MetricScope
    slice_id: str
    parent_id: str | None = None
    t_start_ns: int
    t_end_ns: int
    identity: Mapping[str, typing.Any]
    window_bundle: AlignedEpisodeBundle
    full_episode: AlignedEpisodeBundle


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeExecutionOutcome(Mapping[str, typing.Any]):
    """Output dictionary and per-key missingness provenance of a single scope execution."""

    scope: MetricScope
    results: dict[str, typing.Any]
    provenance: dict[str, str]
    children: Mapping[str, typing.Any] = MappingProxyType({})

    def __getitem__(self, key: str) -> object:
        return self.results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


class BaseMetricCalculator(ABC):
    """Abstract Base Class for all metric calculators."""

    NAME: str = ""
    SERIES_TABLE_STEM: typing.ClassVar[str] = ""
    CATEGORY: str = "general"
    REQUIRES_PEDSIM: bool = False
    DEPENDS_ON: list[str] = []
    REQUIRED_TOPICS: list[str | list[str] | tuple[str, ...] | set[str]] = []
    # Metrics this calculator publishes; each is defined once beside this class with define_metric.
    OUTPUTS: typing.ClassVar[tuple[Metric, ...]] = ()
    # Parameters this calculator reads, so a what-if knows which calculators to rerun.
    PARAMS: typing.ClassVar[tuple[str, ...]] = ()
    # Better side of each output that has one; an output absent here has none.
    DIRECTIONS: typing.ClassVar[Mapping[str, Direction]] = MappingProxyType({})
    # Scopes at which this calculator publishes a metric; absent names use the metric's own scopes.
    OUTPUT_SCOPES: typing.ClassVar[Mapping[str, tuple[MetricScope, ...]]] = (
        MappingProxyType({})
    )
    # Table-valued results (replay phases, collision records, edge-case rows); not metric columns.
    CHILD_OUTPUTS: typing.ClassVar[tuple[str, ...]] = ()
    PRIMARY_OUTPUTS: list[str] = []
    # 1NF series tables this calculator reads. The pipeline publishes every series before scope
    # evaluation; the registry skips a calculator whose declared table is absent or empty and
    # explains the nulls with `not_recorded:<stem>` provenance.
    SERIES_DEPENDS_ON: typing.ClassVar[tuple[str, ...]] = ()
    # Natural key of the series table this calculator publishes: one row per key tuple.
    SERIES_KEY: typing.ClassVar[tuple[str, ...]] = ("t_ns",)
    # Topics a series table needs to be emitted; empty when it derives from the aligned bundle alone.
    SERIES_REQUIRED_TOPICS: typing.ClassVar[tuple[str, ...]] = ()
    world: str | None = None

    def __init__(self, robot_params: RobotParams):
        """Initializes the calculator with robot parameters."""
        self.robot_params = robot_params
        self.metrics_config: dict[str, typing.Any] = {}

    @classmethod
    def effective_scopes(cls) -> frozenset[MetricScope]:
        """All evaluation scopes in which this calculator is registered."""
        return frozenset(
            scope for metric in cls.OUTPUTS for scope in cls.scopes_of_output(metric)
        )

    @classmethod
    def scopes_of_output(cls, metric: Metric) -> tuple[MetricScope, ...]:
        """Scopes at which this calculator publishes one of its metrics."""
        return cls.OUTPUT_SCOPES.get(metric.name, metric.scopes)

    @classmethod
    def required_topics_for_scope(
        cls, scope: MetricScope
    ) -> tuple[str | tuple[str, ...], ...]:
        """Required topic names (or topic alternatives) for executing in `scope`."""
        del scope
        normalized: list[str | tuple[str, ...]] = []
        for req in cls.REQUIRED_TOPICS:
            if isinstance(req, str):
                normalized.append(req)
            else:
                normalized.append(tuple(req))
        return tuple(normalized)

    @classmethod
    def output_keys_for_scope(cls, scope: MetricScope) -> tuple[str, ...]:
        """Names of the metrics this calculator publishes at `scope`."""
        return tuple(
            metric.name
            for metric in cls.OUTPUTS
            if scope in cls.scopes_of_output(metric)
        )

    def _ground_truth_frame(self, episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The aligned native /tf rows with a finite pose, None when no sample is usable."""
        if (
            episode.data is None or not {"x", "y", "yaw"} <= set(episode.data.columns)
        ):  # agent-contracts: ignore optional stream: a recording without a ground-truth track carries none of x, y or yaw
            return None
        frame = episode.data.filter(
            pl.col("x").is_not_null()
            & ~pl.col("x").is_nan()
            & pl.col("y").is_not_null()
            & ~pl.col("y").is_nan()
            & pl.col("yaw").is_not_null()
            & ~pl.col("yaw").is_nan()
        )
        return None if len(frame) == 0 else frame

    def resolve_robot_frame(self, episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The aligned native /tf rows with a finite pose, the one track every robot-scope metric reads."""
        return self._ground_truth_frame(episode)

    def resolve_robot_pose(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map-frame native /tf pose (x, y, yaw) of the episode.

        There is one pose source: the robot's native /tf track. The trailing estimate triple is
        empty, because the aligned frame carries no wheel odometry and odometry never stands in
        for the trajectory (D3).
        """
        empty = np.array([], dtype=np.float64)
        frame = self._ground_truth_frame(episode)
        if frame is None:
            return empty, empty, empty, empty, empty, empty
        return (
            frame["x"].to_numpy().copy(),
            frame["y"].to_numpy().copy(),
            frame["yaw"].to_numpy().copy(),
            empty,
            empty,
            empty,
        )

    def series_table(
        self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = ""
    ) -> pl.DataFrame | None:
        """1NF long-format series this calculator publishes, None when it publishes none."""
        return None

    def native_topics(self, episode: AlignedEpisodeBundle) -> dict:
        """Raw native-rate topic frames keyed by topic name; {} when absent."""
        return episode.topics if isinstance(episode.topics, dict) else {}

    def series_frame(
        self, episode: AlignedEpisodeBundle, stem: str
    ) -> pl.DataFrame | None:
        """The episode's published 1NF series frame for `stem`, None when it is absent or empty."""
        frame = episode.series.get(stem)
        if frame is None or frame.height == 0:
            return None
        return frame

    def missing_input_provenance(self, episode: AlignedEpisodeBundle) -> dict[str, str]:
        """Output key -> typed null_reason for nulls caused by an absent optional input stream.

        The registry uses these codes instead of the generic `no_samples` when a calculator ran but
        an optional input (a world snapshot, markers, an interaction stream) was not recorded.
        """
        del episode
        return {}

    def resolve_native_pose(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ground-truth robot pose on the native clock: (x, y, yaw, t_ns) float64 arrays, empty without ground truth.

        The robot's /tf track is the only trajectory source; wheel odometry never stands in for it.
        """
        empty = np.array([])
        tracks = self.native_topics(episode).get("robot_tracks")
        if tracks is None or not {"x", "y", "yaw"} <= set(tracks.columns):
            return empty, empty, empty, empty
        t_df = tracks.drop_nulls(["x", "y", "yaw"]).sort("t_ns")
        return (
            t_df["x"].to_numpy().astype(np.float64),
            t_df["y"].to_numpy().astype(np.float64),
            t_df["yaw"].to_numpy().astype(np.float64),
            t_df["t_ns"].to_numpy().astype(np.int64),
        )

    def native_ped_frame(self, episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The robot's pedestrians as flat rows, one per pedestrian per frame, sorted by t_ns; None when absent."""
        peds = self.native_topics(episode).get("pedestrian_arena_peds")
        if peds is None:
            return None
        return peds.sort("t_ns")

    def pose_at_times(
        self,
        times_ns: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        yaw: np.ndarray,
        t_ns: np.ndarray,
        tolerance_ns: int | None = TRACK_ALIGN_TOLERANCE_NS,
        continuous: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate pose arrays onto query timestamps ``times_ns``."""
        if len(t_ns) == 0 or len(times_ns) == 0:
            return (
                np.full(len(times_ns), np.nan),
                np.full(len(times_ns), np.nan),
                np.full(len(times_ns), np.nan),
            )

        order = np.argsort(t_ns)
        t_sorted = t_ns[order]
        x_sorted = np.asarray(x, dtype=np.float64)[order]
        y_sorted = np.asarray(y, dtype=np.float64)[order]
        yaw_sorted = np.asarray(yaw, dtype=np.float64)[order]

        if continuous or tolerance_ns is None:
            return (
                interpolate_values(t_sorted, x_sorted, times_ns),
                interpolate_values(t_sorted, y_sorted, times_ns),
                interpolate_values(t_sorted, yaw_sorted, times_ns),
            )
        return (
            asof_values(t_sorted, x_sorted, times_ns, tolerance_ns),
            asof_values(t_sorted, y_sorted, times_ns, tolerance_ns),
            asof_values(t_sorted, yaw_sorted, times_ns, tolerance_ns),
        )

    def values_at_times(
        self,
        values: np.ndarray,
        values_time_ns: np.ndarray,
        query_times_ns: np.ndarray,
        tolerance_ns: int | None = TRACK_ALIGN_TOLERANCE_NS,
        continuous: bool = False,
    ) -> np.ndarray:
        """Interpolate values onto query timestamps."""
        if len(values_time_ns) == 0 or len(query_times_ns) == 0:
            return np.full(len(query_times_ns), np.nan)

        order = np.argsort(values_time_ns)
        t_sorted = values_time_ns[order]
        v_sorted = np.asarray(values, dtype=np.float64)[order]

        if continuous or tolerance_ns is None:
            return interpolate_values(t_sorted, v_sorted, query_times_ns)
        return asof_values(t_sorted, v_sorted, query_times_ns, tolerance_ns)

    @staticmethod
    def speed_from_pose(x: np.ndarray, y: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
        """GT speed (m/s) by finite-differencing GT positions; NaN where no step has a positive duration."""
        n = len(x)
        speed = np.full(n, np.nan)
        if n < 2:
            return speed
        dt = np.diff(t_ns) / 1e9
        speed[1:] = np.divide(
            np.hypot(np.diff(x), np.diff(y)),
            dt,
            out=np.full(n - 1, np.nan),
            where=dt > 0.0,
        )
        return speed

    @staticmethod
    def velocity_from_pose(
        x: np.ndarray, y: np.ndarray, t_ns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """GT velocity vector (m/s) by finite-differencing GT positions; NaN where no step has a positive duration."""
        n = len(x)
        vx = np.full(n, np.nan)
        vy = np.full(n, np.nan)
        if n < 2:
            return vx, vy
        dt = np.diff(t_ns) / 1e9
        vx[1:] = np.divide(np.diff(x), dt, out=np.full(n - 1, np.nan), where=dt > 0.0)
        vy[1:] = np.divide(np.diff(y), dt, out=np.full(n - 1, np.nan), where=dt > 0.0)
        return vx, vy

    @classmethod
    def output_keys(cls) -> list[str]:
        """Names of every metric this calculator publishes."""
        return [metric.name for metric in cls.OUTPUTS]

    @abstractmethod
    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Calculate metrics for a single episode."""
        pass

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """Calculate metrics for a specific scope slice, defaulting to `calculate`."""
        del scope_slice
        return self.calculate(episode, prior_results)
