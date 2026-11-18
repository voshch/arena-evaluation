"""Multi-resolution scoped evaluator: one episode in, five typed fact tables out.

The evaluator drives the scope-aware calculators through the MetricRegistry over
the natural spatiotemporal windows of an episode (episode, encounter, contact,
pedestrian, robot) and pivots their outputs into the registry's fact tables.
Detector windows come from :mod:`processing.windows` and are fully vectorized;
calculators run once per window over an isolated episode view, and fact tables are
materialized only at the serialization boundary.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import typing
from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    MetricScope,
    ScopeExecutionOutcome,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.registry import (
    MetricRegistry,
    load_metric_definitions,
)
from arena_evaluation.processing.metrics.safety.contact_attribution import (
    COLLISION_CONTACTS_TOPIC,
    static_point_of,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    PEDESTRIAN_TRACKS_TOPIC,
    ROBOT_TRACKS_TOPIC,
)
from arena_evaluation.processing.outcome import outcome_verdict
from arena_evaluation.processing.validity import episode_validity
from arena_evaluation.processing.windows import (
    ACTOR_PEDESTRIAN,
    DEFAULT_DETECTOR_PARAMS,
    NS_PER_S,
    DetectorParams,
    detect_contact_windows_lf,
    detect_encounter_windows_lf,
    empty_contact_windows,
    empty_encounter_windows,
    fleet_pairs_lf,
    pedestrian_pairs_lf,
)
from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.registry import (
    FACT_TABLES,
    FactTable,
    Metric,
    NullReason,
    defined_metrics,
)
from arena_evaluation.storage.registry.series import SERIES_TABLES
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

EVALUATOR_VERSION: Final[str] = "wp2-scoped-evaluator-v2"

NO_SAMPLES: Final[str] = NullReason.NO_SAMPLES.value
PEDSIM_UNAVAILABLE: Final[str] = NullReason.PEDSIM_UNAVAILABLE.value
ZERO_DENOMINATOR: Final[str] = NullReason.ZERO_DENOMINATOR.value
TOPIC_MISSING: Final[str] = NullReason.TOPIC_MISSING.value
NOT_RECORDED: Final[str] = NullReason.NOT_RECORDED.value

# Registry provenance kinds that keep the legacy `field:no_samples` base reason; any other code
# (e.g. `reference_exceeds_path`) replaces it as the field's distinct reason.
_GENERIC_PROVENANCE_KINDS: Final[tuple[str, ...]] = (
    NO_SAMPLES,
    PEDSIM_UNAVAILABLE,
    TOPIC_MISSING,
    "exception",
)

ROBOT_POWER_TOPIC: Final[str] = "robot_power"
ROBOT_ACOUSTICS_TOPIC: Final[str] = "robot_acoustics"

CONTACT_PRE_WINDOW_S: Final[float] = 0.5
CONTACT_POST_WINDOW_S: Final[float] = 0.5
METRES_PER_WH_J: Final[float] = 3600.0
_ENV_PREFIX: Final[re.Pattern[str]] = re.compile(r"^env_[0-9]+_")
#: The identity value of a row that has no such parent (no encounter, no contact).
_ABSENT_ID: Final[str] = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeScopeHierarchy:
    """Immutable 5-scope slice hierarchy decomposed from a single episode bundle."""

    contacts: tuple[ScopeSlice, ...]
    encounters: tuple[ScopeSlice, ...]
    pedestrians: tuple[ScopeSlice, ...]
    robots: tuple[ScopeSlice, ...]
    episodes: tuple[ScopeSlice, ...]
    encounter_windows: pl.DataFrame
    contact_windows: pl.DataFrame


@dataclasses.dataclass(frozen=True, slots=True)
class ScopedFactTables:
    """The eleven typed fact tables of one evaluated episode."""

    episodes: pl.DataFrame
    encounters: pl.DataFrame
    contacts: pl.DataFrame
    pedestrians: pl.DataFrame
    robots: pl.DataFrame
    edge_cases: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=_fact_schema(FactTable.EDGE_CASES))
    )
    formations: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=_fact_schema(FactTable.FORMATIONS))
    )
    phase_outcomes: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(
            schema=_fact_schema(FactTable.PHASE_OUTCOMES)
        )
    )
    judge_disagreements: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(
            schema=_fact_schema(FactTable.JUDGE_DISAGREEMENTS)
        )
    )
    mechanism_states: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(
            schema=_fact_schema(FactTable.MECHANISM_STATES)
        )
    )
    collisions: pl.DataFrame = dataclasses.field(
        default_factory=lambda: pl.DataFrame(schema=_fact_schema(FactTable.COLLISIONS))
    )

    def by_name(self) -> dict[str, pl.DataFrame]:
        """Fact tables keyed by their registry table name."""
        return {
            FactTable.EPISODES.value: self.episodes,
            FactTable.ENCOUNTERS.value: self.encounters,
            FactTable.CONTACTS.value: self.contacts,
            FactTable.PEDESTRIANS.value: self.pedestrians,
            FactTable.ROBOTS.value: self.robots,
            FactTable.EDGE_CASES.value: self.edge_cases,
            FactTable.FORMATIONS.value: self.formations,
            FactTable.PHASE_OUTCOMES.value: self.phase_outcomes,
            FactTable.JUDGE_DISAGREEMENTS.value: self.judge_disagreements,
            FactTable.MECHANISM_STATES.value: self.mechanism_states,
            FactTable.COLLISIONS.value: self.collisions,
        }


def write_fact_tables(tables: ScopedFactTables, metrics_dir: pathlib.Path) -> None:
    """Write the fact tables of one episode into its metrics folder, each conformed to its registry declaration."""
    load_metric_definitions()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    named = tables.by_name()
    for member, table in FACT_TABLES.items():
        _conformed(named[member.value], table).write_parquet(
            metrics_dir / table.file_name(), compression="zstd"
        )


def write_series_tables(
    series: Mapping[str, pl.DataFrame], metrics_dir: pathlib.Path
) -> None:
    """Write the series frames of one episode, keyed by stem, each conformed to its registry declaration; an undeclared stem is refused."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in series.items():
        table = SERIES_TABLES.get(stem)
        if table is None:
            raise ContractViolationError(
                f"series {stem!r} is not declared in the registry"
            )
        _conformed(frame, table).write_parquet(
            metrics_dir / table.file_name(), compression="zstd"
        )


def _conformed(frame: pl.DataFrame, table: type) -> pl.DataFrame:
    """The frame in the registry table's column order and dtypes; a missing or undeclared column is refused."""
    schema = table.polars_schema()
    missing = [name for name in schema.names() if name not in frame.columns]
    undeclared = [name for name in frame.columns if name not in schema]
    if missing or undeclared:
        raise ContractViolationError(
            f"{table.table_name}: frame does not match its declaration (missing {missing}, undeclared {undeclared})"
        )
    return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])


def _fact_table(table: FactTable) -> type:
    """The registry table class of a fact table, with every metric definition loaded so its columns are complete."""
    load_metric_definitions()
    return FACT_TABLES[table]


def _fact_schema(table: FactTable) -> pl.Schema:
    """The registry schema of a fact table, its declared metric columns included."""
    return _fact_table(table).polars_schema()


def _nullable(table: FactTable) -> tuple[str, ...]:
    """Fact columns the registry marks as NULL-bearing; each needs a null_reason when it is NULL."""
    return _fact_table(table).null_reason_columns()


def _metric_declarations(table: FactTable) -> tuple[Metric, ...]:
    """Every metric the registry stores at this table's scope, in definition order."""
    scope = _fact_table(table).table_scope
    if scope is None:
        return ()
    return tuple(metric for metric in defined_metrics() if scope in metric.scopes)


def _collect_lazy_tables(
    raw: object,
    kind: str,
    lazy_keys: list[tuple[str, str]],
    lazy_frames: list[pl.LazyFrame],
) -> dict[str, pl.DataFrame]:
    """Split a table mapping into eager DataFrames and queued LazyFrames."""
    tables: dict[str, pl.DataFrame] = {}
    if not isinstance(raw, Mapping):
        return tables
    for name, table in raw.items():
        if isinstance(table, pl.LazyFrame):
            lazy_keys.append((kind, name))
            lazy_frames.append(table)
        elif table is not None:
            tables[name] = table
    return tables


def _materialize_episode_bundle(episode: AlignedEpisodeBundle) -> AlignedEpisodeBundle:
    """Materialize all LazyFrame topic and fleet tables once via pl.collect_all before slicing."""
    lazy_keys: list[tuple[str, str]] = []
    lazy_frames: list[pl.LazyFrame] = []

    topics = _collect_lazy_tables(episode.topics, "topic", lazy_keys, lazy_frames)
    fleet = _collect_lazy_tables(episode.fleet, "fleet", lazy_keys, lazy_frames)

    if isinstance(episode.data, pl.LazyFrame):
        lazy_keys.append(("data", "data"))
        lazy_frames.append(episode.data)

    data_df = episode.data if isinstance(episode.data, pl.DataFrame) else pl.DataFrame()
    if lazy_frames:
        collected = pl.collect_all(lazy_frames)
        for (kind, name), df in zip(lazy_keys, collected, strict=True):
            if kind == "topic":
                topics[name] = df
            elif kind == "fleet":
                fleet[name] = df
            else:
                data_df = df

    return dataclasses.replace(episode, data=data_df, topics=topics, fleet=fleet)


def _table(episode: AlignedEpisodeBundle, topic: str) -> pl.DataFrame | None:
    """Materialized flat table of an episode topic, None when absent."""
    topics = episode.topics
    if topics is None:
        return None
    table = topics.get(topic)
    if table is None:
        return None
    return table.collect() if isinstance(table, pl.LazyFrame) else table


def _has_rows(table: pl.DataFrame | None) -> bool:
    return table is not None and len(table) > 0


def _float(value: object) -> float | None:
    return None if value is None else float(typing.cast(float, value))


def _int(value: object) -> int | None:
    return None if value is None else int(typing.cast(int, value))


_BLIND_ROBOT_AWARENESS: Final[float] = 0.0


def _blind_from_awareness(robot_awareness: float | None) -> bool | None:
    """A pedestrian is blind to robots when its own robot_awareness is 0; None while it is not recorded."""
    return (
        None if robot_awareness is None else robot_awareness == _BLIND_ROBOT_AWARENESS
    )


def _latched(frame: pl.DataFrame, column: str) -> float | None:
    """The first recorded value of a per-pedestrian constant; None when the frame lacks the column or no row has a value."""
    if column not in frame.columns:
        return None
    values = frame.get_column(column).drop_nulls()
    return None if len(values) == 0 else _float(values[0])


def _row_null_reason(
    row: Mapping[str, object],
    nullable: tuple[str, ...],
    provenance: Mapping[str, str] | None = None,
) -> str:
    """``field:code`` provenance for every mapped field that stayed null; a calculator code wins over no_samples."""
    codes: list[str] = []
    for field in nullable:
        if row.get(field) is not None:
            continue
        override = (provenance or {}).get(field)
        if isinstance(override, str) and override:
            kind = override.partition(":")[2].partition(":")[0]
            if kind not in _GENERIC_PROVENANCE_KINDS:
                codes.append(override)
                continue
        codes.append(f"{field}:{NO_SAMPLES}")
    return ";".join(codes)


def _null_reason_expression(nullable: tuple[str, ...]) -> pl.Expr:
    """``field:code`` provenance expression for every nullable column of a child table row."""
    if not nullable:
        return pl.lit("", dtype=pl.String)
    parts = [
        pl.when(pl.col(name).is_null())
        .then(pl.lit(f"{name}:{NO_SAMPLES}", dtype=pl.String))
        .otherwise(pl.lit(None, dtype=pl.String))
        for name in nullable
    ]
    return pl.concat_str(parts, separator=";", ignore_nulls=True).fill_null("")


def _merge_provenance_reasons(
    base_reason: str, provenance: Mapping[str, str], row: Mapping[str, object]
) -> str:
    """Combine field missingness codes with registry topic_missing and exception provenance, on the vocabulary."""
    parts = [part for part in base_reason.split(";") if part]
    seen = set(parts)
    for key, code in provenance.items():
        if (
            ":exception:" in code
            or (":topic_missing:" in code and row.get(key) is None)
            or (":not_recorded:" in code and row.get(key) is None)
        ):
            if code not in seen:
                parts.append(code)
                seen.add(code)
    return ";".join(parts)


def _declared_value(value: object, declaration: Metric) -> object:
    """One calculator value coerced to its declared fact dtype; None stays null for provenance."""
    if value is None:
        return None
    match declaration.dtype:
        case pl.Float64:
            return float(typing.cast(float, value))
        case pl.Int64:
            return int(typing.cast(int, value))
        case pl.Boolean:
            if not isinstance(
                value, (bool, int, float, np.bool_, np.integer, np.floating)
            ):
                raise TypeError(
                    f"{declaration.name}: expected a boolean, got {type(value).__name__}"
                )
            return bool(value)
        case pl.String:
            if not isinstance(value, str):
                raise TypeError(
                    f"{declaration.name}: expected a string, got {type(value).__name__}"
                )
            return value
        case unsupported:
            raise TypeError(f"{declaration.name}: unsupported fact dtype {unsupported}")


def _declared_fact_values(
    results: Mapping[str, object], table: FactTable
) -> dict[str, object]:
    """Every declared fact column of ``table``, copied from the calculator result of its name."""
    return {
        declaration.name: _declared_value(results.get(declaration.name), declaration)
        for declaration in _metric_declarations(table)
    }


_EDGE_CASE_LIST_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("step", "edge_case_steps"),
    ("agent", "edge_case_agents"),
    ("missing", "edge_case_missing"),
    ("unplayed", "edge_case_unplayed"),
)
_FORMATION_LIST_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("member", "edge_case_formation_members"),
    ("absent", "edge_case_formation_absent"),
)
_MECHANISM_FIELDS: Final[tuple[str, ...]] = ("locked", "progress")


def _sequence(value: object) -> list[object]:
    """The list a child-table builder reshapes; a non-list output has no rows."""
    if isinstance(value, list):
        return typing.cast(list[object], value)
    return []


def _event_time_ns(episode: AlignedEpisodeBundle) -> np.ndarray:
    """Nanosecond clock the EPISODE-scope calculators sampled their per-sample lists on."""
    frame = episode.data
    if frame is None or len(frame) == 0:
        return np.array([], dtype=np.int64)
    return frame.get_column("t_ns").to_numpy().astype(np.int64)


def _frame(rows: list[dict[str, object]], table: FactTable) -> pl.DataFrame:
    """Typed fact frame of the given rows, empty with the registry schema when there are none."""
    schema = _fact_schema(table)
    if not rows:
        return pl.DataFrame(schema=schema)
    projected = [{name: row.get(name) for name in schema.names()} for row in rows]
    return pl.DataFrame(projected, schema=schema)


ProgressCallback = Callable[[str, int, int], None]


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluatedEpisode:
    """One robot's episode: its eleven fact tables, its series tables and its episode metrics row."""

    facts: ScopedFactTables
    series: dict[str, pl.DataFrame]
    episode_metrics: dict[str, Any]


def _episode_series_tables(
    registry: MetricRegistry,
    episode: AlignedEpisodeBundle,
    *,
    run_id: str,
    robot_id: str,
) -> dict[str, pl.DataFrame]:
    """1NF series frames the registered calculators publish for one robot episode, keyed by table stem."""
    frames: dict[str, pl.DataFrame] = {}
    for stem, calculator in registry.series_calculators():
        available = episode.topics or {}
        if calculator.SERIES_REQUIRED_TOPICS and any(
            topic not in available for topic in calculator.SERIES_REQUIRED_TOPICS
        ):
            continue
        frame = calculator.series_table(episode, run_id=run_id, robot_id=robot_id)
        if frame is not None and frame.height > 0:
            frames[stem] = frame
    return frames


class ScopedEvaluator:
    """Multi-resolution scoped evaluator delegating metric calculations to the scope-native registry."""

    def __init__(
        self,
        robot_params: RobotParams,
        metrics_config: Mapping[str, Any] | None = None,
        detector_params: DetectorParams = DEFAULT_DETECTOR_PARAMS,
    ) -> None:
        self.robot_params = robot_params
        self.metrics_config: dict[str, Any] = dict(metrics_config or {})
        self.detector_params = detector_params

    @staticmethod
    def _episode_span(robot_tracks: pl.DataFrame | None) -> tuple[int, int, str | None]:
        """(ep_start_ns, ep_end_ns, env_id) from the robot track table."""
        if not _has_rows(robot_tracks):
            return 0, 0, None
        rt_df = typing.cast(pl.DataFrame, robot_tracks)
        t_col = rt_df.get_column("t_ns").drop_nulls()
        min_t = t_col.min()
        max_t = t_col.max()
        envs = rt_df.get_column("env_id").drop_nulls()
        return (
            min_t if isinstance(min_t, int) else 0,
            max_t if isinstance(max_t, int) else 0,
            str(envs[0]) if len(envs) > 0 else None,
        )

    def _build_contact_slices(
        self,
        materialized: AlignedEpisodeBundle,
        contact_windows: pl.DataFrame,
        encounter_ids: Mapping[int, str],
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        robot_token: str,
    ) -> list[ScopeSlice]:
        """Build CONTACT ScopeSlices from detected contact windows."""
        if len(contact_windows) == 0:
            return []
        pre_window_s = float(
            self.metrics_config.get("pre_contact_window_s", CONTACT_PRE_WINDOW_S)
        )
        post_window_s = float(
            self.metrics_config.get("post_contact_window_s", CONTACT_POST_WINDOW_S)
        )
        contact_slices: list[ScopeSlice] = []
        for (
            contact_idx,
            collider_kind,
            obstacle_id,
            t_onset_ns,
            t_end_ns,
            obstacle_x,
            obstacle_y,
            encounter_idx,
        ) in zip(
            contact_windows.get_column("contact_idx").to_numpy(),
            contact_windows.get_column("collider_kind").to_numpy(),
            contact_windows.get_column("obstacle_id").to_numpy(),
            contact_windows.get_column("t_onset_ns").to_numpy(),
            contact_windows.get_column("t_end_ns").to_numpy(),
            contact_windows.get_column("obstacle_x").to_numpy(),
            contact_windows.get_column("obstacle_y").to_numpy(),
            contact_windows.get_column("encounter_idx").fill_null(-1).to_numpy(),
            strict=True,
        ):
            internal_kind = (
                ACTOR_PEDESTRIAN
                if collider_kind in (ACTOR_PEDESTRIAN, "robot")
                else str(collider_kind)
            )
            actor_id = str(obstacle_id)
            win_start_ns = int(t_onset_ns - round(pre_window_s * NS_PER_S))
            win_end_ns = int(t_end_ns + round(post_window_s * NS_PER_S))
            window_bundle = self._window_episode(
                materialized, win_start_ns, win_end_ns, actor_id, internal_kind
            )
            static_pt = (
                None
                if collider_kind in (ACTOR_PEDESTRIAN, "robot")
                else static_point_of(float(obstacle_x), float(obstacle_y))
            )
            enc_index = None if int(encounter_idx) < 0 else int(encounter_idx)
            parent_encounter_id = (
                None if enc_index is None else encounter_ids.get(enc_index)
            )
            contact_id = self._contact_id(episode_id, robot_token, int(contact_idx))
            contact_slices.append(
                ScopeSlice(
                    scope=MetricScope.CONTACT,
                    slice_id=contact_id,
                    parent_id=parent_encounter_id,
                    t_start_ns=int(t_onset_ns),
                    t_end_ns=int(t_end_ns),
                    identity={
                        "run_id": run_id,
                        "episode_id": int(episode_id),
                        "robot_id": robot_id,
                        "contact_id": contact_id,
                        "encounter_id": parent_encounter_id,
                        "collider_kind": str(collider_kind),
                        "obstacle_id": actor_id,
                        "static_point": static_pt,
                        "pre_contact_window_s": pre_window_s,
                        "post_contact_window_s": post_window_s,
                    },
                    window_bundle=window_bundle,
                    full_episode=materialized,
                )
            )
        return contact_slices

    def _build_encounter_slices(
        self,
        materialized: AlignedEpisodeBundle,
        encounter_windows: pl.DataFrame,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        robot_token: str,
        episode_slice_id: str,
    ) -> list[ScopeSlice]:
        """Build ENCOUNTER ScopeSlices from detected encounter windows."""
        if len(encounter_windows) == 0:
            return []
        encounter_slices: list[ScopeSlice] = []
        for (
            encounter_idx,
            actor_id,
            actor_kind,
            t_start_ns,
            t_end_ns,
            min_clearance_m,
            approach_type,
            window_null_reason,
            detector_version,
            detector_params_hash,
        ) in zip(
            encounter_windows.get_column("encounter_idx").fill_null(-1).to_numpy(),
            encounter_windows.get_column("actor_id").to_numpy(),
            encounter_windows.get_column("actor_kind").to_numpy(),
            encounter_windows.get_column("t_start_ns").to_numpy(),
            encounter_windows.get_column("t_end_ns").to_numpy(),
            encounter_windows.get_column("min_clearance_m").to_numpy(),
            encounter_windows.get_column("approach_type").to_numpy(),
            encounter_windows.get_column("null_reason").to_numpy(),
            encounter_windows.get_column("detector_version").to_numpy(),
            encounter_windows.get_column("detector_params_hash").to_numpy(),
            strict=True,
        ):
            actor = str(actor_id)
            kind = str(actor_kind)
            window_bundle = self._window_episode(
                materialized, int(t_start_ns), int(t_end_ns), actor, kind
            )
            encounter_id = self._encounter_id(
                episode_id, robot_token, int(encounter_idx)
            )
            encounter_slices.append(
                ScopeSlice(
                    scope=MetricScope.ENCOUNTER,
                    slice_id=encounter_id,
                    parent_id=episode_slice_id,
                    t_start_ns=int(t_start_ns),
                    t_end_ns=int(t_end_ns),
                    identity={
                        "run_id": run_id,
                        "episode_id": int(episode_id),
                        "robot_id": robot_id,
                        "encounter_id": encounter_id,
                        "actor_1": robot_id,
                        "actor_2": actor,
                        "actor_2_kind": kind,
                        "approach_type": None
                        if approach_type is None
                        else str(approach_type),
                        "min_clearance_m": _float(min_clearance_m),
                        "detector_version": str(detector_version),
                        "detector_params_hash": str(detector_params_hash),
                        "window_null_reason": ""
                        if window_null_reason is None
                        else str(window_null_reason),
                    },
                    window_bundle=window_bundle,
                    full_episode=materialized,
                )
            )
        return encounter_slices

    @staticmethod
    def _build_pedestrian_slices(
        materialized: AlignedEpisodeBundle,
        pedestrian_tracks: pl.DataFrame | None,
        *,
        run_id: str,
        episode_id: int,
        env_id: str | None,
        ep_start_ns: int,
        ep_end_ns: int,
        episode_slice_id: str,
    ) -> list[ScopeSlice]:
        """Build PEDESTRIAN ScopeSlices from pedestrian track tables."""
        if not _has_rows(pedestrian_tracks):
            return []
        pedestrian_slices: list[ScopeSlice] = []
        tracks_sorted = typing.cast(pl.DataFrame, pedestrian_tracks).sort(
            ["ped_id", "t_ns"]
        )
        for ped_df in tracks_sorted.partition_by("ped_id", maintain_order=True):
            ped_id = str(ped_df.get_column("ped_id")[0])
            ped_t_col = ped_df.get_column("t_ns").drop_nulls()
            min_ped_t = ped_t_col.min()
            max_ped_t = ped_t_col.max()
            ped_start_ns = min_ped_t if isinstance(min_ped_t, int) else ep_start_ns
            ped_end_ns = max_ped_t if isinstance(max_ped_t, int) else ep_end_ns
            ped_topics = dict(materialized.topics or {})
            ped_topics[PEDESTRIAN_TRACKS_TOPIC] = ped_df
            ped_bundle = dataclasses.replace(
                materialized, data=pl.DataFrame(), topics=ped_topics, fleet={}
            )

            ped_envs = ped_df.get_column("env_id").drop_nulls()
            ped_names = ped_df.get_column("ped_name").drop_nulls()
            ped_policies = ped_df.get_column("policy").drop_nulls()
            ped_speeds = ped_df.get_column("desired_speed").drop_nulls()
            ped_radii = ped_df.get_column("radius_m").drop_nulls()
            ped_xs = ped_df.get_column("x").drop_nulls()
            ped_ys = ped_df.get_column("y").drop_nulls()
            ped_yaws = ped_df.get_column("yaw").drop_nulls()

            pedestrian_slices.append(
                ScopeSlice(
                    scope=MetricScope.PEDESTRIAN,
                    slice_id=ped_id,
                    parent_id=episode_slice_id,
                    t_start_ns=ped_start_ns,
                    t_end_ns=ped_end_ns,
                    identity={
                        "run_id": run_id,
                        "episode_id": int(episode_id),
                        "env_id": str(ped_envs[0]) if len(ped_envs) > 0 else env_id,
                        "ped_id": ped_id,
                        "ped_name": str(ped_names[0]) if len(ped_names) > 0 else ped_id,
                        "ped_policy": str(ped_policies[0])
                        if len(ped_policies) > 0
                        else None,
                        "is_blind": _blind_from_awareness(
                            _latched(ped_df, "robot_awareness")
                        ),
                        "desired_speed": _float(ped_speeds[0])
                        if len(ped_speeds) > 0
                        else None,
                        "radius_m": _float(ped_radii[0])
                        if len(ped_radii) > 0
                        else None,
                        "spawn_x": _float(ped_xs[0]) if len(ped_xs) > 0 else None,
                        "spawn_y": _float(ped_ys[0]) if len(ped_ys) > 0 else None,
                        "spawn_yaw": _float(ped_yaws[0]) if len(ped_yaws) > 0 else None,
                    },
                    window_bundle=ped_bundle,
                    full_episode=materialized,
                )
            )
        return pedestrian_slices

    def decompose_hierarchy(
        self, episode: AlignedEpisodeBundle, *, run_id: str, episode_id: int
    ) -> EpisodeScopeHierarchy:
        """Stage 1: Materialize episode topics once and decompose into immutable ScopeSlice hierarchy."""
        materialized = _materialize_episode_bundle(episode)
        robot_tracks = _table(materialized, ROBOT_TRACKS_TOPIC)
        pedestrian_tracks = _table(materialized, PEDESTRIAN_TRACKS_TOPIC)
        contact_windows = empty_contact_windows()
        encounter_windows = empty_encounter_windows()

        if _has_rows(robot_tracks):
            encounter_windows = self._detect_encounters(
                materialized, typing.cast(pl.DataFrame, robot_tracks), pedestrian_tracks
            )
            contact_windows = self._detect_contacts(materialized, encounter_windows)

        robot_id = self._robot_actor_id(materialized, robot_tracks)
        robot_token = self._short_robot_token(robot_id)
        episode_slice_id = f"{run_id}:{int(episode_id)}:{robot_id}"
        ep_start_ns, ep_end_ns, env_id = self._episode_span(robot_tracks)
        encounter_ids = {
            int(index): self._encounter_id(episode_id, robot_token, int(index))
            for index in encounter_windows.get_column("encounter_idx").to_numpy()
        }

        contact_slices = self._build_contact_slices(
            materialized,
            contact_windows,
            encounter_ids,
            run_id=run_id,
            episode_id=episode_id,
            robot_id=robot_id,
            robot_token=robot_token,
        )
        encounter_slices = self._build_encounter_slices(
            materialized,
            encounter_windows,
            run_id=run_id,
            episode_id=episode_id,
            robot_id=robot_id,
            robot_token=robot_token,
            episode_slice_id=episode_slice_id,
        )
        pedestrian_slices = self._build_pedestrian_slices(
            materialized,
            pedestrian_tracks,
            run_id=run_id,
            episode_id=episode_id,
            env_id=env_id,
            ep_start_ns=ep_start_ns,
            ep_end_ns=ep_end_ns,
            episode_slice_id=episode_slice_id,
        )

        robot_slice = ScopeSlice(
            scope=MetricScope.ROBOT,
            slice_id=robot_id,
            parent_id=episode_slice_id,
            t_start_ns=ep_start_ns,
            t_end_ns=ep_end_ns,
            identity={
                "run_id": run_id,
                "episode_id": int(episode_id),
                "env_id": env_id,
                "robot_id": robot_id,
                "robot_name": materialized.robot_name,
            },
            window_bundle=materialized,
            full_episode=materialized,
        )

        episode_slice = ScopeSlice(
            scope=MetricScope.EPISODE,
            slice_id=episode_slice_id,
            parent_id=None,
            t_start_ns=ep_start_ns,
            t_end_ns=ep_end_ns,
            identity={
                "run_id": run_id,
                "episode_id": int(episode_id),
                "robot_id": robot_id,
            },
            window_bundle=materialized,
            full_episode=materialized,
        )

        return EpisodeScopeHierarchy(
            contacts=tuple(contact_slices),
            encounters=tuple(encounter_slices),
            pedestrians=tuple(pedestrian_slices),
            robots=(robot_slice,),
            episodes=(episode_slice,),
            encounter_windows=encounter_windows,
            contact_windows=contact_windows,
        )

    def evaluate(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        world: str | None,
        episode_context: Mapping[str, Any],
        pedsim_available: bool,
        available_topics: set[str],
        progress_callback: ProgressCallback | None = None,
    ) -> EvaluatedEpisode:
        """Evaluate one robot's episode: its series, its episode scope and the scopes below it, as eleven fact tables."""
        registry = MetricRegistry(
            self.robot_params, world=world, metrics_config=dict(self.metrics_config)
        )
        series_tables = _episode_series_tables(
            registry,
            episode,
            run_id=run_id,
            robot_id=str(episode.robot_name or "robot"),
        )
        episode = dataclasses.replace(episode, series=series_tables)

        hierarchy = self.decompose_hierarchy(
            episode, run_id=run_id, episode_id=episode_id
        )
        episode_slice = hierarchy.episodes[0]
        materialized = episode_slice.full_episode

        ep_outcome = registry.run_scope(
            MetricScope.EPISODE,
            episode,
            pedsim_available=pedsim_available,
            available_topics=available_topics,
            progress_callback=progress_callback,
        )
        episode_metrics = {**ep_outcome.results, **episode_context}
        provenance: dict[str, str] = dict(ep_outcome.provenance)
        metrics = dict(episode_metrics)
        metrics.update(
            registry.calculators["mechanism_states"].calculate(
                materialized, dict(metrics)
            )
        )

        contact_rows = self._contact_fact_rows(registry, hierarchy.contacts)
        encounter_rows = self._encounter_fact_rows(
            registry, materialized, hierarchy.encounters, contact_rows
        )
        pedestrians = self._pedestrian_fact_frame(
            registry, materialized, hierarchy.pedestrians
        )

        robot_slice = hierarchy.robots[0]
        robot_episode = robot_slice.window_bundle
        robot_outcome: ScopeExecutionOutcome = registry.run_scope(
            MetricScope.ROBOT,
            robot_episode,
            available_topics=set(robot_episode.topics or {}),
            prior_results=dict(metrics),
            scope_slice=robot_slice,
        )
        episode_row = self._episode_fact_row(
            materialized, run_id, episode_id, metrics, provenance
        )
        robots = self._robot_fact_frame(robot_slice, robot_outcome, provenance)
        robot_id = self._robot_actor_id(
            materialized, _table(materialized, ROBOT_TRACKS_TOPIC)
        )
        child_identity = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "robot_id": robot_id,
            "metrics": {**metrics, **robot_outcome.children},
        }

        return EvaluatedEpisode(
            facts=ScopedFactTables(
                episodes=_frame([episode_row], FactTable.EPISODES),
                encounters=_frame(encounter_rows, FactTable.ENCOUNTERS),
                contacts=_frame(contact_rows, FactTable.CONTACTS),
                pedestrians=pedestrians,
                robots=robots,
                edge_cases=self._edge_case_fact_frame(**child_identity),
                formations=self._formation_fact_frame(**child_identity),
                phase_outcomes=self._phase_outcome_fact_frame(**child_identity),
                judge_disagreements=self._judge_disagreement_fact_frame(
                    **child_identity
                ),
                mechanism_states=self._mechanism_state_fact_frame(
                    materialized, **child_identity
                ),
                collisions=self._collision_fact_frame(materialized, **child_identity),
            ),
            series=series_tables,
            episode_metrics=episode_metrics,
        )

    def _detect_encounters(
        self,
        episode: AlignedEpisodeBundle,
        robot_tracks: pl.DataFrame,
        pedestrian_tracks: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Encounter windows of this robot with its pedestrians and with the other fleet robots, from the windows detector."""
        radius_m = float(self.robot_params.robot_radius)
        pairs: list[pl.LazyFrame] = []
        if _has_rows(pedestrian_tracks):
            pairs.append(
                pedestrian_pairs_lf(
                    robot_tracks,
                    typing.cast(pl.DataFrame, pedestrian_tracks),
                    radius_m,
                    self.detector_params,
                )
            )
        if episode.fleet:
            pairs.append(
                fleet_pairs_lf(
                    robot_tracks, episode.fleet, radius_m, self.detector_params
                )
            )
        if not pairs:
            return empty_encounter_windows()
        combined = pl.concat(pairs, how="vertical") if len(pairs) > 1 else pairs[0]
        return detect_encounter_windows_lf(combined, self.detector_params).collect()

    def _detect_contacts(
        self, episode: AlignedEpisodeBundle, encounters: pl.DataFrame
    ) -> pl.DataFrame:
        """Contact windows of the recorded collision stream, each linked to its enclosing encounter."""
        contacts = _table(episode, COLLISION_CONTACTS_TOPIC)
        if not _has_rows(contacts):
            return empty_contact_windows()
        return detect_contact_windows_lf(
            typing.cast(pl.DataFrame, contacts), encounters, self.detector_params
        ).collect()

    def _window_episode(
        self,
        episode: AlignedEpisodeBundle,
        t_start_ns: int,
        t_end_ns: int,
        actor_id: str,
        actor_kind: str,
    ) -> AlignedEpisodeBundle:
        """Episode view cut to ``[t_start_ns, t_end_ns]`` with the pedestrian stream restricted to the window's actor."""
        topics: dict[str, pl.DataFrame] = {}
        for name, frame in (episode.topics or {}).items():
            if not isinstance(frame, pl.DataFrame):
                continue
            if name == PEDESTRIAN_TRACKS_TOPIC:
                frame = self._actor_track_frame(episode, frame, actor_id, actor_kind)
            if len(frame) > 0:
                frame = frame.filter(
                    (pl.col("t_ns") >= t_start_ns) & (pl.col("t_ns") <= t_end_ns)
                )
            topics[name] = frame
        if PEDESTRIAN_TRACKS_TOPIC not in topics and actor_kind != ACTOR_PEDESTRIAN:
            actor_frame = self._actor_track_frame(
                episode, pl.DataFrame(), actor_id, actor_kind
            )
            if len(actor_frame) > 0:
                actor_frame = actor_frame.filter(
                    (pl.col("t_ns") >= t_start_ns) & (pl.col("t_ns") <= t_end_ns)
                )
            topics[PEDESTRIAN_TRACKS_TOPIC] = actor_frame
        return dataclasses.replace(
            episode, data=pl.DataFrame(), topics=topics, fleet={}
        )

    def _actor_track_frame(
        self,
        episode: AlignedEpisodeBundle,
        pedestrian_tracks: pl.DataFrame,
        actor_id: str,
        actor_kind: str,
    ) -> pl.DataFrame:
        """Pedestrian-shaped track of the window's actor: pedestrian rows or another robot's task poses."""
        if actor_kind == ACTOR_PEDESTRIAN:
            if len(pedestrian_tracks) == 0:
                return pedestrian_tracks
            return pedestrian_tracks.filter(pl.col("ped_id") == actor_id)
        task_poses = episode.fleet.get(actor_id)
        if task_poses is None or len(task_poses) < 2:
            return (
                pedestrian_tracks.head(0)
                if len(pedestrian_tracks) > 0
                else pl.DataFrame(
                    schema={
                        "t_ns": pl.Int64,
                        "ped_id": pl.String,
                        "ped_name": pl.String,
                        "x": pl.Float64,
                        "y": pl.Float64,
                        "vx": pl.Float64,
                        "vy": pl.Float64,
                        "radius_m": pl.Float64,
                        "desired_speed": pl.Float64,
                    }
                )
            )
        previous_time_s = pl.col("t_ns").diff().cast(pl.Float64) / 1e9
        dt_s = pl.when(previous_time_s > 0.0).then(previous_time_s).otherwise(None)
        return (
            task_poses.sort("t_ns")
            .with_columns(
                ped_id=pl.lit(actor_id),
                ped_name=pl.lit(actor_id),
                x=pl.col("pos_x").cast(pl.Float64),
                y=pl.col("pos_y").cast(pl.Float64),
                # The first sample has no derivative: it stays NULL, never zero.
                vx=pl.col("pos_x").diff() / dt_s,
                vy=pl.col("pos_y").diff() / dt_s,
                radius_m=pl.col("radius_m").cast(pl.Float64),
                desired_speed=pl.lit(None, dtype=pl.Float64),
            )
            .select(
                "t_ns",
                "ped_id",
                "ped_name",
                "x",
                "y",
                "vx",
                "vy",
                "radius_m",
                "desired_speed",
            )
        )

    @staticmethod
    def _short_robot_token(robot_id: str) -> str:
        """Compact robot token used in fact ids: ``env_0_jackal_0`` -> ``jackal_0``."""
        return _ENV_PREFIX.sub("", robot_id, count=1)

    @staticmethod
    def _robot_actor_id(
        episode: AlignedEpisodeBundle, robot_tracks: pl.DataFrame | None
    ) -> str:
        """Robot identity of the bundle: its robot column, else its bundle name."""
        if _has_rows(robot_tracks):
            robot_ids = (
                typing.cast(pl.DataFrame, robot_tracks)
                .get_column("robot_id")
                .drop_nulls()
            )
            if len(robot_ids) > 0:
                return str(robot_ids[0])
        return str(episode.robot_name or "robot")

    def _episode_fact_row(
        self,
        episode: AlignedEpisodeBundle,
        run_id: str,
        episode_id: int,
        metrics: Mapping[str, Any],
        provenance: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        """Episode row: identity, the runtime result, the declared EPISODE values and the validity columns."""
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        result, _success = outcome_verdict(episode.outcome_state, episode.outcome_info)
        row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "robot_id": self._robot_actor_id(episode, robot_tracks),
            "result": result,
        }
        row.update(_declared_fact_values(metrics, FactTable.EPISODES))
        published_frame = metrics.get("pose_frame")
        row["pose_frame"] = (
            published_frame
            if isinstance(published_frame, str) and published_frame
            else None
        )
        integrity = metrics.get("integrity")
        _is_valid, invalid_reason = episode_validity(
            robot_tracks=robot_tracks,
            outcome_state=episode.outcome_state,
            integrity=integrity if isinstance(integrity, (list, tuple)) else (),
        )
        row["invalid_reason"] = invalid_reason or ""
        row["null_reason"] = self._episode_null_reason(episode, row, provenance or {})
        return row

    def _episode_null_reason(
        self,
        episode: AlignedEpisodeBundle,
        row: Mapping[str, object],
        provenance: Mapping[str, str],
    ) -> str:
        """Episode provenance: missing-field codes, missing stream topics and calculator exceptions."""
        reasons = [_row_null_reason(row, _nullable(FactTable.EPISODES), provenance)]
        if not _has_rows(_table(episode, ROBOT_TRACKS_TOPIC)):
            reasons.append(f"{TOPIC_MISSING}:{ROBOT_TRACKS_TOPIC}")
        if not _has_rows(_table(episode, PEDESTRIAN_TRACKS_TOPIC)):
            reasons.append(PEDSIM_UNAVAILABLE)
        base = ";".join(reason for reason in reasons if reason)
        return _merge_provenance_reasons(base, provenance, row)

    def _contact_fact_rows(
        self, registry: MetricRegistry, contact_slices: tuple[ScopeSlice, ...]
    ) -> list[dict[str, object]]:
        """One fact row per detected contact slice, attributed by the scope-aware contact calculators."""
        rows: list[dict[str, object]] = []
        for contact_slice in contact_slices:
            outcome: ScopeExecutionOutcome = registry.run_scope(
                MetricScope.CONTACT,
                contact_slice.window_bundle,
                available_topics=set(contact_slice.window_bundle.topics or {}),
                scope_slice=contact_slice,
            )
            contact_res = outcome.results
            ident = contact_slice.identity
            row: dict[str, object] = _declared_fact_values(
                contact_res, FactTable.CONTACTS
            )
            row.update(
                {
                    "run_id": ident["run_id"],
                    "episode_id": int(ident["episode_id"]),
                    "robot_id": ident["robot_id"],
                    "contact_id": contact_slice.slice_id,
                    "encounter_id": _ABSENT_ID
                    if contact_slice.parent_id is None
                    else contact_slice.parent_id,
                    "collider_kind": str(ident["collider_kind"]),
                    "obstacle_id": str(ident["obstacle_id"]),
                    "t_onset_ns": int(contact_slice.t_start_ns),
                    "t_end_ns": int(contact_slice.t_end_ns),
                    "duration_s": float(
                        max(
                            0,
                            int(contact_slice.t_end_ns) - int(contact_slice.t_start_ns),
                        )
                    )
                    / NS_PER_S,
                }
            )
            base_reason = _row_null_reason(
                row, _nullable(FactTable.CONTACTS), outcome.provenance
            )
            row["null_reason"] = _merge_provenance_reasons(
                base_reason, outcome.provenance, row
            )
            rows.append(row)
        return rows

    def _encounter_fact_rows(
        self,
        registry: MetricRegistry,
        episode: AlignedEpisodeBundle,
        encounter_slices: tuple[ScopeSlice, ...],
        contact_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """One fact row per encounter slice, from registered encounter calculators."""
        if not encounter_slices:
            return []

        power_tbl = _table(episode, ROBOT_POWER_TOPIC)
        has_power = _has_rows(power_tbl)
        episode_mean_power_w = (
            float(
                typing.cast(pl.DataFrame, power_tbl)
                .get_column("total_power_w")
                .drop_nulls()
                .mean()
            )  # type: ignore[arg-type]
            if has_power
            and len(
                typing.cast(pl.DataFrame, power_tbl)
                .get_column("total_power_w")
                .drop_nulls()
            )
            > 0
            else None
        )

        contacts_by_encounter: dict[str, dict[str, object]] = {}
        for contact in contact_rows:
            encounter_id = contact.get("encounter_id")
            if (
                isinstance(encounter_id, str)
                and encounter_id not in contacts_by_encounter
            ):
                contacts_by_encounter[encounter_id] = contact

        rows: list[dict[str, object]] = []
        for enc_slice in encounter_slices:
            priors = {"episode_mean_power_w": episode_mean_power_w}
            outcome: ScopeExecutionOutcome = registry.run_scope(
                MetricScope.ENCOUNTER,
                enc_slice.window_bundle,
                available_topics=set(enc_slice.window_bundle.topics or {}),
                prior_results=priors,
                scope_slice=enc_slice,
            )
            enc_res = outcome.results
            ident = enc_slice.identity
            encounter_id = enc_slice.slice_id
            linked_contact = contacts_by_encounter.get(encounter_id)
            row: dict[str, object] = _declared_fact_values(
                enc_res, FactTable.ENCOUNTERS
            )
            row.update(
                {
                    "run_id": ident["run_id"],
                    "episode_id": int(ident["episode_id"]),
                    "robot_id": ident["robot_id"],
                    "encounter_id": encounter_id,
                    "actor_1": ident["actor_1"],
                    "actor_2": ident["actor_2"],
                    "actor_2_kind": ident["actor_2_kind"],
                    "t_start_ns": int(enc_slice.t_start_ns),
                    "t_end_ns": int(enc_slice.t_end_ns),
                    "duration_s": float(
                        max(0, int(enc_slice.t_end_ns) - int(enc_slice.t_start_ns))
                    )
                    / NS_PER_S,
                    "approach_type": ident["approach_type"],
                    "min_clearance_m": _float(ident["min_clearance_m"]),
                    "detector_version": ident["detector_version"],
                    "detector_params_hash": ident["detector_params_hash"],
                    "contact_id": _ABSENT_ID
                    if linked_contact is None
                    else str(linked_contact["contact_id"]),
                }
            )
            window_reason = str(ident.get("window_null_reason") or "")
            base_reason = ";".join(
                part
                for part in (
                    self._encounter_null_reason(row, has_power=has_power),
                    window_reason,
                )
                if part
            )
            row["null_reason"] = _merge_provenance_reasons(
                base_reason, outcome.provenance, row
            )
            rows.append(row)
        return rows

    @staticmethod
    def _encounter_null_reason(row: Mapping[str, object], has_power: bool) -> str:
        """Encounter provenance: the power ratio is zero_denominator without power, other nulls have no samples."""
        reasons: list[str] = []
        for field in _nullable(FactTable.ENCOUNTERS):
            if row.get(field) is None:
                if field == "energy_surge_ratio" and has_power:
                    reasons.append(f"{field}:{ZERO_DENOMINATOR}")
                else:
                    reasons.append(f"{field}:{NO_SAMPLES}")
        return ";".join(reasons)

    def _pedestrian_fact_frame(
        self,
        registry: MetricRegistry,
        episode: AlignedEpisodeBundle,
        pedestrian_slices: tuple[ScopeSlice, ...],
    ) -> pl.DataFrame:
        """Per-pedestrian lifetime summaries executed through registered PEDESTRIAN calculators."""
        if not pedestrian_slices:
            return pl.DataFrame(schema=_fact_schema(FactTable.PEDESTRIANS))

        rows: list[dict[str, object]] = []
        for ped_slice in pedestrian_slices:
            ped_id = ped_slice.slice_id
            outcome: ScopeExecutionOutcome = registry.run_scope(
                MetricScope.PEDESTRIAN,
                ped_slice.window_bundle,
                available_topics=set(ped_slice.window_bundle.topics or {}),
                pedsim_available=True,
                scope_slice=ped_slice,
            )
            ped_res = outcome.results
            ident = ped_slice.identity
            row: dict[str, object] = {
                "run_id": ident["run_id"],
                "episode_id": int(ident["episode_id"]),
                "env_id": ident["env_id"],
                "ped_id": ped_id,
                "ped_name": ident["ped_name"],
                "ped_policy": ident["ped_policy"],
                # Blind to robots only at its own robot_awareness 0; unknown while the pedestrian's awareness is not recorded.
                "is_blind": None
                if ident["is_blind"] is None
                else bool(ident["is_blind"]),
                "desired_speed": _float(ident["desired_speed"]),
                "radius_m": _float(ident["radius_m"]),
                "spawn_x": _float(ident["spawn_x"]),
                "spawn_y": _float(ident["spawn_y"]),
                "spawn_yaw": _float(ident["spawn_yaw"]),
            }
            row.update(_declared_fact_values(ped_res, FactTable.PEDESTRIANS))
            provenance = dict(outcome.provenance)
            if row["is_blind"] is None:
                provenance.setdefault("is_blind", f"is_blind:{NOT_RECORDED}")
            base_reason = _row_null_reason(
                row, _nullable(FactTable.PEDESTRIANS), provenance
            )
            row["null_reason"] = _merge_provenance_reasons(base_reason, provenance, row)
            rows.append(row)

        return _frame(rows, FactTable.PEDESTRIANS)

    def _robot_fact_frame(
        self,
        robot_slice: ScopeSlice,
        outcome: ScopeExecutionOutcome,
        episode_provenance: Mapping[str, str],
    ) -> pl.DataFrame:
        """Per-robot fact row: identity, the declared ROBOT values and their provenance."""
        ident = robot_slice.identity
        row: dict[str, object] = {
            "run_id": ident["run_id"],
            "episode_id": int(ident["episode_id"]),
            "env_id": ident["env_id"],
            "robot_id": ident["robot_id"],
            "robot_name": ident["robot_name"],
        }
        row.update(_declared_fact_values(outcome.results, FactTable.ROBOTS))
        base_reason = _row_null_reason(
            row, _nullable(FactTable.ROBOTS), outcome.provenance
        )
        merged = _merge_provenance_reasons(base_reason, outcome.provenance, row)
        row["null_reason"] = _merge_provenance_reasons(merged, episode_provenance, row)
        return _frame([row], FactTable.ROBOTS)

    def _edge_case_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per case step and per scored, missing or unplayed case agent."""
        parts: list[pl.DataFrame] = []
        for kind, key in _EDGE_CASE_LIST_FIELDS:
            elements = _sequence(metrics.get(key))
            if not elements:
                continue
            parts.append(
                pl.DataFrame(
                    {
                        "kind": [kind] * len(elements),
                        "ordinal": pl.int_range(0, len(elements), eager=True),
                        "element": pl.Series("element", elements, dtype=pl.String),
                    }
                )
            )
        if not parts:
            return pl.DataFrame(schema=_fact_schema(FactTable.EDGE_CASES))
        frame = pl.concat(parts, how="vertical").with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            case_idx=pl.lit(0, dtype=pl.Int64),
            null_reason=_null_reason_expression(_nullable(FactTable.EDGE_CASES)),
        )
        return _conformed(frame, _fact_table(FactTable.EDGE_CASES))

    def _formation_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per formation member and per formation agent that was never recorded."""
        parts: list[pl.DataFrame] = []
        for role, key in _FORMATION_LIST_FIELDS:
            elements = _sequence(metrics.get(key))
            if not elements:
                continue
            parts.append(
                pl.DataFrame(
                    {
                        "role": [role] * len(elements),
                        "element": pl.Series("element", elements, dtype=pl.String),
                    }
                )
            )
        if not parts:
            return pl.DataFrame(schema=_fact_schema(FactTable.FORMATIONS))
        frame = pl.concat(parts, how="vertical").with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            formation_idx=pl.lit(0, dtype=pl.Int64),
            null_reason=_null_reason_expression(_nullable(FactTable.FORMATIONS)),
        )
        return _conformed(frame, _fact_table(FactTable.FORMATIONS))

    def _phase_outcome_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per replayed phase with its outcome token and met flag."""
        outcomes = _sequence(metrics.get("phase_outcomes"))
        if not outcomes:
            return pl.DataFrame(schema=_fact_schema(FactTable.PHASE_OUTCOMES))
        frame = (
            pl.DataFrame(
                {
                    "phase_idx": pl.int_range(0, len(outcomes), eager=True),
                    "outcome": pl.Series("outcome", outcomes, dtype=pl.String),
                }
            )
            .with_columns(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                robot_id=pl.lit(robot_id, dtype=pl.String),
                met=pl.col("outcome") == pl.lit("met"),
            )
            .with_columns(
                null_reason=_null_reason_expression(_nullable(FactTable.PHASE_OUTCOMES))
            )
        )
        return _conformed(frame, _fact_table(FactTable.PHASE_OUTCOMES))

    def _judge_disagreement_fact_frame(
        self,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per judge disagreement with its text."""
        disagreements = _sequence(metrics.get("judge_disagreements"))
        if not disagreements:
            return pl.DataFrame(schema=_fact_schema(FactTable.JUDGE_DISAGREEMENTS))
        frame = (
            pl.DataFrame(
                {
                    "disagreement_idx": pl.int_range(0, len(disagreements), eager=True),
                    "disagreement": pl.Series(
                        "disagreement", disagreements, dtype=pl.String
                    ),
                }
            )
            .with_columns(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
                robot_id=pl.lit(robot_id, dtype=pl.String),
            )
            .with_columns(
                null_reason=_null_reason_expression(
                    _nullable(FactTable.JUDGE_DISAGREEMENTS)
                )
            )
        )
        return _conformed(frame, _fact_table(FactTable.JUDGE_DISAGREEMENTS))

    def _mechanism_state_fact_frame(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """Change-point rows of every door and elevator state series, one flag per mechanism."""
        states = _sequence(metrics.get("mechanism_states"))
        if not states:
            return pl.DataFrame(schema=_fact_schema(FactTable.MECHANISM_STATES))
        times = _event_time_ns(episode)
        frames: list[pl.DataFrame] = []
        for state in states:
            if not isinstance(state, Mapping):
                continue
            progress = _sequence(state.get("progress"))
            locked = _sequence(state.get("locked"))
            if not progress:
                continue
            count = len(progress)
            time_column = (
                pl.Series("t_ns", times[:count], dtype=pl.Int64)
                if len(times) >= count
                else pl.Series("t_ns", [None] * count, dtype=pl.Int64)
            )
            series = pl.DataFrame(
                {
                    "t_ns": time_column,
                    "progress": pl.Series("progress", progress).cast(pl.Float64),
                    "locked": pl.Series(
                        "locked", locked[:count] if locked else [None] * count
                    ).cast(pl.Boolean),
                }
            ).with_columns(pl.col("progress").fill_nan(None))
            changed = series.filter(
                (pl.int_range(0, pl.len()) == 0)
                | pl.col("progress").ne_missing(pl.col("progress").shift(1))
                | pl.col("locked").ne_missing(pl.col("locked").shift(1))
            )
            rows = pl.concat(
                [
                    changed.select(
                        "t_ns",
                        pl.lit(field).alias("field"),
                        pl.col(field).cast(pl.Float64).alias("value"),
                    )
                    for field in _MECHANISM_FIELDS
                ],
                how="vertical",
            ).sort("t_ns", "field", nulls_last=True)
            frame = rows.with_columns(
                entity=pl.lit(str(state.get("entity") or ""), dtype=pl.String),
                level=pl.lit(str(state.get("level") or ""), dtype=pl.String),
                kind=pl.lit(str(state.get("kind") or ""), dtype=pl.String),
            )
            frames.append(frame)
        if not frames:
            return pl.DataFrame(schema=_fact_schema(FactTable.MECHANISM_STATES))
        frame = pl.concat(frames, how="vertical").with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            null_reason=_null_reason_expression(_nullable(FactTable.MECHANISM_STATES)),
        )
        return _conformed(frame, _fact_table(FactTable.MECHANISM_STATES))

    def _collision_fact_frame(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        """One row per obstacle the episode hit, with the first contact's kind, time and point."""
        obstacles = _sequence(metrics.get("collision_obstacles"))
        if not obstacles:
            return pl.DataFrame(schema=_fact_schema(FactTable.COLLISIONS))
        identifiers = pl.DataFrame(
            {"obstacle_id": pl.Series("obstacle_id", obstacles, dtype=pl.String)}
        ).with_row_index("collision_idx")
        contacts = _table(episode, COLLISION_CONTACTS_TOPIC)
        if contacts is None or not {"obstacle_id", "t_ns"} <= set(contacts.columns):
            firsts = identifiers.with_columns(
                obstacle_kind=pl.lit(None, dtype=pl.String),
                t_ns=pl.lit(None, dtype=pl.Int64),
                x_m=pl.lit(None, dtype=pl.Float64),
                y_m=pl.lit(None, dtype=pl.Float64),
            )
        else:
            firsts = identifiers.join(
                contacts.filter(
                    pl.col("obstacle_id").is_not_null()
                    & (pl.col("obstacle_id") != pl.lit(""))
                )
                .sort("t_ns")
                .group_by("obstacle_id")
                .agg(
                    pl.col("kind").first().alias("obstacle_kind"),
                    pl.col("t_ns").first().cast(pl.Int64).alias("t_ns"),
                    pl.col("obstacle_x").first().cast(pl.Float64).alias("x_m"),
                    pl.col("obstacle_y").first().cast(pl.Float64).alias("y_m"),
                ),
                on="obstacle_id",
                how="left",
            ).sort("collision_idx")
        frame = firsts.with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            null_reason=_null_reason_expression(_nullable(FactTable.COLLISIONS)),
        )
        return _conformed(frame, _fact_table(FactTable.COLLISIONS))

    @staticmethod
    def _encounter_id(episode_id: int, robot_token: str, encounter_idx: int) -> str:
        return f"enc-{episode_id:03d}-{robot_token}-{encounter_idx:03d}"

    @staticmethod
    def _contact_id(episode_id: int, robot_token: str, contact_idx: int) -> str:
        return f"con-{episode_id:03d}-{robot_token}-{contact_idx:03d}"
