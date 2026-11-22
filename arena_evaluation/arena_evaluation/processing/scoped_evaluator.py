"""Multi-resolution scoped evaluator: one episode in, eleven typed fact tables out."""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
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
    ROBOT_COLLISION_CONTACTS_TOPIC,
    ContactAttributionCalculator,
    ContactEvent,
    detect_contact_events,
    static_point_of,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    PEDESTRIAN_ARENA_PEDS_TOPIC,
    ROBOT_TRACKS_TOPIC,
    extract_episode_ped_tracks,
    resolve_robot_trajectory,
)
from arena_evaluation.processing.outcome import outcome_verdict
from arena_evaluation.processing.validity import episode_validity
from arena_evaluation.processing.windows import (
    ACTOR_PEDESTRIAN,
    ACTOR_ROBOT,
    DEFAULT_DETECTOR_PARAMS,
    NS_PER_S,
    DetectorParams,
    contact_windows_lf,
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

NO_SAMPLES: Final[str] = NullReason.NO_SAMPLES.value
PEDSIM_UNAVAILABLE: Final[str] = NullReason.PEDSIM_UNAVAILABLE.value
ZERO_DENOMINATOR: Final[str] = NullReason.ZERO_DENOMINATOR.value
TOPIC_MISSING: Final[str] = NullReason.TOPIC_MISSING.value
NOT_RECORDED: Final[str] = NullReason.NOT_RECORDED.value
_GENERIC_PROVENANCE_KINDS: Final[tuple[str, ...]] = (
    NO_SAMPLES,
    PEDSIM_UNAVAILABLE,
    TOPIC_MISSING,
    "exception",
)

ROBOT_POWER_TOPIC: Final[str] = "robot_power"
CONTACT_PRE_WINDOW_S: Final[float] = 0.5
CONTACT_POST_WINDOW_S: Final[float] = 0.5
_ENV_PREFIX: Final[re.Pattern[str]] = re.compile(r"^env_[0-9]+_")
_ROBOT_NAMESPACE: Final[re.Pattern[str]] = re.compile(r"^env_[0-9]+[_/]")
_ABSENT_ID: Final[str] = ""
_EDGE_CASE_LIST_FIELDS: Final = (
    ("step", "edge_case_steps"),
    ("agent", "edge_case_agents"),
    ("missing", "edge_case_missing"),
    ("unplayed", "edge_case_unplayed"),
)
_FORMATION_LIST_FIELDS: Final = (
    ("member", "edge_case_formation_members"),
    ("absent", "edge_case_formation_absent"),
)
_MECHANISM_FIELDS: Final[tuple[str, ...]] = ("locked", "progress")
_ACTOR_TRACK_SCHEMA: Final = {
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

ProgressCallback = Callable[[str, int, int], None]


def _fact_table(table: FactTable) -> type:
    load_metric_definitions()
    return FACT_TABLES[table]


def _empty_fact(table: FactTable) -> pl.DataFrame:
    return pl.DataFrame(schema=_fact_table(table).polars_schema())


def _conformed(frame: pl.DataFrame, table: type) -> pl.DataFrame:
    schema = table.polars_schema()
    missing = [c for c in schema.names() if c not in frame.columns]
    undeclared = [c for c in frame.columns if c not in schema]
    if missing or undeclared:
        raise ContractViolationError(f"{table.table_name}: frame does not match its declaration (missing {missing}, undeclared {undeclared})")
    return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])


def _frame(rows: list[dict[str, object]], table: FactTable) -> pl.DataFrame:
    schema = _fact_table(table).polars_schema()
    return pl.DataFrame([{k: r.get(k) for k in schema.names()} for r in rows], schema=schema)


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


def _default_fact_field(table: FactTable) -> pl.DataFrame:
    return dataclasses.field(default_factory=lambda: _empty_fact(table))


@dataclasses.dataclass(frozen=True, slots=True)
class ScopedFactTables:
    """The eleven typed fact tables of one evaluated episode."""

    episodes: pl.DataFrame
    encounters: pl.DataFrame
    contacts: pl.DataFrame
    pedestrians: pl.DataFrame
    robots: pl.DataFrame
    edge_cases: pl.DataFrame = _default_fact_field(FactTable.EDGE_CASES)
    formations: pl.DataFrame = _default_fact_field(FactTable.FORMATIONS)
    phase_outcomes: pl.DataFrame = _default_fact_field(FactTable.PHASE_OUTCOMES)
    judge_disagreements: pl.DataFrame = _default_fact_field(FactTable.JUDGE_DISAGREEMENTS)
    mechanism_states: pl.DataFrame = _default_fact_field(FactTable.MECHANISM_STATES)
    collisions: pl.DataFrame = _default_fact_field(FactTable.COLLISIONS)

    def by_name(self) -> dict[str, pl.DataFrame]:
        return {FactTable[f.name.upper()].value: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluatedEpisode:
    """One robot's episode: its eleven fact tables, its series tables and its episode metrics row."""

    facts: ScopedFactTables
    series: dict[str, pl.DataFrame]
    episode_metrics: dict[str, Any]


def write_fact_tables(tables: ScopedFactTables, metrics_dir: pathlib.Path) -> None:
    load_metric_definitions()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    named = tables.by_name()
    for member, table in FACT_TABLES.items():
        _conformed(named[member.value], table).write_parquet(metrics_dir / table.file_name(), compression="zstd")


def write_series_tables(series: Mapping[str, pl.DataFrame], metrics_dir: pathlib.Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for stem, frame in series.items():
        if (table := SERIES_TABLES.get(stem)) is None:
            raise ContractViolationError(f"series {stem!r} is not declared in the registry")
        _conformed(frame, table).write_parquet(metrics_dir / table.file_name(), compression="zstd")


def _materialize_episode_bundle(episode: AlignedEpisodeBundle) -> AlignedEpisodeBundle:
    """Materialize all LazyFrame topic, fleet, and data tables once via pl.collect_all."""
    lazy: list[tuple[str, str, pl.LazyFrame]] = []
    eager: dict[str, dict[str, pl.DataFrame]] = {"topic": {}, "fleet": {}}
    for kind, raw in (("topic", episode.topics), ("fleet", episode.fleet)):
        for key, value in raw.items() if isinstance(raw, Mapping) else ():
            if isinstance(value, pl.LazyFrame):
                lazy.append((kind, key, value))
            elif value is not None:
                eager[kind][key] = value
    data = episode.data if isinstance(episode.data, pl.DataFrame) else pl.DataFrame()
    if isinstance(episode.data, pl.LazyFrame):
        lazy.append(("data", "data", episode.data))
    for (kind, key, _), frame in zip(lazy, pl.collect_all([f for _, _, f in lazy]), strict=True):
        if kind == "data":
            data = frame
        else:
            eager[kind][key] = frame
    return dataclasses.replace(episode, data=data, topics=eager["topic"], fleet=eager["fleet"])


def _table(episode: AlignedEpisodeBundle, topic: str) -> pl.DataFrame | None:
    frame = (episode.topics or {}).get(topic)
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _has_rows(table: pl.DataFrame | None) -> bool:
    return table is not None and not table.is_empty()


def _float(val: object) -> float | None:
    return None if val is None else float(val)  # type: ignore[arg-type]


def _first_val[T](
    df: pl.DataFrame | None,
    col: str,
    cast: Callable[[object], T],
    default: T | None = None,
) -> T | None:
    """Return the first non-null value of `col` in `df` cast via `cast`, or `default`."""
    if df is None or col not in df.columns:
        return default
    values = df.get_column(col).drop_nulls()
    return cast(values[0]) if len(values) > 0 else default


def _declared_value(value: object, decl: Metric) -> object:
    if value is None:
        return None
    if decl.dtype == pl.Float64:
        return float(value)  # type: ignore[arg-type]
    if decl.dtype == pl.Int64:
        return int(value)  # type: ignore[arg-type]
    if decl.dtype == pl.Boolean:
        if not isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            raise TypeError(f"{decl.name}: expected a boolean, got {type(value).__name__}")
        return bool(value)
    if decl.dtype == pl.String:
        if not isinstance(value, str):
            raise TypeError(f"{decl.name}: expected a string, got {type(value).__name__}")
        return value
    raise TypeError(f"{decl.name}: unsupported fact dtype {decl.dtype}")


def _declared_fact_values(results: Mapping[str, object], table: FactTable) -> dict[str, object]:
    scope = _fact_table(table).table_scope
    if scope is None:
        return {}
    return {m.name: _declared_value(results.get(m.name), m) for m in defined_metrics() if scope in m.scopes}


def _null_reason(
    row: Mapping[str, object],
    nullable: tuple[str, ...],
    provenance: Mapping[str, str] | None = None,
    *,
    codes: list[str] | None = None,
) -> str:
    """The semicolon-joined null reasons of `row`, with provenance overrides folded in once."""
    prov = provenance or {}

    def specific(field: str) -> str | None:
        code = prov.get(field)
        return code if isinstance(code, str) and code and code.partition(":")[2].partition(":")[0] not in _GENERIC_PROVENANCE_KINDS else None

    base = codes if codes is not None else [specific(field) or f"{field}:{NO_SAMPLES}" for field in nullable if row.get(field) is None]
    causes = {field: code for field, code in prov.items() if ":exception:" in code or ((":topic_missing:" in code or ":not_recorded:" in code) and row.get(field) is None)}
    return ";".join(
        dict.fromkeys(
            [
                *(part for part in base if part and part.partition(":")[0] not in causes),
                *causes.values(),
            ]
        )
    )


def _finalize_fact_row(
    row: dict[str, object],
    table: FactTable,
    provenance: Mapping[str, str],
    *,
    extra_provenance: Mapping[str, str] | None = None,
) -> dict[str, object]:
    nullable = _fact_table(table).null_reason_columns()
    merged = _null_reason(row, nullable, provenance)
    if extra_provenance:
        merged = _null_reason(
            row,
            nullable,
            extra_provenance,
            codes=[part for part in merged.split(";") if part],
        )
    row["null_reason"] = merged
    return row


def unaligned_episode_facts(*, run_id: str, episode_id: int, robot_id: str, result: str | None, outcome_state: int | None, integrity: Sequence[str]) -> ScopedFactTables:
    """The fact tables of an episode with no trajectory to evaluate: its invalid episode row with every metric NULL for the missing track."""
    _, invalid_reason = episode_validity(robot_tracks=None, outcome_state=outcome_state, integrity=integrity)
    row: dict[str, object] = {
        "run_id": run_id,
        "episode_id": int(episode_id),
        "robot_id": robot_id,
        "result": result,
        "invalid_reason": invalid_reason or "",
    }
    nullable = _fact_table(FactTable.EPISODES).null_reason_columns()
    row["null_reason"] = _null_reason(row, nullable, codes=[f"{field}:{TOPIC_MISSING}:{ROBOT_TRACKS_TOPIC}" for field in nullable if row.get(field) is None])
    return ScopedFactTables(
        episodes=_frame([row], FactTable.EPISODES),
        encounters=_empty_fact(FactTable.ENCOUNTERS),
        contacts=_empty_fact(FactTable.CONTACTS),
        pedestrians=_empty_fact(FactTable.PEDESTRIANS),
        robots=_empty_fact(FactTable.ROBOTS),
    )


def _fleet_key(robot_name: str, fleet: Mapping[str, pl.DataFrame]) -> str:
    if robot_name in fleet:
        return robot_name
    bare = _ROBOT_NAMESPACE.sub("", robot_name)
    return next((k for k in fleet if _ROBOT_NAMESPACE.sub("", k) == bare), robot_name)


def _contact_intervals(events: Sequence[ContactEvent]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_onset_ns": [e.t0_ns for e in events],
            "t_end_ns": [e.t_end_ns for e in events],
            "collider_kind": [e.obstacle_kind for e in events],
            "obstacle_id": [e.obstacle_id for e in events],
            "obstacle_x": [None if e.static_point is None else e.static_point[0] for e in events],
            "obstacle_y": [None if e.static_point is None else e.static_point[1] for e in events],
        },
        schema={
            "t_onset_ns": pl.Int64,
            "t_end_ns": pl.Int64,
            "collider_kind": pl.String,
            "obstacle_id": pl.String,
            "obstacle_x": pl.Float64,
            "obstacle_y": pl.Float64,
        },
    )


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
        if robot_tracks is None or not _has_rows(robot_tracks):
            return 0, 0, None
        times = robot_tracks.get_column("t_ns").drop_nulls()
        lo, hi = times.min(), times.max()
        return (
            lo if isinstance(lo, int) else 0,
            hi if isinstance(hi, int) else 0,
            _first_val(robot_tracks, "env_id", str),
        )

    @staticmethod
    def _scope_slice(
        scope: MetricScope,
        slice_id: str,
        parent_id: str | None,
        t_start_ns: int,
        t_end_ns: int,
        identity: dict[str, object],
        window_bundle: AlignedEpisodeBundle,
        full_episode: AlignedEpisodeBundle,
    ) -> ScopeSlice:
        return ScopeSlice(
            scope=scope,
            slice_id=slice_id,
            parent_id=parent_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            identity=identity,
            window_bundle=window_bundle,
            full_episode=full_episode,
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
        if contact_windows.is_empty():
            return []
        pre_s = float(self.metrics_config.get("pre_contact_window_s", CONTACT_PRE_WINDOW_S))
        post_s = float(self.metrics_config.get("post_contact_window_s", CONTACT_POST_WINDOW_S))
        common = {"run_id": run_id, "episode_id": int(episode_id), "robot_id": robot_id}
        slices: list[ScopeSlice] = []
        # agent-contracts: ignore AC004 - materializes ScopeSlice dataclasses
        for row in contact_windows.iter_rows(named=True):
            kind = str(row["collider_kind"])
            actor = _fleet_key(str(row["obstacle_id"]), materialized.fleet) if kind == ACTOR_ROBOT else str(row["obstacle_id"])
            onset, end = int(row["t_onset_ns"]), int(row["t_end_ns"])
            idx = row["encounter_idx"]
            parent = encounter_ids.get(int(idx)) if idx is not None and int(idx) >= 0 else None
            contact_id = self._contact_id(episode_id, robot_token, int(row["contact_idx"]))
            slices.append(
                self._scope_slice(
                    MetricScope.CONTACT,
                    contact_id,
                    parent,
                    onset,
                    end,
                    {
                        **common,
                        "contact_id": contact_id,
                        "encounter_id": parent,
                        "collider_kind": kind,
                        "obstacle_id": actor,
                        "static_point": None if kind in (ACTOR_PEDESTRIAN, ACTOR_ROBOT) else static_point_of(_float(row["obstacle_x"]), _float(row["obstacle_y"])),
                        "pre_contact_window_s": pre_s,
                        "post_contact_window_s": post_s,
                    },
                    self._window_episode(
                        materialized,
                        onset - round(pre_s * NS_PER_S),
                        end + round(post_s * NS_PER_S),
                        actor,
                        kind,
                    ),
                    materialized,
                )
            )
        return slices

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
        if encounter_windows.is_empty():
            return []
        common = {"run_id": run_id, "episode_id": int(episode_id), "robot_id": robot_id}
        return [
            self._scope_slice(
                MetricScope.ENCOUNTER,
                encounter_id := self._encounter_id(
                    episode_id,
                    robot_token,
                    -1 if row["encounter_idx"] is None else int(row["encounter_idx"]),
                ),
                episode_slice_id,
                start := int(row["t_start_ns"]),
                end := int(row["t_end_ns"]),
                {
                    **common,
                    "encounter_id": encounter_id,
                    "actor_1": robot_id,
                    "actor_2": str(row["actor_id"]),
                    "actor_2_kind": str(row["actor_kind"]),
                    "approach_type": None if row["approach_type"] is None else str(row["approach_type"]),
                    "min_clearance_m": _float(row["min_clearance_m"]),
                    "detector_version": str(row["detector_version"]),
                    "detector_params_hash": str(row["detector_params_hash"]),
                    "window_null_reason": "" if row["null_reason"] is None else str(row["null_reason"]),
                },
                self._window_episode(
                    materialized,
                    start,
                    end,
                    str(row["actor_id"]),
                    str(row["actor_kind"]),
                ),
                materialized,
            )
            # agent-contracts: ignore AC004 - materializes ScopeSlice dataclasses
            for row in encounter_windows.iter_rows(named=True)
        ]

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
        if pedestrian_tracks is None or pedestrian_tracks.is_empty():
            return []
        common = {"run_id": run_id, "episode_id": int(episode_id)}
        slices: list[ScopeSlice] = []
        for ped_df in pedestrian_tracks.sort(["ped_id", "t_ns"]).partition_by("ped_id", maintain_order=True):
            ped_id = str(ped_df.get_column("ped_id")[0])
            times = ped_df.get_column("t_ns").drop_nulls()
            lo, hi = times.min(), times.max()
            blind = _first_val(ped_df, "robot_blind", bool)
            slices.append(
                ScopedEvaluator._scope_slice(
                    MetricScope.PEDESTRIAN,
                    ped_id,
                    episode_slice_id,
                    lo if isinstance(lo, int) else ep_start_ns,
                    hi if isinstance(hi, int) else ep_end_ns,
                    {
                        **common,
                        "env_id": _first_val(ped_df, "env_id", str, env_id),
                        "ped_id": ped_id,
                        "ped_name": _first_val(ped_df, "ped_name", str, ped_id),
                        "ped_policy": _first_val(ped_df, "policy", str),
                        "is_blind": blind,
                        "desired_speed": _first_val(ped_df, "desired_speed", _float),
                        "radius_m": _first_val(ped_df, "radius_m", _float),
                        "spawn_x": _first_val(ped_df, "x", _float),
                        "spawn_y": _first_val(ped_df, "y", _float),
                        "spawn_yaw": _first_val(ped_df, "yaw", _float),
                    },
                    dataclasses.replace(
                        materialized,
                        data=pl.DataFrame(),
                        topics={
                            **(materialized.topics or {}),
                            PEDESTRIAN_ARENA_PEDS_TOPIC: ped_df,
                        },
                        fleet={},
                    ),
                    materialized,
                )
            )
        return slices

    def decompose_hierarchy(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        contact_calculator: ContactAttributionCalculator,
    ) -> EpisodeScopeHierarchy:
        materialized = _materialize_episode_bundle(episode)
        robot_tracks = _table(materialized, ROBOT_TRACKS_TOPIC)
        ped_tracks = _table(materialized, PEDESTRIAN_ARENA_PEDS_TOPIC)
        if _has_rows(robot_tracks):
            encounter_windows = self._detect_encounters(materialized, robot_tracks, ped_tracks)  # type: ignore[arg-type]
            contact_windows = self._detect_contacts(materialized, encounter_windows, contact_calculator)
        else:
            encounter_windows, contact_windows = (
                empty_encounter_windows(),
                empty_contact_windows(),
            )

        robot_id = _first_val(robot_tracks, "robot_id", str, str(materialized.robot_name or "robot")) or str(materialized.robot_name or "robot")
        robot_token = _ENV_PREFIX.sub("", robot_id, count=1)
        episode_slice_id = f"{run_id}:{int(episode_id)}:{robot_id}"
        start_ns, end_ns, env_id = self._episode_span(robot_tracks)
        encounter_ids = {
            int(idx): self._encounter_id(episode_id, robot_token, int(idx))
            # agent-contracts: ignore AC004 - maps encounter_idx integers to ScopeSlice IDs
            for idx in encounter_windows.get_column("encounter_idx").to_list()
            if idx is not None
        }
        common = {"run_id": run_id, "episode_id": int(episode_id), "robot_id": robot_id}
        robot_slice = self._scope_slice(
            MetricScope.ROBOT,
            robot_id,
            episode_slice_id,
            start_ns,
            end_ns,
            {**common, "env_id": env_id, "robot_name": materialized.robot_name},
            materialized,
            materialized,
        )
        episode_slice = self._scope_slice(
            MetricScope.EPISODE,
            episode_slice_id,
            None,
            start_ns,
            end_ns,
            common,
            materialized,
            materialized,
        )
        return EpisodeScopeHierarchy(
            contacts=tuple(
                self._build_contact_slices(
                    materialized,
                    contact_windows,
                    encounter_ids,
                    robot_token=robot_token,
                    **common,
                )
            ),
            encounters=tuple(
                self._build_encounter_slices(
                    materialized,
                    encounter_windows,
                    robot_token=robot_token,
                    episode_slice_id=episode_slice_id,
                    **common,
                )
            ),
            pedestrians=tuple(
                self._build_pedestrian_slices(
                    materialized,
                    ped_tracks,
                    run_id=run_id,
                    episode_id=episode_id,
                    env_id=env_id,
                    ep_start_ns=start_ns,
                    ep_end_ns=end_ns,
                    episode_slice_id=episode_slice_id,
                )
            ),
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
        materialized_input = _materialize_episode_bundle(episode)
        registry = MetricRegistry(
            self.robot_params,
            world=world,
            metrics_config={
                "contact_merge_gap_s": self.detector_params.contact_merge_gap_s,
                **self.metrics_config,
            },
        )
        robot_key = str(materialized_input.robot_name or "robot")
        topics = materialized_input.topics or {}
        series_tables: dict[str, pl.DataFrame] = {}
        for stem, calc in registry.series_calculators():
            if all(topic in topics for topic in calc.SERIES_REQUIRED_TOPICS):
                frame = calc.series_table(materialized_input, run_id=run_id, robot_id=robot_key)
                if frame is not None and frame.height > 0:
                    series_tables[stem] = frame
        episode = dataclasses.replace(materialized_input, series=series_tables)
        hierarchy = self.decompose_hierarchy(
            episode,
            run_id=run_id,
            episode_id=episode_id,
            contact_calculator=registry.calculators[ContactAttributionCalculator.NAME],  # type: ignore[arg-type]
        )
        materialized = hierarchy.episodes[0].full_episode
        ep_outcome = registry.run_scope(
            MetricScope.EPISODE,
            episode,
            pedsim_available=pedsim_available,
            available_topics=available_topics,
            progress_callback=progress_callback,
        )
        episode_metrics = {**ep_outcome.results, **episode_context}
        provenance = dict(ep_outcome.provenance)
        metrics = {
            **episode_metrics,
            **registry.calculators["mechanism_states"].calculate(materialized, dict(episode_metrics)),
        }

        contact_rows = self._scope_fact_rows(
            registry,
            FactTable.CONTACTS,
            hierarchy.contacts,
            extras=lambda scoped: {
                "encounter_id": _ABSENT_ID if scoped.parent_id is None else scoped.parent_id,
                "t_onset_ns": int(scoped.t_start_ns),
                "t_end_ns": int(scoped.t_end_ns),
                "duration_s": max(0, int(scoped.t_end_ns) - int(scoped.t_start_ns)) / NS_PER_S,
            },
        )
        encounter_rows = self._encounter_fact_rows(registry, materialized, hierarchy.encounters, contact_rows)
        pedestrians = self._pedestrian_fact_frame(registry, hierarchy.pedestrians)
        robot_slice = hierarchy.robots[0]
        robot_outcome = registry.run_scope(
            MetricScope.ROBOT,
            robot_slice.window_bundle,
            available_topics=set(robot_slice.window_bundle.topics or {}),
            prior_results=dict(metrics),
            scope_slice=robot_slice,
        )
        episode_row = self._episode_fact_row(materialized, run_id, episode_id, metrics, provenance)
        robot_row = {
            **robot_slice.identity,
            **_declared_fact_values(robot_outcome.results, FactTable.ROBOTS),
        }
        robots = _frame(
            [
                _finalize_fact_row(
                    robot_row,
                    FactTable.ROBOTS,
                    robot_outcome.provenance,
                    extra_provenance=provenance,
                )
            ],
            FactTable.ROBOTS,
        )
        child_ctx = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "robot_id": robot_slice.identity["robot_id"],
            "metrics": {**metrics, **robot_outcome.children},
        }
        return EvaluatedEpisode(
            facts=ScopedFactTables(
                episodes=_frame([episode_row], FactTable.EPISODES),
                encounters=_frame(encounter_rows, FactTable.ENCOUNTERS),
                contacts=_frame(contact_rows, FactTable.CONTACTS),
                pedestrians=pedestrians,
                robots=robots,
                edge_cases=self._edge_case_fact_frame(**child_ctx),
                formations=self._formation_fact_frame(**child_ctx),
                phase_outcomes=self._phase_outcome_fact_frame(**child_ctx),
                judge_disagreements=self._judge_disagreement_fact_frame(**child_ctx),
                mechanism_states=self._mechanism_state_fact_frame(materialized, **child_ctx),
                collisions=self._collision_fact_frame(materialized, **child_ctx),
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
        radius_m = float(self.robot_params.robot_radius)
        pairs: list[pl.LazyFrame] = []
        if _has_rows(pedestrian_tracks):
            pairs.append(pedestrian_pairs_lf(robot_tracks, pedestrian_tracks, radius_m, self.detector_params))  # type: ignore[arg-type]
        if episode.fleet:
            pairs.append(fleet_pairs_lf(robot_tracks, episode.fleet, radius_m, self.detector_params))
        if not pairs:
            return empty_encounter_windows()
        combined = pairs[0] if len(pairs) == 1 else pl.concat(pairs, how="vertical")
        return detect_encounter_windows_lf(combined, self.detector_params).collect()

    def _detect_contacts(
        self,
        episode: AlignedEpisodeBundle,
        encounters: pl.DataFrame,
        contact_calculator: ContactAttributionCalculator,
    ) -> pl.DataFrame:
        robot, _ = resolve_robot_trajectory(contact_calculator, episode)
        if len(robot.t_ns) == 0:
            return empty_contact_windows()
        gap_s = float(contact_calculator.metrics_config.get("contact_merge_gap_s", DEFAULT_DETECTOR_PARAMS.contact_merge_gap_s))
        events = detect_contact_events(
            episode,
            robot,
            extract_episode_ped_tracks(episode),
            merge_gap_ns=round(gap_s * NS_PER_S),
        )
        if not events:
            return empty_contact_windows()
        return contact_windows_lf(_contact_intervals(events), encounters, self.detector_params).collect()

    def _window_episode(
        self,
        episode: AlignedEpisodeBundle,
        t_start_ns: int,
        t_end_ns: int,
        actor_id: str,
        actor_kind: str,
    ) -> AlignedEpisodeBundle:
        in_window = (pl.col("t_ns") >= t_start_ns) & (pl.col("t_ns") <= t_end_ns)

        def clip(frame: pl.DataFrame) -> pl.DataFrame:
            return frame.filter(in_window) if not frame.is_empty() else frame

        topics = {name: clip(self._actor_track_frame(episode, frame, actor_id, actor_kind) if name == PEDESTRIAN_ARENA_PEDS_TOPIC else frame) for name, frame in (episode.topics or {}).items() if isinstance(frame, pl.DataFrame)}
        if PEDESTRIAN_ARENA_PEDS_TOPIC not in topics and actor_kind != ACTOR_PEDESTRIAN:
            topics[PEDESTRIAN_ARENA_PEDS_TOPIC] = clip(self._actor_track_frame(episode, pl.DataFrame(), actor_id, actor_kind))
        return dataclasses.replace(episode, data=pl.DataFrame(), topics=topics, fleet={})

    def _actor_track_frame(
        self,
        episode: AlignedEpisodeBundle,
        pedestrian_tracks: pl.DataFrame,
        actor_id: str,
        actor_kind: str,
    ) -> pl.DataFrame:
        if actor_kind == ACTOR_PEDESTRIAN:
            return pedestrian_tracks if pedestrian_tracks.is_empty() else pedestrian_tracks.filter(pl.col("ped_id") == actor_id)
        task_poses = episode.fleet.get(actor_id)
        if task_poses is None or len(task_poses) < 2:
            return pedestrian_tracks.head(0) if not pedestrian_tracks.is_empty() else pl.DataFrame(schema=_ACTOR_TRACK_SCHEMA)
        previous_s = pl.col("t_ns").diff().cast(pl.Float64) / 1e9
        dt_s = pl.when(previous_s > 0.0).then(previous_s).otherwise(None)
        return task_poses.sort("t_ns").select(
            "t_ns",
            ped_id=pl.lit(actor_id),
            ped_name=pl.lit(actor_id),
            x=pl.col("pos_x").cast(pl.Float64),
            y=pl.col("pos_y").cast(pl.Float64),
            vx=pl.col("pos_x").diff() / dt_s,
            vy=pl.col("pos_y").diff() / dt_s,
            radius_m=pl.col("radius_m").cast(pl.Float64),
            desired_speed=pl.lit(None, dtype=pl.Float64),
        )

    def _episode_fact_row(
        self,
        episode: AlignedEpisodeBundle,
        run_id: str,
        episode_id: int,
        metrics: Mapping[str, Any],
        provenance: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        robot_tracks = _table(episode, ROBOT_TRACKS_TOPIC)
        result, _ = outcome_verdict(episode.outcome_state, episode.outcome_info)
        pose_frame = metrics.get("pose_frame")
        integrity = metrics.get("integrity")
        _, invalid_reason = episode_validity(
            robot_tracks=robot_tracks,
            outcome_state=episode.outcome_state,
            integrity=integrity if isinstance(integrity, (list, tuple)) else (),
        )
        row: dict[str, object] = {
            "run_id": run_id,
            "episode_id": int(episode_id),
            "robot_id": _first_val(robot_tracks, "robot_id", str, str(episode.robot_name or "robot")),
            **_declared_fact_values(metrics, FactTable.EPISODES),
            "result": result,
            "pose_frame": pose_frame if isinstance(pose_frame, str) and pose_frame else None,
            "invalid_reason": invalid_reason or "",
        }
        nullable = _fact_table(FactTable.EPISODES).null_reason_columns()
        code = _null_reason(row, nullable, provenance or {})
        if not _has_rows(robot_tracks):
            code = ";".join(part for part in (code, f"{TOPIC_MISSING}:{ROBOT_TRACKS_TOPIC}") if part)
        row["null_reason"] = _null_reason(
            row,
            nullable,
            provenance or {},
            codes=[part for part in code.split(";") if part],
        )
        return row

    @staticmethod
    def _scope_outcomes(
        registry: MetricRegistry,
        table: FactTable,
        slices: Sequence[ScopeSlice],
        *,
        prior_results: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[ScopeSlice, ScopeExecutionOutcome]]:
        scope = _fact_table(table).table_scope
        for scope_slice in slices:
            yield (
                scope_slice,
                registry.run_scope(
                    scope,
                    scope_slice.window_bundle,
                    available_topics=set(scope_slice.window_bundle.topics or {}),
                    prior_results=dict(prior_results) if prior_results is not None else None,
                    scope_slice=scope_slice,
                ),
            )

    def _scope_fact_rows(
        self,
        registry: MetricRegistry,
        table: FactTable,
        slices: Sequence[ScopeSlice],
        *,
        prior_results: Mapping[str, Any] | None = None,
        extras: Callable[[ScopeSlice], dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        return [
            _finalize_fact_row(
                {
                    **scoped.identity,
                    **_declared_fact_values(outcome.results, table),
                    **(extras(scoped) if extras else {}),
                },
                table,
                outcome.provenance,
            )
            for scoped, outcome in self._scope_outcomes(registry, table, slices, prior_results=prior_results)
        ]

    def _encounter_fact_rows(
        self,
        registry: MetricRegistry,
        episode: AlignedEpisodeBundle,
        encounter_slices: tuple[ScopeSlice, ...],
        contact_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not encounter_slices:
            return []
        power = _table(episode, ROBOT_POWER_TOPIC)
        has_power = _has_rows(power)
        mean_power = power.get_column("total_power_w").drop_nulls().mean() if has_power else None  # type: ignore[union-attr]
        priors = {"episode_mean_power_w": None if mean_power is None else float(mean_power)}  # type: ignore[arg-type]
        contacts_by_encounter: dict[str, dict[str, object]] = {}
        for contact in contact_rows:
            if isinstance(eid := contact.get("encounter_id"), str) and eid not in contacts_by_encounter:
                contacts_by_encounter[eid] = contact
        nullable = _fact_table(FactTable.ENCOUNTERS).null_reason_columns()
        rows: list[dict[str, object]] = []
        for scoped, outcome in self._scope_outcomes(registry, FactTable.ENCOUNTERS, encounter_slices, prior_results=priors):
            linked = contacts_by_encounter.get(scoped.slice_id)
            row: dict[str, object] = {
                **_declared_fact_values(outcome.results, FactTable.ENCOUNTERS),
                **scoped.identity,
                "t_start_ns": int(scoped.t_start_ns),
                "t_end_ns": int(scoped.t_end_ns),
                "duration_s": max(0, int(scoped.t_end_ns) - int(scoped.t_start_ns)) / NS_PER_S,
                "contact_id": _ABSENT_ID if linked is None else str(linked["contact_id"]),
            }
            codes = [f"{field}:{ZERO_DENOMINATOR if field == 'energy_surge_ratio' and has_power else NO_SAMPLES}" for field in nullable if row.get(field) is None]
            if window_reason := str(scoped.identity.get("window_null_reason") or ""):
                codes.append(window_reason)
            row["null_reason"] = _null_reason(row, nullable, outcome.provenance, codes=codes)
            rows.append(row)
        return rows

    def _pedestrian_fact_frame(self, registry: MetricRegistry, pedestrian_slices: tuple[ScopeSlice, ...]) -> pl.DataFrame:
        if not pedestrian_slices:
            return _empty_fact(FactTable.PEDESTRIANS)
        rows: list[dict[str, object]] = []
        for scoped, outcome in self._scope_outcomes(registry, FactTable.PEDESTRIANS, pedestrian_slices):
            row: dict[str, object] = {
                **scoped.identity,
                **_declared_fact_values(outcome.results, FactTable.PEDESTRIANS),
            }
            row["is_blind"] = None if row["is_blind"] is None else bool(row["is_blind"])
            provenance = dict(outcome.provenance)
            if row["is_blind"] is None:
                provenance.setdefault("is_blind", f"is_blind:{NOT_RECORDED}")
            rows.append(_finalize_fact_row(row, FactTable.PEDESTRIANS, provenance))
        return _frame(rows, FactTable.PEDESTRIANS)

    @staticmethod
    def _finalize_child_frame(
        frame: pl.DataFrame | None,
        table: FactTable,
        run_id: str,
        episode_id: int,
        robot_id: str,
        **extra_cols: pl.Expr,
    ) -> pl.DataFrame:
        if frame is None or frame.is_empty():
            return _empty_fact(table)
        nullable = _fact_table(table).null_reason_columns()
        parts = [pl.when(pl.col(c).is_null()).then(pl.lit(f"{c}:{NO_SAMPLES}")) for c in nullable]
        null_reason = pl.concat_str(parts, separator=";", ignore_nulls=True).fill_null("") if parts else pl.lit("", dtype=pl.String)
        enriched = frame.with_columns(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id, dtype=pl.String),
            **extra_cols,
        ).with_columns(null_reason=null_reason)
        return _conformed(enriched, _fact_table(table))

    def _list_fact_frame(
        self,
        table: FactTable,
        column: str,
        index_column: str,
        values: object,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        extra_columns: Mapping[str, pl.Expr] | None = None,
    ) -> pl.DataFrame:
        if not isinstance(values, list) or not values:
            return _empty_fact(table)
        frame = pl.DataFrame(
            {
                index_column: pl.int_range(0, len(values), eager=True),
                column: pl.Series(column, values, dtype=pl.String),
            }
        )
        return self._finalize_child_frame(frame, table, run_id, episode_id, robot_id, **(extra_columns or {}))

    def _edge_case_fact_frame(self, *, run_id: str, episode_id: int, robot_id: str, metrics: Mapping[str, Any]) -> pl.DataFrame:
        rows = [{"kind": kind, "ordinal": ordinal, "element": element} for kind, key in _EDGE_CASE_LIST_FIELDS if isinstance(elements := metrics.get(key), list) for ordinal, element in enumerate(elements)]
        frame = (
            pl.DataFrame(
                rows,
                schema={"kind": pl.String, "ordinal": pl.Int64, "element": pl.String},
            )
            if rows
            else None
        )
        return self._finalize_child_frame(
            frame,
            FactTable.EDGE_CASES,
            run_id,
            episode_id,
            robot_id,
            case_idx=pl.lit(0, dtype=pl.Int64),
        )

    def _formation_fact_frame(self, *, run_id: str, episode_id: int, robot_id: str, metrics: Mapping[str, Any]) -> pl.DataFrame:
        rows = [{"role": role, "element": element} for role, key in _FORMATION_LIST_FIELDS if isinstance(elements := metrics.get(key), list) for element in elements]
        frame = pl.DataFrame(rows, schema={"role": pl.String, "element": pl.String}) if rows else None
        return self._finalize_child_frame(
            frame,
            FactTable.FORMATIONS,
            run_id,
            episode_id,
            robot_id,
            formation_idx=pl.lit(0, dtype=pl.Int64),
        )

    def _phase_outcome_fact_frame(self, *, run_id: str, episode_id: int, robot_id: str, metrics: Mapping[str, Any]) -> pl.DataFrame:
        return self._list_fact_frame(
            FactTable.PHASE_OUTCOMES,
            "outcome",
            "phase_idx",
            metrics.get("phase_outcomes"),
            run_id=run_id,
            episode_id=episode_id,
            robot_id=robot_id,
            extra_columns={"met": pl.col("outcome") == "met"},
        )

    def _judge_disagreement_fact_frame(self, *, run_id: str, episode_id: int, robot_id: str, metrics: Mapping[str, Any]) -> pl.DataFrame:
        return self._list_fact_frame(
            FactTable.JUDGE_DISAGREEMENTS,
            "disagreement",
            "disagreement_idx",
            metrics.get("judge_disagreements"),
            run_id=run_id,
            episode_id=episode_id,
            robot_id=robot_id,
        )

    def _mechanism_state_fact_frame(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        states = metrics.get("mechanism_states")
        if not isinstance(states, list) or not states:
            return _empty_fact(FactTable.MECHANISM_STATES)
        times = episode.data.get_column("t_ns").to_numpy().astype(np.int64) if _has_rows(episode.data) else np.array([], dtype=np.int64)
        frames: list[pl.DataFrame] = []
        for state in states:
            if not isinstance(state, Mapping) or not isinstance(progress := state.get("progress"), list) or not progress:
                continue
            n = len(progress)
            locked = state.get("locked") if isinstance(state.get("locked"), list) else []
            frame = (
                pl.DataFrame(
                    {
                        "t_ns": pl.Series(
                            "t_ns",
                            times[:n] if len(times) >= n else [None] * n,
                            dtype=pl.Int64,
                        ),
                        "progress": pl.Series("progress", progress).cast(pl.Float64).fill_nan(None),
                        "locked": pl.Series("locked", locked[:n] if locked else [None] * n).cast(pl.Boolean),
                    }
                )
                .filter((pl.int_range(0, pl.len()) == 0) | pl.col("progress").ne_missing(pl.col("progress").shift(1)) | pl.col("locked").ne_missing(pl.col("locked").shift(1)))
                .with_columns(pl.col(field).cast(pl.Float64) for field in _MECHANISM_FIELDS)
            )
            frames.append(
                frame.unpivot(
                    on=list(_MECHANISM_FIELDS),
                    index="t_ns",
                    variable_name="field",
                    value_name="value",
                )
                .sort("t_ns", "field", nulls_last=True)
                .with_columns(
                    entity=pl.lit(str(state.get("entity") or ""), dtype=pl.String),
                    level=pl.lit(str(state.get("level") or ""), dtype=pl.String),
                    kind=pl.lit(str(state.get("kind") or ""), dtype=pl.String),
                )
            )
        return self._finalize_child_frame(
            pl.concat(frames, how="vertical") if frames else None,
            FactTable.MECHANISM_STATES,
            run_id,
            episode_id,
            robot_id,
        )

    def _collision_fact_frame(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str,
        episode_id: int,
        robot_id: str,
        metrics: Mapping[str, Any],
    ) -> pl.DataFrame:
        obstacles = metrics.get("collision_obstacles")
        if not isinstance(obstacles, list) or not obstacles:
            return _empty_fact(FactTable.COLLISIONS)
        ids = pl.DataFrame({"obstacle_id": pl.Series("obstacle_id", obstacles, dtype=pl.String)}).with_row_index("collision_idx")
        contacts = _table(episode, ROBOT_COLLISION_CONTACTS_TOPIC)
        if contacts is None or not {"obstacle_id", "t_ns"} <= set(contacts.columns):
            firsts = ids.with_columns(
                obstacle_kind=pl.lit(None, dtype=pl.String),
                t_ns=pl.lit(None, dtype=pl.Int64),
                x_m=pl.lit(None, dtype=pl.Float64),
                y_m=pl.lit(None, dtype=pl.Float64),
            )
        else:
            firsts = ids.join(
                contacts.filter(pl.col("obstacle_id").is_not_null() & (pl.col("obstacle_id") != ""))
                .sort("t_ns")
                .group_by("obstacle_id")
                .agg(
                    obstacle_kind=pl.col("kind").first(),
                    t_ns=pl.col("t_ns").first().cast(pl.Int64),
                    x_m=pl.col("obstacle_x").first().cast(pl.Float64),
                    y_m=pl.col("obstacle_y").first().cast(pl.Float64),
                ),
                on="obstacle_id",
                how="left",
            ).sort("collision_idx")
        return self._finalize_child_frame(firsts, FactTable.COLLISIONS, run_id, episode_id, robot_id)

    @staticmethod
    def _encounter_id(episode_id: int, robot_token: str, encounter_idx: int) -> str:
        return f"enc-{episode_id:03d}-{robot_token}-{encounter_idx:03d}"

    @staticmethod
    def _contact_id(episode_id: int, robot_token: str, contact_idx: int) -> str:
        return f"con-{episode_id:03d}-{robot_token}-{contact_idx:03d}"
