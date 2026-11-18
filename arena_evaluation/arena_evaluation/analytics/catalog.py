"""Benchmark OLAP Catalog (Layer 3).

Registers the Star Schema base tables (``dim_*``, ``fact_*``, ``series_*``, and
recorded telemetry) from the storage registry and exposes canonical analytical
views as pure Polars ``LazyFrame`` query plans.

Ranking rule of ``v_planner_scorecard``: ``episodes_count`` counts every episode
in the group, ``valid_episodes`` counts those whose ``is_valid_for_ranking`` is TRUE,
and ``excluded_episodes`` is the remainder. Success rates and metric means are
computed over ``valid_episodes`` only (NULL when a group has no valid episode);
encounter and contact totals cover every episode in the group.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Mapping
from typing import Final

import polars as pl

from arena_evaluation.analytics.evidence import (
    EVIDENCE_INDEX_SCHEMA,
    compute_frame_hash,
    compute_snapshot_id,
)
from arena_evaluation.storage.registry import (
    FACT_TABLES,
    NullReason,
    dimensions,
    series,
    table as registry_table,
)
from arena_evaluation.storage.registry.analysis import EvidenceIndex
from arena_evaluation.storage.star_schema import (
    scan_series_tables,
    scan_star_schema,
    scan_telemetry_tables,
)

__all__ = [
    "CANONICAL_VIEW_NAMES",
    "SERIES_TABLE_PREFIX",
    "BenchmarkCatalog",
]

SERIES_TABLE_PREFIX: Final[str] = registry_table.SERIES_PREFIX

CANONICAL_VIEW_NAMES: Final[tuple[str, ...]] = (
    "v_episode_summary",
    "v_planner_scorecard",
    "v_contacts",
    "v_contact_summary",
    "v_encounter_summary",
    "v_pedestrian_summary",
)

_JOIN_KEYS: Final[tuple[str, ...]] = dimensions.DimEpisodes.key

_ENCOUNTER_MEAN_METRICS: Final[tuple[str, ...]] = (
    "duration_s",
    "min_clearance_m",
    "detour_time_s",
    "robot_path_deviation_m",
    "energy_total_wh",
    "energy_surge_ratio",
    "probst_R_robot",
    "probst_R_actor2",
    "probst_R_time",
    "probst_E_robot",
    "probst_E_actor2",
    "sound_exposure_dose_pa2s",
)

_PEDESTRIAN_MEAN_METRICS: Final[tuple[str, ...]] = (
    "ped_max_exposure_dba",
    "ped_leq_exposure_dba",
    "ped_max_startle_factor",
    "ped_path_deflection_m",
    "ped_velocity_delay_ratio",
    "total_distance_m",
    "mean_speed_mps",
)

_STAR_SCHEMAS: Final[dict[str, pl.Schema]] = {
    **{name: tbl.polars_schema() for name, tbl in dimensions.DIMENSION_TABLES.items()},
    **{f"fact_{tbl.table_name}": tbl.polars_schema() for tbl in FACT_TABLES.values()},
}

_PROTECTED_BASE_NAMES: Final[frozenset[str]] = frozenset(
    {
        *_STAR_SCHEMAS.keys(),
        *(tbl.table_name for tbl in series.SERIES_TABLES.values()),
        *(
            tbl.table_name
            for tbl in registry_table.registered_tables()
            if tbl.table_kind is registry_table.TableKind.TELEMETRY
        ),
        "evidence_index",
        *CANONICAL_VIEW_NAMES,
    }
)


def _to_lazy(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def _stream_total(column: str, alias: str) -> pl.Expr:
    """Sum a per-episode stream count, NULL when no episode of the group recorded that stream.

    A sum of zeros and a stream nobody recorded look identical; the NULL keeps the second one visible and
    the companion ``_stream_total_reason`` names the missing stream.
    """
    return (
        pl.when(pl.col(column).is_null().sum() == pl.len())
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col(column).fill_null(0).sum().cast(pl.Int64))
        .alias(alias)
    )


def _stream_total_reason(presence_column: str, token: str, topic: str) -> pl.Expr:
    """The reason a stream total is NULL: the topic was never recorded for the group's episodes."""
    return (
        pl.when(pl.col(presence_column).is_null().sum() == pl.len())
        .then(pl.lit(f"{token}:{NullReason.TOPIC_MISSING.value}"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(f"total_{topic}_null_reason")
    )


def _crowd_condition_view(dim_peds: pl.LazyFrame) -> pl.LazyFrame:
    """Derive per-episode ``crowd_condition`` and its ``null_reason`` from ``dim_pedestrians``."""
    non_null = pl.col("robot_awareness").is_not_null().sum()
    min_aw = pl.col("robot_awareness").min()
    max_aw = pl.col("robot_awareness").max()
    condition = (
        pl.when(non_null == 0)
        .then(pl.lit(None, dtype=pl.String))
        .when((min_aw == 0.0) & (max_aw == 0.0))
        .then(pl.lit("blind"))
        .when((min_aw == 1.0) & (max_aw == 1.0))
        .then(pl.lit("visible"))
        .otherwise(pl.lit("mixed"))
        .alias("crowd_condition")
    )
    reason = (
        pl.when(non_null == 0)
        .then(pl.lit(str(NullReason.NOT_RECORDED)))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("crowd_condition_null_reason")
    )
    return dim_peds.group_by("run_id", "episode_id", "env_id").agg(condition, reason)


class BenchmarkCatalog:
    """Lazy Polars catalog of Star Schema tables, analytical views, and telemetry tables."""

    def __init__(
        self,
        source: pathlib.Path | Mapping[str, pl.LazyFrame | pl.DataFrame] | None = None,
        *,
        variant: str | None = None,
    ) -> None:
        self._tables: dict[str, pl.LazyFrame] = {}
        self._source_dir: pathlib.Path | None = (
            source if isinstance(source, pathlib.Path) else None
        )
        if variant not in (None, "sensitivity"):
            raise ValueError(
                f"Unsupported catalog variant {variant!r}; expected None or 'sensitivity'"
            )
        self._variant = variant
        self._snapshot_id: str | None = None
        self._load_source(source)
        self._register_canonical_views()
        self._register_series_views()

    @property
    def variant(self) -> str | None:
        """Active catalog variant (``None`` for published or ``'sensitivity'``)."""
        return self._variant

    @property
    def is_sensitivity(self) -> bool:
        """True when opened in in-memory sensitivity mode."""
        return self._variant == "sensitivity"

    @property
    def snapshot_id(self) -> str:
        """Lazily computed, cached dataset and registry snapshot identifier."""
        if self._snapshot_id is not None:
            return self._snapshot_id
        if self._source_dir is not None:
            from arena_evaluation.analytics.merge import load_merge_manifest

            merge_manifest = load_merge_manifest(self._source_dir)
            if merge_manifest is not None:
                hasher = hashlib.sha256()
                for src in merge_manifest.sources:
                    hasher.update(src.run_id.encode())
                    hasher.update(
                        compute_snapshot_id(pathlib.Path(src.run_dir)).encode()
                    )
                    for k, v in sorted(src.labels.items()):
                        hasher.update(f"{k}={v}".encode())
                self._snapshot_id = f"snap-{hasher.hexdigest()[:12]}"
                return self._snapshot_id
            self._snapshot_id = compute_snapshot_id(self._source_dir)
            return self._snapshot_id
        hasher = hashlib.sha256()
        for name in sorted(_STAR_SCHEMAS):
            hasher.update(name.encode())
            hasher.update(compute_frame_hash(self._tables[name].collect()).encode())
        self._snapshot_id = f"snap-{hasher.hexdigest()[:12]}"
        return self._snapshot_id

    def _load_source(
        self,
        source: pathlib.Path | Mapping[str, pl.LazyFrame | pl.DataFrame] | None,
    ) -> None:
        if isinstance(source, pathlib.Path):
            self._load_from_directory(source)
        elif isinstance(source, Mapping):
            self._load_from_mapping(source)
        else:
            self._load_empty_star_schema()

    def _load_from_directory(self, source_dir: pathlib.Path) -> None:
        """Scan Star Schema tables, series tables, and telemetry tables from ``source_dir``."""
        from arena_evaluation.analytics.derived import scan_derived_tables
        from arena_evaluation.analytics.merge import load_merge_manifest

        merge_manifest = load_merge_manifest(source_dir)
        if merge_manifest is not None:
            self._load_merged_directory(source_dir, merge_manifest)
            self._tables.update(scan_derived_tables(source_dir))
            return
        self._tables.update(scan_star_schema(source_dir))
        self._tables.update(scan_series_tables(source_dir))
        self._tables.update(scan_telemetry_tables(source_dir))
        self._tables.update(scan_derived_tables(source_dir))
        self._register_evidence_index(source_dir)

    def _load_merged_directory(
        self,
        source_dir: pathlib.Path,
        merge_manifest: object,
    ) -> None:
        """Compose a lazy catalog across multiple source runs without copying episodes."""
        from arena_evaluation.analytics.merge import MergeManifest

        assert isinstance(merge_manifest, MergeManifest)
        all_label_keys = sorted(
            {k for src in merge_manifest.sources for k in src.labels}
        )
        per_table: dict[str, list[pl.LazyFrame]] = {}
        for src in merge_manifest.sources:
            src_path = pathlib.Path(src.run_dir)
            if not src_path.is_dir():
                raise FileNotFoundError(
                    f"Merged benchmark {source_dir.name!r} references source run {src.run_id!r} at "
                    f"{src_path}, which does not exist"
                )
            src_tables: dict[str, pl.LazyFrame] = {}
            src_tables.update(scan_star_schema(src_path))
            src_tables.update(scan_series_tables(src_path))
            src_tables.update(scan_telemetry_tables(src_path))
            if "dim_episodes" in src_tables:
                label_exprs = [
                    pl.lit(src.run_id, dtype=pl.String).alias("source_run_id"),
                    *(
                        pl.lit(src.labels.get(k), dtype=pl.String).alias(k)
                        for k in all_label_keys
                    ),
                ]
                src_tables["dim_episodes"] = src_tables["dim_episodes"].with_columns(
                    label_exprs
                )
            for name, lf in src_tables.items():
                # Every merged frame carries its source-run key, telemetry included: a telemetry row has
                # no run of its own and would otherwise be attributed to whichever run happened to be first.
                lf = lf.with_columns(
                    pl.lit(src.run_id, dtype=pl.String).alias("run_id")
                )
                per_table.setdefault(name, []).append(lf)
        for name, frames in per_table.items():
            self._tables[name] = (
                frames[0]
                if len(frames) == 1
                else pl.concat(frames, how="diagonal_relaxed")
            )
        self._register_evidence_index(source_dir)

    def _load_from_mapping(
        self, source: Mapping[str, pl.LazyFrame | pl.DataFrame]
    ) -> None:
        for name, schema in _STAR_SCHEMAS.items():
            raw = source.get(name)
            self._tables[name] = (
                _to_lazy(raw) if raw is not None else pl.LazyFrame(schema=schema)
            )
        for name, table in source.items():
            if name not in self._tables:
                self._tables[name] = _to_lazy(table)
        self._tables.setdefault(
            "evidence_index", pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)
        )

    def _register_evidence_index(self, benchmark_dir: pathlib.Path) -> None:
        index_path = benchmark_dir / EvidenceIndex.relative_path()
        self._tables["evidence_index"] = (
            pl.scan_parquet(index_path)
            if index_path.is_file()
            else pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)
        )

    def _load_empty_star_schema(self) -> None:
        for name, schema in _STAR_SCHEMAS.items():
            self._tables[name] = pl.LazyFrame(schema=schema)
        self._tables.setdefault(
            "evidence_index", pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)
        )

    @classmethod
    def open(
        cls,
        benchmark: pathlib.Path,
        *,
        variant: str | None = None,
    ) -> BenchmarkCatalog:
        """Open a benchmark directory as a lazy ``BenchmarkCatalog``."""
        return cls(benchmark, variant=variant)

    def metric_dictionary(self) -> pl.LazyFrame:
        """The published metric dictionary: metric, scope, fact table, dtype, unit, direction and description.

        ``dim_metrics`` is the one table every consumer reads metric metadata from, so the catalog offers one
        projection of it instead of each caller reshaping the table itself.
        """
        return self.get_table("dim_metrics").select(
            "metric",
            "scope",
            "fact_table",
            "dtype",
            "unit",
            "description",
            "direction",
        )

    def is_protected_table(self, name: str) -> bool:
        """Return whether ``name`` is a registered base table or canonical view."""
        return name in _PROTECTED_BASE_NAMES

    def register_table(self, name: str, frame: pl.LazyFrame | pl.DataFrame) -> None:
        """Register or replace a lazy table or derived view in the catalog."""
        if not name or not name.strip():
            raise ValueError("Table name must be a non-empty string.")
        self._tables[name] = _to_lazy(frame)
        if name in _STAR_SCHEMAS:
            self._register_canonical_views()
            self._register_series_views()

    def get_table(self, name: str) -> pl.LazyFrame:
        """Return the ``LazyFrame`` registered under ``name``."""
        if name not in self._tables:
            raise KeyError(
                f"Unknown table or view '{name}'. Available: {sorted(self._tables)}"
            )
        return self._tables[name]

    def table_names(self) -> tuple[str, ...]:
        """Return all registered base, view, and derived table names."""
        return tuple(self._tables.keys())

    def tables(self) -> dict[str, pl.LazyFrame]:
        """Return a shallow copy of all registered ``LazyFrame`` tables."""
        return dict(self._tables)

    def _register_series_views(self) -> None:
        """Expose every ``series_*`` table joined with episode identity columns."""
        series_names = [
            name for name in self._tables if name.startswith(SERIES_TABLE_PREFIX)
        ]
        if not series_names:
            return
        identity = self._tables["v_episode_summary"].select(
            [
                *_JOIN_KEYS,
                "local_planner",
                "planner",
                "inter_planner",
                "dispatcher",
                "robot_model",
                "stage",
                "world",
                "scenario",
                "crowd_condition",
                "crowd_condition_null_reason",
                "seed",
                "is_reference",
                "is_valid_for_ranking",
                "dim_episode_null_reason",
                "dim_robot_null_reason",
                "fact_episode_null_reason",
                "fact_robot_null_reason",
            ]
        )
        for name in series_names:
            self._tables[f"v_{name}"] = self._tables[name].join(
                identity, on=list(_JOIN_KEYS), how="left"
            )

    def _register_canonical_views(self) -> None:
        dim_ep = self._tables["dim_episodes"]
        dim_rob = self._tables["dim_robots"]
        dim_ped = self._tables["dim_pedestrians"]
        fact_ep = self._tables["fact_episodes"]
        fact_rob = self._tables["fact_robots"]
        fact_enc = self._tables["fact_encounters"]
        fact_con = self._tables["fact_contacts"]
        fact_ped = self._tables["fact_pedestrians"]

        dim_ep_cols = set(dim_ep.collect_schema().names())
        dim_rob_cols = [
            *_JOIN_KEYS,
            *(
                col
                for col in dim_rob.collect_schema().names()
                if col not in dim_ep_cols and col != "null_reason"
            ),
            pl.col("null_reason").alias("dim_robot_null_reason"),
        ]
        dim_rob_select = [
            pl.col(col) if isinstance(col, str) else col for col in dim_rob_cols
        ]
        dim_ep_select = [
            *(
                pl.col(col)
                for col in dim_ep.collect_schema().names()
                if col != "null_reason"
            ),
            pl.col("null_reason").alias("dim_episode_null_reason"),
        ]

        fact_ep_names = set(fact_ep.collect_schema().names())
        fact_ep_select = [
            *(
                pl.col(col)
                for col in fact_ep.collect_schema().names()
                if col != "null_reason"
            ),
            pl.col("null_reason").alias("fact_episode_null_reason"),
        ]
        fact_rob_select = [
            *_JOIN_KEYS,
            *(
                col
                for col in fact_rob.collect_schema().names()
                if col not in dim_ep_cols
                and col not in fact_ep_names
                and col != "null_reason"
            ),
        ]
        fact_rob_exprs = [
            *(pl.col(col) for col in fact_rob_select),
            pl.col("null_reason").alias("fact_robot_null_reason"),
        ]

        ped_cond = _crowd_condition_view(dim_ped)
        con_counts = fact_con.group_by(_JOIN_KEYS).agg(
            contact_rows=pl.len().cast(pl.Int64),
            pedestrian_contact_rows=(pl.col("collider_kind") == "pedestrian")
            .sum()
            .cast(pl.Int64),
            static_contact_rows=(pl.col("collider_kind").is_in(["wall", "static"]))
            .sum()
            .cast(pl.Int64),
            other_robot_contact_rows=(pl.col("collider_kind") == "other_robot")
            .sum()
            .cast(pl.Int64),
        )

        v_episode_summary = (
            dim_ep.select(dim_ep_select)
            .join(dim_rob.select(dim_rob_select), on=list(_JOIN_KEYS), how="left")
            .join(fact_ep.select(fact_ep_select), on=list(_JOIN_KEYS), how="left")
            .join(fact_rob.select(fact_rob_exprs), on=list(_JOIN_KEYS), how="left")
            .join(ped_cond, on=["run_id", "episode_id", "env_id"], how="left")
            .join(con_counts, on=list(_JOIN_KEYS), how="left")
            .with_columns(
                pl.when(pl.col("planner") == "")
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(pl.col("planner"))
                .alias("local_planner"),
                (
                    pl.col("invalid_reason").is_not_null()
                    & (pl.col("invalid_reason") == "")
                ).alias("is_valid_for_ranking"),
                pl.when(pl.col("success").is_null() | pl.col("contact_rows").is_null())
                .then(pl.lit(None, dtype=pl.Boolean))
                .otherwise(pl.col("success") & (pl.col("pedestrian_contact_rows") == 0))
                .alias("success_without_pedestrian_contact"),
                pl.when(pl.col("success").is_null())
                .then(pl.lit(f"success:{NullReason.NOT_RECORDED}"))
                .when(pl.col("contact_rows").is_null())
                .then(pl.lit(f"contact_count:{NullReason.TOPIC_MISSING}"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("success_without_pedestrian_contact_null_reason"),
                pl.when(pl.col("crowd_condition").is_null())
                .then(
                    pl.col("crowd_condition_null_reason").fill_null(
                        str(NullReason.NOT_RECORDED)
                    )
                )
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("crowd_condition_null_reason"),
                (
                    pl.col("collision_amount_wall") + pl.col("collision_amount_static")
                ).alias("collision_amount_wall_and_static"),
            )
            .drop(
                "contact_rows",
                "pedestrian_contact_rows",
                "static_contact_rows",
                "other_robot_contact_rows",
            )
        )
        self._tables["v_episode_summary"] = v_episode_summary

        enc_counts = fact_enc.group_by(_JOIN_KEYS).agg(
            encounter_count=pl.len().cast(pl.Int64)
        )

        # A merged catalog adds the source-run key and its labels to dim_episodes; they are part of the
        # identity of every aggregate, not a column the scorecard may drop (L3-060).
        merged_extras = tuple(
            name
            for name in dim_ep.collect_schema().names()
            if name not in dimensions.DimEpisodes.polars_schema()
        )
        scorecard_group_cols = (
            *merged_extras,
            "world",
            "scenario",
            "stage",
            "crowd_condition",
            "planner",
            "dispatcher",
            "robot_model",
        )
        valid_for_ranking = pl.col("is_valid_for_ranking").fill_null(False)

        self._tables["v_planner_scorecard"] = (
            v_episode_summary.join(enc_counts, on=list(_JOIN_KEYS), how="left")
            .join(con_counts, on=list(_JOIN_KEYS), how="left")
            .group_by(scorecard_group_cols)
            .agg(
                episodes_count=pl.len().cast(pl.Int64),
                valid_episodes=valid_for_ranking.sum().cast(pl.Int64),
                excluded_episodes=(pl.len() - valid_for_ranking.sum()).cast(pl.Int64),
                success_rate=pl.col("success")
                .cast(pl.Float64)
                .filter(valid_for_ranking)
                .mean(),
                success_without_pedestrian_contact_rate=(
                    pl.col("success_without_pedestrian_contact")
                    .cast(pl.Float64)
                    .filter(valid_for_ranking)
                    .mean()
                ),
                mean_time_to_goal=pl.col("time_to_goal")
                .filter(valid_for_ranking)
                .mean(),
                mean_path_efficiency=pl.col("path_efficiency")
                .filter(valid_for_ranking)
                .mean(),
                mean_energy_total_wh=pl.col("energy_total_wh")
                .filter(valid_for_ranking)
                .mean(),
                episodes_without_contacts=pl.col("contact_rows")
                .is_null()
                .sum()
                .cast(pl.Int64),
                total_encounters=_stream_total("encounter_count", "total_encounters"),
                total_contacts=_stream_total("contact_rows", "total_contacts"),
                pedestrian_contacts=_stream_total(
                    "pedestrian_contact_rows", "pedestrian_contacts"
                ),
                static_contacts=_stream_total("static_contact_rows", "static_contacts"),
                other_robot_contacts=_stream_total(
                    "other_robot_contact_rows", "other_robot_contacts"
                ),
                total_encounters_null_reason=_stream_total_reason(
                    "encounter_count", "encounter_count", "encounters"
                ),
                total_contacts_null_reason=_stream_total_reason(
                    "contact_rows", "contact_count", "contacts"
                ),
            )
            .sort(scorecard_group_cols)
        )

        treatment = v_episode_summary.select(
            [
                *_JOIN_KEYS,
                *merged_extras,
                "crowd_condition",
                "planner",
                "world",
                "scenario",
                "stage",
            ]
        )

        v_robot_close = pl.col("v_close_robot_mps").clip(lower_bound=0.0)
        v_target_close = pl.col("v_close_target_mps").clip(lower_bound=0.0)
        closing_denom = v_robot_close + v_target_close
        robot_closing_share = (
            pl.when(closing_denom > 0.0)
            .then(v_robot_close / closing_denom)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("robot_closing_share")
        )
        target_closing_share = (
            pl.when(closing_denom > 0.0)
            .then(1.0 - (v_robot_close / closing_denom))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("target_closing_share")
        )

        v_contacts = fact_con.with_columns(robot_closing_share, target_closing_share)
        self._tables["v_contacts"] = v_contacts

        contact_group_cols = (
            *merged_extras,
            "world",
            "scenario",
            "stage",
            "crowd_condition",
            "planner",
        )
        self._tables["v_contact_summary"] = (
            v_contacts.join(treatment, on=list(_JOIN_KEYS), how="left")
            .group_by(contact_group_cols)
            .agg(
                total_contacts=pl.len().cast(pl.Int64),
                mean_robot_closing_share=pl.col("robot_closing_share").mean(),
                mean_target_closing_share=pl.col("target_closing_share").mean(),
                mean_relative_impact_speed_mps=pl.col(
                    "relative_impact_speed_mps"
                ).mean(),
                mean_freeze_clearance_m=pl.col("freeze_clearance_m").mean(),
            )
            .sort(contact_group_cols)
        )

        had_contact = pl.col("contact_id").is_not_null() & (pl.col("contact_id") != "")
        encounter_group_cols = (
            *merged_extras,
            "world",
            "scenario",
            "stage",
            "crowd_condition",
            "planner",
            "approach_type",
        )
        self._tables["v_encounter_summary"] = (
            fact_enc.join(treatment, on=list(_JOIN_KEYS), how="left")
            .group_by(encounter_group_cols)
            .agg(
                encounter_count=pl.len().cast(pl.Int64),
                *(
                    pl.col(col).mean().alias(f"mean_{col}")
                    for col in _ENCOUNTER_MEAN_METRICS
                ),
                contact_encounters=had_contact.sum().cast(pl.Int64),
            )
            .sort(encounter_group_cols)
        )

        episode_identity = dim_ep.group_by("run_id", "episode_id").agg(
            world=pl.col("world").max(),
            scenario=pl.col("scenario").max(),
            stage=pl.col("stage").max(),
        )
        pedestrian_group_cols = (
            *merged_extras,
            "world",
            "scenario",
            "stage",
            "ped_policy",
        )
        self._tables["v_pedestrian_summary"] = (
            fact_ped.join(episode_identity, on=["run_id", "episode_id"], how="left")
            .group_by(pedestrian_group_cols)
            .agg(
                pedestrians=pl.len().cast(pl.Int64),
                episodes=pl.col("episode_id").n_unique().cast(pl.Int64),
                *(
                    pl.col(col).mean().alias(f"mean_{col}")
                    for col in _PEDESTRIAN_MEAN_METRICS
                ),
            )
            .sort(pedestrian_group_cols)
        )
