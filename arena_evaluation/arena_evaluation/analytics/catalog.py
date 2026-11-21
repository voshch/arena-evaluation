"""Benchmark OLAP Catalog (Layer 3)."""

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
    "robot_path_deviation_vs_planned_m",
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
        *_STAR_SCHEMAS,
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

#: Per-episode stream total and, for the two mandatory streams, its NULL-reason companion.
_STREAM_TOTALS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("encounter_count", "total_encounters", "encounter_count", "encounters"),
    ("contact_rows", "total_contacts", "contact_count", "contacts"),
    ("pedestrian_contact_rows", "pedestrian_contacts", "", ""),
    ("static_contact_rows", "static_contacts", "", ""),
    ("other_robot_contact_rows", "other_robot_contacts", "", ""),
)

_CONTACT_COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "contact_rows",
    "pedestrian_contact_rows",
    "static_contact_rows",
    "other_robot_contact_rows",
)

_SCORECARD_MEAN_METRICS: Final[tuple[str, ...]] = (
    "time_to_goal",
    "path_efficiency",
    "energy_total_wh",
)

_SERIES_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "local_planner",
    "planner",
    "planner_world",
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
)


def _to_lazy(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def _stream_total(
    column: str, alias: str, token: str, topic: str
) -> tuple[pl.Expr, pl.Expr | None]:
    """Sum a per-episode stream count, NULL when no episode of the group recorded that stream.

    The optional companion names the missing stream: a sum of zeros and a stream nobody recorded
    look identical. It is only emitted for the mandatory encounter and contact streams.
    """
    missing = pl.col(column).is_null().sum() == pl.len()
    total = (
        pl.when(missing)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col(column).fill_null(0).sum().cast(pl.Int64))
        .alias(alias)
    )
    if not token:
        return total, None
    reason = (
        pl.when(missing)
        .then(pl.lit(f"{token}:{NullReason.TOPIC_MISSING.value}"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(f"total_{topic}_null_reason")
    )
    return total, reason


def _crowd_condition_view(dim_peds: pl.LazyFrame) -> pl.LazyFrame:
    """Derive per-episode ``crowd_condition`` and its ``null_reason`` from ``dim_pedestrians``."""
    non_null, min_aw, max_aw = (
        pl.col("robot_awareness").is_not_null().sum(),
        pl.col("robot_awareness").min(),
        pl.col("robot_awareness").max(),
    )
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


def _select_with_reason(
    frame: pl.LazyFrame,
    reason_alias: str,
    *,
    keys: tuple[str, ...] = (),
    exclude: set[str] | frozenset[str] = frozenset(),
) -> list[pl.Expr]:
    """Project every column but the raw ``null_reason``, which is renamed to ``reason_alias``."""
    names = (
        name
        for name in frame.collect_schema().names()
        if name not in exclude and name != "null_reason"
    )
    return [
        *(pl.col(name) for name in keys),
        *(pl.col(name) for name in names),
        pl.col("null_reason").alias(reason_alias),
    ]


def _merged_tables(
    source_dir: pathlib.Path, manifest: object
) -> dict[str, pl.LazyFrame]:
    """Compose the source runs of a merged benchmark lazily, without copying episodes."""
    from arena_evaluation.analytics.merge import MergeManifest

    assert isinstance(manifest, MergeManifest)
    label_keys = sorted({key for src in manifest.sources for key in src.labels})
    per_table: dict[str, list[pl.LazyFrame]] = {}
    for src in manifest.sources:
        src_path = pathlib.Path(src.run_dir)
        if not src_path.is_dir():
            raise FileNotFoundError(
                f"Merged benchmark {source_dir.name!r} references source run {src.run_id!r} at "
                f"{src_path}, which does not exist"
            )
        tables = {
            **scan_star_schema(src_path),
            **scan_series_tables(src_path),
            **scan_telemetry_tables(src_path),
        }
        if "dim_episodes" in tables:
            tables["dim_episodes"] = tables["dim_episodes"].with_columns(
                pl.lit(src.run_id, dtype=pl.String).alias("source_run_id"),
                *(
                    pl.lit(src.labels.get(k), dtype=pl.String).alias(k)
                    for k in label_keys
                ),
            )
        # Every merged frame carries its source-run key, telemetry included: a telemetry row has no
        # run of its own and would otherwise be attributed to whichever run happened to be first.
        for name, frame in tables.items():
            per_table.setdefault(name, []).append(
                frame.with_columns(pl.lit(src.run_id, dtype=pl.String).alias("run_id"))
            )
    return {
        name: frames[0]
        if len(frames) == 1
        else pl.concat(frames, how="diagonal_relaxed")
        for name, frames in per_table.items()
    }


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
        self._register_views()

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
                    for key, value in sorted(src.labels.items()):
                        hasher.update(f"{key}={value}".encode())
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
            self._load_directory(source)
        else:
            self._register_star_tables(source if isinstance(source, Mapping) else None)

    def _register_star_tables(
        self,
        source: Mapping[str, pl.LazyFrame | pl.DataFrame] | None,
    ) -> None:
        """Populate the empty Star Schema, overlaying any frame a mapping source provides."""
        for name, schema in _STAR_SCHEMAS.items():
            raw = None if source is None else source.get(name)
            self._tables[name] = (
                _to_lazy(raw) if raw is not None else pl.LazyFrame(schema=schema)
            )
        if source is not None:
            self._tables.update(
                {
                    name: _to_lazy(frame)
                    for name, frame in source.items()
                    if name not in self._tables
                }
            )
        self._tables.setdefault(
            "evidence_index", pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)
        )

    def _load_directory(self, source_dir: pathlib.Path) -> None:
        """Scan Star Schema, series, telemetry, and derived tables from ``source_dir``."""
        from arena_evaluation.analytics.derived import scan_derived_tables
        from arena_evaluation.analytics.merge import load_merge_manifest

        merge_manifest = load_merge_manifest(source_dir)
        if merge_manifest is None:
            for scan in (scan_star_schema, scan_series_tables, scan_telemetry_tables):
                self._tables.update(scan(source_dir))
            self._tables.update(scan_derived_tables(source_dir))
        else:
            self._tables.update(_merged_tables(source_dir, merge_manifest))
        index_path = source_dir / EvidenceIndex.relative_path()
        self._tables["evidence_index"] = (
            pl.scan_parquet(index_path)
            if index_path.is_file()
            else pl.LazyFrame(schema=EVIDENCE_INDEX_SCHEMA)
        )
        if merge_manifest is not None:
            self._tables.update(scan_derived_tables(source_dir))

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
            "metric", "scope", "fact_table", "dtype", "unit", "description", "direction"
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
            self._register_views()

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

    def _register_views(self) -> None:
        self._register_canonical_views()
        self._register_series_views()

    def _register_series_views(self) -> None:
        """Expose every ``series_*`` table joined with episode identity columns."""
        series_names = [
            name for name in self._tables if name.startswith(SERIES_TABLE_PREFIX)
        ]
        if not series_names:
            return
        identity = self._tables["v_episode_summary"].select(
            *_JOIN_KEYS, *_SERIES_IDENTITY_COLUMNS
        )
        for name in series_names:
            self._tables[f"v_{name}"] = self._tables[name].join(
                identity, on=list(_JOIN_KEYS), how="left"
            )

    def _register_canonical_views(self) -> None:
        tables = self._tables
        dim_ep = tables["dim_episodes"]
        dim_rob = tables["dim_robots"]
        dim_ped = tables["dim_pedestrians"]
        fact_ep = tables["fact_episodes"]
        fact_rob = tables["fact_robots"]
        fact_enc = tables["fact_encounters"]
        fact_con = tables["fact_contacts"]
        fact_ped = tables["fact_pedestrians"]

        dim_ep_cols = set(dim_ep.collect_schema().names())
        dim_ep_select = _select_with_reason(dim_ep, "dim_episode_null_reason")
        dim_rob_select = _select_with_reason(
            dim_rob, "dim_robot_null_reason", keys=_JOIN_KEYS, exclude=dim_ep_cols
        )
        fact_ep_select = _select_with_reason(fact_ep, "fact_episode_null_reason")
        fact_rob_select = _select_with_reason(
            fact_rob,
            "fact_robot_null_reason",
            keys=_JOIN_KEYS,
            exclude=dim_ep_cols | set(fact_ep.collect_schema().names()),
        )

        ped_cond = _crowd_condition_view(dim_ped)
        con_counts = fact_con.group_by(_JOIN_KEYS).agg(
            contact_rows=pl.len().cast(pl.Int64),
            pedestrian_contact_rows=(pl.col("collider_kind") == "pedestrian")
            .sum()
            .cast(pl.Int64),
            static_contact_rows=pl.col("collider_kind")
            .is_in(["wall", "static"])
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
            .join(fact_rob.select(fact_rob_select), on=list(_JOIN_KEYS), how="left")
            .join(ped_cond, on=["run_id", "episode_id", "env_id"], how="left")
            .join(con_counts, on=list(_JOIN_KEYS), how="left")
            .with_columns(
                pl.when(
                    pl.col("local_planner").is_not_null()
                    & (pl.col("local_planner") != "")
                )
                .then(pl.col("local_planner"))
                .when(pl.col("planner") != "")
                .then(pl.col("planner"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("local_planner"),
            )
            .with_columns(
                pl.concat_str(
                    [pl.col("local_planner"), pl.col("world")], separator=" · "
                ).alias("planner_world"),
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
            .with_columns(
                pl.when(
                    pl.col("is_reference").fill_null(False)
                    & pl.col("success").fill_null(False)
                    & (pl.col("time_to_goal") > 0.0)
                )
                .then(pl.col("time_to_goal"))
                .median()
                .over(["planner"])
                .alias("_ref_time_to_goal"),
                pl.when(
                    pl.col("is_reference").fill_null(False)
                    & pl.col("success").fill_null(False)
                    & (pl.col("energy_total_wh") > 0.0)
                )
                .then(pl.col("energy_total_wh"))
                .median()
                .over(["planner"])
                .alias("_ref_energy_total_wh"),
            )
            .with_columns(
                pl.when(
                    ~pl.col("is_reference").fill_null(False)
                    & pl.col("success").fill_null(False)
                    & (pl.col("time_to_goal") > 0.0)
                    & pl.col("_ref_time_to_goal").is_not_null()
                    & (pl.col("_ref_time_to_goal") > 0.0)
                )
                .then(pl.col("_ref_time_to_goal") / pl.col("time_to_goal"))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("relative_throughput"),
                pl.when(pl.col("_ref_time_to_goal").is_null())
                .then(pl.lit(NullReason.REFERENCE_MISSING.value))
                .when(pl.col("is_reference").fill_null(False))
                .then(pl.lit("is_reference"))
                .when(~pl.col("success").fill_null(False))
                .then(pl.lit("not_successful"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("relative_throughput_null_reason"),
                pl.when(
                    ~pl.col("is_reference").fill_null(False)
                    & pl.col("success").fill_null(False)
                    & (pl.col("energy_total_wh") > 0.0)
                    & pl.col("_ref_energy_total_wh").is_not_null()
                    & (pl.col("_ref_energy_total_wh") > 0.0)
                )
                .then(
                    (pl.col("energy_total_wh") - pl.col("_ref_energy_total_wh"))
                    / pl.col("_ref_energy_total_wh")
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("relative_social_compliance_cost"),
                pl.when(pl.col("_ref_energy_total_wh").is_null())
                .then(pl.lit(NullReason.REFERENCE_MISSING.value))
                .when(pl.col("is_reference").fill_null(False))
                .then(pl.lit("is_reference"))
                .when(~pl.col("success").fill_null(False))
                .then(pl.lit("not_successful"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("relative_social_compliance_cost_null_reason"),
            )
            .drop(*_CONTACT_COUNT_COLUMNS, "_ref_time_to_goal", "_ref_energy_total_wh")
        )
        tables["v_episode_summary"] = v_episode_summary

        enc_counts = fact_enc.group_by(_JOIN_KEYS).agg(
            encounter_count=pl.len().cast(pl.Int64)
        )
        merged_extras = tuple(
            n
            for n in dim_ep.collect_schema().names()
            if n not in dimensions.DimEpisodes.polars_schema()
        )
        # A merged catalog adds the source-run key and its labels to dim_episodes; they are part of
        # the identity of every aggregate, not a column the scorecard may drop (L3-060).
        common_group_cols = (
            *merged_extras,
            "world",
            "scenario",
            "stage",
            "crowd_condition",
            "planner",
            "local_planner",
            "inter_planner",
        )
        scorecard_group_cols = (*common_group_cols, "dispatcher", "robot_model")
        valid_for_ranking = pl.col("is_valid_for_ranking").fill_null(False)
        stream_totals = [_stream_total(*spec) for spec in _STREAM_TOTALS]
        tables["v_planner_scorecard"] = (
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
                *(
                    pl.col(metric)
                    .filter(valid_for_ranking)
                    .mean()
                    .alias(f"mean_{metric}")
                    for metric in _SCORECARD_MEAN_METRICS
                ),
                episodes_without_contacts=pl.col("contact_rows")
                .is_null()
                .sum()
                .cast(pl.Int64),
                *(total for total, _ in stream_totals),
                *(reason for _, reason in stream_totals if reason is not None),
            )
            .sort(scorecard_group_cols)
        )

        treatment = v_episode_summary.select(
            *_JOIN_KEYS,
            *merged_extras,
            "crowd_condition",
            "planner",
            "local_planner",
            "inter_planner",
            "world",
            "scenario",
            "stage",
        )

        robot_close = pl.col("v_close_robot_mps").clip(lower_bound=0.0)
        target_close = pl.col("v_close_target_mps").clip(lower_bound=0.0)
        closing_denom = robot_close + target_close
        closing_share = pl.when(closing_denom > 0.0).then(robot_close / closing_denom)
        v_contacts = fact_con.with_columns(
            closing_share.alias("robot_closing_share"),
            (1.0 - closing_share).alias("target_closing_share"),
        )
        tables["v_contacts"] = v_contacts

        tables["v_contact_summary"] = (
            v_contacts.join(treatment, on=list(_JOIN_KEYS), how="left")
            .group_by(common_group_cols)
            .agg(
                total_contacts=pl.len().cast(pl.Int64),
                mean_robot_closing_share=pl.col("robot_closing_share").mean(),
                mean_target_closing_share=pl.col("target_closing_share").mean(),
                mean_relative_impact_speed_mps=pl.col(
                    "relative_impact_speed_mps"
                ).mean(),
                mean_freeze_clearance_m=pl.col("freeze_clearance_m").mean(),
            )
            .sort(common_group_cols)
        )

        had_contact = pl.col("contact_id").is_not_null() & (pl.col("contact_id") != "")
        encounter_group_cols = (*common_group_cols, "approach_type")
        tables["v_encounter_summary"] = (
            fact_enc.join(treatment, on=list(_JOIN_KEYS), how="left")
            .group_by(encounter_group_cols)
            .agg(
                encounter_count=pl.len().cast(pl.Int64),
                *(
                    pl.col(metric).mean().alias(f"mean_{metric}")
                    for metric in _ENCOUNTER_MEAN_METRICS
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
        tables["v_pedestrian_summary"] = (
            fact_ped.join(episode_identity, on=["run_id", "episode_id"], how="left")
            .group_by(pedestrian_group_cols)
            .agg(
                pedestrians=pl.len().cast(pl.Int64),
                episodes=pl.col("episode_id").n_unique().cast(pl.Int64),
                *(
                    pl.col(metric).mean().alias(f"mean_{metric}")
                    for metric in _PEDESTRIAN_MEAN_METRICS
                ),
            )
            .sort(pedestrian_group_cols)
        )
