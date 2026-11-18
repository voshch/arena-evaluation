"""Query store, SQL guard, and shipped query declaration execution tests."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.queries import QueryCatalog, QueryDeclaration
from arena_evaluation.analytics.sql_engine import (
    PolarsSQLEngine,
    SQLGuardError,
    validate_readonly_sql,
)
from arena_evaluation.storage.registry import FACT_TABLES, FactTable, dimensions
from arena_evaluation.storage.registry.telemetry import PedestrianArenaPeds, RobotTracks
from arena_evaluation.storage.star_schema import write_benchmark_dimensions


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def _write_query_fixture_benchmark(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True)
    dims = {
        name: pl.DataFrame(schema=tbl.polars_schema())
        for name, tbl in dimensions.DIMENSION_TABLES.items()
    }
    dims["dim_episodes"] = _frame(
        [
            {
                "run_id": "run-q",
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "world": "hospital",
                "scenario": "corridor",
                "planner": "dwb",
                "dispatcher": "none",
                "robot_model": "jackal",
                "seed": 1,
                "is_reference": False,
            }
        ],
        dimensions.DimEpisodes.polars_schema(),
    )
    dims["dim_pedestrians"] = _frame(
        [
            {
                "run_id": "run-q",
                "episode_id": 0,
                "env_id": "env_0",
                "ped_id": "p0",
                "robot_awareness": 0.0,
            }
        ],
        dimensions.DimPedestrians.polars_schema(),
    )
    write_benchmark_dimensions(dims, root)

    ep_dir = root / "episodes" / "episode_000"
    (ep_dir / "metrics").mkdir(parents=True)
    (ep_dir / "topics").mkdir(parents=True)

    _frame(
        [
            {
                "run_id": "run-q",
                "episode_id": 0,
                "robot_id": "robot_0",
                "result": "SUCCESS",
                "invalid_reason": "",
            }
        ],
        FACT_TABLES[FactTable.EPISODES].polars_schema(),
    ).write_parquet(ep_dir / FACT_TABLES[FactTable.EPISODES].relative_path())

    _frame(
        [
            {
                "run_id": "run-q",
                "episode_id": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "robot_name": "robot_0",
                "success": True,
                "time_to_goal": 12.0,
                "path_efficiency": 0.85,
                "energy_total_wh": 3.0,
            }
        ],
        FACT_TABLES[FactTable.ROBOTS].polars_schema(),
    ).write_parquet(ep_dir / FACT_TABLES[FactTable.ROBOTS].relative_path())

    _frame(
        [
            {
                "run_id": "run-q",
                "episode_id": 0,
                "robot_id": "robot_0",
                "encounter_id": "e0",
                "actor_1": "robot_0",
                "actor_2": "p0",
                "actor_2_kind": "pedestrian",
                "t_start_ns": 0,
                "t_end_ns": 1_000_000_000,
                "contact_id": "c0",
                "approach_type": "head_on",
                "min_clearance_m": 0.8,
                "duration_s": 1.0,
                "detour_time_s": 0.5,
                "robot_path_deviation_m": 0.3,
                "energy_total_wh": 0.6,
                "energy_surge_ratio": 1.8,
                "probst_R_robot": 0.6,
                "probst_R_actor2": 0.4,
                "probst_E_robot": 0.7,
            }
        ],
        FACT_TABLES[FactTable.ENCOUNTERS].polars_schema(),
    ).write_parquet(ep_dir / FACT_TABLES[FactTable.ENCOUNTERS].relative_path())

    _frame(
        [
            {
                "run_id": "run-q",
                "episode_id": 0,
                "robot_id": "robot_0",
                "contact_id": "c0",
                "encounter_id": "e0",
                "collider_kind": "pedestrian",
                "obstacle_id": "p0",
                "t_onset_ns": 100,
                "t_end_ns": 200,
                "duration_s": 0.1,
                "v_close_robot_mps": 0.7,
                "v_close_target_mps": 0.3,
                "relative_impact_speed_mps": 1.0,
                "freeze_clearance_m": 0.25,
            }
        ],
        FACT_TABLES[FactTable.CONTACTS].polars_schema(),
    ).write_parquet(ep_dir / FACT_TABLES[FactTable.CONTACTS].relative_path())

    _frame(
        [
            {
                "run_id": "run-q",
                "episode_id": 0,
                "env_id": "env_0",
                "ped_id": "p0",
                "ped_name": "p0",
                "ped_policy": "social_force",
                "ped_path_deflection_m": 0.4,
            }
        ],
        FACT_TABLES[FactTable.PEDESTRIANS].polars_schema(),
    ).write_parquet(ep_dir / FACT_TABLES[FactTable.PEDESTRIANS].relative_path())

    _frame(
        [
            {
                "t_ns": 0,
                "env_id": "env_0",
                "ped_id": "p0",
                "ped_name": "p0",
                "x": 1.0,
                "y": 1.0,
                "yaw": 0.0,
                "speed_mps": 1.0,
            },
            {
                "t_ns": 1_000_000_000,
                "env_id": "env_0",
                "ped_id": "p0",
                "ped_name": "p0",
                "x": 1.5,
                "y": 1.0,
                "yaw": 0.0,
                "speed_mps": 1.0,
            },
        ],
        PedestrianArenaPeds.polars_schema(),
    ).write_parquet(ep_dir / PedestrianArenaPeds.relative_path())

    _frame(
        [
            {
                "t_ns": 0,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "vx": 0.5,
                "vy": 0.0,
                "speed_mps": 0.5,
            },
            {
                "t_ns": 1_000_000_000,
                "env_id": "env_0",
                "robot_id": "robot_0",
                "x": 0.5,
                "y": 0.0,
                "yaw": 0.0,
                "vx": 0.5,
                "vy": 0.0,
                "speed_mps": 0.5,
            },
        ],
        RobotTracks.polars_schema(),
    ).write_parquet(ep_dir / RobotTracks.relative_path())
    return root


def test_sql_guard_preserves_double_dash_inside_string_literal() -> None:
    cleaned = validate_readonly_sql("SELECT 'a--b' AS literal_val FROM dim_episodes")
    assert "'a--b'" in cleaned


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_parquet('/tmp/secret.parquet')",
        "SELECT * FROM \"read_parquet\"('/tmp/secret.parquet')",
        "SELECT * FROM \"READ_PARQUET\"('/tmp/secret.parquet')",
        "SELECT * FROM read_csv('/tmp/secret.csv')",
        "SELECT * FROM \"read_csv\"('/tmp/secret.csv')",
        "SELECT * FROM \"READ_CSV\"('/tmp/secret.csv')",
        "SELECT * FROM `read_csv`('/tmp/secret.csv')",
        "SELECT * FROM read_csv /* comment */ ('/tmp/secret.csv')",
        "SELECT * FROM read_csv\n('/tmp/secret.csv')",
        "SELECT * FROM read_ipc('/tmp/secret.ipc')",
        "SELECT * FROM \"read_ipc\"('/tmp/secret.ipc')",
        "SELECT * FROM \"READ_IPC\"('/tmp/secret.ipc')",
        "SELECT * FROM read_ndjson('/tmp/secret.ndjson')",
        "SELECT * FROM \"read_ndjson\"('/tmp/secret.ndjson')",
        "SELECT * FROM \"READ_NDJSON\"('/tmp/secret.ndjson')",
        "SELECT * FROM scan_parquet('/tmp/secret.parquet')",
        "SELECT * FROM \"scan_parquet\"('/tmp/secret.parquet')",
        "SELECT * FROM \"SCAN_PARQUET\"('/tmp/secret.parquet')",
    ],
)
def test_sql_guard_refuses_file_reading_table_functions(sql: str) -> None:
    with pytest.raises(SQLGuardError, match="External table function"):
        validate_readonly_sql(sql)


def test_sql_engine_allows_string_literal_containing_table_function_name() -> None:
    engine = PolarsSQLEngine(BenchmarkCatalog())
    res = engine.query("SELECT 'read_csv' AS x")
    assert res.item(0, "x") == "read_csv"


def test_sql_engine_refuses_save_as_onto_base_table_or_canonical_view() -> None:
    engine = PolarsSQLEngine(BenchmarkCatalog())
    with pytest.raises(SQLGuardError, match="protected"):
        engine.query("SELECT 1 AS x", save_as="dim_episodes")
    with pytest.raises(SQLGuardError, match="protected"):
        engine.query("SELECT 1 AS x", save_as="v_episode_summary")


def test_sql_engine_has_no_default_row_cap() -> None:
    many_rows = pl.DataFrame({"id": list(range(150))})
    catalog = BenchmarkCatalog({"many_rows": many_rows})
    engine = PolarsSQLEngine(catalog)
    result = engine.query("SELECT id FROM many_rows")
    assert result.height == 150
    limited = engine.query("SELECT id FROM many_rows", limit=10)
    assert limited.height == 10


def test_query_catalog_surfaces_malformed_yaml_and_validates_save(
    tmp_path: pathlib.Path,
) -> None:
    bad_root = tmp_path / "bad_root"
    bad_bundle = bad_root / "queries" / "broken"
    bad_bundle.mkdir(parents=True)
    (bad_bundle / "query.yaml").write_text("name: broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid query declaration"):
        QueryCatalog([bad_root])

    good_dir = tmp_path / "good_root"
    catalog = QueryCatalog([good_dir])
    with pytest.raises(SQLGuardError):
        catalog.save_query(
            name="bad_sql",
            description="x",
            sql="DROP TABLE dim_episodes",
            directory=good_dir,
        )

    decl = QueryDeclaration(
        name="typed",
        description="typed param",
        sql="SELECT :ep AS ep",
        parameters={"ep": {"type": "integer", "default": 0}},
    )
    with pytest.raises(TypeError, match="expects type 'integer'"):
        decl.render_sql({"ep": "not_an_int"})


def test_query_catalog_searches_run_dir_before_package(tmp_path: pathlib.Path) -> None:
    run_queries = tmp_path / "analysis" / "queries" / "robot_speed_distribution"
    run_queries.mkdir(parents=True)
    (run_queries / "query.yaml").write_text(
        "name: robot_speed_distribution\ndescription: override\nsql: SELECT 42 AS override_val\n",
        encoding="utf-8",
    )
    catalog = QueryCatalog(run_dir=tmp_path)
    assert "42 AS override_val" in catalog.get_query("robot_speed_distribution").sql


def test_all_shipped_queries_execute_on_fixture(tmp_path: pathlib.Path) -> None:
    bench = _write_query_fixture_benchmark(tmp_path / "bench")
    catalog = BenchmarkCatalog.open(bench)
    engine = PolarsSQLEngine(catalog)
    qcat = QueryCatalog()

    shipped = qcat.list_queries()
    assert len(shipped) >= 9
    assert not qcat.has_query("closer_threshold_sensitivity")

    for declaration in shipped:
        df = qcat.execute(declaration, engine)
        assert df.height >= 1, f"{declaration.name} returned 0 rows"


def test_non_reference_trajectory_queries_do_not_fan_out_per_robot(
    tmp_path: pathlib.Path,
) -> None:
    """A two-robot episode repeats neither a robot sample nor a pedestrian sample (L3-010, L3-011)."""
    bench = _write_query_fixture_benchmark(tmp_path / "bench")
    dim_episodes = pl.read_parquet(bench / "dim_episodes.parquet")
    frame = pl.DataFrame(
        [
            dim_episodes.row(0, named=True),
            {**dim_episodes.row(0, named=True), "robot_id": "robot_1"},
        ]
    )
    frame.write_parquet(bench / "dim_episodes.parquet")

    catalog = BenchmarkCatalog.open(bench)
    engine = PolarsSQLEngine(catalog)
    qcat = QueryCatalog()

    # One robot sample and two pedestrian samples exist; the second robot of the episode has none.
    robot_rows = qcat.execute("robot_tracks_non_ref", engine)
    assert robot_rows.height == 2
    assert robot_rows.get_column("robot_id").unique().to_list() == ["robot_0"]

    pedestrian_rows = qcat.execute("pedestrian_arena_peds_non_ref", engine)
    assert pedestrian_rows.height == 2
    assert set(pedestrian_rows.columns) == {
        "run_id",
        "episode_id",
        "ped_id",
        "world",
        "t_ns",
        "x",
        "y",
        "yaw",
    }

    deflection_rows = qcat.execute("pedestrian_deflection_by_inter_planner", engine)
    assert deflection_rows.height == 1

    # The pedestrian-scope exposure query is keyed on the episode, not on the robot (R2-10, L3-010).
    exposure_rows = qcat.execute("pedestrian_exposure_by_planner", engine)
    assert exposure_rows.height == 1
    assert set(exposure_rows.columns) == {
        "planner",
        "world",
        "stage",
        "crowd_condition",
        "ped_id",
        "ped_policy",
        "ped_max_exposure_dba",
        "ped_leq_exposure_dba",
        "ped_max_startle_factor",
        "ped_path_deflection_m",
    }


def test_snapshot_id_tracks_telemetry_bytes_and_the_evaluator_identity(
    tmp_path: pathlib.Path,
) -> None:
    """Series and telemetry inputs and the evaluator version are part of the snapshot (L3-003)."""
    from arena_evaluation.analytics.evidence import compute_snapshot_id

    bench = _write_query_fixture_benchmark(tmp_path / "bench")
    before = compute_snapshot_id(bench)
    assert before.startswith("snap-")

    tracks = next(bench.rglob("robot_tracks.parquet"))
    frame = pl.read_parquet(tracks)
    frame.with_columns(pl.col("x") + 1.0).write_parquet(tracks)
    assert compute_snapshot_id(bench) != before


def test_evidence_index_schema_is_flat_and_registered() -> None:
    from arena_evaluation.analytics.evidence import EVIDENCE_INDEX_SCHEMA
    from arena_evaluation.storage.registry.analysis import EvidenceIndex
    from arena_evaluation.storage.registry.table import TableKind

    assert EvidenceIndex.table_kind is TableKind.ANALYSIS
    assert EVIDENCE_INDEX_SCHEMA == EvidenceIndex.polars_schema()
    nested = (pl.List, pl.Struct, pl.Object, pl.Array)
    assert not any(isinstance(dt, nested) for dt in EVIDENCE_INDEX_SCHEMA.values())


def test_evidence_store_writes_under_analysis_is_order_and_purpose_insensitive_and_allows_empty(
    tmp_path: pathlib.Path,
) -> None:
    from arena_evaluation.analytics.evidence import EvidenceStore, compute_snapshot_id

    bench = _write_query_fixture_benchmark(tmp_path / "bench")
    snap = compute_snapshot_id(bench)
    assert snap.startswith("snap-")

    store = EvidenceStore(benchmark_dir=bench)
    assert store.evidence_dir == bench / "analysis" / "evidence"

    df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pl.DataFrame({"a": [2, 1], "b": ["y", "x"]})
    r1 = store.record(
        df1,
        kind="sql",
        operation="q1",
        purpose="first",
        snapshot_id=snap,
        sql_text="SELECT 1",
    )
    r2 = store.record(
        df2,
        kind="sql",
        operation="q1",
        purpose="second",
        snapshot_id=snap,
        sql_text="SELECT 1",
    )
    assert r1.evidence_id == r2.evidence_id

    empty_df = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
    r_empty = store.record(
        empty_df, kind="computation", operation="empty_op", snapshot_id=snap
    )
    assert r_empty.row_count == 0

    from arena_evaluation.storage.registry.analysis import EvidenceIndex

    assert store.index_path == bench / EvidenceIndex.relative_path()
    assert list((bench / "analysis").rglob("*.json")) == []
    assert not pathlib.Path(r1.artifact_path).is_absolute()

    reopened = EvidenceStore(benchmark_dir=bench)
    listed_ids = {r.evidence_id for r in reopened.list_records()}
    assert listed_ids == {r1.evidence_id, r_empty.evidence_id}
    assert reopened.get(r1.evidence_id).snapshot_id == snap
    assert reopened.get(r1.evidence_id).sql_text == "SELECT 1"
    assert reopened.read_frame(r1.evidence_id).height == 2
    assert reopened.read_frame(r_empty.evidence_id).height == 0

    with pytest.raises(ValueError, match="snapshot_id"):
        store.record(df1, kind="sql", operation="q1", snapshot_id="")


def test_sql_engine_allows_scan_column_names() -> None:
    sql = validate_readonly_sql("SELECT scan_max_m, scan_min_m FROM robot_scan_summary")
    assert sql == "SELECT scan_max_m, scan_min_m FROM robot_scan_summary"


def test_evidence_store_rejects_nested_frames(tmp_path: pathlib.Path) -> None:
    from arena_evaluation.analytics.evidence import EvidenceStore, compute_snapshot_id

    bench = _write_query_fixture_benchmark(tmp_path / "bench")
    store = EvidenceStore(benchmark_dir=bench)
    snap = compute_snapshot_id(bench)
    nested = pl.DataFrame({"ids": [[1, 2], [3]]})
    with pytest.raises(ValueError, match="flat frames"):
        store.record(nested, kind="sql", operation="array_agg_q", snapshot_id=snap)


def test_snapshot_id_determinism_invalidation_caching_and_frame_backed(
    tmp_path: pathlib.Path,
) -> None:
    from arena_evaluation.analytics import computations
    from arena_evaluation.storage.registry import FACT_TABLES, FactTable

    bench = _write_query_fixture_benchmark(tmp_path / "bench")
    c1 = BenchmarkCatalog.open(bench)
    c2 = BenchmarkCatalog.open(bench)
    cached_id = c1.snapshot_id
    assert cached_id == c2.snapshot_id
    assert cached_id.startswith("snap-")

    computations.run_computation("group_summary", c1, {"metrics": ["time_to_goal"]})
    computations.run_computation("distribution_bins", c1, {"metric": "time_to_goal"})
    assert c1.snapshot_id == cached_id

    fact_path = (
        bench
        / "episodes"
        / "episode_000"
        / FACT_TABLES[FactTable.EPISODES].relative_path()
    )
    df_ep = pl.read_parquet(fact_path).with_columns(pl.lit("TIMEOUT").alias("result"))
    df_ep.write_parquet(fact_path)
    assert c1.snapshot_id == cached_id
    assert BenchmarkCatalog.open(bench).snapshot_id != cached_id

    f1 = pl.DataFrame({"episode_id": [0], "success": [True]})
    f2 = pl.DataFrame({"episode_id": [0], "success": [False]})
    cat_a = BenchmarkCatalog({"dim_episodes": f1})
    cat_b = BenchmarkCatalog({"dim_episodes": f1})
    cat_c = BenchmarkCatalog({"dim_episodes": f2})
    assert cat_a.snapshot_id == cat_b.snapshot_id
    assert cat_a.snapshot_id != cat_c.snapshot_id


def test_evidence_store_read_frame_raises_when_artifact_missing(
    tmp_path: pathlib.Path,
) -> None:
    from arena_evaluation.analytics import EvidenceStore

    bench = _write_query_fixture_benchmark(tmp_path / "bench")
    store = EvidenceStore(bench)
    rec = store.record(
        pl.DataFrame({"x": [1, 2]}),
        kind="sql",
        operation="q",
        snapshot_id="snap-aaaaaaaaaaaa",
    )
    (bench / rec.artifact_path).unlink()
    with pytest.raises(KeyError, match="not found"):
        store.read_frame(rec.evidence_id)
