"""Unit tests for the WP5 analyst workflow MCP tools: interaction slices and findings.

Both modules read and write the run's declared Star Schema artefacts only: a slice is a
query over ``fact_*``/``series_*`` Parquet, and a finding is the ``findings.yaml`` document
next to ``notes.yaml``. The fixtures build those artefacts with the package's own writers
(:func:`write_star_schema`), so the tests exercise the published schema rather than a stub.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from collections.abc import Mapping
from collections.abc import Iterator
from typing import Any

import polars as pl
import pytest
import yaml
from arena_mcp.core import ToolError
from polars.testing import assert_frame_equal

from arena_evaluation.mcp.findings import (
    evaluation_finding_verify,
    FindingStatus,
    evaluation_finding_attach,
    evaluation_finding_create,
    evaluation_finding_status,
    evaluation_findings,
)
from arena_evaluation.mcp.slices import (
    EncounterRankingResponse,
    SliceResponse,
    evaluation_slice,
    evaluation_worst_encounters,
)
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES

DIM_EPISODES_SCHEMA = DIMENSION_TABLES["dim_episodes"].polars_schema()
DIM_ROBOTS_SCHEMA = DIMENSION_TABLES["dim_robots"].polars_schema()
DIM_PEDESTRIANS_SCHEMA = DIMENSION_TABLES["dim_pedestrians"].polars_schema()
DIM_WORLD_ENTITIES_SCHEMA = DIMENSION_TABLES["dim_world_entities"].polars_schema()
FACT_EPISODES_SCHEMA = FACT_TABLES["episodes"].polars_schema()
FACT_ROBOTS_SCHEMA = FACT_TABLES["robots"].polars_schema()
FACT_PEDESTRIANS_SCHEMA = FACT_TABLES["pedestrians"].polars_schema()
FACT_ENCOUNTERS_SCHEMA = FACT_TABLES["encounters"].polars_schema()
FACT_CONTACTS_SCHEMA = FACT_TABLES["contacts"].polars_schema()
FACT_COLLISIONS_SCHEMA = FACT_TABLES["collisions"].polars_schema()


def _write_tables(benchmark_dir: pathlib.Path, tables: dict[str, pl.DataFrame]) -> None:
    """Write dimension and per-episode fact tables in canonical Star Schema layout."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for name, table in DIMENSION_TABLES.items():
        frame = tables.get(name, pl.DataFrame(schema=table.polars_schema()))
        frame.write_parquet(benchmark_dir / table.relative_path())
    ep_ids: set[int] = set()
    for name in FACT_TABLES:
        fact_name = f"fact_{name}"
        if (
            fact_name in tables
            and tables[fact_name].height > 0
            and "episode_id" in tables[fact_name].columns
        ):
            ep_ids.update(int(x) for x in tables[fact_name]["episode_id"].to_list())
    if not ep_ids and "dim_episodes" in tables and tables["dim_episodes"].height > 0:
        ep_ids.update(int(x) for x in tables["dim_episodes"]["episode_id"].to_list())
    for ep_id in sorted(ep_ids):
        ep_dir = benchmark_dir / "episodes" / f"episode_{ep_id:03d}"
        (ep_dir / "metrics").mkdir(parents=True, exist_ok=True)
        for name, table in FACT_TABLES.items():
            fact_name = f"fact_{name}"
            frame = tables.get(fact_name, pl.DataFrame(schema=table.polars_schema()))
            if frame.height > 0 and "episode_id" in frame.columns:
                sub = frame.filter(pl.col("episode_id") == ep_id)
            else:
                sub = pl.DataFrame(schema=table.polars_schema())
            sub.write_parquet(ep_dir / table.relative_path())


RUN_ID = "run-wp5-slices"
EPISODE_ID = 0
ENCOUNTER_A = "enc-000-robot_0-000"
ENCOUNTER_B = "enc-000-robot_0-001"
CONTACT_ID = "con-000-robot_0-000"
PEDESTRIAN_NEAR = "Pedestrian_1"
PEDESTRIAN_FAR = "Pedestrian_2"
SECOND_NS = 1_000_000_000
FACT_SCHEMAS_BY_NAME: Mapping[str, pl.Schema] = {
    "fact_episodes": FACT_EPISODES_SCHEMA,
    "fact_encounters": FACT_ENCOUNTERS_SCHEMA,
    "fact_contacts": FACT_CONTACTS_SCHEMA,
    "fact_pedestrians": FACT_PEDESTRIANS_SCHEMA,
    "fact_robots": FACT_ROBOTS_SCHEMA,
    "fact_collisions": FACT_COLLISIONS_SCHEMA,
}
SERIES_EXPOSURE_SCHEMA: pl.Schema = pl.Schema(
    {
        "run_id": pl.String,
        "episode_id": pl.Int64,
        "robot_id": pl.String,
        "t_ns": pl.Int64,
        "ped_id": pl.String,
        "exposure_dba": pl.Float64,
    }
)


def _column_names(schema: pl.Schema | Mapping[str, pl.DataType]) -> list[str]:
    return list(schema) if isinstance(schema, Mapping) else schema.names()


def _row(
    schema: pl.Schema | Mapping[str, pl.DataType], **values: Any
) -> dict[str, Any]:
    """One table row holding every declared column, null unless a value is given."""
    return {name: values.get(name) for name in _column_names(schema)}


def _frame(schema: pl.Schema, rows: list[dict[str, Any]]) -> pl.DataFrame:
    """A schema-typed frame from partial rows."""
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


@contextlib.contextmanager
def _arena_data(tmp_path: pathlib.Path):
    prev = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path / "data")
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("ARENA_DATA_DIR", None)
        else:
            os.environ["ARENA_DATA_DIR"] = prev


def _star_tables() -> dict[str, pl.DataFrame]:
    """One run with one episode, two encounters, one attributed contact and two pedestrians."""
    dim_episodes = _frame(
        DIM_EPISODES_SCHEMA,
        [
            _row(
                DIM_EPISODES_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                env_id="env_0",
                robot_id="robot_0",
                benchmark_id=RUN_ID,
                suite_name="basic",
                contest_name="basic",
                stage="s1",
                world="map_empty",
                planner="dwb",
                inter_planner="",
                dispatcher="",
                crowd_condition="blind",
                crowd_size=2,
                seed=7,
                is_reference=False,
                duration_s=12.0,
                outcome_state=2,
                outcome_info="finished",
            )
        ],
    )
    dim_robots = _frame(
        DIM_ROBOTS_SCHEMA,
        [
            _row(
                DIM_ROBOTS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                env_id="env_0",
                robot_id="robot_0",
                robot_model="jackal",
                planner="dwb",
            )
        ],
    )
    dim_pedestrians = _frame(
        DIM_PEDESTRIANS_SCHEMA,
        [
            _row(
                DIM_PEDESTRIANS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                env_id="env_0",
                ped_id=PEDESTRIAN_NEAR,
                ped_name="alice",
                agent_type="pedestrian",
                ped_policy="sfm",
                is_blind=True,
            ),
            _row(
                DIM_PEDESTRIANS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                env_id="env_0",
                ped_id=PEDESTRIAN_FAR,
                ped_name="bob",
                agent_type="pedestrian",
                ped_policy="sfm",
                is_blind=True,
            ),
        ],
    )
    dim_world_entities = pl.DataFrame(schema=DIM_WORLD_ENTITIES_SCHEMA)
    fact_episodes = _frame(
        FACT_EPISODES_SCHEMA,
        [
            _row(
                FACT_EPISODES_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                robot_id="robot_0",
                result="GOAL_REACHED",
                success=True,
                path_length_m=28.0,
                null_reason="mar:no_samples;pfi:no_samples",
            )
        ],
    )
    fact_encounters = _frame(
        FACT_ENCOUNTERS_SCHEMA,
        [
            _row(
                FACT_ENCOUNTERS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                robot_id="robot_0",
                encounter_id=ENCOUNTER_A,
                actor_1="robot_0",
                actor_2=PEDESTRIAN_NEAR,
                actor_2_kind="pedestrian",
                t_start_ns=1 * SECOND_NS,
                t_end_ns=4 * SECOND_NS,
                duration_s=3.0,
                approach_type="crossing",
                min_clearance_m=0.4,
                contact_id=CONTACT_ID,
            ),
            _row(
                FACT_ENCOUNTERS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                robot_id="robot_0",
                encounter_id=ENCOUNTER_B,
                actor_1="robot_0",
                actor_2=PEDESTRIAN_FAR,
                actor_2_kind="pedestrian",
                t_start_ns=5 * SECOND_NS,
                t_end_ns=10 * SECOND_NS,
                duration_s=5.0,
                approach_type="head_on",
                min_clearance_m=0.9,
            ),
        ],
    )
    fact_contacts = _frame(
        FACT_CONTACTS_SCHEMA,
        [
            _row(
                FACT_CONTACTS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                robot_id="robot_0",
                contact_id=CONTACT_ID,
                encounter_id=ENCOUNTER_A,
                collider_kind="pedestrian",
                obstacle_id=PEDESTRIAN_NEAR,
                t_onset_ns=1_200_000_000,
                t_end_ns=1_500_000_000,
                duration_s=0.3,
                impact_speed_mps=0.8,
                robot_closing_share=1.0,
                null_reason="obstacle_speed_at_impact_mps:no_samples",
            )
        ],
    )
    fact_pedestrians = _frame(
        FACT_PEDESTRIANS_SCHEMA,
        [
            _row(
                FACT_PEDESTRIANS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                env_id="env_0",
                ped_id=PEDESTRIAN_NEAR,
                ped_name="alice",
                is_blind=True,
                round_trips_completed=1,
            ),
            _row(
                FACT_PEDESTRIANS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                env_id="env_0",
                ped_id=PEDESTRIAN_FAR,
                ped_name="bob",
                is_blind=True,
                round_trips_completed=0,
                null_reason="cumulative_sound_dose_pa2s:no_samples",
            ),
        ],
    )
    fact_robots = _frame(
        FACT_ROBOTS_SCHEMA,
        [
            _row(
                FACT_ROBOTS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                env_id="env_0",
                robot_id="robot_0",
                robot_name="jackal",
                path_length_m=28.0,
            )
        ],
    )
    fact_collisions = _frame(
        FACT_COLLISIONS_SCHEMA,
        [
            _row(
                FACT_COLLISIONS_SCHEMA,
                run_id=RUN_ID,
                episode_id=EPISODE_ID,
                robot_id="robot_0",
                collision_idx=0,
                obstacle_kind="pedestrian",
                t_ns=1_300_000_000,
                x_m=1.0,
                y_m=2.0,
            )
        ],
    )
    return dict(
        dim_episodes=dim_episodes,
        dim_robots=dim_robots,
        dim_pedestrians=dim_pedestrians,
        dim_world_entities=dim_world_entities,
        fact_episodes=fact_episodes,
        fact_encounters=fact_encounters,
        fact_contacts=fact_contacts,
        fact_pedestrians=fact_pedestrians,
        fact_robots=fact_robots,
        fact_collisions=fact_collisions,
    )


def _write_series(run_dir: pathlib.Path) -> None:
    """The acoustic-exposure 1NF series, one row inside the contact window and one outside the episode."""
    series = pl.DataFrame(
        [
            {
                "run_id": RUN_ID,
                "episode_id": EPISODE_ID,
                "robot_id": "robot_0",
                "t_ns": 1_300_000_000,
                "ped_id": PEDESTRIAN_NEAR,
                "exposure_dba": 61.0,
            },
            {
                "run_id": RUN_ID,
                "episode_id": EPISODE_ID,
                "robot_id": "robot_0",
                "t_ns": 9_000_000_000,
                "ped_id": PEDESTRIAN_NEAR,
                "exposure_dba": 44.0,
            },
            {
                "run_id": RUN_ID,
                "episode_id": 7,
                "robot_id": "robot_0",
                "t_ns": 1_300_000_000,
                "ped_id": PEDESTRIAN_NEAR,
                "exposure_dba": 70.0,
            },
        ],
        schema=SERIES_EXPOSURE_SCHEMA,
    )
    metrics_dir = run_dir / "episodes" / "episode_000" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    series.write_parquet(metrics_dir / "series_acoustic_exposure.parquet")


@pytest.fixture
def run(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """A written benchmark run that the MCP tools resolve through $ARENA_DATA_DIR."""
    run_dir = tmp_path / "data" / "benchmarks" / RUN_ID
    _write_tables(run_dir, _star_tables())
    _write_series(run_dir)
    from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES

    pl.DataFrame(
        [
            {
                "metric": "min_clearance_m",
                "scope": "encounters",
                "fact_table": "fact_encounters",
                "dtype": "Float64",
                "unit": "m",
                "description": "Minimum clearance",
                "direction": "higher",
                "producer": "proxemics",
                "null_reason": None,
            },
            {
                "metric": "duration_s",
                "scope": "encounters",
                "fact_table": "fact_encounters",
                "dtype": "Float64",
                "unit": "s",
                "description": "Encounter duration",
                "direction": "lower",
                "producer": "proxemics",
                "null_reason": None,
            },
        ],
        schema=DIMENSION_TABLES["dim_metrics"].polars_schema(),
    ).write_parquet(run_dir / "dim_metrics.parquet")
    with _arena_data(tmp_path):
        yield run_dir


def _persisted(run_dir: pathlib.Path, table: str) -> pl.DataFrame:
    return pl.read_parquet(
        run_dir / "episodes" / "episode_000" / "metrics" / f"{table}.parquet"
    )


# --- slices ------------------------------------------------------------------


def test_contact_slice_explains_the_attributed_contact(run: pathlib.Path) -> None:
    response = evaluation_slice(RUN_ID, contact_id=CONTACT_ID, preview_limit=5)

    assert isinstance(response, SliceResponse)
    assert response.identity.episode_id == EPISODE_ID
    assert response.identity.encounter_id == ENCOUNTER_A
    assert response.identity.contact_id == CONTACT_ID
    assert response.identity.pedestrian_ids == [PEDESTRIAN_NEAR]
    assert response.identity.t_start_ns == 1_200_000_000
    assert response.identity.t_end_ns == 1_500_000_000
    assert response.identity.robot_id == "robot_0"

    contact_rows = pl.DataFrame(response.contacts.rows, schema=FACT_CONTACTS_SCHEMA)
    expected = _persisted(run, "fact_contacts").select(
        "contact_id",
        "encounter_id",
        "edge_case_collision_fault",
        "edge_case_collision_with",
    )
    assert_frame_equal(
        contact_rows.select(
            "contact_id",
            "encounter_id",
            "edge_case_collision_fault",
            "edge_case_collision_with",
        ),
        expected,
    )
    assert response.episode is not None
    assert response.episode.rows[0]["result"] == "GOAL_REACHED"
    assert response.collisions.n_rows == 1
    assert "obstacle_speed_at_impact_mps:no_samples" in response.null_reasons
    assert "mar:no_samples" in response.null_reasons


def test_encounter_slice_previews_its_contact_and_pedestrian_rows(
    run: pathlib.Path,
) -> None:
    response = evaluation_slice(RUN_ID, encounter_id=ENCOUNTER_A)

    assert response.encounter is not None
    assert response.encounter.n_rows == 1
    assert response.contacts.n_rows == 1
    assert response.pedestrians.n_rows == 1
    assert response.pedestrians.rows[0]["ped_id"] == PEDESTRIAN_NEAR
    pedestrian_frame = pl.DataFrame(
        response.pedestrians.rows, schema=FACT_PEDESTRIANS_SCHEMA
    )
    assert_frame_equal(
        pedestrian_frame.select("ped_id", "ped_name", "total_intrusions_personal"),
        _persisted(run, "fact_pedestrians")
        .filter(pl.col("ped_id") == PEDESTRIAN_NEAR)
        .select("ped_id", "ped_name", "total_intrusions_personal"),
    )


def test_episode_slice_lists_every_interaction(run: pathlib.Path) -> None:
    response = evaluation_slice(RUN_ID, episode_id=EPISODE_ID)

    assert response.identity.encounter_id is None
    assert response.episode is not None and response.episode.n_rows == 1
    assert response.encounter is not None and response.encounter.n_rows == 2
    assert response.contacts.n_rows == 1
    assert response.identity.pedestrian_ids == sorted([PEDESTRIAN_NEAR, PEDESTRIAN_FAR])
    assert response.pedestrians.n_rows == 2


def test_slice_series_is_the_window_not_the_whole_episode(run: pathlib.Path) -> None:
    response = evaluation_slice(RUN_ID, contact_id=CONTACT_ID, preview_limit=50)

    series = {preview.table: preview for preview in response.series}
    exposure = series["series_acoustic_exposure"]
    assert exposure.n_rows == 1
    assert exposure.rows[0]["ped_id"] == PEDESTRIAN_NEAR
    assert exposure.rows[0]["t_ns"] == 1_300_000_000
    assert_frame_equal(
        pl.DataFrame(exposure.rows, schema=SERIES_EXPOSURE_SCHEMA).select(
            "t_ns", "ped_id", "exposure_dba"
        ),
        pl.DataFrame(
            [
                {
                    "run_id": RUN_ID,
                    "episode_id": EPISODE_ID,
                    "robot_id": "robot_0",
                    "t_ns": 1_300_000_000,
                    "ped_id": PEDESTRIAN_NEAR,
                    "exposure_dba": 61.0,
                }
            ],
            schema=SERIES_EXPOSURE_SCHEMA,
        ).select("t_ns", "ped_id", "exposure_dba"),
    )


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ({"encounter_id": "enc-missing"}, "no encounter"),
        ({"contact_id": "con-missing"}, "no contact"),
        ({}, "pass encounter_id, contact_id or episode_id"),
    ],
    ids=["unknown_encounter", "unknown_contact", "no_ids"],
)
def test_slice_rejects_unknown_ids_with_a_typed_error(
    run: pathlib.Path, ids: dict[str, str], message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        evaluation_slice(RUN_ID, **ids)


def test_slice_needs_a_star_schema(tmp_path: pathlib.Path) -> None:
    (tmp_path / "data" / "benchmarks" / "bare-run").mkdir(parents=True)
    with _arena_data(tmp_path):
        with pytest.raises(ToolError, match="has no Star Schema tables"):
            evaluation_slice("bare-run", episode_id=EPISODE_ID)


def test_worst_encounters_rank_the_closest_first(run: pathlib.Path) -> None:
    response = evaluation_worst_encounters(RUN_ID, limit=2)

    assert isinstance(response, EncounterRankingResponse)
    assert response.metric == "min_clearance_m"
    assert response.direction == "higher"
    assert response.n_encounters == 2
    assert [row["encounter_id"] for row in response.rows] == [ENCOUNTER_A, ENCOUNTER_B]
    assert response.rows[0]["min_clearance_m"] < response.rows[1]["min_clearance_m"]


def test_worst_encounters_rank_a_lower_measure_descending(run: pathlib.Path) -> None:
    response = evaluation_worst_encounters(RUN_ID, metric="duration_s", limit=2)

    assert response.direction == "lower"
    assert [row["encounter_id"] for row in response.rows] == [ENCOUNTER_B, ENCOUNTER_A]


def test_worst_encounters_reject_an_unranked_measure(run: pathlib.Path) -> None:
    with pytest.raises(ToolError, match="direction"):
        evaluation_worst_encounters(RUN_ID, metric="actor_2")


# --- findings ----------------------------------------------------------------


def test_finding_create_list_review_and_attach(run: pathlib.Path) -> None:
    plots = run / "plots"
    plots.mkdir()
    (plots / "episode_000_clearance.png").write_bytes(b"png")
    (run / "report.html").write_text("<html></html>")

    finding = evaluation_finding_create(
        RUN_ID,
        "dwb cuts the corner in a blind crowd, ending in a robot-fault contact.",
        episode_id=EPISODE_ID,
        contact_id=CONTACT_ID,
        numbers={"min_clearance_m": 0.4, "impact_speed_mps": 0.8},
        figure_path="plots/episode_000_clearance.png",
    )
    assert finding.finding_id.startswith("fnd-")
    assert finding.status is FindingStatus.PROPOSED
    assert finding.figure_path == "plots/episode_000_clearance.png"
    assert finding.numbers == {"min_clearance_m": 0.4, "impact_speed_mps": 0.8}

    finding_file = run / "analysis" / "findings" / f"{finding.finding_id}.yaml"
    document = yaml.safe_load(finding_file.read_text(encoding="utf-8"))
    assert document["finding_id"] == finding.finding_id

    listed = evaluation_findings(RUN_ID)
    assert listed.n_findings == 1
    assert listed.findings[0].model_dump() == finding.model_dump()

    evaluation_finding_verify(RUN_ID, finding.finding_id)
    reviewed = evaluation_finding_status(
        RUN_ID, finding.finding_id, "verified", by="verifier"
    )
    assert reviewed.status is FindingStatus.VERIFIED
    assert evaluation_findings(RUN_ID).findings[0].status is FindingStatus.VERIFIED

    attached = evaluation_finding_attach(RUN_ID, finding.finding_id)
    assert attached.finding.report == "report.html"
    assert attached.note_label == f"finding:{finding.finding_id}"
    notes = yaml.safe_load((run / "notes.yaml").read_text(encoding="utf-8"))
    assert [row["label"] for row in notes] == [attached.note_label]
    assert "dwb cuts the corner" in notes[0]["value"]
    assert finding_file.is_file()


def test_finding_rejects_a_missing_artifact(run: pathlib.Path) -> None:
    with pytest.raises(ToolError, match="no figure at"):
        evaluation_finding_create(RUN_ID, "claim", figure_path="plots/missing.png")


def test_finding_rejects_evidence_without_a_pinned_index(run: pathlib.Path) -> None:
    with pytest.raises(ToolError, match="no pinned evidence"):
        evaluation_finding_create(RUN_ID, "claim", evidence_id="ev-missing")


def test_finding_accepts_pinned_evidence_and_a_stored_query(run: pathlib.Path) -> None:
    from arena_evaluation.analytics.evidence import EvidenceStore

    ev_rec = EvidenceStore(run).record(
        pl.DataFrame({"min_clearance_m": [0.4]}),
        kind="sql",
        operation="evaluation_query",
        snapshot_id="snap-test",
        purpose="Clearance evidence",
    )
    from arena_evaluation.storage.config_layout import bundle_path, run_analysis_root

    q_path = bundle_path(run_analysis_root(run), "query", "clearance_claim")
    q_path.parent.mkdir(parents=True, exist_ok=True)
    q_path.write_text(
        yaml.safe_dump(
            {
                "name": "clearance_claim",
                "description": "clearance claim",
                "sql": "SELECT min_clearance_m FROM fact_encounters",
            }
        ),
        encoding="utf-8",
    )

    finding = evaluation_finding_create(
        RUN_ID,
        "the near encounter had 0.4 m clearance",
        episode_id=EPISODE_ID,
        evidence_id=ev_rec.evidence_id,
        stored_query="clearance_claim",
    )
    assert finding.evidence_id == ev_rec.evidence_id
    assert finding.stored_query == "clearance_claim"


def test_finding_attach_needs_a_report(run: pathlib.Path) -> None:
    finding = evaluation_finding_create(RUN_ID, "claim")
    evaluation_finding_verify(RUN_ID, finding.finding_id)
    evaluation_finding_status(RUN_ID, finding.finding_id, "verified", by="verifier")
    with pytest.raises(ToolError, match="has no report"):
        evaluation_finding_attach(RUN_ID, finding.finding_id)


def test_finding_status_rejects_an_unknown_finding(run: pathlib.Path) -> None:
    with pytest.raises(ToolError, match="no finding"):
        evaluation_finding_status(RUN_ID, "fnd-missing", "verified", by="verifier")


def test_finding_rejects_non_finite_numbers(run: pathlib.Path) -> None:
    with pytest.raises(ToolError, match="must be finite"):
        evaluation_finding_create(
            RUN_ID, "claim", numbers={"min_clearance_m": float("nan")}
        )


def test_finding_rejects_an_empty_claim(run: pathlib.Path) -> None:
    with pytest.raises(ToolError, match="needs a claim"):
        evaluation_finding_create(RUN_ID, "   ")
