"""MCP WP5 slice tools: drill down one interaction through the Star Schema."""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

import polars as pl
import pydantic

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.scopes import FactTable
from arena_evaluation.storage.registry.series import SERIES_TABLES
from arena_evaluation.storage.star_schema import has_star_schema
from arena_mcp.core import ToolError

from .base import RunId, domain, run_path

__all__ = [
    "EncounterRankingResponse",
    "SliceResponse",
    "evaluation_slice",
    "evaluation_worst_encounters",
]

PREVIEW_ROWS: Final[int] = 10
SERIES_WINDOW_MARGIN_NS: Final[int] = 1_000_000_000
# The identity contract every calculator's series_table publishes (BaseMetricCalculator.series_table).
SERIES_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "episode_id",
    "robot_id",
    "t_ns",
)
SERIES_TABLE_NAMES: Final[frozenset[str]] = frozenset(table.table_name for table in SERIES_TABLES.values())
PEDESTRIAN_ACTOR_KIND: Final[str] = "pedestrian"
SLICE_FACT_KEYS: Final[tuple[str, ...]] = (
    "episodes",
    "encounters",
    "contacts",
    "pedestrians",
    "collisions",
)

PreviewLimit = Annotated[
    int,
    pydantic.Field(description="rows previewed per Star Schema table", ge=1, le=500),
]
RankLimit = Annotated[int, pydantic.Field(description="encounters returned by the ranking", ge=1, le=100)]


class TablePreview(pydantic.BaseModel):
    """At most ``preview_limit`` typed rows of one Star Schema or series table."""

    table: str
    columns: list[str]
    n_rows: int
    rows: list[dict[str, Any]]
    snapshot_id: str = ""


class SliceIdentity(pydantic.BaseModel):
    """Identity keys and time span of one drill-down slice."""

    run_id: str
    episode_id: int | None = None
    encounter_id: str | None = None
    contact_id: str | None = None
    pedestrian_ids: list[str] = pydantic.Field(default_factory=list)
    robot_id: str | None = None
    t_start_ns: int | None = None
    t_end_ns: int | None = None


class SliceResponse(pydantic.BaseModel):
    """One interaction explained by the Star Schema rows around it."""

    identity: SliceIdentity
    episode: TablePreview | None
    encounter: TablePreview | None
    contacts: TablePreview
    pedestrians: TablePreview
    collisions: TablePreview
    series: list[TablePreview]
    null_reasons: list[str]
    snapshot_id: str = ""


class EncounterRankingResponse(pydantic.BaseModel):
    """Encounters ranked by one declared measure, worst outcome first."""

    run_id: str
    metric: str
    direction: Literal["higher", "lower"]
    n_encounters: int
    preview_limit: int
    columns: list[str]
    rows: list[dict[str, Any]]


def _catalog(run_id: str) -> BenchmarkCatalog:
    """The run's Star Schema catalog, a ToolError when the run has no Star Schema yet."""
    path = run_path(run_id)
    if not has_star_schema(path):
        raise ToolError(
            f"run {run_id!r} has no Star Schema tables",
            hint="evaluation_process builds them",
        )
    return BenchmarkCatalog.open(path)


def _facts(catalog: BenchmarkCatalog, *keys: str) -> tuple[pl.DataFrame, ...]:
    """The declared fact tables named by ``keys``, in order; absent files use the canonical schema."""
    return tuple(catalog.get_table(f"fact_{FACT_TABLES[FactTable(key)].table_name}").collect() for key in keys)


def _preview(table: str, frame: pl.DataFrame, limit: int) -> TablePreview:
    """A typed preview of the first ``limit`` rows of a table."""
    return TablePreview(
        table=table,
        columns=list(frame.columns),
        n_rows=len(frame),
        rows=frame.head(limit).to_dicts(),
    )


def _null_reasons(frames: list[pl.DataFrame]) -> list[str]:
    """Every distinct ``field:code`` provenance token across the sliced tables, sorted.

    A fact row carries its missing fields as one ``field:code;field:code`` string; the slice
    reports the individual tokens so an analyst can filter and count them.
    """
    tokens = (token for frame in frames if len(frame) > 0 for value in frame.get_column("null_reason").drop_nulls() for token in str(value).split(";") if token)
    return sorted(set(tokens))


def _identity_row(frame: pl.DataFrame, column: str, value: str, what: str) -> dict[str, Any]:
    """The one row whose identity column equals ``value``; a ToolError when it is absent."""
    rows = frame.filter(pl.col(column) == value)
    if len(rows) == 0:
        raise ToolError(
            f"no {what} {value!r} in this run",
            hint=f"ids come from fact_{what}s; evaluation_query lists them",
        )
    return rows.row(0, named=True)


def _pedestrian_refs(frame: pl.DataFrame, kind_column: str, id_column: str) -> set[str]:
    """Pedestrian identities named by one ``(kind, id)`` column pair."""
    if len(frame) == 0:
        return set()
    return {str(value) for kind, value in zip(frame.get_column(kind_column), frame.get_column(id_column), strict=True) if kind == PEDESTRIAN_ACTOR_KIND and value}


@domain.tool(read_only=True)
def evaluation_slice(
    run_id: RunId,
    encounter_id: Annotated[
        str | None,
        pydantic.Field(description="encounter to explain, from fact_encounters"),
    ] = None,
    contact_id: Annotated[str | None, pydantic.Field(description="contact to explain, from fact_contacts")] = None,
    pedestrian_id: Annotated[str | None, pydantic.Field(description="pedestrian identity to include")] = None,
    episode_id: Annotated[
        int | None,
        pydantic.Field(description="episode to slice when no encounter or contact id is given"),
    ] = None,
    preview_limit: PreviewLimit = PREVIEW_ROWS,
) -> SliceResponse:
    """One interaction explained by its Star Schema rows: encounter, contacts, pedestrians, episode, collisions and series window.

    Give an ``encounter_id`` ("why was this encounter risky"), a ``contact_id`` ("why was this contact
    attributed to the robot") or just an ``episode_id`` ("everything that happened in this episode").
    Unknown ids are a typed error naming the table, never a stack trace.
    """
    if encounter_id is None and contact_id is None and episode_id is None:
        raise ToolError(
            "pass encounter_id, contact_id or episode_id",
            hint="evaluation_worst_encounters lists encounter ids; evaluation_query reads fact_contacts",
        )
    catalog = _catalog(run_id)
    episodes, encounters, contacts, pedestrians, collisions = _facts(catalog, *SLICE_FACT_KEYS)

    contact_row: dict[str, Any] | None = None
    encounter_row: dict[str, Any] | None = None
    if contact_id is not None:
        contact_row = _identity_row(contacts, "contact_id", contact_id, "contact")
        episode_id = int(contact_row["episode_id"])
        linked_encounter = contact_row["encounter_id"]
        if encounter_id is not None and linked_encounter is not None and linked_encounter != encounter_id:
            raise ToolError(
                f"contact {contact_id!r} belongs to encounter {linked_encounter!r}, not {encounter_id!r}",
                hint=f"pass encounter_id {linked_encounter!r} or omit it",
            )
        if encounter_id is None:
            encounter_id = linked_encounter
    if encounter_id is not None:
        encounter_row = _identity_row(encounters, "encounter_id", encounter_id, "encounter")
        episode_id = int(encounter_row["episode_id"])
    if episode_id is None:
        raise ToolError(
            "no episode resolved from the given ids",
            hint="evaluation_progress lists the run's episode ids",
        )

    contact_rows = contacts.filter(pl.col("episode_id") == episode_id)
    if contact_id is not None:
        contact_rows = contact_rows.filter(pl.col("contact_id") == contact_id)
    elif encounter_id is not None:
        contact_rows = contact_rows.filter(pl.col("encounter_id") == encounter_id)
    contact_rows = contact_rows.sort("t_onset_ns")

    encounter_rows = encounters.filter(pl.col("episode_id") == episode_id).sort("t_start_ns")
    if encounter_id is not None:
        encounter_rows = encounter_rows.filter(pl.col("encounter_id") == encounter_id)
    episode_rows = episodes.filter(pl.col("episode_id") == episode_id)
    if len(episode_rows) == 0:
        raise ToolError(
            f"no episode {episode_id} in this run",
            hint="evaluation_progress lists the recorded episodes",
        )

    pedestrian_ids = sorted((set() if pedestrian_id is None else {pedestrian_id}) | _pedestrian_refs(encounter_rows, "actor_2_kind", "actor_2") | _pedestrian_refs(contact_rows, "collider_kind", "obstacle_id"))
    pedestrian_rows = pedestrians.filter((pl.col("episode_id") == episode_id) & (pl.col("ped_id").is_in(pedestrian_ids) | pl.col("ped_name").is_in(pedestrian_ids))).sort("ped_id") if pedestrian_ids else pedestrians.head(0)
    collision_rows = collisions.filter(pl.col("episode_id") == episode_id).sort("collision_idx")

    anchor, start_column = (contact_row, "t_onset_ns") if contact_row is not None else (encounter_row, "t_start_ns")
    t_start_ns = int(anchor[start_column]) if anchor is not None else None
    t_end_ns = int(anchor["t_end_ns"]) if anchor is not None else None
    if anchor is not None:
        robot_id: str | None = str(anchor["robot_id"])
    else:
        episode_robots = episode_rows.get_column("robot_id").unique()
        robot_id = str(episode_robots[0]) if len(episode_robots) == 1 else None

    series: list[TablePreview] = []
    for name in sorted(catalog.table_names()):
        if name not in SERIES_TABLE_NAMES:
            continue
        window = catalog.get_table(name).collect().filter(pl.col("episode_id") == episode_id)
        if t_start_ns is not None and t_end_ns is not None:
            window = window.filter(
                pl.col("t_ns").is_between(
                    t_start_ns - SERIES_WINDOW_MARGIN_NS,
                    t_end_ns + SERIES_WINDOW_MARGIN_NS,
                )
            )
        if len(window) > 0:
            series.append(_preview(name, window.sort(*SERIES_IDENTITY_COLUMNS), preview_limit))

    sliced = {
        "fact_episodes": episode_rows,
        "fact_encounters": encounter_rows,
        "fact_contacts": contact_rows,
        "fact_pedestrians": pedestrian_rows,
        "fact_collisions": collision_rows,
    }
    previews = {name: _preview(name, frame, preview_limit) for name, frame in sliced.items()}
    encounter_visible = encounter_row is not None or contact_row is None
    return SliceResponse(
        identity=SliceIdentity(
            run_id=run_id,
            episode_id=episode_id,
            encounter_id=encounter_id,
            contact_id=contact_id,
            pedestrian_ids=pedestrian_ids,
            robot_id=robot_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
        ),
        snapshot_id=catalog.snapshot_id,
        episode=previews["fact_episodes"],
        encounter=previews["fact_encounters"] if encounter_visible and len(encounter_rows) > 0 else None,
        contacts=previews["fact_contacts"],
        pedestrians=previews["fact_pedestrians"],
        collisions=previews["fact_collisions"],
        series=series,
        null_reasons=_null_reasons(list(sliced.values())),
    )


@domain.tool(read_only=True)
def evaluation_worst_encounters(
    run_id: RunId,
    metric: Annotated[
        str,
        pydantic.Field(description="declared fact_encounters measure column to rank by"),
    ] = "min_clearance_m",
    limit: RankLimit = PREVIEW_ROWS,
    episode_id: Annotated[int | None, pydantic.Field(description="only this episode's encounters")] = None,
    direction: Annotated[
        Literal["higher", "lower"] | None,
        pydantic.Field(description="override direction when dim_metrics has NULL direction"),
    ] = None,
) -> EncounterRankingResponse:
    """The encounters ranked worst first by one declared measure, reading direction from dim_metrics.

    "Worst" is the opposite of the measure's declared direction, so ``min_clearance_m`` ranks the
    closest encounter first and ``duration_s`` the longest. Nulls never outrank a measured value.
    """
    declared = FACT_TABLES[FactTable.ENCOUNTERS].polars_schema()
    if metric not in declared.names():
        raise ToolError(
            f"{metric!r} is not a declared fact_encounters column",
            hint=f"declared columns: {', '.join(declared.names())}",
        )
    catalog = _catalog(run_id)
    resolved_direction: Literal["higher", "lower"] | None
    if direction is None:
        direction_col = catalog.get_table("dim_metrics").collect().filter(pl.col("metric") == metric).get_column("direction")
        # agent-contracts: ignore AC004 - MCP scalar lookup from collected dim_metrics
        declared_directions = direction_col.to_list()
        resolved_direction = next(
            ("higher" if value == "higher" else "lower" for value in declared_directions if value in ("higher", "lower")),
            None,
        )
        if resolved_direction is None:
            raise ToolError(
                f"encounter metric {metric!r} has no declared direction in dim_metrics",
                hint="pass direction='higher' or direction='lower', or inspect evaluation_metrics",
            )
    elif direction == "higher":
        resolved_direction = "higher"
    else:
        resolved_direction = "lower"
    (encounters,) = _facts(catalog, "encounters")
    if episode_id is not None:
        encounters = encounters.filter(pl.col("episode_id") == episode_id)
    if len(encounters) == 0:
        raise ToolError(
            f"run {run_id!r} has no encounter rows",
            hint="a run without detected encounters has an empty fact_encounters",
        )
    ranked = encounters.sort(metric, descending=resolved_direction == "lower", nulls_last=True).head(limit)
    return EncounterRankingResponse(
        run_id=run_id,
        metric=metric,
        direction=resolved_direction,
        n_encounters=len(encounters),
        preview_limit=limit,
        columns=list(ranked.columns),
        rows=ranked.to_dicts(),
    )
