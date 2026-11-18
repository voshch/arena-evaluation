"""Vectorized scope-window detectors for the multi-resolution evaluator.

Every function here is a pure ``pl.LazyFrame -> pl.LazyFrame`` pipeline stage: encounter
bouts are extracted with run-length ids over boolean clearance masks, never with Python
row loops. The evaluator collects once, at the fact-table boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Final

import polars as pl

from arena_evaluation.processing.alignment import TRACK_ALIGN_TOLERANCE_NS, asof_join
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import NullReason

NS_PER_S: Final[float] = 1e9
DETECTOR_VERSION: Final[str] = "wp2-scope-detectors-v4"

# Encounter membership is surface-to-surface clearance: the exposed 2.0 m edge-to-edge decision.
ENCOUNTER_EDGE_DISTANCE_M: Final[float] = 2.0
ENCOUNTER_MERGE_GAP_S: Final[float] = 1.0
CONTACT_MERGE_GAP_S: Final[float] = 0.25

STATIONARY_SPEED_MPS: Final[float] = 0.15
HEAD_ON_ANGLE_RAD: Final[float] = 3.141592653589793 / 4.0
OVERTAKING_ALIGNMENT: Final[float] = 0.7
PEDESTRIAN_KIND_PATTERN: Final[str] = "ped|human"
ROBOT_KIND_PATTERN: Final[str] = "robot"
WALL_KIND_PATTERN: Final[str] = "wall"
TWO_PI_RAD: Final[float] = 6.283185307179586

ACTOR_PEDESTRIAN: Final[str] = "pedestrian"
ACTOR_ROBOT: Final[str] = "robot"
APPROACH_HEAD_ON: Final[str] = "head_on"
APPROACH_CROSSING: Final[str] = "crossing"
APPROACH_OVERTAKING: Final[str] = "overtaking"
APPROACH_STATIONARY: Final[str] = "stationary"
APPROACH_TYPES: Final[tuple[str, ...]] = (APPROACH_HEAD_ON, APPROACH_CROSSING, APPROACH_OVERTAKING, APPROACH_STATIONARY)
APPROACH_TYPE_DTYPE: Final[pl.Enum] = pl.Enum(APPROACH_TYPES)

PAIR_SCHEMA = pl.Schema(
    {
        "robot_id": pl.String,
        "actor_id": pl.String,
        "actor_name": pl.String,
        "actor_kind": pl.String,
        "t_ns": pl.Int64,
        "robot_x": pl.Float64,
        "robot_y": pl.Float64,
        "robot_yaw": pl.Float64,
        "robot_vx": pl.Float64,
        "robot_vy": pl.Float64,
        "actor_x": pl.Float64,
        "actor_y": pl.Float64,
        "actor_vx": pl.Float64,
        "actor_vy": pl.Float64,
        "actor_radius_m": pl.Float64,
        "distance_m": pl.Float64,
        "edge_clearance_m": pl.Float64,
    }
)

ENCOUNTER_WINDOW_SCHEMA = pl.Schema(
    {
        "encounter_idx": pl.UInt32,
        "robot_id": pl.String,
        "actor_id": pl.String,
        "actor_name": pl.String,
        "actor_kind": pl.String,
        "t_start_ns": pl.Int64,
        "t_end_ns": pl.Int64,
        "min_distance_m": pl.Float64,
        "min_clearance_m": pl.Float64,
        "approach_type": APPROACH_TYPE_DTYPE,
        "detector_version": pl.String,
        "detector_params_hash": pl.String,
        "null_reason": pl.String,
    }
)

CONTACT_WINDOW_SCHEMA = pl.Schema(
    {
        "contact_idx": pl.UInt32,
        "collider_kind": pl.String,
        "obstacle_id": pl.String,
        "t_onset_ns": pl.Int64,
        "t_end_ns": pl.Int64,
        "obstacle_x": pl.Float64,
        "obstacle_y": pl.Float64,
        "encounter_idx": pl.Int64,
        "detector_version": pl.String,
        "detector_params_hash": pl.String,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class DetectorParams:
    """Versioned parameters of the encounter and contact detectors.

    ``encounter_edge_distance_m`` is surface-to-surface clearance: two bodies are in an
    encounter while their edges are at most this far apart. Centre distance is reported in
    ``min_distance_m`` but never decides membership.
    """

    encounter_edge_distance_m: float = ENCOUNTER_EDGE_DISTANCE_M
    encounter_merge_gap_s: float = ENCOUNTER_MERGE_GAP_S
    contact_merge_gap_s: float = CONTACT_MERGE_GAP_S
    align_tolerance_ns: int = TRACK_ALIGN_TOLERANCE_NS
    stationary_speed_mps: float = STATIONARY_SPEED_MPS
    head_on_angle_rad: float = HEAD_ON_ANGLE_RAD
    overtaking_alignment: float = OVERTAKING_ALIGNMENT

    def fingerprint(self) -> str:
        """Stable short hash of the parameters, stored next to every detected window."""
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


DEFAULT_DETECTOR_PARAMS = DetectorParams()


def _require_radius(actors: pl.DataFrame | pl.LazyFrame, column: str) -> None:
    """Refuse actor samples whose radius column is null or NaN: the detector never invents a body size."""
    radius = pl.col(column)
    if actors.lazy().select((radius.is_null() | radius.is_nan()).any()).collect().item():
        raise MetricInputError(f"an actor sample has no usable {column}; each body needs its own recorded radius")


def pedestrian_pairs_lf(robot_tracks: pl.DataFrame | pl.LazyFrame, pedestrian_tracks: pl.DataFrame | pl.LazyFrame, robot_radius_m: float, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """One row per pedestrian sample with the robot pose at or before it: centre distance and edge clearance."""
    _require_radius(pedestrian_tracks, "radius_m")
    actors = (
        pedestrian_tracks.lazy()
        .select(
            pl.col("ped_id").alias("actor_id"),
            pl.col("ped_name").alias("actor_name"),
            pl.lit(ACTOR_PEDESTRIAN).alias("actor_kind"),
            pl.col("t_ns").cast(pl.Int64),
            pl.col("x").alias("actor_x"),
            pl.col("y").alias("actor_y"),
            pl.col("vx").alias("actor_vx"),
            pl.col("vy").alias("actor_vy"),
            pl.col("radius_m").alias("actor_radius_m"),
        )
        .sort("t_ns")
    )
    return _pairs_lf(robot_tracks, actors, robot_radius_m, params)


def fleet_pairs_lf(robot_tracks: pl.DataFrame | pl.LazyFrame, fleet: dict[str, pl.DataFrame], robot_radius_m: float, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """One row per other-robot task-pose sample with this robot's pose at or before it.

    The evaluated robot is dropped from its own fleet by full env-prefixed identity, and each
    actor carries its own recorded ``radius_m`` rather than a fleet-wide default.
    """
    actor_frames = [_fleet_actor_frame(actor_id, frame) for actor_id, frame in fleet.items()]
    if not actor_frames:
        return pl.LazyFrame(schema=PAIR_SCHEMA)
    ego_ids = robot_tracks.lazy().select(pl.col("robot_id").alias("actor_id")).unique()
    actors = pl.concat(actor_frames, how="vertical").join(ego_ids, on="actor_id", how="anti").sort("t_ns")
    _require_radius(actors, "actor_radius_m")
    return _pairs_lf(robot_tracks, actors, robot_radius_m, params)


def _fleet_actor_frame(actor_id: str, frame: pl.DataFrame) -> pl.LazyFrame:
    """Task poses of another robot as a pair-actor frame: position, derived velocity and its own radius."""
    actor = (
        frame.lazy()
        .select(
            pl.col("t_ns").cast(pl.Int64),
            pl.col("pos_x").alias("actor_x"),
            pl.col("pos_y").alias("actor_y"),
            pl.col("radius_m").cast(pl.Float64).alias("actor_radius_m"),
        )
        .sort("t_ns")
    )
    previous_time_s = pl.col("t_ns").diff().cast(pl.Float64) / NS_PER_S
    dt_s = pl.when(previous_time_s > 0.0).then(previous_time_s).otherwise(None)
    return actor.with_columns(
        actor_id=pl.lit(actor_id),
        actor_name=pl.lit(actor_id),
        actor_kind=pl.lit(ACTOR_ROBOT),
        actor_vx=pl.col("actor_x").diff() / dt_s,
        actor_vy=pl.col("actor_y").diff() / dt_s,
    ).select("actor_id", "actor_name", "actor_kind", "t_ns", "actor_x", "actor_y", "actor_vx", "actor_vy", "actor_radius_m")


def _pairs_lf(robot_tracks: pl.DataFrame | pl.LazyFrame, actors: pl.LazyFrame, robot_radius_m: float, params: DetectorParams) -> pl.LazyFrame:
    """As-of join the actor samples onto the robot timeline and derive centre distance and edge clearance."""
    robot = (
        robot_tracks.lazy()
        .select(
            "robot_id",
            pl.col("t_ns").cast(pl.Int64).alias("robot_t_ns"),
            pl.col("x").alias("robot_x"),
            pl.col("y").alias("robot_y"),
            pl.col("yaw").alias("robot_yaw"),
            pl.col("vx").alias("robot_vx"),
            pl.col("vy").alias("robot_vy"),
        )
        .sort("robot_t_ns")
    )
    join_tolerance_ns = params.align_tolerance_ns
    return (
        asof_join(actors, robot, left_on="t_ns", right_on="robot_t_ns", tolerance_ns=join_tolerance_ns)
        .filter(pl.col("robot_x").is_not_null() & pl.col("actor_x").is_not_null())
        .with_columns(
            distance_m=(pl.col("actor_x") - pl.col("robot_x")).pow(2).add((pl.col("actor_y") - pl.col("robot_y")).pow(2)).sqrt(),
            edge_clearance_m=(pl.col("actor_x") - pl.col("robot_x")).pow(2).add((pl.col("actor_y") - pl.col("robot_y")).pow(2)).sqrt() - (pl.lit(robot_radius_m) + pl.col("actor_radius_m")),
        )
        .select(PAIR_SCHEMA.names())
    )


def detect_encounter_windows_lf(pairs: pl.LazyFrame, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """Contiguous bouts whose surface clearance <= encounter_edge_distance_m, near gaps bridged.

    Run-length ids over the boolean clearance mask give the bouts; a second run-length id over the
    gap-bridged mask separates bouts that survive the merge window. A gap is bridged when the time
    between the last in-encounter sample before it and the first in-encounter sample after it is
    shorter than the merge gap. The onset row of each bout keeps the kinematics that classify the approach.
    """
    merge_gap_ns = int(params.encounter_merge_gap_s * NS_PER_S)
    in_encounter = pl.col("edge_clearance_m") <= params.encounter_edge_distance_m

    runs = (
        pairs.sort(["actor_id", "t_ns"])
        .with_columns(in_encounter=in_encounter)
        .with_columns(run_id=pl.col("in_encounter").rle_id().over("actor_id"))
        .group_by(["actor_id", "run_id"])
        .agg(
            run_in_encounter=pl.col("in_encounter").first(),
            run_start_ns=pl.col("t_ns").min(),
            run_end_ns=pl.col("t_ns").max(),
        )
        .sort(["actor_id", "run_id"])
    )
    previous_in_encounter = pl.col("run_in_encounter").shift(1).over("actor_id").fill_null(False)
    next_in_encounter = pl.col("run_in_encounter").shift(-1).over("actor_id").fill_null(False)
    gap_ns = pl.col("run_start_ns").shift(-1).over("actor_id") - pl.col("run_end_ns").shift(1).over("actor_id")
    interior_gap = (~pl.col("run_in_encounter")) & previous_in_encounter & next_in_encounter & (gap_ns < merge_gap_ns)
    runs = runs.with_columns(bridgeable=interior_gap)
    flagged = (
        pairs.sort(["actor_id", "t_ns"])
        .with_columns(in_encounter=in_encounter)
        .with_columns(run_id=pl.col("in_encounter").rle_id().over("actor_id"))
        .join(runs.select("actor_id", "run_id", "bridgeable"), on=["actor_id", "run_id"], how="left")
        .with_columns(bridged=pl.col("in_encounter") | pl.col("bridgeable").fill_null(False))
        .with_columns(encounter_run=pl.col("bridged").rle_id().over("actor_id"))
        .filter(pl.col("bridged"))
    )

    onset = {column: pl.col(column).sort_by("t_ns").first() for column in ("robot_yaw", "robot_vx", "robot_vy", "robot_x", "robot_y", "actor_x", "actor_y", "actor_vx", "actor_vy")}
    windows = flagged.group_by(["actor_id", "encounter_run"]).agg(
        robot_id=pl.col("robot_id").first(),
        actor_name=pl.col("actor_name").first(),
        actor_kind=pl.col("actor_kind").first(),
        t_start_ns=pl.col("t_ns").min(),
        t_end_ns=pl.col("t_ns").max(),
        min_distance_m=pl.col("distance_m").min(),
        min_clearance_m=pl.col("edge_clearance_m").min(),
        **onset,
    )

    bearing_rad = pl.arctan2(pl.col("actor_y") - pl.col("robot_y"), pl.col("actor_x") - pl.col("robot_x"))
    heading_error_rad = ((pl.col("robot_yaw") - bearing_rad + pl.lit(3.141592653589793)) % pl.lit(TWO_PI_RAD) - pl.lit(3.141592653589793)).abs()
    ped_speed_mps = pl.col("actor_vx").pow(2).add(pl.col("actor_vy").pow(2)).sqrt()
    robot_speed_mps = pl.col("robot_vx").pow(2).add(pl.col("robot_vy").pow(2)).sqrt()
    robot_toward_actor = (pl.col("robot_vx") * bearing_rad.cos() + pl.col("robot_vy") * bearing_rad.sin()) > 0.0
    same_direction = (pl.col("actor_vx") * pl.col("robot_vx") + pl.col("actor_vy") * pl.col("robot_vy")) > (pl.lit(params.overtaking_alignment) * ped_speed_mps * robot_speed_mps)
    speeds_known = ped_speed_mps.is_not_null() & robot_speed_mps.is_not_null()
    classified = (
        pl.when(ped_speed_mps < params.stationary_speed_mps)
        .then(pl.lit(APPROACH_STATIONARY))
        .when((heading_error_rad <= params.head_on_angle_rad) & robot_toward_actor)
        .then(pl.lit(APPROACH_HEAD_ON))
        .when(same_direction & (robot_speed_mps > ped_speed_mps))
        .then(pl.lit(APPROACH_OVERTAKING))
        .otherwise(pl.lit(APPROACH_CROSSING))
    )
    approach_type = pl.when(speeds_known).then(classified).otherwise(pl.lit(None, dtype=pl.String))
    approach_null_reason = (
        pl.when(speeds_known)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.lit(f"approach_type:{NullReason.NO_SAMPLES}", dtype=pl.String))
    )

    return (
        windows.with_columns(
            approach_type=approach_type.cast(APPROACH_TYPE_DTYPE),
            null_reason=approach_null_reason,
            detector_version=pl.lit(DETECTOR_VERSION),
            detector_params_hash=pl.lit(params.fingerprint()),
        )
        .sort(["t_start_ns", "actor_kind", "actor_id"])
        .with_row_index("encounter_idx")
        .with_columns(pl.col("encounter_idx").cast(pl.UInt32))
        .select(ENCOUNTER_WINDOW_SCHEMA.names())
    )


def contact_windows_lf(intervals: pl.DataFrame | pl.LazyFrame, encounters: pl.DataFrame | pl.LazyFrame, params: DetectorParams = DEFAULT_DETECTOR_PARAMS) -> pl.LazyFrame:
    """Contact bouts of one episode, each linked to the encounter that encloses its onset.

    ``intervals`` holds one row per contact bout with its raw ``collider_kind`` and ``obstacle_id``. Each obstacle id is
    resolved to the encounter actor that carries it (by id, name or numeric suffix) before the bout is linked to it.
    The bouts come from the contact calculator's detector; each row is stamped with the detector version and parameters.
    """
    normalized_kind = pl.col("collider_kind").fill_null("static").str.to_lowercase()
    collider_kind = (
        pl.when(normalized_kind.str.contains(PEDESTRIAN_KIND_PATTERN))
        .then(pl.lit(ACTOR_PEDESTRIAN))
        .when(normalized_kind.str.contains(ROBOT_KIND_PATTERN))
        .then(pl.lit(ACTOR_ROBOT))
        .when(normalized_kind.str.contains(WALL_KIND_PATTERN))
        .then(pl.lit("wall"))
        .otherwise(pl.lit("static"))
    )
    entity_id = pl.col("obstacle_id").fill_null("").str.replace(r"^env_[0-9]+/", "").str.split("/").list.first().fill_null("")

    enc_lf = encounters.lazy()
    by_id = enc_lf.select(pl.col("actor_id").alias("entity_id"), pl.col("actor_id").alias("resolved_by_id")).unique(subset=["entity_id"], keep="first")
    by_name = (
        enc_lf.filter(pl.col("actor_name").is_not_null() & (pl.col("actor_name") != ""))
        .select(pl.col("actor_name").alias("entity_id"), pl.col("actor_id").alias("resolved_by_name"))
        .unique(subset=["entity_id"], keep="first")
    )
    by_suffix = (
        enc_lf.filter(pl.col("actor_kind") == ACTOR_PEDESTRIAN)
        .with_columns(suffix_id=pl.col("actor_id").str.extract(r"(\d+)$", 1))
        .filter(pl.col("suffix_id").is_not_null())
        .select(pl.col("suffix_id"), pl.col("actor_id").alias("resolved_by_suffix"))
        .unique(subset=["suffix_id"], keep="first")
    )

    resolved = (
        intervals.lazy()
        .select(
            pl.col("t_onset_ns").cast(pl.Int64),
            pl.col("t_end_ns").cast(pl.Int64),
            collider_kind.alias("collider_kind"),
            pl.col("obstacle_id").fill_null(""),
            pl.col("obstacle_x").cast(pl.Float64),
            pl.col("obstacle_y").cast(pl.Float64),
        )
        .with_columns(entity_id=entity_id, suffix_id=entity_id.str.extract(r"(\d+)$", 1))
        .join(by_id, on="entity_id", how="left")
        .join(by_name, on="entity_id", how="left")
        .join(by_suffix, on="suffix_id", how="left")
        .with_columns(
            obstacle_id=pl.when(pl.col("collider_kind") == ACTOR_PEDESTRIAN)
            .then(pl.coalesce(pl.col("resolved_by_id"), pl.col("resolved_by_name"), pl.col("resolved_by_suffix"), pl.col("entity_id")))
            .when(pl.col("collider_kind") == ACTOR_ROBOT)
            .then(pl.coalesce(pl.col("resolved_by_id"), pl.col("entity_id")))
            .otherwise(pl.col("obstacle_id")),
        )
        .sort(["t_onset_ns", "collider_kind", "obstacle_id"])
        .with_row_index("contact_idx")
        .with_columns(pl.col("contact_idx").cast(pl.UInt32))
    )

    enclosing_windows = enc_lf.select(
        pl.col("actor_id").alias("obstacle_id"),
        pl.col("t_start_ns").alias("encounter_start_ns"),
        pl.col("t_end_ns").alias("encounter_end_ns"),
        pl.col("encounter_idx"),
    )
    enclosing = (
        resolved.join(enclosing_windows, on="obstacle_id", how="inner")
        .filter((pl.col("t_onset_ns") >= pl.col("encounter_start_ns")) & (pl.col("t_onset_ns") <= pl.col("encounter_end_ns")))
        .group_by("contact_idx")
        .agg(encounter_idx=pl.col("encounter_idx").min())
    )
    return (
        resolved.join(enclosing, on="contact_idx", how="left")
        .with_columns(
            encounter_idx=pl.col("encounter_idx").cast(pl.Int64),
            detector_version=pl.lit(DETECTOR_VERSION),
            detector_params_hash=pl.lit(params.fingerprint()),
        )
        .select(CONTACT_WINDOW_SCHEMA.names())
    )


def empty_encounter_windows() -> pl.DataFrame:
    """Empty encounter window table with the detector schema."""
    return pl.DataFrame(schema=ENCOUNTER_WINDOW_SCHEMA)


def empty_contact_windows() -> pl.DataFrame:
    """Empty contact window table with the detector schema."""
    return pl.DataFrame(schema=CONTACT_WINDOW_SCHEMA)


def empty_pairs() -> pl.DataFrame:
    """Empty pair table with the detector schema."""
    return pl.DataFrame(schema=PAIR_SCHEMA)
