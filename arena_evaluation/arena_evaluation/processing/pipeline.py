from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import functools
import json
import logging
import multiprocessing
import os
import pathlib
import re
import shutil
import time
import typing
from collections.abc import Callable, MutableMapping

import polars as pl

if typing.TYPE_CHECKING:
    import numpy as np

    from arena_evaluation.processing.scoped_evaluator import ScopedFactTables

from arena_evaluation.processing.dimension_rows import episode_dimensions
from arena_evaluation.processing.scoped_evaluator import write_fact_tables, write_series_tables
from arena_evaluation.processing.world_geometry import benchmark_world_geometry
from arena_evaluation.processing.mcap_reader import MCAPReader, load_bundles
from arena_evaluation.processing.metrics.common.alignment import TRACK_ALIGN_TOLERANCE_NS, asof_join
from arena_evaluation.processing.metrics.dictionary import metric_dictionary
from arena_evaluation.processing.metrics.social.interaction_common import PEDESTRIAN_PHYSICS_TOPIC, PEDESTRIAN_TRACKS_TOPIC
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pose_anchor import pose_frame, resolve_pose_source
from arena_evaluation.processing.outcome import outcome_verdict
from arena_evaluation.processing.pose_segments import consistent_segment, ride_arrival_times
from arena_evaluation.storage.contracts import enforce_dimension_contract
from arena_evaluation.storage.exceptions import ContractViolationError, MetricInputError
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.registry import FactTable
from arena_evaluation.storage.registry.dimensions import DimMetrics, DimWorldGeometry
from arena_evaluation.storage.schemas import (
    AlignedEpisodeBundle,
    EpisodeDescriptor,
    RobotParams,
    RunMetadata,
    TopicBundle,
)
from arena_evaluation.storage.star_schema import stack, write_benchmark_dimensions

_log = logging.getLogger(__name__)

# Frames shared across robots or scoped to the whole episode, never per-robot sample data.
_SHARED_FRAMES = frozenset({"tf", "tf_static", "world_semantics", "episode_lifecycle", "episode_integrity"})
KEPT_EMPTY_TABLES = ("collision_contacts", "pedestrian_gestures")
# Recorded table name -> the topic name the calculators read; the recorder names most of them alike.
_TOPIC_NAMES: typing.Final[dict[str, str]] = {
    "pedestrian_arena_peds": "pedestrian_tracks",
    "pedestrian_agent_states": "pedestrian_tracks_physics",
    "robot_collision_events": "collision_events",
    "robot_collision_contacts": "collision_contacts",
}

# The robot's ground-truth track is the clock; the legacy column names carry it and the `_gt` aliases mirror it.
_PRIMARY = {
    "x": "pos_x",
    "y": "pos_y",
    "yaw": "yaw",
    "speed_mps": "vel_linear",
    "omega_z": "vel_angular",
}
_GROUND_TRUTH_ALIASES = {"x": "pos_x_gt", "y": "pos_y_gt", "yaw": "yaw_gt"}
_SCAN = {"scan_min_m": "scan_min", "range_max_m": "scan_range_max", "n_valid_rays": "n_rays"}
_CMD_VEL = {"linear_x": "linear_x", "linear_y": "linear_y", "linear_z": "linear_z", "angular_x": "angular_x", "angular_y": "angular_y", "angular_z": "angular_z"}
_COLLISIONS = {"event_count": "collision_event", "wall_count": "collision_wall", "static_count": "collision_static", "ped_count": "collision_pedestrian"}
_MONITOR = {"action_type": "action_type", "polygon_name": "polygon_name"}
_POWER = {"total_power_w": "total_power_w", "static_power_w": "static_power_w", "total_mech_power_w": "total_mech_power_w", "total_thermal_power_w": "total_thermal_power_w"}
_ENERGY = {"total_energy_consumed_wh": "total_energy_consumed_wh", "battery_soc_percent": "battery_soc_percent"}
_ACOUSTICS = {
    "total_level_af_dba": "total_level_af_dba",
    "total_level_zf_db": "total_level_zf_db",
    "baseline_level_dba": "baseline_level_dba",
    "drivetrain_level_dba": "drivetrain_level_dba",
    "uncertainty_dba": "uncertainty_1sigma_dba",
    "validity_flags": "validity_flags",
}
_MARKERS = {"label": "label"}

# EpisodeRecord.outcome_state -> (result, success)
_STATUS_EVALUATED = "evaluated"


def _frame(table: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    """The table collected once, None when it was not recorded."""
    if table is None:
        return None
    return table.collect() if isinstance(table, pl.LazyFrame) else table


def _merge_series_frames(groups: list[dict[str, pl.DataFrame]]) -> dict[str, pl.DataFrame]:
    """Concatenate per-robot series frames by table stem."""
    parts: dict[str, list[pl.DataFrame]] = {}
    for group in groups:
        for stem, frame in group.items():
            parts.setdefault(stem, []).append(frame)
    return {stem: frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed") for stem, frames in parts.items()}


def _merge_fact_tables(groups: list[ScopedFactTables]) -> ScopedFactTables | None:
    """Concatenate the fact tables of the robots evaluated for one episode."""
    if not groups:
        return None
    from arena_evaluation.processing.scoped_evaluator import ScopedFactTables

    named: dict[str, pl.DataFrame] = {}
    for member in FactTable:
        frames = [group.by_name()[member.value] for group in groups]
        named[member.value] = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")
    return ScopedFactTables(**named)


def _worker_init():
    import os
    import signal

    os.environ["POLARS_MAX_THREADS"] = "1"
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def _shutdown_executor_cleanly(executor: concurrent.futures.ProcessPoolExecutor):
    """Force terminate all child worker processes without blocking on wait=True."""
    try:
        processes = list(executor._processes.values())
        for proc in processes:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _extract_worker(
    data_root_str: str,
    ep: EpisodeDescriptor,
    force_extract: bool,
    status_dict: _EpisodeStatus | None = None,
) -> int:
    import pathlib
    import time

    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

    if status_dict is not None:
        try:
            status_dict[ep.episode_id] = (ep.planner, ep.stage, "Extracting MCAP / Topics", 1, 1, time.perf_counter())
        except Exception:
            pass
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    pipeline.extract_episode(ep, force_extract=force_extract)
    if status_dict is not None:
        try:
            status_dict.pop(ep.episode_id, None)
        except Exception:
            pass
    return ep.episode_id


def _process_worker(
    data_root_str: str,
    ep: EpisodeDescriptor,
    force_extract: bool,
    status_dict: _EpisodeStatus | None = None,
) -> tuple[int, EpisodeEvaluation]:
    import pathlib
    import time

    from arena_evaluation.processing.pipeline import EpisodeEvaluation, ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

    if status_dict is not None:
        try:
            status_dict[ep.episode_id] = (ep.planner, ep.stage, "Loading Topics", 0, 18, time.perf_counter())
        except Exception:
            pass
    fm = FolderManager(data_root=pathlib.Path(data_root_str))
    pipeline = ProcessingPipeline(fm, profiler=None, workers=1)
    try:
        try:
            evaluation = pipeline.evaluate_episode(ep, force_extract=force_extract, status_dict=status_dict)
        except MetricInputError:
            # A required input is missing: the benchmark must refuse the episode, so the failure
            # travels to process_benchmark instead of becoming a status row.
            _log.exception(f"episode_{ep.episode_id:03d}: required metric input missing")
            raise
        except Exception as e:
            _log.exception(f"episode_{ep.episode_id:03d}: metrics failed")
            evaluation = EpisodeEvaluation(
                episode_id=ep.episode_id,
                rows=[_status_row(ep, None, "", "error", repr(e))],
                metrics_dir=pathlib.Path(ep.episode_dir) / "metrics",
            )
    finally:
        if status_dict is not None:
            try:
                status_dict.pop(ep.episode_id, None)
            except Exception:
                pass
    return ep.episode_id, evaluation


def _peak_pedestrian_count(peds: pl.DataFrame | None, start_ns: int | None = None, end_ns: int | None = None) -> int:
    """Largest number of pedestrians recorded in one frame between start_ns and end_ns, bounds inclusive."""
    if peds is None or len(peds) == 0:
        return 0
    if start_ns is not None:
        peds = peds.filter(pl.col("t_ns") >= start_ns)
    if end_ns is not None:
        peds = peds.filter(pl.col("t_ns") <= end_ns)
    counts = peds.group_by("t_ns").len()["len"]
    return int(counts.max()) if len(counts) > 0 else 0  # pyright: ignore[reportArgumentType]


def _align_bundle(bundle: TopicBundle, start_t_ns: int | None = None, end_t_ns: int | None = None) -> pl.DataFrame | None:
    """Align the robot's recorded tables onto its ground-truth track on native timestamps.

    The source ``processing.native_bundle`` module is absent at the PR1 tip; its one alignment
    implementation is carried here because every episode-scope calculator reads ``episode.data``.
    """
    track = _frame(bundle.robot_tracks)
    if track is None or track.height == 0:
        return None
    if start_t_ns is not None:
        track = track.filter(pl.col("t_ns") >= start_t_ns)
    if end_t_ns is not None:
        track = track.filter(pl.col("t_ns") <= end_t_ns)
    if track.height == 0:
        return None
    available = set(track.columns)
    frame = track.select(
        [
            pl.col("t_ns"),
            *(pl.col(source).alias(target) for source, target in _PRIMARY.items() if source in available),
            *(pl.col(source).alias(target) for source, target in _GROUND_TRUTH_ALIASES.items() if source in available),
        ]
    ).sort("t_ns")
    for field_name, renames in _SECONDARY_TABLES:
        frame = _join_secondary(frame, vars(bundle)[field_name], renames, start_t_ns, end_t_ns)
    markers = _frame(bundle.robot_characterization_phase)
    if markers is not None and markers.height > 0:
        selected = markers.select([pl.col("t_ns"), *(pl.col(source).alias(target) for source, target in _MARKERS.items() if source in markers.columns)]).sort("t_ns")
        frame = asof_join(frame, selected, tolerance_ns=None)
    return frame


_SECONDARY_TABLES: typing.Final[tuple[tuple[str, dict[str, str]], ...]] = (
    ("robot_scan_summary", _SCAN),
    ("robot_cmd_vel", _CMD_VEL),
    ("robot_collision_events", _COLLISIONS),
    ("robot_collision_monitor", _MONITOR),
    ("robot_power", _POWER),
    ("robot_energy", _ENERGY),
    ("robot_acoustics", _ACOUSTICS),
)


def _join_secondary(
    primary: pl.DataFrame,
    table: pl.DataFrame | pl.LazyFrame | None,
    renames: dict[str, str],
    start_t_ns: int | None,
    end_t_ns: int | None,
) -> pl.DataFrame:
    """As-of join one recorded stream onto the track, leaving its columns NULL where it has no sample."""
    frame = _frame(table)
    if frame is None or frame.height == 0:
        return primary
    tolerance = TRACK_ALIGN_TOLERANCE_NS
    if start_t_ns is not None:
        frame = frame.filter(pl.col("t_ns") >= start_t_ns - tolerance)
    if end_t_ns is not None:
        frame = frame.filter(pl.col("t_ns") <= end_t_ns + tolerance)
    if frame.height == 0:
        return primary
    available = set(frame.columns)
    selected = frame.select([pl.col("t_ns"), *(pl.col(source).alias(target) for source, target in renames.items() if source in available)]).sort("t_ns")
    return asof_join(primary, selected, tolerance_ns=tolerance)


def _consistent_pose_segment(aligned_df: pl.DataFrame | None, ride_arrivals_ns: np.ndarray | None = None) -> pl.DataFrame | None:
    """The rows with a finite ground-truth pose, cut to the longest stretch free of teleport jumps (elevator rides and stair crossings excepted)."""
    if aligned_df is None or len(aligned_df) == 0:
        return aligned_df
    aligned_df = aligned_df.filter(pl.col("pos_x_gt").is_not_null() & ~pl.col("pos_x_gt").is_nan() & pl.col("pos_y_gt").is_not_null() & ~pl.col("pos_y_gt").is_nan() & pl.col("yaw_gt").is_not_null() & ~pl.col("yaw_gt").is_nan())
    if len(aligned_df) < 2:
        return aligned_df
    start, end = consistent_segment(aligned_df["pos_x_gt"].to_numpy(), aligned_df["pos_y_gt"].to_numpy(), aligned_df["t_ns"].to_numpy(), ride_arrivals_ns)
    return aligned_df.slice(start, end - start)


def _metric_world(metadata: RunMetadata | None, ep: EpisodeDescriptor) -> str | None:
    """World name the world-geometry metrics resolve their asset by, None when unknown."""
    if metadata is not None and metadata.map:
        return metadata.map
    if ep.map and ep.map != "unknown":
        return ep.map
    return None


def _benchmark_worlds(episodes: list[EpisodeDescriptor]) -> list[str]:
    """The distinct named worlds of a benchmark's episodes, sorted."""
    named = (_metric_world(None, ep) for ep in episodes)
    return sorted({world for world in named if world})


_RECORD_RUNNING = 1  # arena_runtime_msgs/msg/EpisodeRecord.RUNNING
_RECORD_SUCCESS = 2  # arena_runtime_msgs/msg/EpisodeRecord.SUCCESS


def _episode_window(record: pl.DataFrame | pl.LazyFrame | None) -> tuple[int | None, int | None]:
    """[RUNNING stamp, terminal stamp] of the episode record, a bound is None when its row is missing."""
    if record is None:
        return None, None
    df = record.collect() if isinstance(record, pl.LazyFrame) else record
    if len(df) == 0 or "t_ns" not in df.columns:
        return None, None
    df = df.sort("t_ns", maintain_order=True)
    if "outcome_state" not in df.columns:
        return int(df["t_ns"][0]), int(df["t_ns"][-1])
    df = _episode_rows(df)
    states = df["outcome_state"].to_list()
    stamps = df["t_ns"].to_list()
    start = next((int(t) for t, s in zip(stamps, states, strict=True) if s == _RECORD_RUNNING), None)
    end = None if states[-1] == _RECORD_RUNNING else int(stamps[-1])
    return start, end


def _episode_endpoints(
    aligned_df: pl.DataFrame,
    plan: pl.DataFrame | pl.LazyFrame | None,
    phases: dict | None = None,
) -> tuple[list[float], list[float]]:
    """Map-frame (x, y, yaw) of the first sample and of the goal: the last goto of the recorded phases, else the last planned pose, else the last sample."""
    first_row = aligned_df.row(0, named=True)
    last_row = aligned_df.row(-1, named=True)
    # Ground truth is map frame, odom starts wherever the robot's odometry happened to be.
    x, y, yaw = ("pos_x_gt", "pos_y_gt", "yaw_gt") if "pos_x_gt" in first_row else ("pos_x", "pos_y", "yaw")

    start_pos: list[float] = []
    goal_pos: list[float] = []
    if x in first_row and y in first_row and first_row[x] is not None:
        start_pos = [first_row[x], first_row[y], first_row.get(yaw, 0.0)]
    if x in last_row and y in last_row and last_row[x] is not None:
        goal_pos = [last_row[x], last_row[y], last_row.get(yaw, 0.0)]

    plan_df = plan.collect() if isinstance(plan, pl.LazyFrame) else plan
    if plan_df is not None and "t_ns" in plan_df.columns:
        # A recording can outlive its episode by a beat and catch the next episode's first plan.
        plan_df = plan_df.filter(pl.col("t_ns") <= aligned_df["t_ns"].max())
    if plan_df is not None and len(plan_df) > 0:
        latest_plan = plan_df.filter(pl.col("plan_id") == plan_df["plan_id"].max()).sort("waypoint_idx")
        goal_pos = [latest_plan["x"][-1], latest_plan["y"][-1], latest_plan["yaw"][-1]]

    gotos = [pose for pose in phases["map_poses"] if pose is not None] if phases else []
    if gotos:
        goal_pos = [float(v) for v in gotos[-1]]
    return start_pos, goal_pos


def _robot_phases(record: pl.DataFrame, robot_dir: str) -> dict | None:
    """This robot's {phases, conditions, map_poses, goal_inputs, instructions} from the last EpisodeRecord row that carries any, None without."""
    if "phases_yaml" not in record.columns:
        return None
    from arena_evaluation.processing.metrics.ecological.condition_metrics import bare_robot_name

    bare = bare_robot_name(robot_dir)
    for raw in reversed(record["phases_yaml"].to_list()):
        if not raw:
            continue
        try:
            entry = json.loads(raw).get(bare)
        except (ValueError, AttributeError):
            return None
        if entry is not None:
            return entry
    return None


def _fleet_poses(
    robots: dict[str, TopicBundle],
    window: tuple[int | None, int | None],
    *,
    ego_robot_id: str,
    params_by_robot: typing.Mapping[str, RobotParams],
) -> dict[str, pl.DataFrame]:
    """Recorded judge poses of every other robot, keyed by its full env-prefixed id and cut to the window.

    The evaluated robot is excluded by identity, so it can never become its own encounter actor, and
    every actor frame carries the radius recorded for that robot, never the evaluated robot's.
    """
    start_ns, end_ns = window
    fleet: dict[str, pl.DataFrame] = {}
    for name, bundle in robots.items():
        if name == ego_robot_id:
            continue
        frame = _frame(bundle.robot_task_pose)
        if frame is None:
            continue
        if start_ns is not None:
            frame = frame.filter(pl.col("t_ns") >= start_ns - TRACK_ALIGN_TOLERANCE_NS)
        if end_ns is not None:
            frame = frame.filter(pl.col("t_ns") <= end_ns + TRACK_ALIGN_TOLERANCE_NS)
        if len(frame) > 0:
            fleet[name] = frame.with_columns(radius_m=pl.lit(float(params_by_robot[name].robot_radius), dtype=pl.Float64))
    return fleet


def _calculator_topic_names(bundle: TopicBundle) -> set[str]:
    """The bundle's carried tables under the topic names the calculators read, so a topic gate sees what the bundle holds."""
    return {_TOPIC_NAMES.get(name, name) for name in bundle.available()}


def _collect_native_topics(bundle: TopicBundle) -> dict[str, pl.DataFrame]:
    """Raw native-rate topic frames keyed by the name the calculators read (collected)."""
    topics: dict[str, pl.DataFrame] = {}
    for field_name in sorted(bundle.available() - _SHARED_FRAMES):
        df = _frame(vars(bundle)[field_name])
        if df is None:
            continue
        topic = _TOPIC_NAMES.get(field_name, field_name)
        if len(df) > 0 or topic in KEPT_EMPTY_TABLES:
            topics[topic] = df
    return _with_pedestrian_constants(topics)


# Per-pedestrian constants only the engine table records; joined onto the arena rows by ped_id.
_PEDESTRIAN_CONSTANT_COLUMNS: typing.Final[tuple[str, ...]] = ("radius_m", "robot_awareness", "robot_fov_gain", "robot_repulsion_gain", "robot_reaction_gain")


def _with_pedestrian_constants(topics: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """The arena pedestrian rows with the engine table's latched constants joined on by ped_id, so the track frame carries the recorded radius and crowd settings."""
    arena = topics.get(PEDESTRIAN_TRACKS_TOPIC)
    engine = topics.get(PEDESTRIAN_PHYSICS_TOPIC)
    if arena is None or engine is None or engine.height == 0:
        return topics
    available = [column for column in _PEDESTRIAN_CONSTANT_COLUMNS if column in engine.columns]
    constants = engine.group_by("ped_id").agg([pl.col(column).drop_nulls().first() for column in available])
    arena_rows = arena.drop([column for column in available if column in arena.columns])
    return {**topics, PEDESTRIAN_TRACKS_TOPIC: arena_rows.join(constants, on="ped_id", how="left", maintain_order="left")}


def _bundle_robot_id(bundle: TopicBundle) -> str:
    """The recorded identity of the robot a bundle belongs to."""
    for table in (bundle.robot_descriptor, bundle.robot_tracks, bundle.robot_odom):
        frame = _frame(table)
        if frame is not None and frame.height and "robot_id" in frame.columns:
            return str(frame.get_column("robot_id")[0])
    raise MetricInputError("a robot bundle has no recorded table that names its robot")


def _robot_parameters(bundle: TopicBundle, robot_id: str) -> RobotParams:
    """The parameters recorded for this robot, never the installed definition or a default."""
    from arena_evaluation.storage.schemas import robot_params_from_recorded

    descriptor = _frame(bundle.robot_descriptor)
    params = _frame(bundle.robot_params)
    recorded = [frame for frame in (descriptor, params) if frame is not None]
    rows = pl.concat(recorded, how="diagonal") if recorded else pl.DataFrame()
    return robot_params_from_recorded(rows, robot_id)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeEvaluation:
    """One episode's metric rows and fact tables, held until the benchmark's publication step."""

    episode_id: int
    rows: list[dict[str, typing.Any]]
    metrics_dir: pathlib.Path
    facts: ScopedFactTables | None = None


def _episode_rows(record: pl.DataFrame) -> pl.DataFrame:
    """The episode's own records in publish order, up to its first terminal record after it ran, without the next episode's queued records that share its last timestamp."""
    df = record.sort("t_ns", maintain_order=True)
    if "outcome_state" not in df.columns:
        return df
    states = df["outcome_state"].to_list()
    started = next((i for i, state in enumerate(states) if state == _RECORD_RUNNING), None)
    if started is None:
        return df
    if df["episode_id"][started] is not None:
        # the previous episode's latched result is not this episode's record
        df = df.filter(pl.col("episode_id") == df["episode_id"][started])
        states = df["outcome_state"].to_list()
        started = states.index(_RECORD_RUNNING)
    end = next((i for i in range(started, len(states)) if states[i] is not None and states[i] >= _RECORD_SUCCESS), None)
    return df if end is None else df.head(end + 1)


def _record_conditions(record: pl.DataFrame) -> list[dict] | None:
    """The episode's conditions: the last of its own records whose `conditions` JSON is a non-empty list."""
    if "conditions_yaml" not in record.columns or "t_ns" not in record.columns:
        return None
    for raw in reversed(_episode_rows(record)["conditions_yaml"].to_list()):
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            continue
        if parsed:
            return parsed
    return None


def _record_outcome(record: pl.DataFrame | pl.LazyFrame | None, metadata: RunMetadata | None) -> tuple[int | None, str | None]:
    """(outcome_state, outcome_info): the episode's own terminal record, else the runner's yaml outcome, else the last record."""
    df = _frame(record)
    rows = df is not None and len(df) > 0 and "outcome_state" in df.columns and "t_ns" in df.columns
    if rows:
        assert df is not None
        df = _episode_rows(df)
        running = df.filter(pl.col("outcome_state") == _RECORD_RUNNING)
        after = df.filter(pl.col("t_ns") > running["t_ns"][0]) if len(running) else df
        terminal = after.filter(pl.col("outcome_state") != _RECORD_RUNNING)
        if len(terminal):
            last = terminal.row(-1, named=True)
            return int(last["outcome_state"]), last.get("outcome_info")
    if metadata is not None and metadata.outcome_state is not None:
        return metadata.outcome_state, metadata.outcome_info
    if rows:
        assert df is not None
        last = df.row(-1, named=True)
        return int(last["outcome_state"]), last.get("outcome_info")
    return None, None


def _record_integrity(
    record: pl.DataFrame | pl.LazyFrame | None,
    flags: pl.DataFrame | pl.LazyFrame | None,
    metadata: RunMetadata | None,
) -> list[str]:
    """Integrity reasons of the last recorded record, from the episode yaml when no record carries any."""
    df = _frame(record)
    reasons = _frame(flags)
    if df is not None and len(df) > 0 and "t_ns" in df.columns and reasons is not None and len(reasons) > 0:
        last_stamp = df["t_ns"].max()
        return reasons.filter(pl.col("t_ns") == last_stamp).sort("flag_idx")["flag"].to_list()
    if metadata is not None:
        return list(metadata.integrity)
    return []


def _planner_split(ep: EpisodeDescriptor, metadata: RunMetadata | None) -> tuple[str, str]:
    if metadata is not None and metadata.local_planner:
        return metadata.local_planner, metadata.inter_planner or ""
    from arena_evaluation.storage.planner_names import split_planner_name

    return split_planner_name(ep.planner)


def _episode_identity(
    ep: EpisodeDescriptor,
    metadata: RunMetadata | None,
    robot: str,
    *,
    start: typing.Any,
    goal: typing.Any,
) -> dict[str, typing.Any]:
    """The identity columns carried by every row of one episode, evaluated or not."""
    local_planner, inter_planner = _planner_split(ep, metadata)
    return {
        "episode": ep.episode_id,
        "planner": ep.planner,
        "local_planner": local_planner,
        "inter_planner": inter_planner,
        "robot": robot,
        "map": ep.map,
        "stage": ep.stage,
        "benchmark_id": ep.benchmark_id,
        "start": start,
        "goal": goal,
        "is_reference": ep.is_reference,
        "reference_type": ep.reference_type,
    }


def _status_row(
    ep: EpisodeDescriptor,
    metadata: RunMetadata | None,
    robot: str,
    status: str,
    reason: str,
    record: pl.DataFrame | pl.LazyFrame | None = None,
    flags: pl.DataFrame | pl.LazyFrame | None = None,
) -> dict:
    """Identity and runtime verdict of an episode without metrics."""
    outcome_state, outcome_info = _record_outcome(record, metadata)
    result, success = outcome_verdict(outcome_state, outcome_info)
    return {
        **_episode_identity(ep, metadata, robot, start=[], goal=[]),
        "status": status,
        "status_reason": reason,
        "pose_source": None,
        "pose_frame": None,
        "outcome_info": outcome_info,
        "integrity": _record_integrity(record, flags, metadata),
        "result": result,
        "success": success,
    }


def _print_status_summary(rows: list[dict]) -> None:
    """One line per status, then every row the pipeline could not fully evaluate."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("status") or "unknown"] = counts.get(row.get("status") or "unknown", 0) + 1
    print("Episode status: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())), flush=True)
    flagged = [r for r in rows if r.get("status") != _STATUS_EVALUATED]
    for r in sorted(flagged, key=lambda r: (str(r.get("stage")), r.get("episode", 0))):
        print(f"  episode {r.get('episode')} {r.get('planner')}/{r.get('stage')} [{r.get('robot')}] {r.get('status')}: {r.get('status_reason')}", flush=True)


def _metadata_robot(metadata: RunMetadata | None, bundles: dict[str, TopicBundle] | None = None) -> str:
    """Bundle-style robot key (env_0_jackal) for a row without a robot bundle."""
    model = metadata.robot_model[0].partition("[")[0] if metadata is not None and metadata.robot_model else ""
    env = next((match.group(0) for name in (bundles or {}) if (match := re.match(r"env_\d+", name))), None)
    return f"{env}_{model}" if env and model else model


def _suite_metrics(episode_dir: pathlib.Path, data_root: pathlib.Path) -> dict[str, typing.Any]:
    """Suite-level metric overrides from the run's manifest.yaml, {} for bare run dirs and junk values."""
    import yaml

    for parent in episode_dir.parents:
        manifest_path = parent / "manifest.yaml"
        if manifest_path.exists():
            try:
                suite = yaml.safe_load(manifest_path.read_text()).get("suite") or {}
                raw = suite.get("metrics") or {}
            except Exception as e:
                _log.warning(f"{manifest_path}: unreadable, ignoring suite metrics: {e!r}")
                return {}
            metrics: dict[str, typing.Any] = {}
            v = raw.get("max_collisions")
            if v is not None:
                try:
                    metrics["max_collisions"] = int(v)
                except (TypeError, ValueError):
                    _log.warning(f"{manifest_path}: suite metrics.max_collisions {v!r} is not an int, ignoring")
            barn = raw.get("barn")
            if barn is not None:
                from .metrics.performance.barn_metrics import parse_reference

                try:
                    metrics["barn"] = parse_reference(barn)
                except ValueError as e:
                    _log.warning(f"{manifest_path}: suite metrics.barn ignored: {e}")
            return metrics
        if parent == data_root:
            break
    return {}


def _failed_evaluation(ep: EpisodeDescriptor, error: Exception) -> EpisodeEvaluation:
    """The evaluation row of an episode whose metrics could not be computed."""
    return EpisodeEvaluation(
        episode_id=ep.episode_id,
        rows=[_status_row(ep, None, "", "error", repr(error))],
        metrics_dir=pathlib.Path(ep.episode_dir) / "metrics",
    )


_EpisodeStatus = MutableMapping[int, tuple[str, str, str, int, int, float]]


def _extract_job(data_root_str: str, force_extract: bool, ep: EpisodeDescriptor, status_dict: _EpisodeStatus) -> int:
    return _extract_worker(data_root_str, ep, force_extract, status_dict)


def _process_job(
    data_root_str: str,
    ep: EpisodeDescriptor,
    status_dict: _EpisodeStatus,
) -> tuple[int, EpisodeEvaluation]:
    return _process_worker(data_root_str, ep, False, status_dict)


class RunProgress(typing.Protocol):
    """Reports one phase's progress; the CLI supplies a terminal display, without one the phase runs silently."""

    def log_error(self, episode_id: int, message: str) -> None: ...

    def log_completed(self, episode_id: int, label: str, seconds: float) -> None: ...


RunProgressFactory = Callable[[str, int, int, _EpisodeStatus], contextlib.AbstractContextManager[RunProgress]]


class _SilentProgress:
    def log_error(self, episode_id: int, message: str) -> None:
        """Errors reach the caller through on_failed."""

    def log_completed(self, episode_id: int, label: str, seconds: float) -> None:
        """Completions reach the caller through on_done."""


@contextlib.contextmanager
def _silent_progress(title: str, episode_count: int, workers: int, status: _EpisodeStatus) -> typing.Iterator[RunProgress]:
    yield _SilentProgress()


def _run_phase(
    title: str,
    episodes: list[EpisodeDescriptor],
    workers: int,
    job: Callable[[EpisodeDescriptor, _EpisodeStatus], typing.Any],
    *,
    progress: RunProgressFactory,
    on_done: Callable[[EpisodeDescriptor, typing.Any], None],
    on_failed: Callable[[EpisodeDescriptor, Exception], None],
) -> None:
    """Run ``job`` once per episode, inline for one worker and in a spawned process pool otherwise."""
    if workers <= 1:
        local_status: _EpisodeStatus = {}
        with progress(title, len(episodes), 1, local_status) as display:
            for ep in episodes:
                t_start = time.perf_counter()
                try:
                    result = job(ep, local_status)
                except MetricInputError:
                    raise
                except Exception as error:
                    on_failed(ep, error)
                    display.log_error(ep.episode_id, str(error))
                    continue
                on_done(ep, result)
                display.log_completed(ep.episode_id, f"{ep.planner}/{ep.stage}", time.perf_counter() - t_start)
        return

    mp_context = multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    status_dict = manager.dict()
    with progress(title, len(episodes), workers, status_dict) as display:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_context, initializer=_worker_init)
        try:
            futures = {executor.submit(job, ep, status_dict): (ep, time.perf_counter()) for ep in episodes}
            for future in concurrent.futures.as_completed(futures):
                ep, t_start = futures[future]
                try:
                    result = future.result()
                except MetricInputError:
                    raise
                except Exception as error:
                    on_failed(ep, error)
                    display.log_error(ep.episode_id, str(error))
                    continue
                on_done(ep, result)
                display.log_completed(ep.episode_id, f"{ep.planner}/{ep.stage}", time.perf_counter() - t_start)
        except KeyboardInterrupt:
            print(f"\n[KeyboardInterrupt] {title} interrupted by user. Terminating workers...", flush=True)
            raise
        finally:
            _shutdown_executor_cleanly(executor)


class PhaseProfiler(typing.Protocol):
    """Times the named phases of a run; benchmark.profiler.PipelineProfiler implements it."""

    def phase(self, name: str) -> contextlib.AbstractContextManager[None]: ...


class ProcessingPipeline:
    """
    Orchestrates the data processing pipeline:
    MCAP -> Extract -> Align -> Split -> Metrics -> Parquet
    """

    def __init__(
        self,
        folder_manager: FolderManager,
        profiler: PhaseProfiler | None = None,
        workers: int | None = None,
        progress: RunProgressFactory = _silent_progress,
    ):
        self.folder_manager = folder_manager
        self.profiler = profiler
        self.progress = progress
        # -1 (or None) = auto-detect the CPU count; any value < 1 falls back to it.
        self.workers = (os.cpu_count() or 1) if (workers is None or workers < 1) else workers

    def extract_episode(self, ep: EpisodeDescriptor, force_extract: bool = False) -> dict[str, TopicBundle] | None:
        """Extract topics from a single episode MCAP into episode_dir/topics/."""
        episode_dir = pathlib.Path(ep.episode_dir)
        mcap_path = self.folder_manager.mcap_path_for_episode(episode_dir)
        topics_dir = self.folder_manager.extracted_topics_path_for_episode(episode_dir)

        if not mcap_path.exists():
            _log.warning(f"episode_{ep.episode_id:03d} ({ep.planner}/{ep.stage}): no MCAP found, so it is not extracted")
            return None

        if not force_extract and TopicParquetStore.is_current(topics_dir):
            bundles = TopicParquetStore.read(topics_dir)
            if bundles:
                return bundles

        _ctx = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx:
            if topics_dir.exists():
                shutil.rmtree(topics_dir)
            bundles = MCAPReader(mcap_path).read(topics_dir)
            TopicParquetStore.mark_complete(topics_dir)
            return bundles

    def _describe_run_dir(self, run_dir: pathlib.Path) -> EpisodeDescriptor:
        ep = self.folder_manager.describe_episode(pathlib.Path(run_dir))
        if ep is None:
            raise FileNotFoundError(f"{run_dir} is not an episode directory (episode_NNN with episode_NNN.yaml)")
        return ep

    def extract_run_dir(self, run_dir: pathlib.Path, force_extract: bool = False) -> dict[str, TopicBundle] | None:
        """Extract one `episode_NNN` directory into its topics/ folder."""
        return self.extract_episode(self._describe_run_dir(run_dir), force_extract=force_extract)

    def process_run_dir(self, run_dir: pathlib.Path, force_extract: bool = False) -> pathlib.Path | None:
        """Extract, evaluate and publish one `episode_NNN` directory."""
        episode = self._describe_run_dir(run_dir)
        self.process_episode(episode, force_extract=force_extract)
        out = pathlib.Path(run_dir) / "metrics"
        if not out.exists():
            return None
        return out

    def evaluate_episode(
        self,
        ep: EpisodeDescriptor,
        force_extract: bool = False,
        status_dict: _EpisodeStatus | None = None,
    ) -> EpisodeEvaluation:
        """Extract and evaluate one episode, holding its fact tables for the publication step."""
        from arena_evaluation.processing.metrics.ecological.mechanism_common import robot_env_snapshot
        from arena_evaluation.processing.scoped_evaluator import ScopedEvaluator

        episode_dir = pathlib.Path(ep.episode_dir)

        if not force_extract and (episode_dir / "metrics" / "fact_episodes.parquet").exists():
            return EpisodeEvaluation(
                episode_id=ep.episode_id,
                rows=[],
                metrics_dir=episode_dir / "metrics",
            )

        # Load episode metadata
        yaml_path = episode_dir / f"{episode_dir.name}.yaml"
        if not yaml_path.exists():
            yaml_path = episode_dir / "metadata.yaml"

        metadata = None
        if yaml_path.exists():
            try:
                metadata = MetadataWriter.read(yaml_path)
            except Exception as e:
                _log.warning(f"episode_{ep.episode_id:03d}: could not read metadata: {e}")

        def finish(
            rows: list[dict],
            facts: list[ScopedFactTables] | None = None,
            series: list[dict[str, pl.DataFrame]] | None = None,
        ) -> EpisodeEvaluation:
            """Fold the scoped fact tables into the evaluation and publish its series tables."""
            metrics_dir = episode_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            write_series_tables(_merge_series_frames(series or []), metrics_dir)
            return EpisodeEvaluation(
                episode_id=ep.episode_id,
                rows=rows,
                metrics_dir=metrics_dir,
                facts=_merge_fact_tables(facts or []),
            )

        bundles = self.extract_episode(ep, force_extract=force_extract)
        if bundles is None or len(bundles) == 0:
            return finish([_status_row(ep, metadata, _metadata_robot(metadata), "no_recording", "no MCAP or empty extraction")])

        _ctx = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx:
            pedsim_avail = False
            if metadata is not None:
                pedsim_avail = metadata.pedsim_available or False

            suite_metrics_cfg = _suite_metrics(episode_dir, self.folder_manager.data_root)

            all_results: list[dict] = []
            episode_facts: list[ScopedFactTables] = []
            episode_series: list[dict[str, pl.DataFrame]] = []
            robots = {name: bundle for name, bundle in bundles.items() if bundle.robot_tracks is not None or bundle.robot_odom is not None}
            if not robots:
                any_record = next((b.episode_lifecycle for b in bundles.values() if b.episode_lifecycle is not None), None)
                return finish([_status_row(ep, metadata, _metadata_robot(metadata, bundles), "no_trajectory", "no tf ground-truth or odometry rows", any_record)])
            robot_ids = {name: _bundle_robot_id(bundle) for name, bundle in robots.items()}
            params_by_robot = {name: _robot_parameters(bundle, robot_ids[name]) for name, bundle in robots.items()}
            for robot_name, bundle in robots.items():
                robot_params = params_by_robot[robot_name]
                window = _episode_window(bundle.episode_lifecycle)
                tracks, pose_source = resolve_pose_source(bundle, window)
                if tracks is None:
                    reason = "no tf ground-truth rows inside the episode window"
                    all_results.append(_status_row(ep, metadata, robot_name, "no_trajectory", reason, bundle.episode_lifecycle, bundle.episode_integrity))
                    continue

                if bundle.pedestrian_arena_peds is not None or bundle.pedestrian_agent_states is not None:
                    pedsim_avail = True
                robot_pedsim = pedsim_avail

                available_topics = _calculator_topic_names(bundle)

                aligned_df = _align_bundle(bundle, *window)
                if aligned_df is None or len(aligned_df) < 5:
                    reason = f"{0 if aligned_df is None else len(aligned_df)} ground-truth rows inside the episode window"
                    _log.warning(f"episode_{ep.episode_id:03d} [{robot_name}]: {reason}")
                    all_results.append(_status_row(ep, metadata, robot_name, "no_trajectory", reason, bundle.episode_lifecycle, bundle.episode_integrity))
                    continue

                # Resolve the pose frame once so all calculators share it
                aligned_df = _consistent_pose_segment(aligned_df, ride_arrival_times(robot_env_snapshot(bundle.world_semantics, robot_name)))
                if aligned_df is None or len(aligned_df) < 5:
                    reason = f"{0 if aligned_df is None else len(aligned_df)} pose rows after resolution"
                    _log.warning(f"episode_{ep.episode_id:03d} [{robot_name}]: {reason}")
                    all_results.append(_status_row(ep, metadata, robot_name, "no_trajectory", reason, bundle.episode_lifecycle, bundle.episode_integrity))
                    continue

                topics = _collect_native_topics(bundle)
                if len(aligned_df) > 0:
                    t0 = int(aligned_df["t_ns"][0])
                    t1 = int(aligned_df["t_ns"][-1])
                    topics = {name: df.filter((pl.col("t_ns") >= t0 - TRACK_ALIGN_TOLERANCE_NS) & (pl.col("t_ns") <= t1 + TRACK_ALIGN_TOLERANCE_NS)) if "t_ns" in df.columns else df for name, df in topics.items()}

                peds_df = _frame(bundle.pedestrian_arena_peds)
                if peds_df is None or peds_df.height == 0:
                    peds_df = _frame(bundle.pedestrian_agent_states)
                num_pedestrians = _peak_pedestrian_count(peds_df, None if window[0] is None else window[0] - TRACK_ALIGN_TOLERANCE_NS, window[1])

                semantic_snapshot_df = _frame(bundle.world_semantics)

                conditions = None
                phases = None
                outcome_state = None
                outcome_info = None
                if bundle.episode_lifecycle is not None:
                    er = _frame(bundle.episode_lifecycle)
                    if er is not None and len(er) > 0:
                        conditions = _record_conditions(er)
                        if "outcome_state" in er.columns and "t_ns" in er.columns:
                            phases = _robot_phases(_episode_rows(er), robot_name)
                        outcome_state, outcome_info = _record_outcome(er, metadata)

                integrity = _record_integrity(bundle.episode_lifecycle, bundle.episode_integrity, metadata)
                start_pos, goal_pos = _episode_endpoints(aligned_df, bundle.robot_plans, phases)

                fleet = _fleet_poses(robots, window, ego_robot_id=robot_name, params_by_robot=params_by_robot)

                aligned_ep = AlignedEpisodeBundle(
                    episode_id=ep.episode_id,
                    data=aligned_df,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    num_pedestrians=num_pedestrians,
                    robot_name=robot_name,
                    semantic_snapshot=semantic_snapshot_df,
                    conditions=conditions,
                    phases=phases,
                    fleet=fleet,
                    outcome_state=outcome_state,
                    outcome_info=outcome_info,
                    run=metadata,
                    peds=peds_df,
                    map=ep.map,
                    topics=topics,
                    run_dir=episode_dir.parent.parent,
                )
                def on_calc_progress(calc_name: str, calc_idx: int, total_calcs: int):
                    if status_dict is not None:
                        try:
                            status_dict[ep.episode_id] = (
                                ep.planner,
                                ep.stage,
                                calc_name,
                                calc_idx,
                                total_calcs,
                                time.perf_counter(),
                            )
                        except Exception:
                            pass

                evaluated = ScopedEvaluator(robot_params, metrics_config=suite_metrics_cfg).evaluate(
                    aligned_ep,
                    run_id=ep.benchmark_id,
                    episode_id=ep.episode_id,
                    world=_metric_world(metadata, ep),
                    episode_context={
                        **_episode_identity(ep, metadata, aligned_ep.robot_name or robot_name, start=aligned_ep.start_pos, goal=aligned_ep.goal_pos),
                        "integrity": integrity,
                        "pose_source": pose_source.kind,
                        "pose_frame": pose_frame(bundle.tf, window),
                        "pose_samples": pose_source.samples,
                        "status": _STATUS_EVALUATED,
                        "status_reason": None,
                    },
                    pedsim_available=robot_pedsim,
                    available_topics=available_topics,
                    progress_callback=on_calc_progress,
                )
                all_results.append(evaluated.episode_metrics)
                episode_facts.append(evaluated.facts)
                episode_series.append(evaluated.series)

            return finish(all_results, episode_facts, episode_series)

    def process_episode(
        self,
        ep: EpisodeDescriptor,
        force_extract: bool = False,
        status_dict: _EpisodeStatus | None = None,
    ) -> list[dict]:
        """Extract, evaluate and publish one episode."""
        evaluation = self.evaluate_episode(ep, force_extract=force_extract, status_dict=status_dict)
        self.publish_evaluation(evaluation)
        return evaluation.rows

    @staticmethod
    def publish_evaluation(evaluation: EpisodeEvaluation) -> None:
        """Write one evaluation's fact tables next to its already published series; a failed write raises."""
        if evaluation.facts is None:
            return
        write_fact_tables(evaluation.facts, evaluation.metrics_dir)

    def _publish_and_consolidate(
        self,
        benchmark_id: str,
        episodes: list[EpisodeDescriptor],
        evaluations: list[EpisodeEvaluation],
    ) -> pathlib.Path | None:
        """Publish every episode's fact tables once, then check the benchmark."""
        from arena_evaluation.storage.contracts import check_benchmark

        for evaluation in evaluations:
            self.publish_evaluation(evaluation)
        benchmark_dir = self.folder_manager.benchmark_dir(benchmark_id)
        report = check_benchmark(benchmark_dir)
        if not report.ok:
            raise ContractViolationError(report.summary())
        print(f"Done. {len(episodes)} episodes -> {benchmark_dir}")
        return benchmark_dir

    def extract_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """Extract all episodes of a benchmark, then write its dimensions once."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)
        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return
        self._extract_episodes(benchmark_id, episodes, force_extract)
        self._write_benchmark_dimensions(benchmark_id, episodes)

    def _write_benchmark_dimensions(self, benchmark_id: str, episodes: list[EpisodeDescriptor]) -> None:
        """Stack the dimension rows of every extracted episode, add the metric dictionary and the geometry of every named world, and write each dimension once at the benchmark root."""
        contributions = []
        for ep in episodes:
            episode_dir = pathlib.Path(ep.episode_dir)
            topics = episode_dir / "topics"
            if not topics.is_dir():
                _log.warning(f"episode_{ep.episode_id:03d}: no extracted topics, so its dimension rows are missing")
                continue
            metadata = MetadataWriter.read(episode_dir / f"{episode_dir.name}.yaml")
            bundles = load_bundles(topics, {})
            contributions.append(episode_dimensions(run_id=benchmark_id, episode_id=ep.episode_id, metadata=metadata, bundles=bundles))
        benchmark_dir = self.folder_manager.benchmark_dir(benchmark_id)
        dimensions = stack(contributions)
        dimensions[DimMetrics.table_name] = metric_dictionary()
        dimensions[DimWorldGeometry.table_name] = benchmark_world_geometry(_benchmark_worlds(episodes), benchmark_dir)
        enforce_dimension_contract(dimensions)
        write_benchmark_dimensions(dimensions, benchmark_dir)

    def _extract_episodes(self, benchmark_id: str, episodes: list[EpisodeDescriptor], force_extract: bool) -> None:
        _run_phase(
            f"Phase 1: Extracting MCAP topics for {benchmark_id}",
            episodes,
            self.workers,
            functools.partial(_extract_job, str(self.folder_manager.data_root), force_extract),
            progress=self.progress,
            on_done=lambda _ep, _episode_id: None,
            on_failed=lambda _ep, _error: None,
        )

    def process_benchmark(self, benchmark_id: str, force_extract: bool = False) -> pathlib.Path | None:
        """Process all episodes into their fact tables, returning the benchmark directory, None when nothing was found."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)

        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return None

        _ctx_extract = self.profiler.phase("extract") if self.profiler else contextlib.nullcontext()
        with _ctx_extract:
            self.extract_benchmark(benchmark_id, force_extract=force_extract)

        evaluations: list[EpisodeEvaluation] = []

        def on_failed(ep: EpisodeDescriptor, error: Exception) -> None:
            evaluations.append(_failed_evaluation(ep, error))

        _ctx_process = self.profiler.phase("process") if self.profiler else contextlib.nullcontext()
        with _ctx_process:
            _run_phase(
                f"Phase 2: Calculating metrics across {len(episodes)} episodes",
                episodes,
                self.workers,
                functools.partial(_process_job, str(self.folder_manager.data_root)),
                progress=self.progress,
                on_done=lambda _ep, result: evaluations.append(result[1]),
                on_failed=on_failed,
            )

        _print_status_summary([row for evaluation in evaluations for row in evaluation.rows])
        return self._publish_and_consolidate(benchmark_id, episodes, evaluations)
