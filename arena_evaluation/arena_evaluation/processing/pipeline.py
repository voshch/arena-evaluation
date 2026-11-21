from __future__ import annotations

import collections
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
    from arena_evaluation.processing.scoped_evaluator import ScopedFactTables

from arena_evaluation.processing.dimension_rows import episode_dimensions
from arena_evaluation.processing.metrics.registry import load_metric_definitions
from arena_evaluation.processing.scoped_evaluator import (
    write_fact_tables,
    write_series_tables,
)
from arena_evaluation.processing.world_geometry import benchmark_world_geometry
from arena_evaluation.processing.mcap_reader import MCAPReader, load_bundles
from arena_evaluation.processing.metrics.common.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
    asof_join,
)
from arena_evaluation.processing.metrics.dictionary import metric_dictionary
from arena_evaluation.processing.metrics.social.interaction_common import (
    PEDESTRIAN_AGENT_STATES_TOPIC,
    PEDESTRIAN_ARENA_PEDS_TOPIC,
)
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pose_anchor import pose_frame, resolve_pose_source
from arena_evaluation.processing.outcome import outcome_verdict
from arena_evaluation.storage.contracts import enforce_benchmark_contract
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.layout import (
    check_benchmark_layout,
    episode_is_complete,
    retire_legacy_outputs,
    write_benchmark_stamp,
    write_episode_stamp,
)
from arena_evaluation.storage.manifest import MetadataWriter
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

_EpisodeStatus = MutableMapping[int, tuple[str, str, str, int, int, float]]

# Frames shared across robots or scoped to the whole episode, never per-robot sample data.
_SHARED_FRAMES = frozenset(
    {"tf", "tf_static", "world_semantics", "episode_lifecycle", "episode_integrity"}
)
KEPT_EMPTY_TABLES = ("robot_collision_contacts", "pedestrian_gestures")

# The robot track's registry columns that every episode-scope calculator reads off the aligned frame.
_TRACK_COLUMNS: typing.Final[tuple[str, ...]] = (
    "x",
    "y",
    "yaw",
    "speed_mps",
    "omega_z",
)

# EpisodeRecord.outcome_state -> (result, success)
_STATUS_EVALUATED = "evaluated"

_RECORD_RUNNING = 1  # arena_runtime_msgs/msg/EpisodeRecord.RUNNING
_RECORD_SUCCESS = 2  # arena_runtime_msgs/msg/EpisodeRecord.SUCCESS


def _frame(table: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    """The table collected once, None when it was not recorded."""
    if table is None:
        return None
    return table.collect() if isinstance(table, pl.LazyFrame) else table


def _report_status(
    status: _EpisodeStatus | None,
    ep: EpisodeDescriptor,
    label: str,
    index: int = 1,
    total: int = 1,
) -> None:
    """Publish this episode's progress row; a vanished manager dict is not an error."""
    if status is None:
        return
    try:
        status[ep.episode_id] = (
            ep.planner,
            ep.stage,
            label,
            index,
            total,
            time.perf_counter(),
        )
    except Exception:
        pass


def _clear_status(status: _EpisodeStatus | None, episode_id: int) -> None:
    if status is None:
        return
    try:
        status.pop(episode_id, None)
    except Exception:
        pass


def _clipped(
    frame: pl.DataFrame,
    start_ns: int | None,
    end_ns: int | None,
    pad: int = 0,
) -> pl.DataFrame:
    """The frame's rows inside the bounded window, padded by ``pad`` nanoseconds on both sides."""
    if start_ns is not None:
        frame = frame.filter(pl.col("t_ns") >= start_ns - pad)
    if end_ns is not None:
        frame = frame.filter(pl.col("t_ns") <= end_ns + pad)
    return frame


def _merge_frames(
    groups: list[dict[str, pl.DataFrame]], how: str
) -> dict[str, pl.DataFrame]:
    """Concatenate per-robot frames by table stem."""
    parts: dict[str, list[pl.DataFrame]] = collections.defaultdict(list)
    for group in groups:
        for stem, frame in group.items():
            parts[stem].append(frame)
    return {
        stem: frames[0] if len(frames) == 1 else pl.concat(frames, how=how)
        for stem, frames in parts.items()
    }


def _merge_series_frames(
    groups: list[dict[str, pl.DataFrame]],
) -> dict[str, pl.DataFrame]:
    """Concatenate per-robot series frames by table stem."""
    return _merge_frames(groups, "vertical_relaxed")


def _merge_fact_tables(groups: list[ScopedFactTables]) -> ScopedFactTables | None:
    """Concatenate the fact tables of the robots evaluated for one episode."""
    if not groups:
        return None
    from arena_evaluation.processing.scoped_evaluator import ScopedFactTables

    return ScopedFactTables(
        **_merge_frames([group.by_name() for group in groups], "vertical")
    )


def _worker_init():
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


def _episode_pipeline(data_root_str: str) -> ProcessingPipeline:
    """A single-worker pipeline rooted at the worker's data root."""
    from arena_evaluation.processing.pipeline import ProcessingPipeline

    return ProcessingPipeline(
        FolderManager(data_root=pathlib.Path(data_root_str)),
        profiler=None,
        workers=1,
    )


def _extract_job(
    data_root_str: str,
    force_extract: bool,
    ep: EpisodeDescriptor,
    status_dict: _EpisodeStatus,
) -> int:
    """Extract one episode's MCAP in a spawned worker."""
    _report_status(status_dict, ep, "Extracting MCAP / Topics")
    try:
        _episode_pipeline(data_root_str).extract_episode(
            ep, force_extract=force_extract
        )
    finally:
        _clear_status(status_dict, ep.episode_id)
    return ep.episode_id


def _process_job(
    data_root_str: str,
    force_extract: bool,
    force_process: bool,
    ep: EpisodeDescriptor,
    status_dict: _EpisodeStatus,
) -> tuple[int, EpisodeEvaluation]:
    """Evaluate one episode in a spawned worker, keeping a benchmark-refusing failure fatal."""
    _report_status(status_dict, ep, "Loading Topics", 0, 18)
    try:
        pipeline = _episode_pipeline(data_root_str)
        try:
            evaluation = pipeline.evaluate_episode(
                ep,
                force_extract=force_extract,
                status_dict=status_dict,
                force_process=force_process,
            )
        except MetricInputError:
            # A required input is missing: the benchmark must refuse the episode, so the failure
            # travels to process_benchmark instead of becoming a status row.
            _log.exception(
                f"episode_{ep.episode_id:03d}: required metric input missing"
            )
            raise
        except Exception as e:
            _log.exception(f"episode_{ep.episode_id:03d}: metrics failed")
            evaluation = _failed_evaluation(ep, e)
    finally:
        _clear_status(status_dict, ep.episode_id)
    return ep.episode_id, evaluation


def _align_bundle(
    bundle: TopicBundle, start_t_ns: int | None = None, end_t_ns: int | None = None
) -> pl.DataFrame | None:
    """Align the robot's recorded tables onto its native /tf track in registry vocabulary.

    The source ``processing.native_bundle`` module is absent at the PR1 tip; its one alignment
    implementation is carried here because every episode-scope calculator reads ``episode.data``.
    """
    track = _frame(bundle.robot_tracks)
    if track is None or track.height == 0:
        return None
    frame = _clipped(track, start_t_ns, end_t_ns)
    available = set(frame.columns)
    frame = frame.select(
        [pl.col("t_ns"), *(pl.col(c) for c in _TRACK_COLUMNS if c in available)]
    ).sort("t_ns")
    # A recording can carry a null or NaN sample; no metric may read one as a pose.
    frame = frame.filter(
        pl.col("x").is_not_null()
        & ~pl.col("x").is_nan()
        & pl.col("y").is_not_null()
        & ~pl.col("y").is_nan()
        & pl.col("yaw").is_not_null()
        & ~pl.col("yaw").is_nan()
    ).sort("t_ns")
    if frame.height == 0:
        return None
    for field_name, columns in _SECONDARY_TABLES:
        table = _frame(vars(bundle)[field_name])
        if table is None or table.height == 0:
            continue
        table = _clipped(table, start_t_ns, end_t_ns, TRACK_ALIGN_TOLERANCE_NS)
        if table.height == 0:
            continue
        available = set(table.columns)
        selected = table.select(
            [pl.col("t_ns"), *(pl.col(c) for c in columns if c in available)]
        ).sort("t_ns")
        frame = asof_join(frame, selected, tolerance_ns=TRACK_ALIGN_TOLERANCE_NS)
    markers = _frame(bundle.robot_characterization_phase)
    if markers is not None and markers.height > 0:
        selected = markers.select(["t_ns", "label"]).sort("t_ns")
        frame = asof_join(frame, selected, tolerance_ns=None)
    return frame


_SECONDARY_TABLES: typing.Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("robot_scan_summary", ("scan_min_m", "range_max_m", "n_valid_rays")),
    (
        "robot_cmd_vel",
        ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"),
    ),
    (
        "robot_collision_events",
        ("event_count", "wall_count", "static_count", "ped_count"),
    ),
    ("robot_collision_monitor", ("action_type", "polygon_name")),
    (
        "robot_power",
        (
            "total_power_w",
            "static_power_w",
            "total_mech_power_w",
            "total_thermal_power_w",
        ),
    ),
    ("robot_energy", ("total_energy_consumed_wh", "battery_soc_percent")),
    (
        "robot_acoustics",
        (
            "total_level_af_dba",
            "total_level_zf_db",
            "baseline_level_dba",
            "drivetrain_level_dba",
            "uncertainty_1sigma_dba",
            "validity_flags",
        ),
    ),
)


def _metric_world(metadata: RunMetadata | None, ep: EpisodeDescriptor) -> str | None:
    """World name the world-geometry metrics resolve their asset by, None when unknown."""
    if metadata is not None and metadata.map:
        return metadata.map
    if ep.map and ep.map != "unknown":
        return ep.map
    return None


def _episode_window(
    record: pl.DataFrame | pl.LazyFrame | None,
) -> tuple[int | None, int | None]:
    """[RUNNING stamp, terminal stamp] of the episode record, a bound is None when its row is missing."""
    df = _frame(record)
    if df is None or len(df) == 0:
        return None, None
    df = _episode_rows(df)
    states = df["outcome_state"].to_numpy()
    stamps = df["t_ns"].to_numpy()
    start = next(
        (int(t) for t, s in zip(stamps, states, strict=True) if s == _RECORD_RUNNING),
        None,
    )
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
    # The native /tf track is map frame.
    start_pos = (
        [first_row["x"], first_row["y"], first_row.get("yaw", 0.0)]
        if first_row.get("x") is not None and first_row.get("y") is not None
        else []
    )
    goal_pos = (
        [last_row["x"], last_row["y"], last_row.get("yaw", 0.0)]
        if last_row.get("x") is not None and last_row.get("y") is not None
        else []
    )

    plan_df = _frame(plan)
    if plan_df is not None and len(plan_df) > 0:
        # A recording can outlive its episode by a beat and catch the next episode's first plan.
        plan_df = plan_df.filter(pl.col("t_ns") <= aligned_df["t_ns"].max())
        if len(plan_df) > 0:
            latest_plan = plan_df.filter(
                pl.col("plan_id") == plan_df["plan_id"].max()
            ).sort("waypoint_idx")
            goal_pos = [
                latest_plan["x"][-1],
                latest_plan["y"][-1],
                latest_plan["yaw"][-1],
            ]

    gotos = [pose for pose in (phases or {}).get("map_poses", []) if pose is not None]
    if gotos:
        goal_pos = [float(v) for v in gotos[-1]]
    return start_pos, goal_pos


def _record_json_values(
    record: pl.DataFrame | pl.LazyFrame, column: str
) -> typing.Iterator[typing.Any]:
    """The parsed non-empty JSON values of the episode's own records, newest first."""
    frame = _frame(record)
    assert frame is not None
    for raw in reversed(_episode_rows(frame)[column].to_numpy()):
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (ValueError, AttributeError):
            continue
        if parsed:
            yield parsed


def _robot_phases(record: pl.DataFrame | pl.LazyFrame, robot_dir: str) -> dict | None:
    """This robot's {phases, conditions, map_poses, goal_inputs, instructions} from the last EpisodeRecord row that carries any, None without."""
    from arena_evaluation.processing.metrics.ecological.condition_metrics import (
        bare_robot_name,
    )

    bare = bare_robot_name(robot_dir)
    return next(
        (
            entry
            for value in _record_json_values(record, "phases_yaml")
            if isinstance(value, dict) and (entry := value.get(bare)) is not None
        ),
        None,
    )


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
        frame = _clipped(frame, start_ns, end_ns, TRACK_ALIGN_TOLERANCE_NS)
        if len(frame) > 0:
            fleet[name] = frame.with_columns(
                radius_m=pl.lit(
                    float(params_by_robot[name].robot_radius), dtype=pl.Float64
                )
            )
    return fleet


def _calculator_topic_names(bundle: TopicBundle) -> set[str]:
    """The bundle's carried tables, already named as the calculators read them, so a topic gate sees what the bundle holds."""
    return set(bundle.available())


# Per-pedestrian constants only the engine table records; joined onto the arena rows by ped_id.
_PEDESTRIAN_CONSTANT_COLUMNS: typing.Final[tuple[str, ...]] = (
    "radius_m",
    "robot_awareness",
    "robot_fov_gain",
    "robot_repulsion_gain",
    "robot_reaction_gain",
)


def _with_pedestrian_constants(
    topics: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """The arena pedestrian rows with the engine table's latched constants joined on by ped_id, so the track frame carries the recorded radius and crowd settings."""
    arena = topics.get(PEDESTRIAN_ARENA_PEDS_TOPIC)
    engine = topics.get(PEDESTRIAN_AGENT_STATES_TOPIC)
    if arena is None or engine is None or engine.height == 0:
        return topics
    available = [
        column for column in _PEDESTRIAN_CONSTANT_COLUMNS if column in engine.columns
    ]
    constants = engine.group_by("ped_id").agg(
        [pl.col(column).drop_nulls().first() for column in available]
    )
    arena_rows = arena.drop([column for column in available if column in arena.columns])
    return {
        **topics,
        PEDESTRIAN_ARENA_PEDS_TOPIC: arena_rows.join(
            constants, on="ped_id", how="left", maintain_order="left"
        ),
    }


def _robot_identity(bundle: TopicBundle) -> tuple[str, RobotParams]:
    """This robot's recorded identity and parameters, never the installed definition or a default."""
    from arena_evaluation.storage.schemas import robot_params_from_recorded

    observed = [
        _frame(table)
        for table in (bundle.robot_descriptor, bundle.robot_tracks, bundle.robot_odom)
    ]
    robot_id = next(
        (
            str(frame.get_column("robot_id")[0])
            for frame in observed
            if frame is not None and frame.height
        ),
        None,
    )
    if robot_id is None:
        raise MetricInputError(
            "a robot bundle has no recorded table that names its robot"
        )
    recorded = [
        _frame(table) for table in (bundle.robot_descriptor, bundle.robot_params)
    ]
    rows = (
        pl.concat([frame for frame in recorded if frame is not None], how="diagonal")
        if any(frame is not None for frame in recorded)
        else pl.DataFrame()
    )
    return robot_id, robot_params_from_recorded(rows, robot_id)


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
    states = df["outcome_state"].to_numpy()
    started = next(
        (i for i, state in enumerate(states) if state == _RECORD_RUNNING), None
    )
    if started is None:
        return df
    if df["episode_id"][started] is not None:
        # the previous episode's latched result is not this episode's record
        df = df.filter(pl.col("episode_id") == df["episode_id"][started])
        states = df["outcome_state"].to_numpy()
        started = next(i for i, state in enumerate(states) if state == _RECORD_RUNNING)
    end = next(
        (
            i
            for i in range(started, len(states))
            if states[i] is not None and states[i] >= _RECORD_SUCCESS
        ),
        None,
    )
    return df if end is None else df.head(end + 1)


def _record_conditions(record: pl.DataFrame | pl.LazyFrame) -> list[dict] | None:
    """The episode's conditions: the last of its own records whose `conditions` JSON is a non-empty list."""
    return next(_record_json_values(record, "conditions_yaml"), None)


def _record_outcome(
    record: pl.DataFrame | pl.LazyFrame | None, metadata: RunMetadata | None
) -> tuple[int | None, str | None]:
    """(outcome_state, outcome_info): the episode's own terminal record, else the runner's yaml outcome, else the last record."""
    df = _frame(record)
    has_rows = df is not None and len(df) > 0
    if has_rows:
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
    if has_rows:
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
    if df is not None and len(df) > 0 and reasons is not None and len(reasons) > 0:
        last_stamp = df["t_ns"].max()
        return [
            str(v)
            for v in reasons.filter(pl.col("t_ns") == last_stamp)
            .sort("flag_idx")["flag"]
            .to_numpy()
        ]
    return list(metadata.integrity) if metadata is not None else []


def _episode_identity(
    ep: EpisodeDescriptor,
    metadata: RunMetadata | None,
    robot: str,
    *,
    start: typing.Any,
    goal: typing.Any,
) -> dict[str, typing.Any]:
    """The identity columns carried by every row of one episode, evaluated or not."""
    from arena_evaluation.storage.planner_names import split_planner_name

    if metadata is not None and metadata.local_planner:
        local_planner = metadata.local_planner
        inter_planner = metadata.inter_planner or ""
    else:
        local_planner, inter_planner = split_planner_name(ep.planner)
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
    counts = collections.Counter(row.get("status") or "unknown" for row in rows)
    print(
        "Episode status: "
        + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())),
        flush=True,
    )
    flagged = sorted(
        (row for row in rows if row.get("status") != _STATUS_EVALUATED),
        key=lambda row: (str(row.get("stage")), row.get("episode", 0)),
    )
    for row in flagged:
        print(
            f"  episode {row.get('episode')} {row.get('planner')}/{row.get('stage')} [{row.get('robot')}] {row.get('status')}: {row.get('status_reason')}",
            flush=True,
        )


def _metadata_robot(
    metadata: RunMetadata | None, bundles: dict[str, TopicBundle] | None = None
) -> str:
    """Bundle-style robot key (env_0_jackal) for a row without a robot bundle."""
    model = (
        metadata.robot_model[0].partition("[")[0]
        if metadata is not None and metadata.robot_model
        else ""
    )
    env = next(
        (m.group(0) for name in (bundles or {}) if (m := re.match(r"env_\d+", name))),
        None,
    )
    return f"{env}_{model}" if env and model else model


def _suite_metrics(
    episode_dir: pathlib.Path, data_root: pathlib.Path
) -> dict[str, typing.Any]:
    """Suite-level metric overrides from the run's manifest.yaml, {} for bare run dirs and junk values."""
    import yaml

    for parent in episode_dir.parents:
        manifest_path = parent / "manifest.yaml"
        if manifest_path.exists():
            try:
                suite = yaml.safe_load(manifest_path.read_text()).get("suite") or {}
                raw = suite.get("metrics") or {}
            except Exception as e:
                _log.warning(
                    f"{manifest_path}: unreadable, ignoring suite metrics: {e!r}"
                )
                return {}
            metrics: dict[str, typing.Any] = {}
            v = raw.get("max_collisions")
            if v is not None:
                try:
                    metrics["max_collisions"] = int(v)
                except (TypeError, ValueError):
                    _log.warning(
                        f"{manifest_path}: suite metrics.max_collisions {v!r} is not an int, ignoring"
                    )
            if (barn := raw.get("barn")) is not None:
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
    return _status_evaluation(
        ep, pathlib.Path(ep.episode_dir), None, "", "error", repr(error)
    )


def _finish_evaluation(
    episode_dir: pathlib.Path,
    ep_id: int,
    rows: list[dict],
    facts: list[ScopedFactTables] | None = None,
    series: list[dict[str, pl.DataFrame]] | None = None,
) -> EpisodeEvaluation:
    """Fold the scoped fact tables into the evaluation and publish its series tables."""
    metrics_dir = episode_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_series_tables(_merge_series_frames(series or []), metrics_dir)
    return EpisodeEvaluation(
        episode_id=ep_id,
        rows=rows,
        metrics_dir=metrics_dir,
        facts=_merge_fact_tables(facts or []),
    )


def _status_evaluation(
    ep: EpisodeDescriptor,
    episode_dir: pathlib.Path,
    metadata: RunMetadata | None,
    robot: str,
    status: str,
    reason: str,
    record: pl.DataFrame | pl.LazyFrame | None = None,
) -> EpisodeEvaluation:
    """An episode's single status row, published without fact tables."""
    return _finish_evaluation(
        episode_dir,
        ep.episode_id,
        [_status_row(ep, metadata, robot, status, reason, record)],
    )


class RunProgress(typing.Protocol):
    """Reports one phase's progress; the CLI supplies a terminal display, without one the phase runs silently."""

    def log_error(self, episode_id: int, message: str) -> None: ...

    def log_completed(self, episode_id: int, label: str, seconds: float) -> None: ...


RunProgressFactory = Callable[
    [str, int, int, _EpisodeStatus], contextlib.AbstractContextManager[RunProgress]
]


class _SilentProgress:
    def log_error(self, episode_id: int, message: str) -> None:
        """Errors reach the caller through on_failed."""

    def log_completed(self, episode_id: int, label: str, seconds: float) -> None:
        """Completions reach the caller through on_done."""


@contextlib.contextmanager
def _silent_progress(
    title: str, episode_count: int, workers: int, status: _EpisodeStatus
) -> typing.Iterator[RunProgress]:
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

    def drain(
        display: RunProgress,
        pending: typing.Iterable[
            tuple[EpisodeDescriptor, float, Callable[[], typing.Any]]
        ],
    ) -> None:
        for ep, started, result_of in pending:
            try:
                result = result_of()
            except MetricInputError:
                raise
            except Exception as error:
                on_failed(ep, error)
                display.log_error(ep.episode_id, str(error))
                continue
            on_done(ep, result)
            display.log_completed(
                ep.episode_id,
                f"{ep.planner}/{ep.stage}",
                time.perf_counter() - started,
            )

    if workers <= 1:
        local_status: _EpisodeStatus = {}
        with progress(title, len(episodes), 1, local_status) as display:
            drain(
                display,
                (
                    (ep, time.perf_counter(), functools.partial(job, ep, local_status))
                    for ep in episodes
                ),
            )
        return

    mp_context = multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    status_dict = manager.dict()
    with progress(title, len(episodes), workers, status_dict) as display:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp_context, initializer=_worker_init
        )
        try:
            futures = {
                executor.submit(job, ep, status_dict): (ep, time.perf_counter())
                for ep in episodes
            }
            drain(
                display,
                (
                    (*futures[future], future.result)
                    for future in concurrent.futures.as_completed(futures)
                ),
            )
        except KeyboardInterrupt:
            print(
                f"\n[KeyboardInterrupt] {title} interrupted by user. Terminating workers...",
                flush=True,
            )
            raise
        finally:
            _shutdown_executor_cleanly(executor)


class PhaseProfiler(typing.Protocol):
    """Times the named phases of a run; benchmark.profiler.PipelineProfiler implements it."""

    def phase(self, name: str) -> contextlib.AbstractContextManager[None]: ...


def _phase_context(
    profiler: PhaseProfiler | None, name: str
) -> contextlib.AbstractContextManager[None]:
    """The profiler's phase, or a no-op context when the run carries no profiler."""
    return profiler.phase(name) if profiler is not None else contextlib.nullcontext()


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
        load_metric_definitions()
        self.folder_manager = folder_manager
        self.profiler = profiler
        self.progress = progress
        # -1 (or None) = auto-detect the CPU count; any value < 1 falls back to it.
        self.workers = (
            (os.cpu_count() or 1) if (workers is None or workers < 1) else workers
        )

    def extract_episode(
        self, ep: EpisodeDescriptor, force_extract: bool = False
    ) -> dict[str, TopicBundle] | None:
        """Extract topics from a single episode MCAP into episode_dir/topics/."""
        episode_dir = pathlib.Path(ep.episode_dir)
        mcap_path = self.folder_manager.mcap_path_for_episode(episode_dir)
        topics_dir = self.folder_manager.extracted_topics_path_for_episode(episode_dir)

        if not mcap_path.exists():
            _log.warning(
                f"episode_{ep.episode_id:03d} ({ep.planner}/{ep.stage}): no MCAP found, so it is not extracted"
            )
            return None

        if not force_extract and TopicParquetStore.is_current(topics_dir):
            bundles = TopicParquetStore.read(topics_dir)
            if bundles:
                return bundles

        with _phase_context(self.profiler, "extract"):
            if topics_dir.exists():
                shutil.rmtree(topics_dir)
            bundles = MCAPReader(mcap_path).read(topics_dir)
            TopicParquetStore.mark_complete(topics_dir)
            return bundles

    def _describe_run_dir(self, run_dir: pathlib.Path) -> EpisodeDescriptor:
        ep = self.folder_manager.describe_episode(pathlib.Path(run_dir))
        if ep is None:
            raise FileNotFoundError(
                f"{run_dir} is not an episode directory (episode_NNN with episode_NNN.yaml)"
            )
        return ep

    def extract_run_dir(
        self, run_dir: pathlib.Path, force_extract: bool = False
    ) -> dict[str, TopicBundle] | None:
        """Extract one `episode_NNN` directory into its topics/ folder."""
        return self.extract_episode(
            self._describe_run_dir(run_dir), force_extract=force_extract
        )

    def process_run_dir(
        self,
        run_dir: pathlib.Path,
        force_extract: bool = False,
        *,
        force_process: bool = False,
    ) -> pathlib.Path | None:
        """Extract, evaluate and publish one `episode_NNN` directory."""
        episode = self._describe_run_dir(run_dir)
        self.process_episode(
            episode, force_extract=force_extract, force_process=force_process
        )
        out = pathlib.Path(run_dir) / "metrics"
        if not out.exists():
            return None
        return out

    @staticmethod
    def _evaluate_robot_bundle(
        ep: EpisodeDescriptor,
        metadata: RunMetadata | None,
        suite_metrics_cfg: dict[str, typing.Any],
        robots: dict[str, TopicBundle],
        params_by_robot: dict[str, RobotParams],
        robot_name: str,
        bundle: TopicBundle,
        pedsim_avail: bool,
        status_dict: _EpisodeStatus | None,
    ) -> tuple[dict, ScopedFactTables | None, dict[str, pl.DataFrame] | None]:
        """Align and evaluate one robot's TopicBundle within an episode."""
        from arena_evaluation.processing.scoped_evaluator import ScopedEvaluator

        robot_params = params_by_robot[robot_name]
        window = _episode_window(bundle.episode_lifecycle)
        tracks, pose_source = resolve_pose_source(bundle, window)
        aligned_df = None if tracks is None else _align_bundle(bundle, *window)
        row_count = 0 if aligned_df is None else len(aligned_df)
        if tracks is None or row_count < 5:
            reason = (
                "no tf ground-truth rows inside the episode window"
                if tracks is None
                else f"{row_count} ground-truth rows inside the episode window"
            )
            if tracks is not None:
                _log.warning(f"episode_{ep.episode_id:03d} [{robot_name}]: {reason}")
            return (
                _status_row(
                    ep,
                    metadata,
                    robot_name,
                    "no_trajectory",
                    reason,
                    bundle.episode_lifecycle,
                    bundle.episode_integrity,
                ),
                None,
                None,
            )

        robot_pedsim = (
            pedsim_avail
            or bundle.pedestrian_arena_peds is not None
            or bundle.pedestrian_agent_states is not None
        )
        available_topics = _calculator_topic_names(bundle)

        t0 = int(aligned_df["t_ns"][0])
        t1 = int(aligned_df["t_ns"][-1])
        native = {
            name: frame
            for name in sorted(bundle.available() - _SHARED_FRAMES)
            if (frame := _frame(vars(bundle)[name])) is not None
            and (len(frame) > 0 or name in KEPT_EMPTY_TABLES)
        }
        topics = {
            name: _clipped(df, t0, t1, TRACK_ALIGN_TOLERANCE_NS)
            for name, df in _with_pedestrian_constants(native).items()
        }

        peds_df = _frame(bundle.pedestrian_arena_peds)
        if peds_df is None or peds_df.height == 0:
            peds_df = _frame(bundle.pedestrian_agent_states)
        peds_df = (
            None
            if peds_df is None
            else _clipped(
                peds_df,
                None if window[0] is None else window[0] - TRACK_ALIGN_TOLERANCE_NS,
                window[1],
            )
        )
        num_pedestrians = (
            0
            if peds_df is None or len(peds_df) == 0
            else int(peds_df.group_by("t_ns").len()["len"].max() or 0)
        )

        conditions = None
        phases = None
        outcome_state = None
        outcome_info = None
        er = _frame(bundle.episode_lifecycle)
        if er is not None and len(er) > 0:
            conditions = _record_conditions(er)
            phases = _robot_phases(er, robot_name)
            outcome_state, outcome_info = _record_outcome(er, metadata)

        integrity = _record_integrity(
            bundle.episode_lifecycle, bundle.episode_integrity, metadata
        )
        start_pos, goal_pos = _episode_endpoints(aligned_df, bundle.robot_plans, phases)

        aligned_ep = AlignedEpisodeBundle(
            episode_id=ep.episode_id,
            data=aligned_df,
            start_pos=start_pos,
            goal_pos=goal_pos,
            num_pedestrians=num_pedestrians,
            robot_name=robot_name,
            semantic_snapshot=_frame(bundle.world_semantics),
            conditions=conditions,
            phases=phases,
            fleet=_fleet_poses(
                robots, window, ego_robot_id=robot_name, params_by_robot=params_by_robot
            ),
            outcome_state=outcome_state,
            outcome_info=outcome_info,
            run=metadata,
            peds=peds_df,
            map=ep.map,
            topics=topics,
            run_dir=pathlib.Path(ep.episode_dir).parent.parent,
        )

        def on_calc_progress(calc_name: str, calc_idx: int, total_calcs: int) -> None:
            _report_status(status_dict, ep, calc_name, calc_idx, total_calcs)

        evaluated = ScopedEvaluator(
            robot_params, metrics_config=suite_metrics_cfg
        ).evaluate(
            aligned_ep,
            run_id=ep.benchmark_id,
            episode_id=ep.episode_id,
            world=_metric_world(metadata, ep),
            episode_context={
                **_episode_identity(
                    ep,
                    metadata,
                    aligned_ep.robot_name or robot_name,
                    start=aligned_ep.start_pos,
                    goal=aligned_ep.goal_pos,
                ),
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
        return evaluated.episode_metrics, evaluated.facts, evaluated.series

    def evaluate_episode(
        self,
        ep: EpisodeDescriptor,
        force_extract: bool = False,
        *,
        status_dict: _EpisodeStatus | None = None,
        force_process: bool = False,
    ) -> EpisodeEvaluation:
        """Extract and evaluate one episode, holding its fact tables for the publication step."""
        episode_dir = pathlib.Path(ep.episode_dir)
        metrics_dir = episode_dir / "metrics"

        if not (force_extract or force_process) and episode_is_complete(metrics_dir):
            return EpisodeEvaluation(
                episode_id=ep.episode_id, rows=[], metrics_dir=metrics_dir
            )

        yaml_path = episode_dir / f"{episode_dir.name}.yaml"
        if not yaml_path.exists():
            yaml_path = episode_dir / "metadata.yaml"
        try:
            metadata = MetadataWriter.read(yaml_path) if yaml_path.exists() else None
        except Exception as e:
            _log.warning(f"episode_{ep.episode_id:03d}: could not read metadata: {e}")
            metadata = None
        bundles = self.extract_episode(ep, force_extract=force_extract)
        if not bundles:
            return _status_evaluation(
                ep,
                episode_dir,
                metadata,
                _metadata_robot(metadata),
                "no_recording",
                "no MCAP or empty extraction",
            )

        with _phase_context(self.profiler, "process"):
            pedsim_avail = (
                bool(metadata.pedsim_available) if metadata is not None else False
            )
            suite_metrics_cfg = _suite_metrics(
                episode_dir, self.folder_manager.data_root
            )
            robots = {
                name: bundle
                for name, bundle in bundles.items()
                if bundle.robot_tracks is not None or bundle.robot_odom is not None
            }
            if not robots:
                any_record = next(
                    (
                        b.episode_lifecycle
                        for b in bundles.values()
                        if b.episode_lifecycle is not None
                    ),
                    None,
                )
                return _status_evaluation(
                    ep,
                    episode_dir,
                    metadata,
                    _metadata_robot(metadata, bundles),
                    "no_trajectory",
                    "no tf ground-truth or odometry rows",
                    any_record,
                )

            params_by_robot = {
                name: _robot_identity(bundle)[1] for name, bundle in robots.items()
            }
            evaluated = [
                self._evaluate_robot_bundle(
                    ep,
                    metadata,
                    suite_metrics_cfg,
                    robots,
                    params_by_robot,
                    robot_name,
                    bundle,
                    pedsim_avail,
                    status_dict,
                )
                for robot_name, bundle in robots.items()
            ]
            return _finish_evaluation(
                episode_dir,
                ep.episode_id,
                [row for row, _facts, _series in evaluated],
                [facts for _row, facts, _series in evaluated if facts is not None],
                [series for _row, _facts, series in evaluated if series is not None],
            )

    def process_episode(
        self,
        ep: EpisodeDescriptor,
        force_extract: bool = False,
        status_dict: _EpisodeStatus | None = None,
        *,
        force_process: bool = False,
    ) -> list[dict]:
        """Extract, evaluate and publish one episode."""
        evaluation = self.evaluate_episode(
            ep,
            force_extract=force_extract,
            status_dict=status_dict,
            force_process=force_process,
        )
        self.publish_evaluation(evaluation)
        return evaluation.rows

    @staticmethod
    def publish_evaluation(evaluation: EpisodeEvaluation) -> None:
        """Write one evaluation's fact tables, then stamp its metrics directory as complete."""
        if evaluation.facts is not None:
            write_fact_tables(evaluation.facts, evaluation.metrics_dir)
        write_episode_stamp(evaluation.metrics_dir)

    def extract_benchmark(self, benchmark_id: str, force_extract: bool = False) -> None:
        """Extract all episodes of a benchmark, then write its dimensions once."""
        episodes = self.folder_manager.discover_episodes(benchmark_id)
        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return
        _run_phase(
            f"Phase 1: Extracting MCAP topics for {benchmark_id}",
            episodes,
            self.workers,
            functools.partial(
                _extract_job, str(self.folder_manager.data_root), force_extract
            ),
            progress=self.progress,
            on_done=lambda _ep, _episode_id: None,
            on_failed=lambda _ep, _error: None,
        )
        self._write_benchmark_dimensions(benchmark_id, episodes)

    def _write_benchmark_dimensions(
        self, benchmark_id: str, episodes: list[EpisodeDescriptor]
    ) -> None:
        """Stack the dimension rows of every extracted episode, add the metric dictionary and the geometry of every named world, and write each dimension once at the benchmark root."""
        contributions = []
        for ep in episodes:
            episode_dir = pathlib.Path(ep.episode_dir)
            topics = episode_dir / "topics"
            if not topics.is_dir():
                _log.warning(
                    f"episode_{ep.episode_id:03d}: no extracted topics, so its dimension rows are missing"
                )
                continue
            metadata = MetadataWriter.read(episode_dir / f"{episode_dir.name}.yaml")
            bundles = load_bundles(topics)
            contributions.append(
                episode_dimensions(
                    run_id=benchmark_id,
                    episode_id=ep.episode_id,
                    metadata=metadata,
                    bundles=bundles,
                )
            )
        benchmark_dir = self.folder_manager.benchmark_dir(benchmark_id)
        dimensions = stack(contributions)
        dimensions[DimMetrics.table_name] = metric_dictionary()
        dimensions[DimWorldGeometry.table_name] = benchmark_world_geometry(
            sorted(
                {
                    world
                    for ep in episodes
                    if (world := _metric_world(None, ep)) is not None
                }
            ),
            benchmark_dir,
        )
        enforce_benchmark_contract(benchmark_dir, dimensions=dimensions)
        write_benchmark_dimensions(dimensions, benchmark_dir)
        write_benchmark_stamp(benchmark_dir)

    def process_benchmark(
        self,
        benchmark_id: str,
        force_extract: bool = False,
        *,
        force_process: bool = False,
    ) -> pathlib.Path | None:
        """Process all episodes into their fact tables, returning the benchmark directory, None when nothing was found."""
        benchmark_dir = self.folder_manager.benchmark_dir(benchmark_id)
        if force_process:
            for moved in retire_legacy_outputs(benchmark_dir):
                print(f"Retired legacy output: {moved}")
        else:
            check_benchmark_layout(benchmark_dir)

        episodes = self.folder_manager.discover_episodes(benchmark_id)

        if not episodes:
            print(f"No episodes found for benchmark '{benchmark_id}'")
            return None

        with _phase_context(self.profiler, "extract"):
            self.extract_benchmark(benchmark_id, force_extract=force_extract)

        evaluations: list[EpisodeEvaluation] = []

        def on_failed(ep: EpisodeDescriptor, error: Exception) -> None:
            evaluations.append(_failed_evaluation(ep, error))

        with _phase_context(self.profiler, "process"):
            _run_phase(
                f"Phase 2: Calculating metrics across {len(episodes)} episodes",
                episodes,
                self.workers,
                functools.partial(
                    _process_job,
                    str(self.folder_manager.data_root),
                    force_extract,
                    force_process,
                ),
                progress=self.progress,
                on_done=lambda _ep, result: evaluations.append(result[1]),
                on_failed=on_failed,
            )

        _print_status_summary(
            [row for evaluation in evaluations for row in evaluation.rows]
        )
        for evaluation in evaluations:
            self.publish_evaluation(evaluation)
        benchmark_dir = self.folder_manager.benchmark_dir(benchmark_id)
        enforce_benchmark_contract(benchmark_dir)
        write_benchmark_stamp(benchmark_dir)
        print(f"Done. {len(episodes)} episodes -> {benchmark_dir}")
        return benchmark_dir
