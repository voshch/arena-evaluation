"""MCAP ingestion coordinator: streams recorded messages into flat telemetry tables and loads per-robot topic bundles."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import re
import typing
from collections import defaultdict
from collections.abc import Callable, Mapping

import polars as pl
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

from arena_evaluation.processing.telemetry import context, events, sensors, tracks
from arena_evaluation.processing.telemetry.messages import (
    Envelope,
    RobotRef,
    env_id_in,
    is_controller_topic,
    robot_id_of,
    topic_parts,
    yaw_from_orientation,
)
from arena_evaluation.processing.telemetry.peds import ARENA_TABLE, ENGINE_TABLE, PedestrianRecorder
from arena_evaluation.processing.telemetry.tracks import Pose3, is_base_frame, is_ground_truth, is_world_frame
from arena_evaluation.processing.telemetry.writer import ENV_TABLES, FLUSH_ROW_LIMIT, GLOBAL_TABLES, ROBOT_TABLES, TableSink
from arena_evaluation.storage.schemas import TopicBundle

if typing.TYPE_CHECKING:
    from mcap.records import Schema
    from tf2_msgs.msg import TFMessage

_log = logging.getLogger(__name__)
Anchor = tuple[int, float, float]
Predicate = Callable[[str, "Schema"], bool]
Handler = Callable[[str, Envelope], None]
Decoder = Callable[[bytes], object]

_AGENT_META_SCHEMA = "arena_humansim_msgs/msg/AgentMeta"
_INTERACTION_FRAME_SCHEMA = "arena_humansim_msgs/msg/InteractionFrame"
_ANIMATION_STATES_SCHEMA = "arena_people_msgs/msg/AnimationStates"
_STATIC_ANCHOR_CHILD = re.compile(r"(env_\d+)(?:/map)?")
_MAP_ANCHOR_CHILD = re.compile(r"(env_\d+)/map")
_UNMAPPED_FRAMES = ("", "map")
_POSITION_COLUMNS: dict[str, tuple[str, str]] = {
    "robot_tracks": ("x", "y"),
    "robot_initialpose": ("pos_x", "pos_y"),
    "robot_plans": ("x", "y"),
    "pedestrian_arena_peds": ("x", "y"),
    "pedestrian_agent_states": ("x", "y"),
    "pedestrian_gestures": ("at_x", "at_y"),
}


def anchor_at(anchors: list[Anchor], t_ns: int) -> tuple[float, float]:
    """The anchor in effect at t_ns: the last one at or before it, the first one before any."""
    ox, oy = anchors[0][1:]
    for t, ax, ay in anchors[1:]:
        if t > t_ns:
            break
        ox, oy = ax, ay
    return ox, oy


def by_anchor(anchors: list[Anchor], value: Callable[[float, float], pl.Expr]) -> pl.Expr:
    """`value(ox, oy)` of the anchor in effect at each row's t_ns, as in anchor_at."""
    (_, ox, oy), *later = anchors
    if not later:
        return value(ox, oy)
    chain = None
    for t, ax, ay in reversed(later):
        cond = pl.col("t_ns") >= t
        chain = pl.when(cond).then(value(ax, ay)) if chain is None else chain.when(cond).then(value(ax, ay))
    assert chain is not None
    return chain.otherwise(value(ox, oy))


class MCAPReader:
    """Streams MCAP recordings into flat, episode-wide telemetry tables and loads one TopicBundle per robot."""

    def __init__(
        self,
        data_path: pathlib.Path,
        *,
        include_scan_rays: bool = False,
        flush_rows: int = FLUSH_ROW_LIMIT,
        environment: str | None = None,
    ) -> None:
        self._flush_rows = flush_rows
        self._environment = environment
        self.data_path = data_path
        self._include_scan_rays = include_scan_rays

    def read(self, out_dir: pathlib.Path) -> dict[str, TopicBundle]:
        """Stream every recording into flat tables under out_dir and return one TopicBundle per robot."""
        files = self._mcap_files()
        ingest = _Ingest(TableSink(out_dir, flush_rows=self._flush_rows), self._include_scan_rays, self._environment)
        try:
            for path in files:
                self._stream(path, ingest)
        finally:
            ingest.finish()
        return load_bundles(out_dir, ingest.pedestrian_frames())

    def _mcap_files(self) -> list[pathlib.Path]:
        if self.data_path.is_dir():
            files = sorted(self.data_path.glob("*.mcap"))
            if not files:
                raise FileNotFoundError(f"No MCAP file found in directory: {self.data_path}")
        else:
            files = [self.data_path]
        for path in files:
            if not path.exists():
                raise FileNotFoundError(f"MCAP file not found: {path}")
        return files

    def _stream(self, path: pathlib.Path, ingest: _Ingest) -> None:
        factory = DecoderFactory()
        decoders: dict[int, Decoder | None] = {}
        with path.open("rb") as handle:
            for schema, channel, message in NonSeekingReader(handle).iter_messages(log_time_order=False):
                if schema is None:
                    continue
                if message.channel_id not in decoders:
                    decoders[message.channel_id] = factory.decoder_for(channel.message_encoding, schema)
                    if decoders[message.channel_id] is None and ingest.accepts(channel.topic, schema):
                        _log.warning(f"{channel.topic}: schema {schema.name!r} ({schema.encoding}) cannot be decoded; its messages are skipped")
                decoder = decoders[message.channel_id]
                if decoder is None:
                    continue
                body = _decode(decoder, channel.topic, message.data)
                if body is not None:
                    ingest.route(channel.topic, schema, body, message.log_time)


def _decode(decoder: Decoder, topic: str, data: bytes) -> object | None:
    """The decoded body of a recorded message, or None when its bytes are corrupt, so the message is skipped."""
    try:
        return decoder(data)
    except Exception as error:
        _log.warning(f"Error reading MCAP message on {topic}: {error}")
        return None


class _Ingest:
    """Routes decoded messages to their recorders and writes the tables that need the whole recording."""

    def __init__(self, sink: TableSink, include_scan_rays: bool, environment: str | None = None) -> None:
        self._sink = sink
        self._environment = environment
        self._kinematics = tracks.KinematicsRecorder(sink)
        self._pedestrians = PedestrianRecorder(sink)
        self._scans = sensors.ScanRecorder(sink, include_scan_rays)
        self._plans = context.PlanRecorder(sink)
        self._schedules = context.ScheduleRecorder()
        self._routes: tuple[tuple[Predicate, Handler], ...] = (
            (_ends_with("/odom"), self._odometry),
            (_ends_with("/scan", "/lidar"), self._scan),
            (_ends_with("/cmd_vel"), self._command),
            (_ends_with("/joint_states"), self._joint_states),
            (_both(_ends_with("/agent_meta"), _has_schema(_AGENT_META_SCHEMA)), self._agent_meta),
            (_ends_with("/arena_peds", "/peds"), self._arena_peds_frame),
            (_ends_with("/agent_states", "/agent_frame"), self._agent_states_frame),
            (_both(_ends_with("/interactions"), _has_schema(_INTERACTION_FRAME_SCHEMA)), self._interactions),
            (_both(_ends_with("/animation_states"), _has_schema(_ANIMATION_STATES_SCHEMA)), self._animations),
            (_ends_with("/acoustics"), self._acoustics),
            (_ends_with("/characterization_phase"), self._characterization_phase),
            (_ends_with("/characterization_schedule"), self._characterization_schedule),
            (_ends_with("/state/episode"), self._episode),
            (_ends_with("/state/semantics"), self._semantics),
            (_ends_with("/collision_events"), self._collision_events),
            (_ends_with("/collision_monitor_state"), self._collision_monitor),
            (_ends_with("/power_publisher/power"), self._power),
            (_ends_with("/power_publisher/energy"), self._energy),
            (_ends_with("/plan", "/global_plan"), self._plan),
            (_ends_with("/task_pose"), self._task_pose),
            (_ends_with("/initialpose"), self._initial_pose),
            (_is_transform_topic, self._transforms),
        )

    def accepts(self, topic: str, schema: Schema) -> bool:
        """Whether the reader routes this recorded topic to a recorder."""
        return any(matches(topic, schema) for matches, _handle in self._routes)

    def route(self, topic: str, schema: Schema, body: object, t_ns: int) -> None:
        env_id = env_id_in(topic) or self._environment
        if env_id is None and not (_is_transform_topic(topic, schema) or topic.endswith("/state/semantics")):
            raise ValueError(f"topic {topic!r} names no environment; the native reader never guesses one")
        ref = None if env_id is None or _is_transform_topic(topic, schema) else RobotRef(env_id=env_id, robot_id=robot_id_of(topic_parts(topic), env_id))
        envelope = Envelope(t_ns=t_ns, ref=ref, schema=schema, body=body)
        for matches, handle in self._routes:
            if matches(topic, schema):
                handle(topic, envelope)
                return

    def pedestrian_frames(self) -> dict[tuple[str, str], frozenset[str]]:
        return self._pedestrians.frames()

    def finish(self) -> None:
        self._kinematics.finalize()
        self._schedules.write(self._sink)
        self._sink.close()

    def _odometry(self, topic: str, envelope: Envelope) -> None:
        table = "robot_odom_controller" if is_controller_topic(topic) else "robot_odom"
        self._kinematics.record_odometry(table, envelope)

    def _scan(self, _topic: str, envelope: Envelope) -> None:
        self._scans.record(envelope)

    def _command(self, topic: str, envelope: Envelope) -> None:
        table = "robot_cmd_vel_controller" if is_controller_topic(topic) else "robot_cmd_vel"
        self._kinematics.record_command(table, envelope)

    def _joint_states(self, _topic: str, envelope: Envelope) -> None:
        sensors.record_joint_states(self._sink, envelope)

    def _agent_meta(self, _topic: str, envelope: Envelope) -> None:
        self._pedestrians.record_meta(envelope)

    def _arena_peds_frame(self, _topic: str, envelope: Envelope) -> None:
        self._pedestrians.record_frame(ARENA_TABLE, envelope)

    def _agent_states_frame(self, _topic: str, envelope: Envelope) -> None:
        self._pedestrians.record_frame(ENGINE_TABLE, envelope)

    def _interactions(self, _topic: str, envelope: Envelope) -> None:
        events.record_interactions(self._sink, envelope)
        events.record_interaction_events(self._sink, envelope)

    def _animations(self, _topic: str, envelope: Envelope) -> None:
        events.record_animations(self._sink, envelope)

    def _acoustics(self, _topic: str, envelope: Envelope) -> None:
        sensors.record_acoustics(self._sink, envelope)

    def _characterization_phase(self, _topic: str, envelope: Envelope) -> None:
        context.record_characterization_phase(self._sink, envelope)

    def _characterization_schedule(self, _topic: str, envelope: Envelope) -> None:
        self._schedules.record(envelope)

    def _episode(self, _topic: str, envelope: Envelope) -> None:
        context.record_episode(self._sink, envelope)

    def _semantics(self, _topic: str, envelope: Envelope) -> None:
        context.record_semantics(self._sink, envelope)

    def _collision_events(self, _topic: str, envelope: Envelope) -> None:
        events.record_collision_events(self._sink, envelope)

    def _collision_monitor(self, _topic: str, envelope: Envelope) -> None:
        events.record_collision_monitor(self._sink, envelope)

    def _power(self, _topic: str, envelope: Envelope) -> None:
        sensors.record_power(self._sink, envelope)

    def _energy(self, _topic: str, envelope: Envelope) -> None:
        sensors.record_energy(self._sink, envelope)

    def _plan(self, _topic: str, envelope: Envelope) -> None:
        self._plans.record(envelope)

    def _task_pose(self, _topic: str, envelope: Envelope) -> None:
        context.record_task_pose(self._sink, envelope)

    def _initial_pose(self, _topic: str, envelope: Envelope) -> None:
        context.record_initial_pose(self._sink, envelope)

    def _transforms(self, topic: str, envelope: Envelope[TFMessage]) -> None:
        context.record_transforms(self._sink, "tf_static" if "static" in topic else "tf", envelope)
        for transform in envelope.body.transforms:
            parent, child = transform.header.frame_id, transform.child_frame_id
            if not is_ground_truth(parent, child):
                continue
            child_env = env_id_in(child)
            if child_env is None:
                raise ValueError(f"ground-truth frame {child!r} names no environment")
            ref = RobotRef(env_id=child_env, robot_id=robot_id_of(child.strip("/").split("/"), child_env))
            translation, rotation = transform.transform.translation, transform.transform.rotation
            pose = Pose3(x=float(translation.x), y=float(translation.y), z=float(translation.z), yaw=yaw_from_orientation(rotation))
            self._kinematics.record_ground_truth(Envelope(t_ns=envelope.t_ns, ref=ref, schema=envelope.schema, body=transform), pose)


def _ends_with(*suffixes: str) -> Predicate:
    return lambda topic, _schema: topic.endswith(suffixes)


def _has_schema(name: str) -> Predicate:
    return lambda _topic, schema: schema.name == name


def _both(first: Predicate, second: Predicate) -> Predicate:
    return lambda topic, schema: first(topic, schema) and second(topic, schema)


def _is_transform_topic(topic: str, _schema: Schema) -> bool:
    """The global and per-environment /tf and /tf_static topics; human skeleton transforms are never robot ground truth."""
    return topic in ("tf", "tf_static") or (topic.endswith(("/tf", "/tf_static")) and "/humans/" not in topic)


@dataclasses.dataclass(frozen=True, slots=True)
class _Tables:
    """The flat tables of one episode directory, the map anchors per environment, and the pedestrian frames seen."""

    out_dir: pathlib.Path
    global_frames: Mapping[str, pl.LazyFrame | None]
    anchors: Mapping[str, list[Anchor]]
    pedestrian_frames: Mapping[tuple[str, str], frozenset[str]]

    def scoped(self, name: str, column: str, value: str) -> pl.LazyFrame | None:
        path = self.out_dir / f"{name}.parquet"
        if not path.exists():
            return None
        frame = pl.scan_parquet(path).filter(pl.col(column) == value)
        return frame if frame.head(1).collect().height else None


def load_bundles(out_dir: pathlib.Path, pedestrian_frames: Mapping[tuple[str, str], frozenset[str]]) -> dict[str, TopicBundle]:
    """One TopicBundle per robot: the episode's flat tables filtered to that robot and its environment."""
    global_frames = {name: _whole_table(out_dir, name) for name in GLOBAL_TABLES}
    tables = _Tables(out_dir, global_frames, _env_anchors(global_frames["tf"], global_frames["tf_static"]), pedestrian_frames)
    return {robot_id: _robot_bundle(tables, env_id, robot_id) for env_id, robot_id in _robot_refs(out_dir)}


def _robot_bundle(tables: _Tables, env_id: str, robot_id: str) -> TopicBundle:
    if tables.anchors and env_id not in tables.anchors:
        raise ValueError(f"{env_id}: no map -> {env_id}/map anchor in tf_static, anchors recorded: {dict(sorted(tables.anchors.items()))}")
    shifts = tables.anchors.get(env_id, [])
    frames: dict[str, pl.LazyFrame | None] = {name: tables.scoped(name, "robot_id", robot_id) for name in ROBOT_TABLES}
    frames |= {name: tables.scoped(name, "env_id", env_id) for name in ENV_TABLES}
    if any(ox != 0.0 or oy != 0.0 for _, ox, oy in shifts):
        _check_map_frames(env_id, shifts, tables.pedestrian_frames)
        _shift_to_map(frames, shifts)
    odom_controller = frames.pop("robot_odom_controller")
    cmd_vel_controller = frames.pop("robot_cmd_vel_controller")
    frames["robot_odom"] = _first_present(odom_controller, frames["robot_odom"])
    frames["robot_cmd_vel"] = _first_present(frames["robot_cmd_vel"], cmd_vel_controller)
    return TopicBundle(**tables.global_frames, **frames)


def _check_map_frames(env_id: str, shifts: list[Anchor], pedestrian_frames: Mapping[tuple[str, str], frozenset[str]]) -> None:
    for table in (ARENA_TABLE, ENGINE_TABLE):
        unmapped = sorted(frame for frame in pedestrian_frames.get((env_id, table), frozenset()) if frame not in _UNMAPPED_FRAMES)
        if unmapped:
            offsets = [anchor[1:] for anchor in shifts]
            raise ValueError(f"{env_id}: {table} stamped {unmapped}, offsets {offsets} are only valid for frame map")


def _shift_to_map(frames: dict[str, pl.LazyFrame | None], shifts: list[Anchor]) -> None:
    for name, columns in _POSITION_COLUMNS.items():
        frame = frames[name]
        if frame is not None:
            frames[name] = _shift_positions(frame, shifts, columns)
    lifecycle = frames.get("episode_lifecycle")
    if lifecycle is not None:
        frames["episode_lifecycle"] = _shift_episode_phases(lifecycle, shifts)


def _shift_positions(frame: pl.LazyFrame, anchors: list[Anchor], columns: tuple[str, str]) -> pl.LazyFrame:
    x_column, y_column = columns
    return frame.with_columns(
        by_anchor(anchors, lambda ox, _: pl.col(x_column) - ox).alias(x_column),
        by_anchor(anchors, lambda _, oy: pl.col(y_column) - oy).alias(y_column),
    )


def _shift_episode_phases(frame: pl.LazyFrame, anchors: list[Anchor]) -> pl.LazyFrame:
    return frame.with_columns(
        pl.struct(["t_ns", "phases_yaml"])
        .map_elements(
            lambda row: _shift_map_poses(row["phases_yaml"], *anchor_at(anchors, int(row["t_ns"]))),
            return_dtype=pl.String,
        )
        .alias("phases_yaml")
    )


def _shift_map_poses(payload: str, ox: float, oy: float) -> str:
    """EpisodeRecord.phases with every robot's map_poses moved by (-ox, -oy)."""
    if not payload:
        return payload
    try:
        robots = json.loads(payload)
        for entry in robots.values():
            entry["map_poses"] = [None if pose is None else [pose[0] - ox, pose[1] - oy, *pose[2:]] for pose in entry["map_poses"]]
        return json.dumps(robots)
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        _log.warning(f"Failed to offset episode phases: {error}")
        return payload


def _env_anchors(tf: pl.LazyFrame | None, tf_static: pl.LazyFrame | None) -> dict[str, list[Anchor]]:
    anchors = _anchors_by_env(tf_static, _STATIC_ANCHOR_CHILD)
    for env_id, dynamic in _anchors_by_env(tf, _MAP_ANCHOR_CHILD).items():
        anchors.setdefault(env_id, dynamic)
    return {env_id: _without_repeats(values) for env_id, values in anchors.items()}


def _anchors_by_env(frame: pl.LazyFrame | None, child_pattern: re.Pattern[str]) -> dict[str, list[Anchor]]:
    anchors: defaultdict[str, list[Anchor]] = defaultdict(list)
    if frame is None:
        return dict(anchors)
    for row in frame.collect().iter_rows(named=True):
        parent = row["parent_frame"].strip("/").lower()
        child = row["child_frame"].strip("/").lower()
        match = child_pattern.fullmatch(child)
        if match and (parent == "" or is_world_frame(parent)) and not is_base_frame(child):
            anchors[match.group(1)].append((row["t_ns"], row["tx"], row["ty"]))
    return dict(anchors)


def _without_repeats(anchors: list[Anchor]) -> list[Anchor]:
    return [anchor for index, anchor in enumerate(anchors) if index == 0 or anchor[1:] != anchors[index - 1][1:]]


def _robot_refs(out_dir: pathlib.Path) -> list[tuple[str, str]]:
    paths = [out_dir / f"{name}.parquet" for name in ROBOT_TABLES]
    frames = [pl.scan_parquet(path).select("env_id", "robot_id") for path in paths if path.exists()]
    if not frames:
        return []
    refs = pl.concat(frames).unique().sort("robot_id").collect()
    return [(env_id, robot_id) for env_id, robot_id in refs.iter_rows()]


def _whole_table(out_dir: pathlib.Path, name: str) -> pl.LazyFrame | None:
    path = out_dir / f"{name}.parquet"
    return pl.scan_parquet(path) if path.exists() else None


def _first_present(*frames: pl.LazyFrame | None) -> pl.LazyFrame | None:
    return next((frame for frame in frames if frame is not None), None)


