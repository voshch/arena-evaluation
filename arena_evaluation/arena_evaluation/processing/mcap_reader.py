"""MCAP ingestion coordinator: streams recorded messages into flat telemetry tables and loads per-robot topic bundles."""

from __future__ import annotations

import json
import logging
import pathlib
import re
import typing
from collections import defaultdict
from collections.abc import Callable, Mapping

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

from arena_evaluation.processing import legacy_schemas
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
from arena_evaluation.processing.telemetry.peds import (
    ARENA_TABLE,
    ENGINE_TABLE,
    PedestrianRecorder,
    agent_meta,
    agent_meta_without_robot_fields,
)
from arena_evaluation.processing.telemetry.tracks import (
    _BASE_FRAME_MARKERS,
    _WORLD_FRAMES,
    Pose3,
    is_ground_truth,
)
from arena_evaluation.processing.telemetry.writer import (
    ENV_TABLES,
    FLUSH_ROW_LIMIT,
    GLOBAL_TABLES,
    ROBOT_TABLES,
    TableSink,
)
from arena_evaluation.storage.schemas import TopicBundle

if typing.TYPE_CHECKING:
    from arena_runtime_msgs.msg import RobotFleet
    from mcap.records import Schema
    from tf2_msgs.msg import TFMessage

_log = logging.getLogger(__name__)
Anchor = tuple[int, float, float]
Predicate = Callable[[str, "Schema"], bool]
Handler = Callable[["_Ingest", str, Envelope], None]
Decoder = Callable[[bytes], object]

_AGENT_META_SCHEMA = "arena_humansim_msgs/msg/AgentMeta"
_INTERACTION_FRAME_SCHEMA = "arena_humansim_msgs/msg/InteractionFrame"
_ANIMATION_STATES_SCHEMA = "arena_people_msgs/msg/AnimationStates"
_GOAL_POSE_SCHEMA = "geometry_msgs/msg/PoseStamped"
_OCCUPANCY_GRID_SCHEMA = "nav_msgs/msg/OccupancyGrid"
_STATIC_ANCHOR_CHILD = re.compile(r"(env_\d+)(?:/map)?")
_MAP_ANCHOR_CHILD = re.compile(r"(env_\d+)/map")
_UNMAPPED_FRAMES = ("", "map")
# Yaw (degrees) and translation (metres) below which a recorded map-frame transform counts as the identity.
_ALIGNED_TOLERANCE = 1e-6
_MAP_PARENT = pl.col("parent_frame").is_in(list(_UNMAPPED_FRAMES))
_CHILD_ENVIRONMENT = pl.col("child_frame").str.extract(r"(env_\d+)", 1)
# Stored tables whose x/y columns are map-frame positions. Odometry is map-aligned in this runtime: the recorded map -> odom transform is the identity.
# The engine's agent states are already in the environment frame, so they are not listed.
_POSITION_COLUMNS: dict[str, tuple[str, str]] = {
    "robot_tracks": ("x", "y"),
    "robot_odom": ("odom_x", "odom_y"),
    "robot_odom_controller": ("odom_x", "odom_y"),
    "robot_initialpose": ("pos_x", "pos_y"),
    "robot_goal_pose": ("pos_x", "pos_y"),
    "robot_task_pose": ("pos_x", "pos_y"),
    "robot_plans": ("x", "y"),
    "robot_collision_contacts": ("obstacle_x", "obstacle_y"),
    "pedestrian_arena_peds": ("x", "y"),
    "pedestrian_gestures": ("at_x", "at_y"),
    "environment_map": ("origin_x", "origin_y"),
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


def _topic(*suffixes: str, schema: str | None = None) -> Predicate:
    """Match a topic suffix, and optionally the schema name that the recorder understands."""
    return lambda topic, message_schema: topic.endswith(suffixes) and (schema is None or message_schema.name == schema)


def _is_transform_topic(topic: str, _schema: Schema) -> bool:
    """The global and per-environment /tf and /tf_static topics; human skeleton transforms are never robot ground truth."""
    return topic in ("tf", "tf_static") or (topic.endswith(("/tf", "/tf_static")) and "/humans/" not in topic)


class MCAPReader:
    """Streams MCAP recordings into flat, episode-wide telemetry tables and loads one TopicBundle per robot."""

    def __init__(
        self,
        data_path: pathlib.Path,
        *,
        flush_rows: int = FLUSH_ROW_LIMIT,
        environment: str | None = None,
    ) -> None:
        self._flush_rows = flush_rows
        self._environment = environment
        self.data_path = data_path

    def read(self, out_dir: pathlib.Path) -> dict[str, TopicBundle]:
        """Stream every recording into flat tables under out_dir and return one TopicBundle per robot."""
        if self.data_path.is_dir():
            files = sorted(self.data_path.glob("*.mcap"))
            if not files:
                raise FileNotFoundError(f"No MCAP file found in directory: {self.data_path}")
        else:
            files = [self.data_path]
        for path in files:
            if not path.exists():
                raise FileNotFoundError(f"MCAP file not found: {path}")
        ingest = _Ingest(
            TableSink(out_dir, flush_rows=self._flush_rows),
            environment=self._environment,
        )
        try:
            for path in files:
                self._stream(path, ingest)
        finally:
            ingest.finish()
        align_to_environment(out_dir, ingest.pedestrian_frames())
        return load_bundles(out_dir)

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
                try:
                    body = decoder(message.data)
                except Exception as error:
                    _log.warning(f"Error reading MCAP message on {channel.topic}: {error}")
                    continue
                ingest.route(channel.topic, schema, body, message.log_time)


def _sink(record: Callable[..., None]) -> Handler:
    """Route a message to a module-level recorder that takes the episode sink first."""
    return lambda ingest, _topic, envelope: record(ingest._sink, envelope)


def _recorder(attribute: str) -> Handler:
    """Route a message to one ingest recorder's ``record`` method."""
    return lambda ingest, _topic, envelope: getattr(ingest, attribute).record(envelope)


def _frame(table: str) -> Handler:
    """Route an agent frame to a pedestrian table."""
    return lambda ingest, _topic, envelope: ingest._pedestrians.record_frame(table, envelope)


def _grid(kind: str) -> Handler:
    """Route an occupancy grid to its table."""
    return lambda ingest, _topic, envelope: context.record_occupancy_grid(ingest._sink, envelope, kind)


def _both_events(first: Callable[..., None], second: Callable[..., None]) -> Handler:
    """Route one interaction frame to both of its recorders."""
    return lambda ingest, _topic, envelope: (
        first(ingest._sink, envelope),
        second(ingest._sink, envelope),
    )


def _odometry(ingest: _Ingest, topic: str, envelope: Envelope) -> None:
    table = "robot_odom_controller" if is_controller_topic(topic) else "robot_odom"
    ingest._kinematics.record_odometry(table, envelope)


def _command(ingest: _Ingest, topic: str, envelope: Envelope) -> None:
    table = "robot_cmd_vel_controller" if is_controller_topic(topic) else "robot_cmd_vel"
    ingest._kinematics.record_command(table, envelope)


def _agent_meta(ingest: _Ingest, _topic: str, envelope: Envelope) -> None:
    schema_text = envelope.schema.data.decode("utf-8", "replace")
    meta = agent_meta(envelope.body) if legacy_schemas.agent_meta_has_crowd(schema_text) else agent_meta_without_robot_fields(envelope.body)
    ingest._pedestrians.record_meta(envelope, meta)


def _episode(ingest: _Ingest, _topic: str, envelope: Envelope) -> None:
    context.record_episode(
        ingest._sink,
        envelope,
        legacy_schemas.episode_fields(envelope.body, envelope.schema),
    )


def _fleet_snapshot(ingest: _Ingest, topic: str, envelope: Envelope[RobotFleet]) -> None:
    env_id = env_id_in(topic) or ingest._environment
    if env_id is None:
        raise ValueError(f"fleet topic {topic!r} names no environment")
    ingest._fleet.record(
        envelope.t_ns,
        env_id,
        legacy_schemas.fleet_robots(envelope.body, envelope.schema),
    )


def _initial_pose(ingest: _Ingest, topic: str, envelope: Envelope) -> None:
    parts = topic_parts(topic)
    if len(parts) >= 2 and parts[-2] == env_id_in(topic):
        ingest._warn_skipped(topic, "an environment-level initial pose names no robot")
        return
    context.record_initial_pose(ingest._sink, envelope)


def _transforms(ingest: _Ingest, topic: str, envelope: Envelope[TFMessage]) -> None:
    context.record_transforms(ingest._sink, "tf_static" if "static" in topic else "tf", envelope)
    for transform in envelope.body.transforms:
        parent, child = transform.header.frame_id, transform.child_frame_id
        if not is_ground_truth(parent, child):
            continue
        child_env = env_id_in(child)
        if child_env is None:
            raise ValueError(f"ground-truth frame {child!r} names no environment")
        translation, rotation = (
            transform.transform.translation,
            transform.transform.rotation,
        )
        ingest._kinematics.record_ground_truth(
            Envelope(
                t_ns=envelope.t_ns,
                ref=RobotRef(
                    env_id=child_env,
                    robot_id=robot_id_of(child.strip("/").split("/"), child_env),
                ),
                schema=envelope.schema,
                body=transform,
            ),
            Pose3(
                x=float(translation.x),
                y=float(translation.y),
                z=float(translation.z),
                yaw=yaw_from_orientation(rotation),
            ),
        )


_ROUTES: tuple[tuple[Predicate, Handler], ...] = (
    (_topic("/odom"), _odometry),
    (_topic("/scan", "/lidar"), _recorder("_scans")),
    (_topic("/cmd_vel"), _command),
    (_topic("/joint_states"), _sink(sensors.record_joint_states)),
    (_topic("/agent_meta", schema=_AGENT_META_SCHEMA), _agent_meta),
    (_topic("/arena_peds", "/peds", "/pedestrians"), _frame(ARENA_TABLE)),
    (_topic("/agent_states", "/agent_frame"), _frame(ENGINE_TABLE)),
    (
        _topic("/interactions", schema=_INTERACTION_FRAME_SCHEMA),
        _both_events(events.record_interactions, events.record_interaction_events),
    ),
    (
        _topic("/animation_states", schema=_ANIMATION_STATES_SCHEMA),
        _sink(events.record_animations),
    ),
    (_topic("/acoustics"), _sink(sensors.record_acoustics)),
    (_topic("/characterization_phase"), _sink(context.record_characterization_phase)),
    (_topic("/characterization_schedule"), _recorder("_schedules")),
    (_topic("/state/episode"), _episode),
    (_topic("/state/semantics"), _sink(context.record_semantics)),
    (_topic("/state/robots"), _fleet_snapshot),
    (_topic("/collision_events"), _sink(events.record_collision_events)),
    (_topic("/collision_monitor_state"), _sink(events.record_collision_monitor)),
    (_topic("/power_publisher/power"), _sink(sensors.record_power)),
    (_topic("/power_publisher/energy"), _sink(sensors.record_energy)),
    (_topic("/plan", "/global_plan"), _recorder("_plans")),
    (_topic("/task_pose"), _sink(context.record_task_pose)),
    (_topic("/goal_pose", schema=_GOAL_POSE_SCHEMA), _sink(context.record_goal_pose)),
    (_topic("/map", schema=_OCCUPANCY_GRID_SCHEMA), _grid("map")),
    (_topic("/door_mask", schema=_OCCUPANCY_GRID_SCHEMA), _grid("door_mask")),
    (_topic("/initialpose"), _initial_pose),
    (_is_transform_topic, _transforms),
)


class _Ingest:
    """Routes decoded messages to their recorders and writes the tables that need the whole recording."""

    def __init__(self, sink: TableSink, *, environment: str | None = None) -> None:
        self._sink = sink
        self._environment = environment
        self._warned_topics: set[str] = set()
        self._kinematics = tracks.KinematicsRecorder(sink)
        self._pedestrians = PedestrianRecorder(sink)
        self._scans = sensors.ScanRecorder(sink)
        self._plans = context.PlanRecorder(sink)
        self._fleet = context.FleetRecorder(sink)
        self._schedules = context.ScheduleRecorder()

    def accepts(self, topic: str, schema: Schema) -> bool:
        """Whether the reader routes this recorded topic to a recorder."""
        return self._handler(topic, schema) is not None

    def _handler(self, topic: str, schema: Schema) -> Handler | None:
        return next((handle for accepts, handle in _ROUTES if accepts(topic, schema)), None)

    def route(self, topic: str, schema: Schema, body: object, t_ns: int) -> None:
        """Dispatch a decoded message to the recorder that accepts its topic, or skip it."""
        handler = self._handler(topic, schema)
        if handler is None:
            return
        env_id = env_id_in(topic) or self._environment
        if env_id is None and not _is_transform_topic(topic, schema) and not topic.endswith("/state/semantics"):
            self._warn_skipped(topic, "names no environment")
            return
        ref = None if env_id is None or _is_transform_topic(topic, schema) else RobotRef(env_id=env_id, robot_id=robot_id_of(topic_parts(topic), env_id))
        handler(self, topic, Envelope(t_ns=t_ns, ref=ref, schema=schema, body=body))

    def _warn_skipped(self, topic: str, reason: str) -> None:
        """Log one warning per skipped topic, naming it and the reason."""
        if topic in self._warned_topics:
            return
        self._warned_topics.add(topic)
        _log.warning(f"{topic}: {reason}; skipped")

    def pedestrian_frames(self) -> dict[tuple[str, str], frozenset[str]]:
        return self._pedestrians.frames()

    def finish(self) -> None:
        self._kinematics.finalize()
        self._schedules.write(self._sink)
        self._sink.close()


def _scoped(out_dir: pathlib.Path, name: str, column: str, value: str) -> pl.LazyFrame | None:
    """The table when it exists and holds a row for this scope value."""
    path = out_dir / f"{name}.parquet"
    if not path.exists():
        return None
    frame = pl.scan_parquet(path).filter(pl.col(column) == value)
    return frame if frame.head(1).collect().height else None


def load_bundles(out_dir: pathlib.Path) -> dict[str, TopicBundle]:
    """One TopicBundle per robot: the episode's stored tables, filtered to that robot and its environment."""
    global_frames = {name: _lazy_table(out_dir, name) for name in GLOBAL_TABLES}
    return {robot_id: _robot_bundle(out_dir, global_frames, env_id, robot_id) for env_id, robot_id in _robot_refs(out_dir)}


def _robot_bundle(
    out_dir: pathlib.Path,
    global_frames: Mapping[str, pl.LazyFrame | None],
    env_id: str,
    robot_id: str,
) -> TopicBundle:
    frames: dict[str, pl.LazyFrame | None] = {name: _scoped(out_dir, name, "robot_id", robot_id) for name in ROBOT_TABLES}
    frames |= {name: _scoped(out_dir, name, "env_id", env_id) for name in ENV_TABLES}
    odom_controller = frames.pop("robot_odom_controller")
    cmd_vel_controller = frames.pop("robot_cmd_vel_controller")
    frames["robot_odom"] = _first_present(odom_controller, frames["robot_odom"])
    frames["robot_cmd_vel"] = _first_present(frames["robot_cmd_vel"], cmd_vel_controller)
    return TopicBundle(**global_frames, **frames)


def _check_map_frames(
    env_id: str,
    shifts: list[Anchor],
    pedestrian_frames: Mapping[tuple[str, str], frozenset[str]],
) -> None:
    for table in (ARENA_TABLE,):
        unmapped = sorted(frame for frame in pedestrian_frames.get((env_id, table), frozenset()) if frame not in _UNMAPPED_FRAMES)
        if unmapped:
            offsets = [anchor[1:] for anchor in shifts]
            raise ValueError(f"{env_id}: {table} stamped {unmapped}, offsets {offsets} are only valid for frame map")


def align_to_environment(out_dir: pathlib.Path, pedestrian_frames: Mapping[tuple[str, str], frozenset[str]]) -> None:
    """Move the map-frame positions of the stored tables into their environment frame, once, in place.

    An environment's offset is its recorded map -> env/map transform (tf_static, else tf). Each position
    column in _POSITION_COLUMNS and the episode phases leave the map frame here, so the stored tables,
    the metrics and the plots all read one frame. An episode without a recorded transform keeps its
    positions as recorded. The recorded transforms whose parent is the map frame (tf_static and tf) move
    with the anchor too; a rotated environment frame, or an odometry frame that is not the map frame, is refused.
    """
    anchors = _env_anchors(_lazy_table(out_dir, "tf"), _lazy_table(out_dir, "tf_static"))
    if not anchors:
        return
    _refuse_rotated_or_unaligned_frames(out_dir, anchors)
    for env_id in sorted({env_id for env_id, _robot_id in _robot_refs(out_dir)}):
        if env_id not in anchors:
            raise ValueError(f"{env_id}: no map -> {env_id}/map anchor in tf_static, anchors recorded: {dict(sorted(anchors.items()))}")
        shifts = anchors[env_id]
        if any(ox != 0.0 or oy != 0.0 for _, ox, oy in shifts):
            _check_map_frames(env_id, shifts, pedestrian_frames)
    for name, (x_column, y_column) in _POSITION_COLUMNS.items():
        path = out_dir / f"{name}.parquet"
        if path.exists():
            _rewrite_columns(
                path,
                [x_column, y_column],
                lambda frame, x_col=x_column, y_col=y_column: frame.with_columns(
                    _in_environment(anchors, x_col, lambda value, ox, _oy: value - ox),
                    _in_environment(anchors, y_col, lambda value, _ox, oy: value - oy),
                ),
            )
    lifecycle = out_dir / "episode_lifecycle.parquet"
    if lifecycle.exists():
        _rewrite_columns(
            lifecycle,
            ["phases_yaml"],
            lambda frame: _shift_episode_phases(frame, anchors),
        )
    for name in ("tf_static", "tf"):
        if (out_dir / f"{name}.parquet").exists():
            _rewrite_columns(
                out_dir / f"{name}.parquet",
                ["tx", "ty"],
                lambda frame: frame.with_columns(
                    _in_environment(
                        anchors,
                        "tx",
                        lambda value, ox, _oy: value - ox,
                        env=_CHILD_ENVIRONMENT,
                        where=_MAP_PARENT,
                    ),
                    _in_environment(
                        anchors,
                        "ty",
                        lambda value, _ox, oy: value - oy,
                        env=_CHILD_ENVIRONMENT,
                        where=_MAP_PARENT,
                    ),
                ),
                keys=("parent_frame", "child_frame", "t_ns"),
            )


def _in_environment(
    anchors: Mapping[str, list[Anchor]],
    column: str,
    shifted: Callable[[pl.Expr, float, float], pl.Expr],
    *,
    env: pl.Expr | None = None,
    where: pl.Expr | None = None,
) -> pl.Expr:
    """`column` of each row selected by `where`, moved by the offset of its environment in effect at the row's t_ns."""
    env_expr = pl.col("env_id") if env is None else env
    where_expr = pl.lit(True) if where is None else where
    expr = pl.col(column)
    for env_id, shifts in anchors.items():
        moved = by_anchor(shifts, lambda ox, oy: shifted(pl.col(column), ox, oy))
        expr = pl.when(where_expr & (env_expr == env_id)).then(moved).otherwise(expr)
    return expr.alias(column)


def _shift_episode_phases(frame: pl.DataFrame, anchors: Mapping[str, list[Anchor]]) -> pl.DataFrame:
    def shifted(row: Mapping[str, typing.Any]) -> str:
        env_anchors = anchors.get(row["env_id"])
        if env_anchors is None:
            return row["phases_yaml"]
        return _shift_map_poses(row["phases_yaml"], *anchor_at(env_anchors, int(row["t_ns"])))

    return frame.with_columns(pl.struct(["env_id", "t_ns", "phases_yaml"]).map_elements(shifted, return_dtype=pl.String).alias("phases_yaml"))


def _rewrite_columns(
    path: pathlib.Path,
    names: list[str],
    transform: Callable[[pl.DataFrame], pl.DataFrame],
    keys: tuple[str, ...] = ("env_id", "t_ns"),
) -> None:
    """Replace the named columns of a stored table with their transformed values; every other column and the arrow schema stay as written."""
    table = pq.read_table(path)
    needed = list(dict.fromkeys([*keys, *names]))
    shifted = transform(pl.from_arrow(table.select(needed))).to_arrow()
    for name in names:
        index = table.schema.get_field_index(name)
        field = table.schema.field(index)
        table = table.set_column(index, field, shifted.column(name).cast(field.type))
    pq.write_table(table, path)


def _map_parented_transforms(out_dir: pathlib.Path) -> pl.DataFrame | None:
    """Every recorded transform whose parent is the map frame, static and dynamic, with its yaw in degrees."""
    frames = [frame.select("parent_frame", "child_frame", "t_ns", "tx", "ty", "qz", "qw") for name in ("tf_static", "tf") if (frame := _lazy_table(out_dir, name)) is not None]
    if not frames:
        return None
    transforms = pl.concat(frames).filter(_MAP_PARENT).collect()
    yaw_deg = np.degrees(2.0 * np.arctan2(transforms["qz"].to_numpy(), transforms["qw"].to_numpy()))
    return transforms.with_columns(pl.Series("yaw_deg", yaw_deg))


def _refuse_rotated_or_unaligned_frames(out_dir: pathlib.Path, anchors: Mapping[str, list[Anchor]]) -> None:
    """Refuse the recorded map frames a translation-only alignment cannot place: a rotated environment frame, or an odometry frame that is not the map frame."""
    transforms = _map_parented_transforms(out_dir)
    if transforms is None:
        return
    for env_id in anchors:
        rotated = transforms.filter((pl.col("child_frame") == f"{env_id}/map") & (pl.col("yaw_deg").abs() > _ALIGNED_TOLERANCE)).head(1)
        if rotated.height > 0:
            yaw_deg = float(rotated["yaw_deg"][0])
            raise ValueError(f"{env_id}: the recorded map -> {env_id}/map transform is rotated ({yaw_deg:.3f} deg); only translated environment frames are supported")
    unaligned = transforms.filter(pl.col("child_frame").str.contains("odom") & ((pl.col("tx").abs() > _ALIGNED_TOLERANCE) | (pl.col("ty").abs() > _ALIGNED_TOLERANCE) | (pl.col("yaw_deg").abs() > _ALIGNED_TOLERANCE)))
    if unaligned.height:
        row = unaligned.row(0, named=True)
        raise ValueError(f"{row['child_frame']}: the recorded map -> odom transform is not the identity (translation {row['tx']}, {row['ty']}; yaw {row['yaw_deg']:.3f} deg), so odometry and ground truth cannot be placed in the environment frame by the anchor alone")


def _lazy_table(out_dir: pathlib.Path, name: str) -> pl.LazyFrame | None:
    path = out_dir / f"{name}.parquet"
    return pl.scan_parquet(path) if path.exists() else None


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
    return {env_id: [anchor for index, anchor in enumerate(values) if index == 0 or anchor[1:] != values[index - 1][1:]] for env_id, values in anchors.items()}


def _anchors_by_env(frame: pl.LazyFrame | None, child_pattern: re.Pattern[str]) -> dict[str, list[Anchor]]:
    anchors: defaultdict[str, list[Anchor]] = defaultdict(list)
    if frame is None:
        return dict(anchors)
    parent = pl.col("parent_frame").str.strip_chars("/").str.to_lowercase()
    child = pl.col("child_frame").str.strip_chars("/").str.to_lowercase()
    env = child.str.extract(f"^{child_pattern.pattern}$", 1)
    world_re = "|".join(re.escape(name) for name in _WORLD_FRAMES)
    base_re = "|".join(re.escape(marker) for marker in _BASE_FRAME_MARKERS)
    world_match = parent.is_in(list(_WORLD_FRAMES)) | parent.str.contains(rf"/(?:{world_re})$")
    filtered = frame.with_columns(parent.alias("_parent"), child.alias("_child"), env.alias("_env")).filter(pl.col("_env").is_not_null() & ((pl.col("_parent") == "") | world_match) & ~child.str.contains(base_re)).select("_env", "t_ns", "tx", "ty").collect()
    # agent-contracts: ignore materialize anchor tuples from filtered tf frame
    for env_id, t_ns, tx, ty in filtered.iter_rows():
        anchors[env_id].append((t_ns, tx, ty))
    return dict(anchors)


def _robot_refs(out_dir: pathlib.Path) -> list[tuple[str, str]]:
    frames = [pl.scan_parquet(out_dir / f"{name}.parquet").select("env_id", "robot_id") for name in ROBOT_TABLES if (out_dir / f"{name}.parquet").exists()]
    if not frames:
        return []
    refs = pl.concat(frames).unique().sort("robot_id").collect()
    # agent-contracts: ignore materialize (env_id, robot_id) keys for bundle dict
    return [(env_id, robot_id) for env_id, robot_id in refs.iter_rows()]


def _first_present(*frames: pl.LazyFrame | None) -> pl.LazyFrame | None:
    return next((frame for frame in frames if frame is not None), None)
