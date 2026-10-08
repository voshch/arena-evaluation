"""MuSoHu rosbag1 runs from the GMU Dataverse recorded as evaluation episodes, one run directory per bag."""

from __future__ import annotations

import dataclasses
import datetime
import heapq
import json
import logging
import shutil
import struct
import typing
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs
import numpy as np

from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.providers import Http, Provider

logger = logging.getLogger(__name__)

DATAVERSE = "https://dataverse.orc.gmu.edu"
PERSISTENT_ID = "doi:10.13021/orc2020/HZI4LJ"
DATASET_URL = f"{DATAVERSE}/dataset.xhtml?persistentId={PERSISTENT_ID}"
DATASET_API = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={PERSISTENT_ID}"
ACCESS_URL = f"{DATAVERSE}/api/access/datafile"
LICENSE = "CC0-1.0"

ROBOT = "human"
PLANNER = "human"
SUITE = "musohu"
REFERENCE_TYPE = "human_demonstration"
ENV_NAMESPACE = "arena/env_0/task_generator_node"
ROBOT_NAMESPACE = f"{ENV_NAMESPACE}/{ROBOT}"
FRAME = f"env_0/{ROBOT}"
EPISODE = "episode_000"
NS = 1_000_000_000
BAG_MAGIC = b"#ROSBAG V2.0\n"
PROBE_HEAD = 4096
PROBE_SPAN = 65536
TF_TOPICS = ("/tf", "/tf_static")
MAPPED_TOPICS = frozenset(f"/{ROBOT_NAMESPACE}/{key}" for key in ("scan", "odom", "cmd_vel"))

Message = tuple[str, int, typing.Any, typing.Any]
Passed = tuple[int, str, type, typing.Any]
Record = tuple[int, str, type, bytes]


class Stamp(typing.Protocol):
    sec: int
    nanosec: int


class Header(typing.Protocol):
    stamp: Stamp
    frame_id: str


class Ros1Message(typing.Protocol):
    __dataclass_fields__: typing.ClassVar[dict[str, dataclasses.Field]]


class Ros2Message(typing.Protocol):
    def get_fields_and_field_types(self) -> dict[str, str]: ...


@attrs.frozen
class Topics:
    scan: str = "/scan"
    odom: str = "/zed2/zed_node/odom"
    velocity: str | None = "/action"
    cmd_vel: str | None = None


DEFAULT_TOPICS = Topics()


class MuSoHu(Importer):
    """One Dataverse bag becomes out/<bag stem>/episodes/episode_000, a reference episode of the robot model human in the recorder's layout."""

    meta = IMPORTERS.meta("episode", "musohu")

    def __init__(self, cache: Path | None = None, provider: Provider | None = None, *, topics: Topics = DEFAULT_TOPICS):
        super().__init__(cache, provider)
        self.topics = topics
        self._listing: dict | None = None

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(ACCESS_URL)

    def listing(self) -> dict:
        if self._listing is None:
            path = self.cache / "dataset.json"
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.provider.read(DATASET_API)[0])
            with open(path) as f:
                self._listing = json.load(f)["data"]["latestVersion"]
        return self._listing

    def files(self) -> dict[str, dict]:
        return {Path(f["dataFile"]["filename"]).stem: f["dataFile"] for f in self.listing()["files"] if f["dataFile"]["filename"].endswith(".bag")}

    def items(self) -> Iterable[str]:
        return sorted(self.files())

    def fetch(self, item: str) -> Path:
        dest = self.cache / f"{item}.bag"
        if dest.is_file():
            return dest
        entry = self.files()[item]
        available = self.probe(item, entry)
        missing = wanted_topics(self.topics) - available
        if missing:
            raise ValueError(f"{dest.name} lacks topics {sorted(missing)}, has {sorted(available)}")
        body = self.provider.get(str(entry["id"]), self.cache / f"{item}.body")
        complete(self.provider, str(entry["id"]), body, int(entry["filesize"]))
        body.replace(dest)
        return dest

    def probe(self, item: str, entry: dict) -> set[str]:
        """Topics of the bag's connection records, read once by range request and cached beside the download."""
        path = self.cache / f"{item}.topics.json"
        if not path.is_file():
            topics, fetched = probe_topics(self.provider, str(entry["id"]), int(entry["filesize"]), self.cache / f"{item}.probe")
            path.write_text(json.dumps({"topics": sorted(topics), "bytes": fetched}))
        return set(json.loads(path.read_text())["topics"])

    def convert(self, item: str, src: Path, out: Path) -> str:
        run = out / item
        if run.exists():
            shutil.rmtree(run)
        types, skipped, start_ns = passthrough_types(src, wanted_topics(self.topics))
        if skipped:
            logger.warning("%s: no ROS 2 message type for %s, not recorded", src.name, ", ".join(f"{topic} ({kind})" for topic, kind in skipped))
        record_episode(run, read_bag(src, self.topics), passthrough_messages(src, types), start_ns)
        return item

    def attribution(self, item: str) -> Attribution:
        return Attribution(
            source="MuSoHu",
            url=DATASET_URL,
            license=LICENSE,
            authors="Duc M. Nguyen, Mohammad Nazeri, Amirreza Payandeh, Aniket Datar, Xuesu Xiao",
            citation="Nguyen et al., Toward Human-Like Social Robot Navigation: A Large-Scale, Multi-Modal, Social Human Navigation Dataset, IROS 2023, doi:10.13021/orc2020/HZI4LJ",
        )


def stamp_ns(header: Header) -> int:
    return header.stamp.sec * NS + header.stamp.nanosec


def wanted_topics(topics: Topics) -> set[str]:
    return {topic for topic in (topics.scan, topics.odom, topics.velocity, topics.cmd_vel) if topic}


def record_fields(header: bytes) -> dict[bytes, bytes]:
    fields: dict[bytes, bytes] = {}
    pos = 0
    while pos + 4 <= len(header):
        (n,) = struct.unpack_from("<I", header, pos)
        name, _, value = header[pos + 4 : pos + 4 + n].partition(b"=")
        fields[name] = value
        pos += 4 + n
    return fields


def connection_topics(blob: bytes, count: int) -> set[str] | None:
    """Topics of the first count connection records in blob, None when blob ends before the last one."""
    topics: list[str] = []
    pos = 0
    while len(topics) < count:
        if pos + 4 > len(blob):
            return None
        (header_len,) = struct.unpack_from("<I", blob, pos)
        if pos + 8 + header_len > len(blob):
            return None
        fields = record_fields(blob[pos + 4 : pos + 4 + header_len])
        if fields[b"op"] != b"\x07":
            raise ValueError(f"record {len(topics)} of the index is op {fields[b'op'].hex()}, not a connection")
        (data_len,) = struct.unpack_from("<I", blob, pos + 4 + header_len)
        topics.append(fields[b"topic"].decode())
        pos += 8 + header_len + data_len
    return set(topics)


def probe_topics(provider: Provider, relpath: str, size: int, scratch: Path) -> tuple[set[str], int]:
    """Topics of a rosbag1 file from its header and the connection records of its index, fetched by range, and the bytes fetched."""
    head = provider.get(relpath, scratch, range=(0, min(PROBE_HEAD, size) - 1)).read_bytes()
    if len(head) > PROBE_HEAD:
        scratch.unlink()
        raise RuntimeError(f"{provider.url(relpath)} ignored the range request, probing it would download the whole bag")
    if not head.startswith(BAG_MAGIC):
        scratch.unlink()
        raise ValueError(f"{provider.url(relpath)} is not a rosbag v2 file")
    (header_len,) = struct.unpack_from("<I", head, len(BAG_MAGIC))
    header = record_fields(head[len(BAG_MAGIC) + 4 : len(BAG_MAGIC) + 4 + header_len])
    (index_pos,) = struct.unpack("<Q", header[b"index_pos"])
    (count,) = struct.unpack("<I", header[b"conn_count"])
    if index_pos == 0:
        scratch.unlink()
        raise ValueError(f"{provider.url(relpath)} has no index, the bag was not closed")
    fetched = len(head)
    span = PROBE_SPAN
    while True:
        blob = provider.get(relpath, scratch, range=(index_pos, min(index_pos + span, size) - 1)).read_bytes()
        fetched += len(blob)
        topics = connection_topics(blob, count)
        if topics is not None or index_pos + span >= size:
            break
        span *= 4
    scratch.unlink()
    if topics is None:
        raise ValueError(f"{provider.url(relpath)}: {count} connection records announced, the index ends early")
    return topics, fetched


def complete(provider: Provider, relpath: str, body: Path, size: int) -> None:
    """Extend body by range requests until it holds size bytes."""
    piece = body.with_name(f"{body.name}.piece")
    while (have := body.stat().st_size) < size:
        if not provider.get(relpath, piece, range=(have, size - 1)).stat().st_size:
            raise ConnectionError(f"{provider.url(relpath)} sent nothing from byte {have} of {size}")
        with open(body, "ab") as f, open(piece, "rb") as g:
            shutil.copyfileobj(g, f)
        piece.unlink()
    if have != size:
        raise ValueError(f"{provider.url(relpath)} sent {have} bytes, its listing names {size}")


def read_bag(bag: Path, topics: Topics) -> list[Message]:
    """Mapped messages as (topic key, time ns, message, twist), timed by header stamp, odom kept only where the velocity topic shares its stamp."""
    from rosbags.highlevel import AnyReader

    keys = {topics.scan: "scan", topics.odom: "odom"}
    wanted = wanted_topics(topics)
    messages: list[Message] = []
    velocity: dict[int, typing.Any] = {}
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic in wanted]
        missing = wanted - {c.topic for c in connections}
        if missing:
            raise ValueError(f"{bag.name} lacks topics {sorted(missing)}, has {sorted(reader.topics)}")
        for connection, t, raw in reader.messages(connections=connections):
            msg: typing.Any = reader.deserialize(raw, connection.msgtype)
            if connection.topic == topics.velocity:
                velocity[stamp_ns(msg.header)] = msg.twist
            elif connection.topic == topics.cmd_vel:
                messages.append(("cmd_vel", t, msg, None))
            else:
                messages.append((keys[connection.topic], stamp_ns(msg.header), msg, None))
    if topics.velocity:
        paired = [(key, t, msg, velocity.get(t) if key == "odom" else None) for key, t, msg, _ in messages]
        messages = [message for message in paired if message[0] != "odom" or message[3] is not None]
    if not any(key == "odom" for key, *_ in messages):
        raise ValueError(f"{bag.name}: no {topics.odom} message" + (f" shares a header stamp with {topics.velocity}" if topics.velocity else ""))
    return sorted(messages, key=lambda message: message[1])


def element_type(kind: str) -> str | None:
    """Element type of a sequence<T> or sequence<T, N> field kind, None for any other kind."""
    if not kind.startswith("sequence<"):
        return None
    return kind[len("sequence<") : -1].split(",")[0].strip()


def to_ros2[M: Ros2Message](src: Ros1Message, dst: M) -> M:
    """Copy a rosbags message into the ROS 2 message dst field by field, ROS 1 field names matched without case."""
    names = {field.name.lower(): field.name for field in dataclasses.fields(src)}
    for name, kind in dst.get_fields_and_field_types().items():
        value = getattr(src, names[name])
        element = element_type(kind)
        target = getattr(dst, name)
        if element is not None and "/" in element:
            package, _, message = element.partition("/")
            cls = ros2_message_type(f"{package}/msg/{message}")
            if cls is None:
                raise TypeError(f"{type(dst).__name__}.{name}: no ROS 2 message type {element}")
            setattr(dst, name, [to_ros2(item, cls()) for item in value])
        elif element is not None and isinstance(target, array):
            setattr(dst, name, array(target.typecode, np.ascontiguousarray(value, dtype=target.typecode).tobytes()))
        elif element is not None:
            setattr(dst, name, list(value))
        elif "/" in kind:
            to_ros2(value, target)
        else:
            setattr(dst, name, value)
    return dst


def robot_frame(frame: str) -> str:
    """frame under the robot's frame prefix, an empty frame left empty."""
    return f"{FRAME}/{frame.strip('/')}" if frame.strip("/") else frame


def rebase(header: Header, origin_ns: int) -> None:
    header.stamp.sec, header.stamp.nanosec = divmod(stamp_ns(header) - origin_ns, NS)
    header.frame_id = robot_frame(header.frame_id)


def rebase_all(msg: Ros2Message, origin_ns: int) -> None:
    """Rebase every header in msg, nested and listed ones included, and put every child_frame_id under the robot's frame prefix."""
    for name, kind in msg.get_fields_and_field_types().items():
        value = getattr(msg, name)
        if kind == "std_msgs/Header":
            rebase(value, origin_ns)
        elif kind == "string" and name == "child_frame_id":
            setattr(msg, name, robot_frame(value))
        elif kind.startswith("sequence<") and "/" in kind:
            for item in value:
                rebase_all(item, origin_ns)
        elif "/" in kind and not kind.startswith("builtin_interfaces/"):
            rebase_all(value, origin_ns)


def recorded_topic(topic: str) -> str:
    """Episode topic of a passed-through bag topic, the recorder's own topic for tf and the robot namespace for the rest."""
    return topic if topic in TF_TOPICS else f"/{ROBOT_NAMESPACE}{topic}"


def ros2_message_type(msgtype: str) -> type | None:
    """ROS 2 message class of a pkg/msg/Name type, None when no installed package defines it."""
    from rosidl_runtime_py.utilities import get_message

    try:
        return get_message(msgtype)
    except (ImportError, AttributeError, ValueError):
        return None


def passthrough_types(bag: Path, mapped: set[str]) -> tuple[dict[str, type], list[tuple[str, str | None]], int]:
    """ROS 2 type of every bag topic outside mapped and off the mapped episode topics, the topics without one as (topic, type), and the bag's first receive time."""
    from rosbags.highlevel import AnyReader

    types: dict[str, type] = {}
    skipped: list[tuple[str, str | None]] = []
    with AnyReader([bag]) as reader:
        for topic, info in sorted(reader.topics.items()):
            if topic in mapped or recorded_topic(topic) in MAPPED_TOPICS:
                continue
            cls = None if info.msgtype is None else ros2_message_type(info.msgtype)
            if cls is None:
                skipped.append((topic, info.msgtype))
            else:
                types[topic] = cls
        return types, skipped, reader.start_time


def passthrough_messages(bag: Path, types: dict[str, type]) -> Iterator[Passed]:
    """Messages of the topics in types as (receive time ns, topic, ROS 2 type, rosbags message), in receive order, read lazily."""
    from rosbags.highlevel import AnyReader

    if not types:
        return
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic in types]
        for connection, t, raw in reader.messages(connections=connections):
            yield t, connection.topic, types[connection.topic], reader.deserialize(raw, connection.msgtype)


def iso(ns: int) -> str:
    return (datetime.datetime.fromtimestamp(0, datetime.UTC) + datetime.timedelta(microseconds=ns // 1000)).isoformat()


def record_episode(run: Path, messages: list[Message], passthrough: Iterable[Passed] = (), start_ns: int | None = None) -> Path:
    """Write the mapped and the passed-through messages as run/episodes/episode_000 through the recorder's episode writer, time counted from the earlier of the first mapped message and start_ns."""
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message

    from arena_evaluation.ingestion.episode_writer import EpisodeWriter
    from arena_evaluation.ingestion.metadata import IngestionMetadata
    from arena_evaluation.ingestion.topics import get_topics
    from arena_evaluation.storage.manifest import MetadataWriter

    definitions = get_topics(namespace=ROBOT_NAMESPACE, parent_namespace=ENV_NAMESPACE)
    types = {key: (definitions[key].name_template, definitions[key].msg_type) for key in ("scan", "cmd_vel")}
    types["odom"] = (f"/{ROBOT_NAMESPACE}/odom", Odometry)

    origin_ns = min(t for _, t, *_ in messages)
    if start_ns is not None:
        origin_ns = min(origin_ns, start_ns)
    episode = run / "episodes" / EPISODE
    metadata = IngestionMetadata.create_episode_metadata(
        benchmark_id=run.name,
        planner=PLANNER,
        stage=run.name,
        map_name="unknown",
        episode_id=0,
        robot_model=ROBOT,
        env_ns_root=f"/{ENV_NAMESPACE}",
        is_reference=True,
        reference_type=REFERENCE_TYPE,
        suite_name=SUITE,
        episodes_requested=1,
        agent_name=ROBOT,
    )
    metadata.recording_started_at = iso(origin_ns)

    def mapped() -> Iterator[Record]:
        for key, t, src, twist in messages:
            topic, msg_type = types[key]
            msg = to_ros2(src, msg_type())
            if key != "cmd_vel":
                rebase(msg.header, origin_ns)
            if key == "odom":
                msg.child_frame_id = robot_frame(msg.child_frame_id)
                if twist is not None:
                    to_ros2(twist, msg.twist)
            yield t, topic, msg_type, serialize_message(msg)

    def passed() -> Iterator[Record]:
        for t, topic, msg_type, src in passthrough:
            msg = to_ros2(src, msg_type())
            rebase_all(msg, origin_ns)
            yield t, recorded_topic(topic), msg_type, serialize_message(msg)

    writer = EpisodeWriter()
    writer.open_episode(episode, metadata)
    written = 0
    last_ns = origin_ns
    for t, topic, msg_type, data in heapq.merge(mapped(), passed(), key=lambda record: record[0]):
        writer.write_message(topic, msg_type, data, t - origin_ns)
        written += 1
        last_ns = max(last_ns, t)
    writer.close_episode()
    if metadata.rosbag2_message_count != written:
        raise RuntimeError(f"{episode}: {metadata.rosbag2_message_count} of {written} messages recorded, the log names the cause")

    metadata.recording_ended_at = iso(last_ns)
    MetadataWriter.write(metadata, episode / f"{EPISODE}.yaml")
    return episode
