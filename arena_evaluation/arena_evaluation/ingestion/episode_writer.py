"""Episode directory writer shared by the recorder node and offline importers."""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import threading
from datetime import UTC, datetime

import rosbag2_py
from ament_index_python.packages import get_package_share_directory

from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata

_log = logging.getLogger(__name__)


class EpisodeWriter:
    """Writes one episode at a time as <dir>/<name>.mcap beside <name>.yaml, the layout the processing pipeline discovers."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir if base_dir is not None else get_package_share_directory("arena_evaluation")
        self.current_episode_dir: pathlib.Path | None = None
        self.current_metadata_path: pathlib.Path | None = None
        self.current_metadata: RunMetadata | None = None
        self.writer: rosbag2_py.SequentialWriter | None = None
        self.writer_lock = threading.Lock()
        self.topics_metadata: dict[str, rosbag2_py.TopicMetadata] = {}
        self._topic_registry: dict[str, rosbag2_py.TopicMetadata] = {}
        self._pre_episode_buffer: list[tuple[str, bytes, int]] = []
        self.latched_topic_names: set[str] = set()
        self.recorded_topics: set[str] = set()
        self.last_recorded_times: dict[str, int] = {}
        self._write_drop_count = 0
        self._write_success_count = 0

    def open_episode(self, episode_dir: pathlib.Path, metadata: RunMetadata) -> None:
        """Open episode_dir for writing, metadata is finalized and written on close."""
        self._enter_episode_dir(episode_dir)
        self.current_metadata = metadata
        self._open_writer()
        if self.writer is None:
            raise RuntimeError(f"no MCAP writer for {episode_dir}, the log names the cause")

    def write_message(self, topic_name: str, msg_type: type, serialized: bytes, timestamp_ns: int) -> None:
        self._register_topic(topic_name, msg_type)
        self._write_serialized(topic_name, serialized, timestamp_ns)

    def close_episode(self) -> None:
        self._close_current_writer()

    def _log_info(self, msg: str):
        _log.info(msg)

    def _log_warn(self, msg: str):
        _log.warning(msg)

    def _log_error(self, msg: str):
        _log.error(msg)

    def _enter_episode_dir(self, ep_dir: pathlib.Path):
        """Create the episode directory and point the writer state at it."""
        self._write_success_count = 0
        self._write_drop_count = 0
        self.recorded_topics = set()

        ep_dir.mkdir(parents=True, exist_ok=True)
        try:
            ep_dir.chmod(0o777)
        except Exception:
            pass

        self.current_episode_dir = ep_dir
        self.current_metadata_path = ep_dir / f"{ep_dir.name}.yaml"

    def _open_writer(self):
        """Open the MCAP writer of the current episode directory and flush the pre-episode buffer into it."""
        ep_dir = self.current_episode_dir
        assert ep_dir is not None
        ep_name = ep_dir.name

        bag_dir = ep_dir / ep_name
        if bag_dir.exists():
            index = 0
            while (kept := ep_dir / f"{ep_name}_unclosed_{index}").exists():
                index += 1
            bag_dir.rename(kept)
            self._log_warn(f"{ep_name} held a bag an earlier recorder left unclosed, kept as {kept.name}")

        mcap_uri = str(bag_dir)
        self._log_info(f"Opening MCAP writer for {ep_name}: uri={mcap_uri}")

        mcap_config_path = os.path.join(self.base_dir, "config", "mcap_writer_options.yaml")
        if not os.path.exists(mcap_config_path):
            self._log_warn("mcap_writer_options.yaml not found - recording without compression")
            mcap_config_path = ""

        storage_options = rosbag2_py.StorageOptions(
            uri=mcap_uri,
            storage_id='mcap',
            max_bagfile_size=0,
            max_cache_size=0,
            storage_config_uri=mcap_config_path,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )

        try:
            with self.writer_lock:
                self.writer = rosbag2_py.SequentialWriter()
                self.writer.open(storage_options, converter_options)
                self.topics_metadata = {}
                self.last_recorded_times = {}

                if self._pre_episode_buffer:
                    self._log_info(f"Flushing {len(self._pre_episode_buffer)} pre-episode buffered messages to writer")
                    for topic_name, serialized_msg, timestamp_ns in self._pre_episode_buffer:
                        self._ensure_topic_in_bag(topic_name)
                        self.writer.write(topic_name.strip('/'), serialized_msg, timestamp_ns)  # pyright: ignore[reportArgumentType]
                        self.recorded_topics.add(topic_name.strip('/'))
                    self._pre_episode_buffer.clear()
            self._log_info(f"MCAP writer opened for {ep_name}")
        except Exception as e:
            self._log_error(f"Failed to open MCAP writer for {ep_name}: {e}")
            self.writer = None

    def _close_current_writer(self):
        """Flush, close the current writer, and flatten the rosbag2 subdirectory."""

        with self.writer_lock:
            if self.writer is None:
                return
            try:
                self.writer.close()
            except Exception as e:
                self._log_error(f"Error closing MCAP writer: {e}")
            finally:
                self.writer = None

        if self.current_episode_dir is not None:
            ep_name = self.current_episode_dir.name
            inner_dir = self.current_episode_dir / ep_name
            if inner_dir.is_dir():
                for mcap_file in sorted(inner_dir.glob("*.mcap")):
                    dest = self.current_episode_dir / f"{ep_name}.mcap"
                    if dest.exists():
                        dest = self.current_episode_dir / f"{ep_name}_{mcap_file.stem}.mcap"
                    try:
                        shutil.move(str(mcap_file), str(dest))
                        self._log_info(f"Flattened {mcap_file.name} -> {dest.name}")
                    except Exception as e:
                        self._log_error(f"Failed to flatten {mcap_file}: {e}")

                rosbag_meta = inner_dir / "metadata.yaml"
                if rosbag_meta.exists():
                    try:
                        import yaml

                        with open(rosbag_meta) as f:
                            bag_info = yaml.safe_load(f).get("rosbag2_bagfile_information", {})

                        if self.current_metadata is not None:
                            self.current_metadata.rosbag2_message_count = bag_info.get("message_count")
                            self.current_metadata.rosbag2_topics = bag_info.get("topics_with_message_count")
                            self._log_info("Merged rosbag2 metadata into episode metadata.")

                        rosbag_meta.unlink()
                    except Exception as e:
                        self._log_error(f"Failed to merge rosbag metadata: {e}")

                try:
                    inner_dir.rmdir()
                except Exception as e:
                    self._log_warn(f"Failed to remove inner dir {inner_dir.name}: {e}")

        self._finalize_episode_metadata()

    def _write_serialized(self, topic_name: str, serialized_msg: bytes, timestamp_ns: int):
        """Write serialized bytes to the open episode, buffer latched topics while none is open, drop the rest."""
        with self.writer_lock:
            if self.writer is None:
                topic_stripped = topic_name.strip('/')
                if topic_stripped in self.latched_topic_names:
                    if len(self._pre_episode_buffer) < 10000:
                        self._pre_episode_buffer.append((topic_name, serialized_msg, timestamp_ns))
                        return

                self._write_drop_count += 1
                if self._write_drop_count <= 10 or self._write_drop_count % 100 == 0:
                    self._log_warn(f"DROP (writer=None) topic={topic_name} drop_count={self._write_drop_count}")
                return
            try:
                self._ensure_topic_in_bag(topic_name)
                self.writer.write(topic_name.strip('/'), serialized_msg, timestamp_ns)  # pyright: ignore[reportArgumentType]
                self.recorded_topics.add(topic_name.strip('/'))
                self._write_success_count += 1
                if self._write_success_count <= 5 or self._write_success_count % 500 == 0:
                    self._log_info(f"Write success count reached {self._write_success_count}")
            except Exception as e:
                self._log_error(f"WRITE ERROR topic={topic_name} ts={timestamp_ns} err={e}")

    def _register_topic(self, topic_name: str, msg_type: type):
        """Pre-register a topic so we know its type string before the first message."""
        if topic_name in self._topic_registry:
            return
        try:
            parts = msg_type.__module__.split('.')
            type_str = f"{parts[0]}/msg/{msg_type.__name__}"
        except AttributeError:
            type_str = str(msg_type)

        self._topic_registry[topic_name] = rosbag2_py.TopicMetadata(
            id=0,
            name=topic_name.strip('/'),
            type=type_str,
            serialization_format='cdr',
        )

    def _ensure_topic_in_bag(self, topic_name: str):
        """Lazily call create_topic on first write. Must be called under writer_lock."""
        assert self.writer is not None
        strip = topic_name.strip('/')
        if strip not in self.topics_metadata and topic_name in self._topic_registry:
            self.writer.create_topic(self._topic_registry[topic_name])
            self.topics_metadata[strip] = self._topic_registry[topic_name]

    def _finalize_episode_metadata(self):
        """Update episode yaml with final recording stats after writer is closed."""
        if self.current_metadata is None or self.current_metadata_path is None:
            return
        try:
            self.current_metadata.recording_ended_at = datetime.now(UTC).isoformat()
            self.current_metadata.pedsim_available = any("arena_peds" in t or "agent_states" in t for t in self.recorded_topics)
            self.current_metadata.recorded_topics = sorted(self.recorded_topics)
            MetadataWriter.write(self.current_metadata, self.current_metadata_path)
            self._log_info(f"Finalized episode metadata: {self.current_metadata_path}")
        except Exception as e:
            self._log_error(f"Failed to finalize episode metadata: {e}")
