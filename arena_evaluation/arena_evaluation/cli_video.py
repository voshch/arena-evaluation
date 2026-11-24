"""CLI subcommand that exports recorded camera images of each episode bag as mp4 videos."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Iterator

import numpy as np
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

DEFAULT_TOPIC_SUFFIX = "/vln_camera/image"
_IMAGE_SCHEMA = "sensor_msgs/msg/Image"
_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}


def image_to_rgb(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """An 8-bit sensor_msgs/Image payload as a (height, width, 3) RGB array."""
    channels = _CHANNELS.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {encoding!r}, expected one of {sorted(_CHANNELS)}")
    rows = np.frombuffer(data, dtype=np.uint8, count=height * step).reshape(height, step)
    pixels = rows[:, : width * channels].reshape(height, width, channels)
    if channels == 1:
        return np.repeat(pixels, 3, axis=2)
    if encoding.startswith("bgr"):
        return np.ascontiguousarray(pixels[:, :, 2::-1])
    return np.ascontiguousarray(pixels[:, :, :3])


def camera_frames(bag: pathlib.Path, topic_suffix: str = DEFAULT_TOPIC_SUFFIX) -> dict[str, list[tuple[int, np.ndarray]]]:
    """The (log time ns, RGB frame) pairs of every Image topic in one bag that ends with topic_suffix, keyed by topic."""
    frames: dict[str, list[tuple[int, np.ndarray]]] = {}
    with bag.open("rb") as stream:
        reader = NonSeekingReader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages():
            if schema is None or schema.name != _IMAGE_SCHEMA or not channel.topic.endswith(topic_suffix):
                continue
            frame = image_to_rgb(decoded.encoding, decoded.height, decoded.width, decoded.step, bytes(decoded.data))
            frames.setdefault(channel.topic, []).append((message.log_time, frame))
    return frames


def resample(frames: list[tuple[int, np.ndarray]], fps: float) -> Iterator[np.ndarray]:
    """Frames on a fixed fps clock over the recorded time span, each tick showing the latest frame at or before it."""
    ordered = sorted(frames, key=lambda item: item[0])
    start, end = ordered[0][0], ordered[-1][0]
    ticks = int((end - start) * fps / 1e9) + 1
    index = 0
    for tick in range(ticks):
        t_ns = start + int(tick * 1e9 / fps)
        while index + 1 < len(ordered) and ordered[index + 1][0] <= t_ns:
            index += 1
        yield ordered[index][1]


def write_mp4(frames: list[tuple[int, np.ndarray]], path: pathlib.Path, fps: float) -> int:
    """Encode the frames as an H.264 mp4 at fps in recorded time and return the number of video frames written."""
    height, width = frames[0][1].shape[:2]
    command = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(path),
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for frame in resample(frames, fps):
            if frame.shape[:2] != (height, width):
                continue
            encoder.stdin.write(frame.tobytes())
            written += 1
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to write {path}")
    return written


def video_name(topic: str) -> str:
    """The mp4 file name of a camera topic: its path without the leading slash, slashes as underscores."""
    return topic.strip("/").replace("/", "_") + ".mp4"


def export_episode(bag: pathlib.Path, out_dir: pathlib.Path, topic_suffix: str, fps: float, force: bool) -> list[pathlib.Path]:
    """Write one mp4 per matching camera topic of one episode bag and return the paths written."""
    written = []
    for topic, frames in sorted(camera_frames(bag, topic_suffix).items()):
        target = out_dir / video_name(topic)
        if target.exists() and not force:
            print(f"skip {target} (exists, pass --force to overwrite)")
            continue
        count = write_mp4(frames, target, fps)
        print(f"{target}: {count} frames at {fps:g} fps from {len(frames)} images")
        written.append(target)
    return written


def handle_video(args: argparse.Namespace) -> None:
    """Handle 'evaluation video': export camera videos for every episode bag of a benchmark run."""
    benchmark_dir = args.benchmark_dir[0] if isinstance(args.benchmark_dir, list) else args.benchmark_dir
    if not benchmark_dir.is_dir():
        print(f"Error: benchmark directory does not exist: {benchmark_dir}")
        sys.exit(1)
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not on PATH.")
        sys.exit(1)
    bags = sorted(benchmark_dir.glob("episodes/*/*.mcap"))
    if args.episode is not None:
        bags = [bag for bag in bags if bag.parent.name == args.episode]
    if not bags:
        print(f"Error: no episode bags under {benchmark_dir}/episodes")
        sys.exit(1)
    total = 0
    for bag in bags:
        out_dir = bag.parent / "video" if args.out_dir is None else args.out_dir / bag.parent.name
        total += len(export_episode(bag, out_dir, args.topic, args.fps, args.force))
    if total == 0:
        print(f"No videos written: no {_IMAGE_SCHEMA} topic ending in {args.topic!r}. Record it with optional_topics: [planner_camera] in data_recorder_config.yaml.")


def add_video_commands(subparsers: argparse._SubParsersAction) -> None:
    """Add the video subcommand to the main parser."""
    parser = subparsers.add_parser("video", help="Export recorded camera images of each episode as mp4 videos.")
    parser.add_argument("--benchmark-dir", type=pathlib.Path, required=True, metavar="DIR", help="Benchmark run directory with episodes/<id>/*.mcap.")
    parser.add_argument("--episode", type=str, default=None, metavar="ID", help="Only this episode folder (default: all).")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC_SUFFIX, help=f"Image topic suffix to export (default: {DEFAULT_TOPIC_SUFFIX}).")
    parser.add_argument("--fps", type=float, default=10.0, help="Output frame rate in recorded time (default: 10, the recording rate).")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, metavar="DIR", help="Write <DIR>/<episode>/*.mp4 instead of episodes/<id>/video/.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing videos.")
