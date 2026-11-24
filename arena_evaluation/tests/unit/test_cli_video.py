from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("mcap_ros2.writer")

from mcap_ros2.writer import Writer

from arena_evaluation.cli_video import add_video_commands, camera_frames, handle_video, image_to_rgb, resample, video_name

SEC = 1_000_000_000
IMAGE = (
    "sensor_msgs/msg/Image",
    "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data"
    + "\n" + "=" * 80 + "\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id"
    + "\n" + "=" * 80 + "\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec",
)  # fmt: skip


def _image(width: int, height: int, value: int, encoding: str = "rgb8") -> dict:
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}[encoding]
    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "camera"},
        "height": height,
        "width": width,
        "encoding": encoding,
        "is_bigendian": 0,
        "step": width * channels,
        "data": bytes([value]) * (width * height * channels),
    }


def _bag(path: pathlib.Path, topics: dict[str, list[tuple[int, dict]]]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Writer(stream)
        schema = writer.register_msgdef(*IMAGE)
        for topic, messages in topics.items():
            for log_time, message in messages:
                writer.write_message(topic, schema, message, log_time=log_time)
        writer.finish()
    return path


def test_image_to_rgb_swaps_bgr_and_expands_mono() -> None:
    bgr = image_to_rgb("bgr8", 1, 2, 6, bytes([1, 2, 3, 4, 5, 6]))
    assert bgr.tolist() == [[[3, 2, 1], [6, 5, 4]]]
    mono = image_to_rgb("mono8", 1, 2, 2, bytes([7, 9]))
    assert mono.tolist() == [[[7, 7, 7], [9, 9, 9]]]


def test_image_to_rgb_drops_row_padding() -> None:
    padded = image_to_rgb("rgb8", 2, 1, 4, bytes([1, 2, 3, 0, 4, 5, 6, 0]))
    assert padded.tolist() == [[[1, 2, 3]], [[4, 5, 6]]]


def test_image_to_rgb_refuses_16_bit_depth() -> None:
    with pytest.raises(ValueError, match="16UC1"):
        image_to_rgb("16UC1", 1, 1, 2, bytes(2))


def test_resample_holds_the_last_frame_over_a_gap() -> None:
    frames = [(0, np.full((1, 1, 3), 1, np.uint8)), (SEC, np.full((1, 1, 3), 2, np.uint8))]
    values = [int(frame[0, 0, 0]) for frame in resample(frames, fps=4.0)]
    assert values == [1, 1, 1, 1, 2]


def test_camera_frames_keeps_robots_apart_and_ignores_other_topics(tmp_path: pathlib.Path) -> None:
    bag = _bag(
        tmp_path / "ep.mcap",
        {
            "/env_0/jackal/vln_camera/image": [(0, _image(4, 2, 10)), (SEC, _image(4, 2, 20))],
            "/env_0/burger/vln_camera/image": [(0, _image(4, 2, 30))],
            "/env_0/jackal/vln_camera/rec_image": [(0, _image(4, 2, 40))],
        },
    )
    frames = camera_frames(bag)
    assert sorted(frames) == ["/env_0/burger/vln_camera/image", "/env_0/jackal/vln_camera/image"]
    assert [int(f[0, 0, 0]) for _, f in frames["/env_0/jackal/vln_camera/image"]] == [10, 20]
    assert video_name("/env_0/jackal/vln_camera/image") == "env_0_jackal_vln_camera_image.mp4"


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffmpeg and ffprobe on PATH")
def test_video_command_writes_one_mp4_per_episode_in_recorded_time(tmp_path: pathlib.Path) -> None:
    for episode in ("0", "1"):
        messages = [(i * SEC // 10, _image(33, 24, 10 * i)) for i in range(21)]
        _bag(tmp_path / "run" / "episodes" / episode / f"episode_{episode}.mcap", {"/env_0/jackal/vln_camera/image": messages})
    parser = argparse.ArgumentParser()
    add_video_commands(parser.add_subparsers(dest="command"))
    handle_video(parser.parse_args(["video", "--benchmark-dir", str(tmp_path / "run")]))
    for episode in ("0", "1"):
        video = tmp_path / "run" / "episodes" / episode / "video" / "env_0_jackal_vln_camera_image.mp4"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames,width", "-of", "json", str(video)],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert (int(stream["nb_read_frames"]), stream["width"]) == (21, 34)
