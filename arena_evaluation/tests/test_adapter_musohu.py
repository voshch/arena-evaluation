from __future__ import annotations

import http.server
import json
import re
import shutil
import threading
from pathlib import Path

import attrs
import numpy as np
import pytest
import yaml
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution

from arena_assets.importers.providers import Http

from arena_evaluation.adapters.musohu import ACCESS_URL, DATASET_URL, PROBE_HEAD, MuSoHu, Topics
from musohu_bag import BEAMS, CAMERA, CLOUD_POINTS, CMD_OFFSET_NS, CMD_VEL, COUNT, DISTORTION, FOCAL, HEIGHT, IMU, ODOM, PATH, POINTS, RANGE_MAX, SCAN, STAMP_LAG_NS, STEP_NS, VELOCITY, cloud_data, cmd_vel, convert, pose, scan_ranges, velocity, write_bag

EPISODE = "episodes/episode_000"
MCAP = f"{EPISODE}/episode_000.mcap"
META = f"{EPISODE}/episode_000.yaml"
ROBOT_NS = "arena/env_0/task_generator_node/human"
LIVE_ITEM = "02182023_frontal_approach_13"
LIVE_REFUSED = "02152023_front_avoid_passenger_1"
LIVE_FILE_ID = 564


def extract(run: Path) -> dict:
    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

    folders = FolderManager(run.parent)
    (episode,) = folders.discover_episodes(run.name)
    ((robot, bundle),) = ProcessingPipeline(folders, workers=1).extract_episode(episode).items()
    assert robot == "env_0_human"
    return {name: getattr(bundle, name).collect() for name in sorted(bundle.available())}


def decoded(run: Path) -> dict[str, list]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    messages: dict[str, list] = {}
    with open(run / MCAP, "rb") as f:
        for _, channel, message, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            messages.setdefault(channel.topic, []).append((message.log_time, msg))
    return messages


def test_registry_resolves_musohu():
    assert IMPORTERS.load("episode", "musohu") is MuSoHu
    assert MuSoHu.meta.kind == "episode"
    assert MuSoHu.default_provider().url("902") == f"{ACCESS_URL}/902"


def test_convert_writes_recorder_layout(converted: Path):
    assert converted.name == "run"
    assert sorted(p.relative_to(converted).as_posix() for p in converted.rglob("*") if p.is_file()) == [MCAP, META]


def test_topic_rows_keep_their_own_clock(converted: Path):
    topics = extract(converted)
    assert sorted(topics) == ["cmd_vel", "odom", "scan"]
    scan = topics["scan"]
    assert scan.columns == ["time_ns", "scan_ranges", "scan_min", "scan_range_max"]
    ranges = np.asarray(scan["scan_ranges"].to_list(), dtype=np.float32)
    assert ranges.shape == (COUNT, BEAMS)
    assert np.isinf(ranges[:, 0]).all()
    np.testing.assert_allclose(ranges[3, 1:], scan_ranges(3)[1:])
    assert scan["scan_min"][3] == pytest.approx(scan_ranges(3)[1])
    assert scan["scan_range_max"].unique().to_list() == [RANGE_MAX]
    odom = topics["odom"]
    assert odom.columns == ["time_ns", "stamp_ns", "pos_x", "pos_y", "yaw", "vel_linear", "vel_angular"]
    row = odom.row(7, named=True)
    assert (row["pos_x"], row["pos_y"], row["yaw"]) == pytest.approx(pose(7))
    assert (row["vel_linear"], row["vel_angular"]) == pytest.approx((velocity(7)[0], velocity(7)[2]))
    command = topics["cmd_vel"]
    assert (command["linear_x"][5], command["angular_z"][5]) == pytest.approx(cmd_vel(5))
    assert odom["stamp_ns"].to_list() == [i * STEP_NS for i in range(COUNT)]
    assert odom["time_ns"].to_list() == odom["stamp_ns"].to_list()
    assert scan["time_ns"].to_list() == odom["time_ns"].to_list()
    assert command["time_ns"].to_list() == [i * STEP_NS + STAMP_LAG_NS + CMD_OFFSET_NS for i in range(COUNT)]


def test_messages_carry_robot_frames_and_the_paired_twist(converted: Path):
    messages = decoded(converted)
    assert sorted(messages) == [f"{ROBOT_NS}/cmd_vel", f"{ROBOT_NS}/odom", f"{ROBOT_NS}/scan"]
    log_time, scan = messages[f"{ROBOT_NS}/scan"][0]
    assert (log_time, scan.header.stamp.sec, scan.header.stamp.nanosec) == (0, 0, 0)
    assert scan.header.frame_id == "env_0/human/lidar_link"
    assert (scan.range_min, scan.range_max, len(scan.intensities)) == (0.0, RANGE_MAX, BEAMS)
    _, odom = messages[f"{ROBOT_NS}/odom"][2]
    assert (odom.header.frame_id, odom.child_frame_id) == ("env_0/human/odom", "env_0/human/base_link")
    assert odom.pose.pose.position.z == HEIGHT
    assert (odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z) == pytest.approx(velocity(2))


def _stamp(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def test_other_bag_topics_pass_through_in_robot_frames(tmp_path: Path):
    pytest.importorskip("rosbags")
    run = convert(write_bag(tmp_path / "run.bag", cmd=True, sensors=True), tmp_path / "musohu", cmd_vel=CMD_VEL)
    messages = decoded(run)
    assert sorted(messages) == sorted(["tf", "tf_static", f"{ROBOT_NS}/cmd_vel", f"{ROBOT_NS}/odom", f"{ROBOT_NS}/scan", f"{ROBOT_NS}{CAMERA}", f"{ROBOT_NS}{IMU}", f"{ROBOT_NS}{PATH}", f"{ROBOT_NS}{POINTS}"])
    log_time, imu = messages[f"{ROBOT_NS}{IMU}"][3]
    assert (imu.header.frame_id, _stamp(imu.header), log_time) == ("env_0/human/zed2_imu_link", 3 * STEP_NS, 3 * STEP_NS + STAMP_LAG_NS + 1)
    assert (imu.angular_velocity.z, imu.linear_acceleration.z, list(imu.orientation_covariance)) == (velocity(3)[2], 9.81, [0.0] * 9)
    _, cloud = messages[f"{ROBOT_NS}{POINTS}"][3]
    assert (cloud.header.frame_id, cloud.width, [field.name for field in cloud.fields], cloud.point_step) == ("env_0/human/velodyne", CLOUD_POINTS, ["x", "y", "z"], 12)
    assert bytes(cloud.data) == cloud_data(3)
    _, tf = messages["tf"][3]
    (transform,) = tf.transforms
    assert (transform.header.frame_id, transform.child_frame_id, _stamp(transform.header)) == ("env_0/human/odom", "env_0/human/base_link", 3 * STEP_NS)
    assert transform.transform.translation.x == pytest.approx(pose(3)[0])
    ((_, static),) = messages["tf_static"]
    (mount,) = static.transforms
    assert (mount.header.frame_id, mount.child_frame_id, mount.transform.translation.z) == ("env_0/human/base_link", "env_0/human/velodyne", 0.5)
    _, trail = messages[f"{ROBOT_NS}{PATH}"][3]
    assert [(pose_.header.frame_id, _stamp(pose_.header)) for pose_ in trail.poses] == [("env_0/human/odom", j * STEP_NS) for j in range(4)]
    _, camera = messages[f"{ROBOT_NS}{CAMERA}"][3]
    assert (camera.header.frame_id, list(camera.d), camera.k[0], camera.p[5], list(camera.r)) == ("env_0/human/zed2_left_camera_optical_frame", list(DISTORTION), FOCAL, FOCAL, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    with open(run / META) as f:
        meta = yaml.safe_load(f)
    counts = {entry["topic_metadata"]["name"]: (entry["topic_metadata"]["type"], entry["message_count"]) for entry in meta["rosbag2_topics"]}
    assert counts[f"{ROBOT_NS}{POINTS}"] == ("sensor_msgs/msg/PointCloud2", COUNT)
    assert counts["tf_static"] == ("tf2_msgs/msg/TFMessage", 1)
    assert meta["rosbag2_message_count"] == sum(len(listed) for listed in messages.values()) == 3 * COUNT + 5 * COUNT + 1
    assert {"cmd_vel", "odom", "scan"} <= set(extract(run))


def test_passed_through_messages_are_written_in_time_order(tmp_path: Path):
    pytest.importorskip("rosbags")
    from mcap.reader import make_reader

    run = convert(write_bag(tmp_path / "run.bag", cmd=True, sensors=True), tmp_path / "musohu", cmd_vel=CMD_VEL)
    with open(run / MCAP, "rb") as f:
        times = [message.log_time for _, _, message in make_reader(f).iter_messages(log_time_order=False)]
    assert times == sorted(times)
    assert len(times) == 8 * COUNT + 1


def test_metadata_agrees_with_recording(converted: Path):
    with open(converted / META) as f:
        meta = yaml.safe_load(f)
    expected = {
        "benchmark_id": "run",
        "stage": "run",
        "suite_name": "musohu",
        "planner": "human",
        "local_planner": "human",
        "robot_model": ["human"],
        "agent_name": "human",
        "map": "unknown",
        "episode_id": 0,
        "episodes_requested": 1,
        "is_reference": True,
        "reference_type": "human_demonstration",
        "env_ns_root": "/arena/env_0/task_generator_node",
        "pedsim_available": False,
        "recording_started_at": "2023-11-14T22:13:19.980000+00:00",
        "recording_ended_at": "2023-11-14T22:13:23.905000+00:00",
        "recorded_topics": [f"{ROBOT_NS}/cmd_vel", f"{ROBOT_NS}/odom", f"{ROBOT_NS}/scan"],
        "rosbag2_message_count": 3 * COUNT,
    }
    assert {key: meta[key] for key in expected} == expected
    counts = {entry["topic_metadata"]["name"]: (entry["topic_metadata"]["type"], entry["message_count"]) for entry in meta["rosbag2_topics"]}
    assert counts == {
        f"{ROBOT_NS}/cmd_vel": ("geometry_msgs/msg/Twist", COUNT),
        f"{ROBOT_NS}/odom": ("nav_msgs/msg/Odometry", COUNT),
        f"{ROBOT_NS}/scan": ("sensor_msgs/msg/LaserScan", COUNT),
    }
    assert sum(frame.height for frame in extract(converted).values()) == meta["rosbag2_message_count"]


def test_metadata_reads_as_run_metadata(converted: Path):
    from arena_evaluation.storage.folder_manager import FolderManager
    from arena_evaluation.storage.manifest import MetadataWriter

    (episode,) = FolderManager(converted.parent).discover_episodes("run")
    assert (episode.episode_id, episode.planner, episode.stage, episode.map) == (0, "human", "run", "unknown")
    assert (episode.is_reference, episode.reference_type) == (True, "human_demonstration")
    assert MetadataWriter.read(converted / META).robot_model == ["human"]


def test_evaluation_pipeline_scores_the_episode(converted: Path):
    from arena_evaluation.processing.parquet_store import ParquetStore
    from arena_evaluation.processing.pipeline import ProcessingPipeline
    from arena_evaluation.storage.folder_manager import FolderManager

    folders = FolderManager(converted.parent)
    (episode,) = folders.discover_episodes(converted.name)
    (row,) = ProcessingPipeline(folders, workers=1).process_episode(episode)
    assert (row["status"], row["pose_source"]) == ("path_only", "odom")
    assert (row["robot"], row["planner"], row["stage"], row["map"]) == ("env_0_human", "human", "run", "unknown")
    assert (row["is_reference"], row["reference_type"]) == (True, "human_demonstration")
    assert row["path_length"] == pytest.approx(0.1 * (COUNT - 1))
    assert row["time_to_goal"] == pytest.approx(0.1 * (COUNT - 1))
    assert row["velocity_mean"] == pytest.approx(np.mean([velocity(i)[0] for i in range(COUNT)]))
    assert row["start"] == pytest.approx(pose(0))
    assert row["goal"] == pytest.approx(pose(COUNT - 1))
    assert row["num_pedestrians"] == 0
    stored, _ = ParquetStore.read(converted / EPISODE / "metrics.parquet")
    assert stored["status"].to_list() == ["path_only"]
    assert stored["path_length"].to_list() == [row["path_length"]]


def test_stray_cmd_vel_adds_no_frame(tmp_path: Path):
    pytest.importorskip("rosbags")
    from arena_evaluation.processing.recorded_episodes import RecordedEpisodes

    run = convert(write_bag(tmp_path / "stray.bag", cmd=True, stray=3), tmp_path / "musohu", cmd_vel=CMD_VEL)
    assert extract(run)["cmd_vel"].height == COUNT + 3
    assert RecordedEpisodes(run).frames().height == COUNT


def test_without_cmd_vel_the_velocity_stays_in_odom(tmp_path: Path, bag: Path):
    topics = extract(convert(bag, tmp_path / "musohu"))
    assert sorted(topics) == ["odom", "scan"]
    np.testing.assert_allclose(topics["odom"]["vel_linear"].to_numpy(), [velocity(i)[0] for i in range(COUNT)])
    assert topics["scan"]["time_ns"].to_list() == [i * STEP_NS for i in range(COUNT)]


def test_odom_without_velocity_partner_is_dropped(tmp_path: Path):
    pytest.importorskip("rosbags")
    odom = extract(convert(write_bag(tmp_path / "gaps.bag", unpaired=(7, 8)), tmp_path / "musohu"))["odom"]
    assert odom["stamp_ns"].to_list() == [i * STEP_NS for i in range(COUNT) if i not in (7, 8)]
    np.testing.assert_allclose(odom["vel_angular"].to_numpy(), velocity(0)[2])


def test_velocity_none_keeps_the_odom_twist(tmp_path: Path):
    pytest.importorskip("rosbags")
    odom = extract(convert(write_bag(tmp_path / "twist.bag", twist_in_odom=True), tmp_path / "musohu", velocity=None))["odom"]
    np.testing.assert_allclose(odom["vel_linear"].to_numpy(), [velocity(i)[0] for i in range(COUNT)])


def test_default_topics_name_the_live_bag_layout():
    assert Topics() == Topics(scan="/scan", odom="/zed2/zed_node/odom", velocity="/action", cmd_vel=None)
    assert (SCAN, ODOM, VELOCITY) == attrs.astuple(Topics())[:3]


def test_missing_topic_lists_available(tmp_path: Path):
    pytest.importorskip("rosbags")
    bag = write_bag(tmp_path / "bare.bag", twist_in_odom=True)
    with pytest.raises(ValueError, match=r"lacks topics \['/action'\], has \['/scan', '/zed2/zed_node/odom'\]"):
        MuSoHu(cache=tmp_path / "cache").convert("bare", bag, tmp_path / "musohu")


def test_import_item_from_cache_writes_attribution(tmp_path: Path, bag: Path):
    pytest.importorskip("rosbag2_py")
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy(bag, cache / "run.bag")
    importer = MuSoHu(cache=cache, topics=Topics(cmd_vel=CMD_VEL))
    target = importer.import_item("run", tmp_path / "musohu")
    assert target == tmp_path / "musohu" / "run"
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert attribution.license == "CC0-1.0"
    assert attribution.url == DATASET_URL
    assert "HZI4LJ" in attribution.citation
    assert (target / MCAP).is_file()
    assert importer.import_item("run", tmp_path / "musohu") == target
    assert len(extract(target)["odom"]) == COUNT


@pytest.mark.network
def test_live_listing_names_bags_and_license(tmp_path: Path):
    importer = MuSoHu(cache=tmp_path / "cache")
    items = list(importer.items())
    assert len(items) >= 287
    assert LIVE_ITEM in items
    assert importer.files()[LIVE_ITEM]["id"] == LIVE_FILE_ID
    assert importer.listing()["license"]["rightsIdentifier"] == importer.attribution(LIVE_ITEM).license
    assert (tmp_path / "cache" / "dataset.json").stat().st_size < 1_000_000


@pytest.mark.network
def test_live_probe_refuses_a_bag_without_action(tmp_path: Path):
    cache = tmp_path / "cache"
    importer = MuSoHu(cache=cache)
    entry = importer.files()[LIVE_REFUSED]
    with pytest.raises(ValueError, match=rf"{LIVE_REFUSED}\.bag lacks topics \['/action'\], has \[.*'/scan'.*'/zed2/zed_node/odom'"):
        importer.fetch(LIVE_REFUSED)
    probe = json.loads((cache / f"{LIVE_REFUSED}.topics.json").read_text())
    assert probe["bytes"] < 200_000 < entry["filesize"]
    assert sorted(p.name for p in cache.iterdir()) == sorted(["dataset.json", f"{LIVE_REFUSED}.topics.json"])


class RangeServer(http.server.ThreadingHTTPServer):
    def __init__(self, root: Path):
        super().__init__(("127.0.0.1", 0), RangeHandler)
        self.root = root
        self.served: list[tuple[str, str | None, int]] = []
        self.cap: int | None = None


class RangeHandler(http.server.BaseHTTPRequestHandler):
    server: RangeServer

    def do_GET(self):
        data = (self.server.root / self.path.lstrip("/")).read_bytes()
        ranged = self.headers.get("Range")
        if ranged:
            first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", ranged).groups())
            body = data[first : last + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{first + len(body) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        cap = self.server.cap
        if cap is not None and len(body) > cap:
            body = body[:cap]
        else:
            self.send_header("Content-Length", str(len(body)))
        self.server.served.append((self.path, ranged, len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object):
        pass


@pytest.fixture
def host(tmp_path: Path):
    pytest.importorskip("rosbags")
    server = RangeServer(tmp_path / "host")
    server.root.mkdir()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def hosted(host: RangeServer, cache: Path, item: str, **bag: object) -> tuple[MuSoHu, Path]:
    source = write_bag(host.root / "41", **bag)
    cache.mkdir()
    listing = {"data": {"latestVersion": {"files": [{"dataFile": {"id": 41, "filename": f"{item}.bag", "filesize": source.stat().st_size}}]}}}
    (cache / "dataset.json").write_text(json.dumps(listing))
    return MuSoHu(cache=cache, provider=Http(f"http://127.0.0.1:{host.server_address[1]}")), source


def test_probe_refuses_a_bag_without_velocity_before_its_body(tmp_path: Path, host):
    cache = tmp_path / "cache"
    importer, source = hosted(host, cache, "gaps", twist_in_odom=True)
    with pytest.raises(ValueError, match=r"gaps\.bag lacks topics \['/action'\], has \['/scan', '/zed2/zed_node/odom'\]"):
        importer.fetch("gaps")
    assert sorted(p.name for p in cache.iterdir()) == ["dataset.json", "gaps.topics.json"]
    assert [ranged is not None for _, ranged, _ in host.served] == [True, True]
    assert host.served[0][2] == PROBE_HEAD
    probe = json.loads((cache / "gaps.topics.json").read_text())
    assert probe == {"topics": ["/scan", "/zed2/zed_node/odom"], "bytes": sum(n for *_, n in host.served)}
    assert probe["bytes"] < source.stat().st_size
    with pytest.raises(ValueError, match="lacks topics"):
        importer.fetch("gaps")
    assert len(host.served) == 2


def test_probe_admits_a_complete_bag_and_downloads_it_once(tmp_path: Path, host):
    cache = tmp_path / "cache"
    importer, source = hosted(host, cache, "full")
    bag = importer.fetch("full")
    assert bag == cache / "full.bag"
    assert bag.read_bytes() == source.read_bytes()
    assert [ranged is not None for _, ranged, _ in host.served] == [True, True, False]
    assert json.loads((cache / "full.topics.json").read_text())["topics"] == ["/action", "/scan", "/zed2/zed_node/odom"]
    assert importer.fetch("full") == bag
    assert len(host.served) == 3


def test_fetch_resumes_a_body_the_host_ends_early(tmp_path: Path, host):
    cache = tmp_path / "cache"
    importer, source = hosted(host, cache, "cut")
    size = source.stat().st_size
    importer.probe("cut", importer.files()["cut"])
    host.cap = size // 3 + 1
    bag = importer.fetch("cut")
    assert bag.read_bytes() == source.read_bytes()
    body = [(ranged, n) for _, ranged, n in host.served[2:]]
    assert body == [(None, host.cap), (f"bytes={host.cap}-{size - 1}", host.cap), (f"bytes={2 * host.cap}-{size - 1}", size - 2 * host.cap)]
    assert sorted(p.name for p in cache.iterdir()) == ["cut.bag", "cut.topics.json", "dataset.json"]
