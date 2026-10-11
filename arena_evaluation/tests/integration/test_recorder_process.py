import pathlib
import re
import signal
import subprocess
import sys
import time

import pytest
import yaml

NAMESPACE = "/recorder_process_test/env"
RESET_NAMESPACE = "/recorder_reset_test/env"
TICKS = 3000
RATE_HZ = 1000.0
RESET_AT_S = 100.0
ROSTER_HZ = 30.0
HISTORY_PUBLISHED = "history published"


def _drive() -> None:
    """Start an episode, publish TICKS clock messages at RATE_HZ, stop the episode."""
    import rclpy
    import rclpy.executors
    from arena_evaluation_msgs.srv import RecordEpisode
    from rosgraph_msgs.msg import Clock

    rclpy.init()
    node = rclpy.create_node("recorder_process_test_driver")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def call(**fields: object) -> None:
        future = client.call_async(RecordEpisode.Request(**fields))
        executor.spin_until_future_complete(future, timeout_sec=20.0)
        response = future.result()
        assert response is not None and response.success, response

    client = node.create_client(RecordEpisode, f"{NAMESPACE}/start_episode")
    assert client.wait_for_service(timeout_sec=30.0)
    publisher = node.create_publisher(Clock, "/clock", 10)
    deadline = time.monotonic() + 10.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert publisher.get_subscription_count() > 0

    call(command=RecordEpisode.Request.COMMAND_START, episode_id=0)
    start = time.monotonic()
    for tick in range(1, TICKS + 1):
        stamp_ns = tick * 1_000_000
        message = Clock()
        message.clock.sec = stamp_ns // 1_000_000_000
        message.clock.nanosec = stamp_ns % 1_000_000_000
        publisher.publish(message)
        time.sleep(max(0.0, start + tick / RATE_HZ - time.monotonic()))
    time.sleep(0.5)
    call(command=RecordEpisode.Request.COMMAND_STOP, episode_id=0, outcome_state=2, outcome_info="done")

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


def _drive_reset(clock: str) -> None:
    """Latch the records of four finished episodes, then run a fifth through a paused reset as the task generator and the runner do, with the clock frozen or silent during the reset."""
    import rclpy
    import rclpy.executors
    from arena_evaluation_msgs.srv import RecordEpisode
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from task_generator_msgs.msg import EpisodeRecord

    rclpy.init()
    node = rclpy.create_node("recorder_reset_test_driver")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def call(**fields: object) -> None:
        future = client.call_async(RecordEpisode.Request(**fields))
        executor.spin_until_future_complete(future, timeout_sec=20.0)
        response = future.result()
        assert response is not None and response.success, response

    def record(episode_id: int, state: int) -> None:
        records.publish(EpisodeRecord(episode_id=episode_id, outcome_state=state))

    def tick(sim_s: float) -> None:
        stamp_ns = round(sim_s * 1e9)
        message = Clock()
        message.clock.sec = stamp_ns // 1_000_000_000
        message.clock.nanosec = stamp_ns % 1_000_000_000
        clock.publish(message)

    def roster(name: str) -> None:
        peds.publish(Pedestrians(pedestrians=[Pedestrian(name=name)]))
        time.sleep(1.0 / ROSTER_HZ)

    client = node.create_client(RecordEpisode, f"{RESET_NAMESPACE}/start_episode")
    records = node.create_publisher(EpisodeRecord, f"{RESET_NAMESPACE}/state/episode", QoSProfile(depth=20, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    clock = node.create_publisher(Clock, "/clock", 10)
    peds = node.create_publisher(Pedestrians, f"{RESET_NAMESPACE.rsplit('/', 1)[0]}/arena_peds", 10)

    record(1, EpisodeRecord.RUNNING)
    record(1, EpisodeRecord.FAILED)
    for episode_id in (2, 3, 4):
        for state in (EpisodeRecord.QUEUED, EpisodeRecord.QUEUED, EpisodeRecord.QUEUED, EpisodeRecord.RUNNING, EpisodeRecord.SUCCESS):
            record(episode_id, state)
    print(HISTORY_PUBLISHED, flush=True)

    assert client.wait_for_service(timeout_sec=60.0)
    deadline = time.monotonic() + 10.0
    while min(records.get_subscription_count(), clock.get_subscription_count(), peds.get_subscription_count()) == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert min(records.get_subscription_count(), clock.get_subscription_count(), peds.get_subscription_count()) > 0

    for _ in range(3):
        record(5, EpisodeRecord.QUEUED)
        for _ in range(10):
            if clock == "frozen":
                tick(RESET_AT_S)
            roster("previous")
    record(5, EpisodeRecord.RUNNING)
    call(command=RecordEpisode.Request.COMMAND_START, episode_id=5)
    for step in range(1, 61):
        tick(RESET_AT_S + step / ROSTER_HZ)
        roster("current")
    record(5, EpisodeRecord.SUCCESS)
    call(command=RecordEpisode.Request.COMMAND_STOP, episode_id=5, outcome_state=EpisodeRecord.SUCCESS, outcome_info="done")

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


def _recorder(record_dir: pathlib.Path, namespace: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "arena_evaluation.ingestion.recorder", "--ros-args", "-p", f"record_data_dir:={record_dir}", "-p", "benchmark_id:=recorder_process_test", "-r", f"__ns:={namespace}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _terminate(recorder: subprocess.Popen) -> str:
    recorder.send_signal(signal.SIGTERM)
    try:
        output, _ = recorder.communicate(timeout=30.0)
    except subprocess.TimeoutExpired:
        recorder.kill()
        output, _ = recorder.communicate()
    return output


def _messages(mcap: pathlib.Path, topic: str, msg_type: type) -> list:
    import rosbag2_py
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(mcap), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    messages = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name == topic:
            messages.append(deserialize_message(data, msg_type))
    return messages


def test_recorder_keeps_up_with_a_fast_clock_and_finalizes_on_sigterm(tmp_path: pathlib.Path) -> None:
    recorder = _recorder(tmp_path, NAMESPACE)
    try:
        driver = subprocess.run([sys.executable, __file__, "clock"], capture_output=True, text=True, timeout=120.0, check=False)
    finally:
        output = _terminate(recorder)

    assert driver.returncode == 0, driver.stdout + driver.stderr
    assert recorder.returncode == 0, output
    received = re.search(r"clock_ticks=(\d+)", output)
    assert received is not None, output
    assert int(received.group(1)) >= 0.97 * TICKS, output
    assert list(tmp_path.rglob("episode_000.mcap")), output
    assert list(tmp_path.rglob("episode_000.yaml")), output


@pytest.mark.parametrize("clock", ["frozen", "silent"])
def test_recorder_opens_one_episode_when_the_reset_is_over(tmp_path: pathlib.Path, clock: str) -> None:
    from arena_people_msgs.msg import Pedestrians
    from task_generator_msgs.msg import EpisodeRecord

    driver = subprocess.Popen([sys.executable, __file__, "reset", clock], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert driver.stdout is not None
    published = driver.stdout.readline().strip()
    recorder = _recorder(tmp_path, RESET_NAMESPACE)
    try:
        try:
            driven, _ = driver.communicate(timeout=120.0)
        except subprocess.TimeoutExpired:
            driver.kill()
            driven, _ = driver.communicate()
    finally:
        output = _terminate(recorder)

    assert published == HISTORY_PUBLISHED, driven
    assert driver.returncode == 0, driven
    assert recorder.returncode == 0, output
    episodes = tmp_path / "episodes"
    assert sorted(path.name for path in episodes.iterdir() if path.is_dir()) == ["episode_000"], output
    assert sorted(path.name for path in (episodes / "episode_000").iterdir()) == ["episode_000.mcap", "episode_000.yaml"], output
    metadata = yaml.safe_load((episodes / "episode_000" / "episode_000.yaml").read_text())
    assert metadata["task_generator_episode_id"] == 5
    assert metadata["outcome_state"] == 2
    mcap = episodes / "episode_000" / "episode_000.mcap"
    batches = [[ped.name for ped in msg.pedestrians] for msg in _messages(mcap, f"{RESET_NAMESPACE.rsplit('/', 1)[0]}/arena_peds".strip("/"), Pedestrians)]
    assert len(batches) >= 20, output
    assert all(batch == ["current"] for batch in batches), batches
    records = [(msg.episode_id, msg.outcome_state) for msg in _messages(mcap, f"{RESET_NAMESPACE}/state/episode".strip("/"), EpisodeRecord)]
    assert (5, EpisodeRecord.RUNNING) in records, records


def test_recorder_records_into_an_episode_whose_bag_was_left_unclosed(tmp_path: pathlib.Path) -> None:
    episode = tmp_path / "episodes" / "episode_000"
    unclosed = episode / "episode_000"
    unclosed.mkdir(parents=True)
    (unclosed / "episode_000_0.mcap").write_bytes(b"")

    recorder = _recorder(tmp_path, NAMESPACE)
    try:
        driver = subprocess.run([sys.executable, __file__, "clock"], capture_output=True, text=True, timeout=120.0, check=False)
    finally:
        output = _terminate(recorder)

    assert driver.returncode == 0, driver.stdout + driver.stderr
    assert recorder.returncode == 0, output
    assert "Bag directory already exists" not in output
    assert (episode / "episode_000.mcap").stat().st_size > 0, output
    assert yaml.safe_load((episode / "episode_000.yaml").read_text())["outcome_state"] == 2
    assert not unclosed.exists()
    assert [path.name for path in (episode / "episode_000_unclosed_0").iterdir()] == ["episode_000_0.mcap"]


if __name__ == "__main__":
    if sys.argv[1] == "reset":
        _drive_reset(sys.argv[2])
    else:
        _drive()
