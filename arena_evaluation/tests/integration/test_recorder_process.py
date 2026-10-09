import pathlib
import re
import signal
import subprocess
import sys
import time

NAMESPACE = "/recorder_process_test/env"
TICKS = 3000
RATE_HZ = 1000.0


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


def test_recorder_keeps_up_with_a_fast_clock_and_finalizes_on_sigterm(tmp_path: pathlib.Path) -> None:
    recorder = subprocess.Popen(
        [sys.executable, "-m", "arena_evaluation.ingestion.recorder", "--ros-args", "-p", f"record_data_dir:={tmp_path}", "-p", "benchmark_id:=recorder_process_test", "-r", f"__ns:={NAMESPACE}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        driver = subprocess.run([sys.executable, __file__], capture_output=True, text=True, timeout=120.0, check=False)
    finally:
        recorder.send_signal(signal.SIGTERM)
        try:
            output, _ = recorder.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            recorder.kill()
            output, _ = recorder.communicate()

    assert driver.returncode == 0, driver.stdout + driver.stderr
    assert recorder.returncode == 0, output
    received = re.search(r"clock_ticks=(\d+)", output)
    assert received is not None, output
    assert int(received.group(1)) >= 0.97 * TICKS, output
    assert list(tmp_path.rglob("episode_000.mcap")), output
    assert list(tmp_path.rglob("episode_000.yaml")), output


if __name__ == "__main__":
    _drive()
