from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rclpy")

from mcp_fixtures import error_text, payload, run_session, server_env

PEER = r"""
import json
import sys
import threading

import rclpy
from arena_evaluation_msgs.msg import BenchmarkState
from arena_evaluation_msgs.srv import RecordEpisode
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from task_generator_msgs.msg import RecordedTopic, RecordedTopics

rclpy.init()
node = rclpy.create_node("task_generator_node", namespace="/arena/env_0")
latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def serve(request, response):
    print(json.dumps({"command": request.command, "episode_id": request.episode_id, "outcome_state": request.outcome_state, "outcome_info": request.outcome_info}), flush=True)
    response.success = True
    response.message = f"recording episode {request.episode_id}"
    return response


node.create_service(RecordEpisode, "/arena/env_0/task_generator_node/start_episode", serve)
topics = node.create_publisher(RecordedTopics, "/arena/env_0/task_generator_node/state/recorded_topics", latched)
topics.publish(RecordedTopics(topics=[RecordedTopic(key="odom", topic="{ns}/odom", msg_type="nav_msgs/msg/Odometry", robot_scoped=True, recorded=True)]))
state = node.create_publisher(BenchmarkState, "/arena/benchmark/state", latched)
state.publish(BenchmarkState(run_id="r1", suite="basic", contest="basic", steps_total=4, steps_done=1, active_keys=["dwb/s1"]))
done = threading.Event()
threading.Thread(target=lambda: (sys.stdin.read(), done.set()), daemon=True).start()
executor = SingleThreadedExecutor()
executor.add_node(node)
print("ready", flush=True)
while not done.is_set():
    executor.spin_once(timeout_sec=0.1)
node.destroy_node()
rclpy.shutdown()
"""


def test_recording_and_benchmark_state_reach_the_env_nodes(tmp_path):
    ros = {"ROS_DOMAIN_ID": str(90 + os.getpid() % 10), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
    peer = subprocess.Popen([sys.executable, "-c", PEER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env={**os.environ, **ros})
    assert peer.stdout.readline().strip() == "ready"

    async def body(client):
        out = {}
        out["watch"] = payload(await client.call_tool("evaluation_watch", {"timeout_s": 15}))
        out["topics"] = payload(await client.call_tool("evaluation_recorded_topics", {}))
        out["start"] = payload(await client.call_tool("evaluation_record_start", {"episode_id": 3}))
        out["stop"] = payload(await client.call_tool("evaluation_record_stop", {"episode_id": 3, "outcome": "failed", "info": "timeout"}))
        out["other_env"] = error_text(await client.call_tool("evaluation_record_start", {"episode_id": 1, "env_id": 5}))
        return out

    try:
        out = run_session(body, server_env(tmp_path, **ros), ("evaluation",))
    finally:
        calls, _ = peer.communicate("", timeout=30)
    assert out["watch"]["run_id"] == "r1" and out["watch"]["steps_total"] == 4 and out["watch"]["active_keys"] == ["dwb/s1"]
    assert out["topics"]["task_generator"] == "/arena/env_0/task_generator_node"
    assert out["topics"]["topics"][0]["key"] == "odom" and out["topics"]["topics"][0]["robot_scoped"] is True
    assert out["start"]["message"] == "recording episode 3"
    assert "cannot pick an env" in out["other_env"]
    requests = [json.loads(line) for line in calls.splitlines() if line.startswith("{")]
    assert requests == [{"command": 1, "episode_id": 3, "outcome_state": 0, "outcome_info": ""}, {"command": 2, "episode_id": 3, "outcome_state": 3, "outcome_info": "timeout"}]
