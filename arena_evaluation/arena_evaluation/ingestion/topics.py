from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TopicDefinition:
    """Definition of a topic to record."""

    name_template: str
    msg_type: type
    throttled: bool = True
    throttle_rate_hz: float = 10.0
    qos_transient_local: bool = False


def get_topics(namespace: str, parent_namespace: str = "") -> dict[str, TopicDefinition]:
    """Return the dictionary of topics to subscribe to."""
    from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
    from sensor_msgs.msg import JointState, LaserScan
    from nav_msgs.msg import OccupancyGrid, Path
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
    from arena_people_msgs.msg import Pedestrians
    from arena_humansim_msgs.msg import AgentStates
    from task_generator_msgs.msg import AudioFrame, ContinuousHeardSoundState, EpisodeRecord, HeardSoundEvent, RobotFleet, SemanticSnapshot
    from arena_robots_msgs.msg import CollisionEvents, Power, Energy, Acoustics
    from nav2_msgs.msg import CollisionMonitorState

    ns = f"/{namespace}" if namespace else ""
    p_ns = f"/{parent_namespace}" if parent_namespace else ""
    if ns == "/":
        ns = ""
    if p_ns == "/":
        p_ns = ""
    # The human simulator publishes under the task generator's namespace, one level above its node name.
    env_ns = p_ns.rsplit("/", 1)[0]

    topics = {
        "cmd_vel": TopicDefinition(f"{ns}/cmd_vel", Twist, throttled=True),
        "scan": TopicDefinition(f"{ns}/scan", LaserScan, throttled=True),
        "lidar": TopicDefinition(f"{ns}/lidar", LaserScan, throttled=True),
        "joint_states": TopicDefinition(f"{ns}/joint_states", JointState, throttled=True),
        "plan": TopicDefinition(f"{ns}/plan", Path, throttled=False),
        "goal_pose": TopicDefinition(f"{ns}/goal_pose", PoseStamped, throttled=False),
        "initialpose": TopicDefinition(f"{p_ns}/initialpose", PoseWithCovarianceStamped, throttled=False),
        "tf": TopicDefinition("/tf", TFMessage, throttled=True),
        "tf_static": TopicDefinition("/tf_static", TFMessage, throttled=False, qos_transient_local=True),
        "peds": TopicDefinition(f"{env_ns}/arena_peds", Pedestrians, throttled=True),
        "agent_states": TopicDefinition(f"{p_ns}/agent_states", AgentStates, throttled=True),
        "episode_record": TopicDefinition(f"{p_ns}/state/episode", EpisodeRecord, throttled=False, qos_transient_local=True),
        "robots_fleet": TopicDefinition(f"{p_ns}/state/robots", RobotFleet, throttled=False, qos_transient_local=True),
        "semantic_snapshot": TopicDefinition(f"{p_ns}/state/semantics", SemanticSnapshot, throttled=False, qos_transient_local=True),
        "collision_events": TopicDefinition(f"{ns}/collision_events", CollisionEvents, throttled=False),
        "power": TopicDefinition(f"{ns}/power_publisher/power", Power, throttled=True),
        "energy": TopicDefinition(f"{ns}/power_publisher/energy", Energy, throttled=True),
        "acoustics": TopicDefinition(f"{ns}/acoustics", Acoustics, throttled=True),
        "characterization_phase": TopicDefinition(f"{ns}/characterization_phase", String, throttled=False),
        "characterization_schedule": TopicDefinition(f"{ns}/characterization_schedule", String, throttled=False, qos_transient_local=True),
        "collision_monitor_state": TopicDefinition(f"{ns}/collision_monitor_state", CollisionMonitorState, throttled=False),
        # AudioFrame.header.stamp is the simulation time of its first sample.
        # These high-rate streams must never be throttled by the evaluator.
        "audio_raw": TopicDefinition(f"{ns}/audio/raw_array", AudioFrame, throttled=False),
        "audio_stem_motor": TopicDefinition(f"{ns}/audio/stem_motor", AudioFrame, throttled=False),
        "audio_stem_pedestrian": TopicDefinition(f"{ns}/audio/stem_pedestrian", AudioFrame, throttled=False),
        "audio_rendered": TopicDefinition(f"{ns}/audio/headphones/stereo", AudioFrame, throttled=False),
        # One JSON object per audio block, so a replay can rebuild the mix. Throttling would drop blocks.
        "audio_render_inputs": TopicDefinition(f"{ns}/audio/diagnostics/render_inputs", String, throttled=False),
        # Carry the per-event `occluded` flag. Task generator level, not per robot.
        "heard_sound_events": TopicDefinition(f"{p_ns}/heard_sound_events", HeardSoundEvent, throttled=False),
        "four_mic_heard_sound_events": TopicDefinition(f"{p_ns}/four_mic_heard_sound_events", HeardSoundEvent, throttled=False),
        "continuous_heard_sounds": TopicDefinition(f"{p_ns}/continuous_heard_sounds", ContinuousHeardSoundState, throttled=False),
        "map": TopicDefinition(f"{p_ns}/map", OccupancyGrid, throttled=False, qos_transient_local=True),
        "door_mask": TopicDefinition(f"{p_ns}/door_mask", OccupancyGrid, throttled=False, qos_transient_local=True),
    }

    return topics
