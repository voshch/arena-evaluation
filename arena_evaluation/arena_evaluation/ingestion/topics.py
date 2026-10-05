from __future__ import annotations

import dataclasses
import functools
import types


@dataclasses.dataclass
class TopicDefinition:
    """Definition of a topic to record."""

    name_template: str
    msg_type: type
    throttled: bool = True
    throttle_rate_hz: float = 10.0
    qos_transient_local: bool = False
    recorded: bool = True


@functools.cache
def _auditory() -> tuple[types.ModuleType, types.ModuleType] | None:
    """The arena_auditory constants and message modules, None without the auditory feature."""
    try:
        import arena_auditory_msgs.msg

        import arena_auditory.constants
    except ImportError:
        return None
    return arena_auditory.constants, arena_auditory_msgs.msg


def get_topics(namespace: str, parent_namespace: str = "") -> dict[str, TopicDefinition]:
    """Return the dictionary of topics to subscribe to."""
    from arena_humansim_msgs.msg import AgentFrame, AgentMeta
    from arena_people_msgs.msg import Pedestrians
    from arena_robots_msgs.msg import Acoustics, CollisionEvents, Energy, Power
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
    from nav2_msgs.msg import CollisionMonitorState
    from nav_msgs.msg import OccupancyGrid, Path
    from sensor_msgs.msg import JointState, LaserScan
    from std_msgs.msg import String
    from task_generator_msgs.msg import EpisodeRecord, RobotFleet, SemanticSnapshot
    from tf2_msgs.msg import TFMessage

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
        "tf_humans": TopicDefinition(f"{env_ns}/humans/tf", TFMessage, throttled=True),
        "peds": TopicDefinition(f"{env_ns}/arena_peds", Pedestrians, throttled=True),
        "agent_states": TopicDefinition(f"{p_ns}/agent_states", AgentFrame, throttled=True),
        "agent_meta": TopicDefinition(f"{p_ns}/agent_meta", AgentMeta, throttled=False, qos_transient_local=True),
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
        "map": TopicDefinition(f"{p_ns}/map", OccupancyGrid, throttled=False, qos_transient_local=True),
        "door_mask": TopicDefinition(f"{p_ns}/door_mask", OccupancyGrid, throttled=False, qos_transient_local=True),
    }

    auditory = _auditory()
    if auditory is not None:
        constants, msgs = auditory
        stream = constants.ArrayStream
        topics.update(
            {
                # AudioFrame.header.stamp is the simulation time of its first sample
                "audio_raw": TopicDefinition(constants.array_stream(ns, stream.RAW), msgs.AudioFrame, throttled=False),
                "audio_stem_motor": TopicDefinition(constants.array_stream(ns, stream.STEM_MOTOR), msgs.AudioFrame, throttled=False),
                "audio_stem_pedestrian": TopicDefinition(constants.array_stream(ns, stream.STEM_PEDESTRIAN), msgs.AudioFrame, throttled=False),
                "audio_stem_ambient": TopicDefinition(constants.array_stream(ns, stream.STEM_AMBIENT), msgs.AudioFrame, throttled=False),
                "audio_rendered": TopicDefinition(constants.array_stream(ns, stream.MONITOR), msgs.AudioFrame, throttled=False),
                "audio_render_inputs": TopicDefinition(constants.array_stream(ns, stream.RENDER_INPUTS), String, throttled=False),
                "heard_sound_events": TopicDefinition(f"{p_ns}/{constants.HEARD_SOUND_EVENTS}", msgs.HeardSoundEvent, throttled=False),
                "continuous_heard_sounds": TopicDefinition(f"{p_ns}/{constants.CONTINUOUS_HEARD_SOUNDS}", msgs.ContinuousHeardSoundState, throttled=False),
                "room_impulses": TopicDefinition(f"{p_ns}/{constants.ROOM_IMPULSES}", msgs.RoomImpulse, throttled=False, qos_transient_local=True, recorded=False),
            }
        )

    return topics
