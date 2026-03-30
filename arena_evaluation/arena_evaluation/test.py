#!/usr/bin/env python3

import os
import re
import csv
import time
import yaml
import math
import argparse
import traceback
import numpy  as np
from pathlib  import Path
from datetime import datetime

from std_msgs.msg       import Int16
from nav_msgs.msg       import Odometry
from sensor_msgs.msg    import LaserScan
from geometry_msgs.msg  import Twist
from rosgraph_msgs.msg  import Clock
# for transformations
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.node                  import Node
from rclpy.qos                   import QoSProfile
from rclpy.qos                   import QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.time                  import Time
from rclpy.parameter             import Parameter
from rclpy.executors             import MultiThreadedExecutor
from rclpy.serialization         import serialize_message
from rosgraph_msgs.msg           import Clock
from ament_index_python.packages import get_package_share_directory

# from arena_evaluation.scripts.utils import Pedestrian
# import pedsim_msgs.msg           as pedsim_msgs
import arena_evaluation_msgs.srv as arena_evaluation_srvs
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

from hunav_msgs.msg import Agents



class DataCollector(Node):

    def __init__(self, topic, unique_name):

        super().__init__(f'data_collector{unique_name}')

        topic_callbacks = [
            ("scan", self.laserscan_callback),
            ("odom", self.odometry_callback),
            ("cmd_vel", self.action_callback)
            # ("pedsim_agents_data", self.pedsim_callback)
        ]

        try:
            callback = lambda msg: [t[1] for t in topic_callbacks if t[0] == topic[1]][0](msg)
        except Exception as e:
            self.get_logger().error(f"Error in callback setup: {e}")
            traceback.print_exc()
            return

        self.full_topic_name = topic[1]
        self.data = None

        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.subscription = self.create_subscription(
            topic[2],
            topic[0],
            callback,
            self.qos  
        )
    
    def laserscan_callback(self, msg_laserscan: LaserScan):

        self.data = [msg_laserscan.range_max if math.isnan(val) else round(val, 3) for val in msg_laserscan.ranges]

    def odometry_callback(self, msg_odometry: Odometry):
        
        pose3d = msg_odometry.pose.pose
        twist = msg_odometry.twist.twist

        roll, pitch, yaw = euler_from_quaternion([
            pose3d.orientation.x, 
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w
        ])

        self.data = {
            "position": [
                round(pose3d.position.x, 3),
                round(pose3d.position.y, 3),
                round(yaw, 3)  
            ],
            "velocity": [
                round(twist.linear.x, 3),
                round(twist.linear.y, 3),
                round(twist.angular.z, 3)
            ],
        }

    def action_callback(self, msg_action: Twist): # variables will be written to csv whenever an action is published

        self.data = [
            round(msg_action.linear.x, 3),
            round(msg_action.linear.y, 3),
            round(msg_action.angular.z, 3)
        ]

    def get_data(self):
        
        return (
            self.full_topic_name,
            self.data 
        )
    
    def episode_callback(self, msg_scenario_reset):
        
        print(msg_scenario_reset)
        
        self.data = msg_scenario_reset.data

    # def pedsim_callback(self, msg_pedsim: pedsim_msgs.PedsimAgentsDataframe):
    #     self.data = [
    #         Pedestrian(
    #             id = agent.id,
    #             type = agent.type,
    #             social_state = agent.social_state,
    #             position = [agent.pose.position.x, agent.pose.position.y],
    #             theta = np.arctan2(agent.forces.force.y, agent.forces.force.x),
    #             destination = [agent.destination.x, agent.destination.y]
    #         )._asdict()
    #         for agent
    #         in msg_pedsim.agent_states
    #     ]

class Recorder(Node):

    def __init__(self, result_dir):

        super().__init__("data_recorder_node")

        self.declare_parameter("data_recorder_autoprefix", "")
        self.result_dir = self.get_directory(result_dir)

        self.declare_parameter("model", "")
        self.model = self.get_parameter("model").value                                                                          

        self.base_dir = get_package_share_directory("arena_evaluation")
        self.result_dir = os.path.join(self.base_dir, "data", "self.result_dir")
        #current_script_dir = os.path.dirname(os.path.abspath(__file__))
        #self.base_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..', 'src', 'arena', 'evaluation', 'arena_evaluation'))
        #self.result_dir = os.path.join(self.base_dir, "data", self.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.write_params()

        topics_to_monitor = self.get_topics_to_monitor()
        published_topics = [topic[0] for topic in self.get_topic_names_and_types()] # self.get_topic_names_and_types() is a list of tuples each tuple contain the topic name and a list of types

        topic_matcher = re.compile(f"({'|'.join([t[0] for t in topics_to_monitor])})$")

        topics_to_sub = []

        for topic_name in published_topics:
            
            match = re.search(topic_matcher, topic_name)

            if not match: 
                continue

            topics_to_sub.append([topic_name, *self.get_class_for_topic_name(topic_name)])

        self.data_collectors = []

        self.declare_parameter('start', [0.0, 0.0, 0.0])
        self.declare_parameter('goal' , [0.0, 0.0, 0.0])

        for topic in topics_to_sub:
            topic_name = topic[0]
            unique_name = topic_name.replace('/', '_')
            data_collector = DataCollector(topic, unique_name)
            self.data_collectors.append(data_collector)
            self.write_data(
                topic[1], 
                ["time", "data"],
                mode="w"
            )

        self.write_data("episode", ["time", "episode"], mode="w")
        self.write_data("start_goal", ["episode", "start", "goal"], mode="w")

        self.config = self.read_config()

        self.current_episode = 0
        self.current_time = None

        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.clock_sub = self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            self.qos  
        )

        self.scenario_reset_sub = self.create_subscription(
            Int16,
            "/scenario_reset",
            self.scenario_reset_callback,
            self.qos
        )

        # Define the service for changing directory
        self.change_directory_service = self.create_service(
            arena_evaluation_srvs.ChangeDirectory,
            '~/change_directory',
            self.change_directory_callback
        )

    def get_directory(self, directory: str):
        AUTO_PREFIX = "auto:/"
        PARAM_AUTO_PREFIX = "data_recorder_autoprefix"
    
        if directory.startswith(AUTO_PREFIX):
            set_prefix = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            print(f"Generated timestamp: {set_prefix}")
    
            param_value = self.get_parameter(PARAM_AUTO_PREFIX).value
    
            if param_value == "":
                self.set_parameters([rclpy.parameter.Parameter(PARAM_AUTO_PREFIX, rclpy.Parameter.Type.STRING, set_prefix)])
            else:
                set_prefix = param_value
    
            directory = os.path.join(str(set_prefix), directory[len(AUTO_PREFIX):])
    
        return directory

    def write_params(self):

        with open(self.result_dir + "/params.yaml", "w") as file:

            # Declare the parameters locally in the method
            self.declare_parameter("map_file", "")
            self.declare_parameter("scenario_file", "")
            self.declare_parameter("inter_planner", "")
            self.declare_parameter("local_planner", "")
            self.declare_parameter("agent_name", "")

            # Get the parameter values
            map_file = self.get_parameter("map_file").value
            scenario_file = self.get_parameter("scenario_file").value
            inter_planner = self.get_parameter("inter_planner").value
            local_planner = self.get_parameter("local_planner").value
            agent_name = self.get_parameter("agent_name").value
            namespace = self.get_namespace().strip('/')

            yaml.dump({
                "model": self.model,
                "map_file": map_file,
                "scenario_file": scenario_file,
                "inter_planner": inter_planner,
                "local_planner": local_planner,
                "agent_name": agent_name,
                "namespace": namespace
            }, file)

    def get_topics_to_monitor(self):

        namespace = self.get_namespace().strip("/")

        return [
            (f"{namespace}/scan", LaserScan),
            (f"{namespace}/scenario_reset", Int16),
            (f"{namespace}/odom", Odometry),
            (f"{namespace}/cmd_vel", Twist),
            # ("/pedsim_simulator/pedsim_agents_data", pedsim_msgs.PedsimAgentsDataframe)
        ]
    
    def get_class_for_topic_name(self, topic_name):
        if "/scan" in topic_name:
            return ["scan", LaserScan]
        if "/odom" in topic_name:
            return ["odom", Odometry]
        if "/cmd_vel" in topic_name:
            return ["cmd_vel", Twist]
        # if "/pedsim_agents_data" in topic_name:
        #     return ["pedsim_agents_data", pedsim_msgs.PedsimAgentsDataframe]
        
    def write_data(self, file_name, data, mode="a"):
        with open(f"{self.result_dir}/{file_name}.csv", mode, newline = "") as file:
            writer = csv.writer(file, delimiter = ',')
            writer.writerow(data)
            file.close()

    def read_config(self):
        with open(self.base_dir + "/config" + "/data_recorder_config.yaml") as file:
            return yaml.safe_load(file)

    def clock_callback(self, clock: Clock):
        
        current_simulation_action_time = clock.clock.sec * 10e9 + clock.clock.nanosec

        if not self.current_time:
            self.current_time = current_simulation_action_time

        time_diff = (current_simulation_action_time - self.current_time) / 1e6 ## in ms

        if time_diff < self.config["record_frequency"]:
            return

        self.current_time = current_simulation_action_time

        for collector in self.data_collectors:

            topic_name, data = collector.get_data()
            
            self.write_data(topic_name, [self.current_time, data])
        
        self.write_data("episode", [self.current_time, self.current_episode])
        self.write_data("start_goal", [
            self.current_episode, 
            self.get_parameter('start').value, 
            self.get_parameter('goal').value
        ])

    def scenario_reset_callback(self, data: Int16):
        self.current_episode = data.data

    def change_directory_callback(self, request, response): # ROS2: Change parameters and update configurations on the fly without needing to restart the node
        new_directory = request.data
        self.result_dir = self.get_directory(new_directory)
        response.result = True
        return response    

class BagRecorder(Node):
    def __init__(self, result_dir: str):
        super().__init__("bag_recorder_node")

        self.declare_parameter("data_recorder_autoprefix", "")
        self.result_dir = self.get_directory(result_dir)

        self.declare_parameter("model", "")
        self.model = self.get_parameter("model").value
        
        self.declare_parameter("world", "")
        self.world = self.get_parameter("world").value

        self.base_dir = get_package_share_directory("arena_evaluation")
        self.result_dir = os.path.join(self.base_dir, "data", self.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.write_params()

        topics_to_monitor = self.get_topics_to_monitor()
        published_topics = [t[0] for t in topics_to_monitor]

        topic_matcher = re.compile(f"({'|'.join([t[0] for t in topics_to_monitor])})$")

        topics_to_sub = []
        for topic_name in published_topics:
            match = re.search(topic_matcher, topic_name)
            if not match:
                continue
            # Append a list: [full_topic_name, topic_id, topic_type]
            topics_to_sub.append([topic_name, topic_name, self.get_class_for_topic_name(topic_name)[1]])


        self.data_collectors = []

        self.declare_parameter('start', [0.0, 0.0, 0.0])
        self.declare_parameter('goal', [0.0, 0.0, 0.0])
        for topic in topics_to_sub:
            topic_name = topic[0]
            unique_name = topic_name.replace('/', '_')
            collector = DataCollector(topic, unique_name)
            self.data_collectors.append(collector)

        # Write extra information as needed (episode and start_goal can be recorded as parameters or in a separate bag topic)
        self.current_episode = 0
        self.current_time = None

        # Setup QoS for clock and scenario reset subscriptions
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.clock_sub = self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            self.qos
        )

        self.scenario_reset_sub = self.create_subscription(
            Int16,
            "/scenario_reset",
            self.scenario_reset_callback,
            self.qos
        )

        self.change_directory_service = self.create_service(
            arena_evaluation_srvs.ChangeDirectory,
            '~/change_directory',
            self.change_directory_callback
        )

        # --- Setup rosbag2 writer ---

        bag_uri = os.path.join(self.result_dir, "my_rosbag")
        storage_options = StorageOptions(uri=bag_uri, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.writer = SequentialWriter()
        self.writer.open(storage_options, converter_options)
        # Create topic metadata for each topic that will be recorded.
        self.topics_metadata = {}
        for topic in topics_to_sub:
            topic_name = topic[0]
            msg_type = topic[2]
            # Construct the type string. This follows the convention "package/msg/MessageType"
            type_str = f"{msg_type.__module__.replace('.', '/')}/{msg_type.__name__}"
            metadata = TopicMetadata(
                name=topic_name,
                type=type_str,
                serialization_format='cdr',
                offered_qos_profiles=''
            )
            self.writer.create_topic(metadata)
            self.topics_metadata[topic_name] = metadata

        self.get_logger().info(f"Started recording to rosbag at: {bag_uri}")

    def get_directory(self, directory: str) -> str:
        AUTO_PREFIX = "auto:/"
        PARAM_AUTO_PREFIX = "data_recorder_autoprefix"
        if directory.startswith(AUTO_PREFIX):
            set_prefix = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            self.get_logger().info(f"Generated timestamp: {set_prefix}")
            param_value = self.get_parameter(PARAM_AUTO_PREFIX).value
            if param_value == "":
                self.set_parameters([Parameter(PARAM_AUTO_PREFIX, Parameter.Type.STRING, set_prefix)])
            else:
                set_prefix = param_value
            directory = os.path.join(str(set_prefix), directory[len(AUTO_PREFIX):])
        return directory

    def write_params(self):
        # Write runtime parameters to a YAML file (for record keeping)
        params_path = os.path.join(self.result_dir, "params.yaml")
        with open(params_path, "w") as file:
            self.declare_parameter("map_file", "")
            self.declare_parameter("scenario_file", "")
            self.declare_parameter("inter_planner", "")
            self.declare_parameter("local_planner", "")
            self.declare_parameter("agent_name", "")

            map_file = self.get_parameter("map_file").value
            scenario_file = self.get_parameter("scenario_file").value
            inter_planner = self.get_parameter("inter_planner").value
            local_planner = self.get_parameter("local_planner").value
            agent_name = self.get_parameter("agent_name").value
            namespace = self.get_namespace().strip('/')
            yaml.dump({
                "model": self.model,
                "world": self.world,
                "map_file": map_file,
                "scenario_file": scenario_file,
                "inter_planner": inter_planner,
                "local_planner": local_planner,
                "agent_name": agent_name,
                "namespace": namespace
            }, file)

    def get_topics_to_monitor(self):
        namespace = self.get_namespace().strip("/")
        return [
            (f"{namespace}/scan", LaserScan),
            (f"{namespace}/scenario_reset", Int16),
            (f"{namespace}/odom", Odometry),
            (f"{namespace}/cmd_vel", Twist),
            (f"{namespace}/human_states", Agents),
        ]

    def get_class_for_topic_name(self, topic_name: str):
        if "/scan" in topic_name:
            return ["scan", LaserScan]
        if "/odom" in topic_name:
            return ["odom", Odometry]
        if "/cmd_vel" in topic_name:
            return ["cmd_vel", Twist]
        if "/scenario_reset" in topic_name:
            return ["scenario_reset", Int16]
        if "/human_states" in topic_name:   
            return ["human_states", Agents] #hunav topic
        # if "/pedsim_agents_data" in topic_name:
        #     return ["pedsim_agents_data", pedsim_msgs.PedsimAgentsDataframe]

    def clock_callback(self, clock: Clock):
        current_simulation_action_time = clock.clock.sec * int(1e9) + clock.clock.nanosec
        if self.current_time is None:
            self.current_time = current_simulation_action_time

        # Record at the configured frequency (in ms) from the configuration file
        time_diff = (current_simulation_action_time - self.current_time) / 1e6  # in ms
        # Read record frequency from config (assuming key "record_frequency" exists)
        if not hasattr(self, 'config'):
            self.config = self.read_config()
        if time_diff < self.config["record_frequency"]:
            return

        self.current_time = current_simulation_action_time

        # For each DataCollector, retrieve the last message and record it into the rosbag.
        for collector in self.data_collectors:
            topic_name, msg = collector.get_data()
            if msg is None:
                continue
            try:
                serialized_msg = serialize_message(msg)

                self.writer.write(topic_name, serialized_msg, self.current_time)
            except Exception as e:
                self.get_logger().error(f"Error writing message on topic {topic_name}: {e}")


    def scenario_reset_callback(self, data: Int16):
        self.current_episode = data.data

    def change_directory_callback(self, request, response):
        new_directory = request.data
        self.result_dir = self.get_directory(new_directory)
        response.result = True
        return response

    def read_config(self):
        config_path = os.path.join(self.base_dir, "config", "data_recorder_config.yaml")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)


def main(args=None):
    rclpy.init(args=args)
    
    # Create an instance of BagRecorder with your desired directory (e.g., "auto:/")
    bag_recorder = BagRecorder("auto:/")
    
    try:
        # Spin the node so that it keeps listening to topics and recording data
        rclpy.spin(bag_recorder)
    except KeyboardInterrupt:
        print("Shutting down rosbag recorder...")
    finally:
        bag_recorder.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()