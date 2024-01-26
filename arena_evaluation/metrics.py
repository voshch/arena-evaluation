"""
This file is used to calculate from the simulation data, various metrics, such as
- did a collision occur
- how long did the robot take form start to goal
the metrics / evaluation data will be saved to be preproccesed in the next step
"""
import enum
import typing
from typing import List
import numpy as np
import pandas as pd
import os
from pandas.core.api import DataFrame as DataFrame
import yaml
import rospkg
import json

from arena_evaluation.utils import Utils


class Action(str, enum.Enum):
    STOP = "STOP"
    ROTATE = "ROTATE"
    MOVE = "MOVE"


class DoneReason(str, enum.Enum):
    TIMEOUT = "TIMEOUT"
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION = "COLLISION"


class Config:
    TIMEOUT_TRESHOLD = 180e9
    MAX_COLLISIONS = 3
    MIN_EPISODE_LENGTH = 5
    
    PERSONAL_SPACE_RADIUS = 1 # personal space is estimated at around 1'-4'
    ROBOT_GAZE_ANGLE = np.radians(5) # size of (one half of) direct robot gaze cone
    PEDESTRIAN_GAZE_ANGLE = np.radians(5) # size of (one half of) direct ped gaze cone

class Math:

    @classmethod
    def round_values(cls, values, digits=3):
        return [round(v, digits) for v in values]

    @classmethod
    def curvature(cls, first, second, third):
        triangle_area = cls.triangle_area(first, second, third)

        divisor = (
            np.abs(np.linalg.norm(first - second)) 
            * np.abs(np.linalg.norm(second - third))
            * np.abs(np.linalg.norm(third - first))
        )

        if divisor == 0:
            return 0, 0

        curvature = 4 * triangle_area / divisor

        normalized = (
            curvature * (
                np.abs(np.linalg.norm(first - second)) 
                + np.abs(np.linalg.norm(second - third))
            )
        )

        return curvature, normalized

    @classmethod
    def roughness(cls, first, second, third):
        triangle_area = cls.triangle_area(first, second, third)

        if np.abs(np.linalg.norm(third - first)) == 0:
            return 0

        return 2 * triangle_area / np.abs(np.linalg.norm(third - first)) ** 2

    @classmethod
    def jerk(cls, first, second, third):
        a1 = second - first
        a2 = third - second

        jerk = np.abs(a2 - a1)

        return jerk

    @classmethod
    def triangle_area(cls, first, second, third):
        return (
            0.5 * np.abs(
                first[0] * (second[1] - third[1]) 
                + second[0] * (third[1] - first[1]) 
                + third[0] * (first[1] - second[1])
            )
        )
    
    @classmethod
    def angle_difference(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            return np.pi - np.abs(np.abs(x1 - x2) - np.pi)

class Metrics:

    dir: str
    _episode_data: typing.Dict

    def _load_data(self) -> typing.List[pd.DataFrame]:
        episode = pd.read_csv(os.path.join(self.dir, "episode.csv"), converters={
            "data": lambda val: 0 if len(val) <= 0 else int(val) 
        })

        laserscan = pd.read_csv(os.path.join(self.dir, "scan.csv"), converters={
            "data": Utils.string_to_float_list
        }).rename(columns={"data": "laserscan"})

        odom = pd.read_csv(os.path.join(self.dir, "odom.csv"), converters={
            "data": lambda col: json.loads(col.replace("'", "\""))
        }).rename(columns={"data": "odom"})

        cmd_vel = pd.read_csv(os.path.join(self.dir, "cmd_vel.csv"), converters={
            "data": Utils.string_to_float_list
        }).rename(columns={"data": "cmd_vel"})

        start_goal = pd.read_csv(os.path.join(self.dir, "start_goal.csv"), converters={
            "start": Utils.string_to_float_list,
            "goal": Utils.string_to_float_list
        })

        return [episode, laserscan, odom, cmd_vel, start_goal]

    def __init__(self, dir: str):

        self.dir = dir
        self.robot_params = self._get_robot_params()

        data = pd.concat(self._load_data(), axis=1, join="inner")
        data = data.loc[:,~data.columns.duplicated()].copy()

        i = 0

        episode_data = self._episode_data = {}

        while True:
            current_episode = data[data["episode"] == i]

            if len(current_episode) < Config.MIN_EPISODE_LENGTH:
                break
            
            episode_data[i] = self._analyze_episode(current_episode, i)
            i = i + 1

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame(self._episode_data).transpose().set_index("episode")

    
    def _analyze_episode(self, episode: pd.DataFrame, index):
        positions, velocities = [], []

        for odom in episode["odom"]:
            positions.append(np.array(odom["position"]))
            velocities.append(np.array(odom["velocity"]))

        curvature, normalized_curvature = self._get_curvature(np.array(positions))
        roughness = self._get_roughness(np.array(positions))

        vel_absolute = self._get_velocity_abs(velocities)
        acceleration = self._get_acceleration(vel_absolute)
        jerk = self._get_jerk(vel_absolute)

        collisions, collision_amount = self._get_collisions(
            episode["laserscan"],
            self.robot_params["robot_radius"]
        )

        path_length, path_length_per_step = self._get_path_length(positions)

        time = int(list(episode["time"])[-1] - list(episode["time"])[0])

        start_position = self._get_mean_position(episode, "start")
        goal_position = self._get_mean_position(episode, "goal")

        # print("PATH LENGTH", path_length, path_length_per_step)

        return {
            "curvature": Math.round_values(curvature),
            "normalized_curvature": Math.round_values(normalized_curvature),
            "roughness": Math.round_values(roughness),
            "path_length_values": Math.round_values(path_length_per_step),
            "path_length": path_length,
            "acceleration": Math.round_values(acceleration),
            "jerk": Math.round_values(jerk),
            "velocity": Math.round_values(vel_absolute),
            "collision_amount": collision_amount,
            "collisions": list(collisions),
            "path": [list(p) for p in positions],
            "angle_over_length": self._get_angle_over_length(path_length, positions),
            "action_type": list(self._get_action_type(episode["cmd_vel"])),
            ## Ros time in ns
            "time_diff": time,
            "time": list(map(int, episode["time"].tolist())),
            "episode": index,
            "result": self._get_success(time, collision_amount),
            "cmd_vel": list(map(list, episode["cmd_vel"].to_list())),
            "goal": goal_position,
            "start": start_position
        }
    
    def _get_robot_params(self):
        with open(os.path.join(self.dir, "params.yaml")) as file:
            content = yaml.safe_load(file)

            model = content["model"]

        robot_model_params_file = os.path.join(
            rospkg.RosPack().get_path("arena-simulation-setup"), 
            "robot", 
            model, 
            "model_params.yaml"
        )

        with open(robot_model_params_file, "r") as file:
            return yaml.safe_load(file)

    def _get_mean_position(self, episode, key):
        positions = episode[key].to_list()
        counter = {}

        for p in positions:
            hash = ":".join([str(pos) for pos in p])

            counter[hash] = counter.get(hash, 0) + 1

        sorted_positions = dict(sorted(counter.items(), key=lambda x: x))

        return [float(r) for r in list(sorted_positions.keys())[0].split(":")]

    def _get_position_for_collision(self, collisions, positions):
        for i, collision in enumerate(collisions):
            collisions[i][2] = positions[collision[0]]

        return collisions

    def _get_angle_over_length(self, path_length, positions):
        total_yaw = 0

        for i, yaw in enumerate(positions[:-1]):
            yaw = yaw[2]
            next_yaw = positions[i + 1][2]

            total_yaw += abs(next_yaw - yaw)

        return total_yaw / path_length

    def _get_success(self, time, collisions):
        if time >= Config.TIMEOUT_TRESHOLD:
            return DoneReason.TIMEOUT

        if collisions >= Config.MAX_COLLISIONS:
            return DoneReason.COLLISION

        return DoneReason.GOAL_REACHED

    def _get_path_length(self, positions):
        path_length = 0
        path_length_per_step = []

        for i, position in enumerate(positions[:-1]):
            next_position = positions[i + 1]

            step_path_length = np.linalg.norm(position - next_position)

            path_length_per_step.append(step_path_length)
            path_length += step_path_length

        return path_length, path_length_per_step
    
    def _get_collisions(self, laser_scans, lower_bound):
        """
        Calculates the collisions. Therefore, 
        the laser scans is examinated and all values below a 
        specific range are marked as collision.

        Argument:
            - Array laser scans representing the scans over
            time
            - the lower bound for which a collisions are counted

        Returns tupel of:
            - Array of tuples with indexs and time in which
            a collision happened
        """
        collisions = []
        collisions_marker = []

        for i, scan in enumerate(laser_scans):

            is_collision = len(scan[scan <= lower_bound]) > 0

            collisions_marker.append(int(is_collision))
            
            if is_collision:
                collisions.append(i)

        collision_amount = 0

        for i, coll in enumerate(collisions_marker[1:]):
            prev_coll = collisions_marker[i]

            if coll - prev_coll > 0:
                collision_amount += 1

        return collisions, collision_amount

    def _get_action_type(self, actions):
        action_type = []

        for action in actions:
            if sum(action) == 0:
                action_type.append(Action.STOP)
            elif action[0] == 0 and action[1] == 0:
                action_type.append(Action.ROTATE)
            else:
                action_type.append(Action.MOVE)

        return action_type

    def _get_curvature(self, positions):
        """
        Calculates the curvature and the normalized curvature
        for all positions in the list

        Returns an array of tuples with (curvature, normalized_curvature)
        """
        curvature_list = []
        normalized_curvature = []

        for i, position in enumerate(positions[:-2]):
            first = position
            second = positions[i + 1]
            third = positions[i + 2]

            curvature, normalized = Math.curvature(first, second, third)

            curvature_list.append(curvature)
            normalized_curvature.append(normalized)

        return curvature_list, normalized_curvature

    def _get_roughness(self, positions):
        roughness_list = []

        for i, position in enumerate(positions[:-2]):
            first = position
            second = positions[i + 1]
            third = positions[i + 2]

            roughness_list.append(Math.roughness(first, second, third))

        return roughness_list

    def _get_velocity_abs(self, velocities):
        return [(i ** 2 + j ** 2) ** 0.5 for i, j, z in velocities]

    def _get_acceleration(self, vel_abs):
        acc_list = []

        for i, vel in enumerate(vel_abs[:-1]):
            acc_list.append(vel_abs[i + 1] - vel)

        return acc_list

    def _get_jerk(self, vel_abs):
        """
        jerk is the rate at which an objects acceleration changes with respect to time
        """
        jerk_list = []

        for i, velocity in enumerate(vel_abs[:-2]):
            first = velocity
            second = vel_abs[i + 1]
            third = vel_abs[i + 2]

            jerk = Math.jerk(first, second, third)

            jerk_list.append(jerk)

        return jerk_list

    
        
class PedsimMetrics(Metrics):

    def _load_data(self) -> List[DataFrame]:
        pedsim_data = pd.read_csv(
            os.path.join(self.dir, "pedsim_agents_data.csv"),
            converters = {"data": Utils.parse_pedsim}
        ).rename(columns={"data": "peds"})
        
        return super()._load_data() + [pedsim_data]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze_episode(self, episode: pd.DataFrame, index):

        super_analysis = super()._analyze_episode(episode, index)

        robot_position = np.array([odom["position"][:2] for odom in episode["odom"]])
        peds_position = np.array([[ped.position for ped in peds] for peds in episode["peds"]])

        # list of (timestamp, ped) indices, duplicate timestamps allowed
        personal_space_frames = np.array(np.where(np.linalg.norm(peds_position - robot_position[:,None], axis=-1) <= Config.PERSONAL_SPACE_RADIUS))

        # time in personal space
        time = np.diff(np.array(episode["time"]), prepend=0)
        time_in_personal_space = time[personal_space_frames[0,:]].sum()

        # v_avg in personal space
        velocity = np.array(super_analysis["velocity"])
        velocity = velocity[personal_space_frames[0,:]]
        avg_velocity_in_personal_space = velocity.mean() if velocity.size else 0


        # gazes
        robot_direction = np.array([odom["position"][2] for odom in episode["odom"]])
        peds_direction = np.array([[ped.theta for ped in peds] for peds in episode["peds"]])
        angle_robot_peds = np.squeeze(np.angle(np.array(peds_position - robot_position[:,np.newaxis]).view(np.complex128)))

        # time looking at pedestrians
        robot_gaze = np.abs(Math.angle_difference(robot_direction[:,np.newaxis], angle_robot_peds))
        looking_at_frames = np.array(np.where(robot_gaze <= Config.ROBOT_GAZE_ANGLE))
        time_looking_at_pedestrians = time[np.unique(looking_at_frames[0,:])].sum()
        cumulative_time_looking_at_pedestrians = time[looking_at_frames].sum()
        
        # time being looked at by pedestrians
        ped_gaze = Math.angle_difference(peds_direction, np.pi - angle_robot_peds)
        looked_at_frames = np.array(np.where(ped_gaze <= Config.PEDESTRIAN_GAZE_ANGLE))
        time_looked_at_by_pedestrians = time[np.unique(looked_at_frames[0,:])].sum()
        cumulative_time_looked_at_by_pedestrians = time[looked_at_frames[0,:]].sum()

        return {
            **super_analysis,
            "time_in_personal_space": int(time_in_personal_space),
            "time_looking_at_pedestrians": int(time_looking_at_pedestrians),
            "cumulative_time_looking_at_pedestrians": int(cumulative_time_looking_at_pedestrians),
            "time_looked_at_by_pedestrians": int(time_looked_at_by_pedestrians),
            "cumulative_time_looked_at_by_pedestrians": int(cumulative_time_looked_at_by_pedestrians),
            "avg_velocity_in_personal_space": avg_velocity_in_personal_space,
            "num_pedestrians": peds_position.shape[0]
        }