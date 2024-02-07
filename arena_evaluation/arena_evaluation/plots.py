

import os
from matplotlib import pyplot as plt
import rospkg
import seaborn as sns

base_dir = os.path.join(rospkg.RosPack().get_path("arena_evaluation"), "plots")
dir = base_dir

def set_dir(_dir):
    global dir
    dir = os.path.join(base_dir, _dir)
    os.makedirs(dir, exist_ok=True)

ext = "png"
def set_ext(_ext):
    global ext
    ext = _ext

def save_path(name):
    global dir, ext
    return os.path.join(dir, f"{name}.{ext}")

def plot(data):
    for metric, title, unit in (
        ("curvature", "Curvature", "m"),
        ("normalized_curvature", "Normalized Curvature", "m"),
        ("roughness", "Roughness", ""),
        ("path_length_values", "Path Length", "m"),
        ("path_length", "Path Length", "m"),
        ("acceleration", "Acceleration", "m/s$^2$"),
        ("jerk", "Jerk", "m/s$^4$"),
        ("velocity", "Velocity", "m/s"),
        ("collision_amount", "Number of Collisions", ""),
        ("collisions", "Collisions", ""),
        ("path", "Path", ""),
        ("angle_over_length", "Angle over Length", "rad/m"),
        ("action_type", "Action Type", ""),
        ("time_diff", "Time to Reach Goal", "s"),
        ("time", "Time", "s"),
        ("episode", "Episode", ""),
        ("result", "Result", ""),
        ("cmd_vel", "Velocity Command", ""),
        ("goal", "Goal", ""),
        ("start", "Start", ""),
        ("avg_velocity_in_personal_space", "Velocity in Personal Space (avg.)", "m/s"),
        ("total_time_in_personal_space", "Time in Personal Space", "s"),
        ("time_in_personal_space", "Time in Personal Space", "s"),
        ("total_time_looking_at_pedestrians", "Time Looking at Pedestrians", "s"),
        ("time_looking_at_pedestrians", "Time Looking at Pedestrians", "s"),
        ("total_time_looked_at_by_pedestrians", "Time Seen by Pedestrians", "s"),
        ("time_looked_at_by_pedestrians", "Time Seen by Pedestrians", "s"),
        ("num_pedestrians", "Number of Pedestrians", "")
    ):
        for kind in ('strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count'):
            try:
                save_name = f"{kind}: {metric} [h]"
                plt.close()
                sns.catplot(
                    data,
                    y = "planner",
                    x = metric,
                    hue = "planner",
                    kind=kind,
                )\
                .set(
                    title = title,
                    xlabel = unit,
                    ylabel = "",
                    ylabels = []
                )\
                .savefig(save_path(save_name))
                print(f"saving {save_name}")
            except Exception as e:
                ...

            try:
                save_name = f"{kind}: {metric} [v]"
                plt.close()
                sns.catplot(
                    data,
                    x = "planner",
                    y = metric,
                    #hue = "inter",
                    kind=kind,
                )\
                .set(
                    title = title,
                    ylabel = unit,
                    xlabel = "",
                )\
                .savefig(save_path(save_name))
                print(f"saving {save_name}")
            except Exception as e:
                ...