#!/usr/bin/env python3

import argparse
import enum
import os
import pandas as pd
import rospkg

import arena_evaluation.metrics as metrics
import arena_evaluation.utils as utils
import arena_evaluation.plots as plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path")
    parser.add_argument("--output", "-o")

    args = parser.parse_args()

    class Key(str, enum.Enum):
        Contestant = enum.auto()
        Stage = enum.auto()
        Robot = enum.auto()

    all_metrics = utils.MultiDict[metrics.PedsimMetrics](Key.Contestant, Key.Stage, Key.Robot)

    base_path = os.path.join(rospkg.RosPack().get_path("arena_evaluation"), "data", args.path)
    
    data_dump = []

    for contestant in next(os.walk(os.path.join(base_path)))[1]:
        for stage in next(os.walk(os.path.join(base_path, contestant)))[1]:
            for robot in next(os.walk(os.path.join(base_path, contestant, stage)))[1]:
                all_metrics.insert(
                    (contestant, stage, robot),
                    (metric:=metrics.PedsimMetrics(dir = os.path.join(base_path, contestant, stage, robot)))
                )
                d = metric.data.copy()
                planner, *modal = contestant.split("-")
                d["planner"] = planner
                d["inter"] = "".join(modal)
                d["stage"] = stage
                d["robot"] = robot
                data_dump.append(d)

    combined_data = pd.concat(data_dump)

    print(combined_data.iloc[0])

    plots.set_dir(args.output)
    plots.plot(combined_data)

    

    # with open(os.path.join("plot_declarations", args.declaration_file)) as file:
    #     declaration_file = yaml.safe_load(file)


    # create_plots_from_declaration_file(declaration_file)