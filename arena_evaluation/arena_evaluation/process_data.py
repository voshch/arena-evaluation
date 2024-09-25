#!/usr/bin/env python3

import os
import enum
import argparse
import numpy  as np
import pandas as pd

import arena_evaluation.scripts.utils   as utils
import arena_evaluation.scripts.plots   as plots
import arena_evaluation.scripts.metrics as metrics

from ament_index_python.packages import get_package_share_directory

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path"  , "-p", help="Data directory for plots")
    parser.add_argument("--output", "-o", help="location of the plotted directory")
    args = parser.parse_args()

    class Key(str, enum.Enum):
        Contestant = enum.auto() 
        Stage      = enum.auto()
        Robot      = enum.auto()

    #all_metrics = utils.MultiDict[metrics.PedsimMetrics](Key.Contestant, Key.Stage, Key.Robot)
    all_metrics = utils.MultiDict[metrics.Metrics](Key.Contestant, Key.Stage, Key.Robot)

    base_path = os.path.join(
        get_package_share_directory(
            "arena_evaluation"),
            "data",
            args.path
    ) 
    
    data_dump = []

    for contestant in next(os.walk(os.path.join(base_path)))[1]:
        for stage in next(os.walk(os.path.join(base_path, contestant)))[1]:
            for robot in next(os.walk(os.path.join(base_path, contestant, stage)))[1]:
                # all_metrics.insert(
                #     (contestant, stage, robot),
                #     #(metric:=metrics.PedsimMetrics(dir = os.path.join(base_path, contestant, stage, robot)))
                #     (metric:=metrics.Metrics(dir = os.path.join(base_path, contestant, stage, robot)))
                # )
                metric = metrics.Metrics(dir=os.path.join(base_path, contestant, stage, robot))
                all_metrics.insert((contestant, stage, robot), metric)
                d = metric.data.copy()
                planner, *modal = contestant.split("-")
                d["planner"] = planner
                d["inter"] = "".join(modal)
                d["stage"] = stage
                d["robot"] = robot
                data_dump.append(d)

    combined_data = pd.concat(data_dump) 

    # Detect columns with list or ndarray values 
    list_columns = []
    for column in combined_data.columns:
        if combined_data[column].apply(lambda x: isinstance(x, list) or isinstance(x, np.ndarray)).any():
            #print(f"Column '{column}' contains list-like values.")
            list_columns.append(column)
            combined_data[column] = combined_data[column].apply(lambda x: np.array(x)[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan)
    #print(f"List-like columns converted: {list_columns}")

    # print(combined_data)

    if args.output:
        plots.set_dir(args.output)
    else:
        plots.set_dir("output")

    plots.plot(combined_data)

    # with open(os.path.join("plot_declarations", args.declaration_file)) as file:
    #     declaration_file = yaml.safe_load(file)

    # create_plots_from_declaration_file(declaration_file)

if __name__ == "__main__":
    main()