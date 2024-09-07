#!/usr/bin/env python3

import argparse
import os
import arena_evaluation.scripts.metrics as Metrics
from ament_index_python.packages import get_package_share_directory


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory where the data is stored")
    parser.add_argument("--pedsim", action="store_const", const=True, default=False, help="Flag to enable Pedsim metrics")
    arguments = parser.parse_args()

    dir_arg = os.path.join(
        get_package_share_directory(
            "arena_evaluation"),
            "data",
            arguments.dir
    ) 

    if arguments.pedsim:
        metrics = Metrics.PedsimMetrics(dir=dir_arg)
    else:
        metrics = Metrics.Metrics(dir=dir_arg)

    # Save the calculated metrics to a CSV file
    metrics.data.to_csv(os.path.join(dir_arg, "metrics.csv"))

if __name__ == "__main__":
    main()