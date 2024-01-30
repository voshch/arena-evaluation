#!/usr/bin/env python3

import argparse
import os 
import arena_evaluation.metrics as Metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--pedsim", action="store_const", const=True, default=False)
    arguments = parser.parse_args()

    if arguments.pedsim:
        metrics = Metrics.PedsimMetrics(
            dir = arguments.dir
        )

    else:
        metrics = Metrics.Metrics(
            dir = arguments.dir
        )

    metrics.data.to_csv(os.path.join(arguments.dir, "metrics.csv"))
