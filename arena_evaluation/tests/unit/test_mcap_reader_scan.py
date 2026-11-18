import math
import pathlib
import types

import polars as pl

from arena_evaluation.processing.telemetry.messages import Envelope, RobotRef
from arena_evaluation.processing.telemetry.sensors import SCAN_SUMMARY_TABLE, ScanRecorder
from arena_evaluation.processing.telemetry.writer import TableSink

_REF = RobotRef(env_id="env_0", robot_id="env_0_jackal")


def _summary(tmp_path: pathlib.Path, ranges: list[float], range_min: float = 0.1, range_max: float = 10.0) -> dict:
    out = tmp_path / "telemetry"
    sink = TableSink(out)
    scan = types.SimpleNamespace(range_min=range_min, range_max=range_max, ranges=ranges, angle_min=0.0, angle_increment=0.1)
    ScanRecorder(sink, include_rays=False).record(Envelope(t_ns=1, ref=_REF, schema=None, body=scan))
    sink.close()
    return pl.read_parquet(out / f"{SCAN_SUMMARY_TABLE}.parquet").row(0, named=True)


def test_the_summary_keeps_the_nearest_and_farthest_valid_return(tmp_path: pathlib.Path):
    row = _summary(tmp_path, [3.0, 1.2, 5.0])

    assert (row["scan_min_m"], row["scan_max_m"], row["n_valid_rays"]) == (1.2, 5.0, 3)


def test_readings_outside_the_range_or_not_numbers_are_not_valid(tmp_path: pathlib.Path):
    row = _summary(tmp_path, [0.05, math.nan, math.inf, 2.5])

    assert (row["scan_min_m"], row["scan_max_m"], row["n_valid_rays"]) == (2.5, 2.5, 1)


def test_a_scan_without_a_detection_reads_as_the_range_max(tmp_path: pathlib.Path):
    row = _summary(tmp_path, [math.inf, 10.0, math.nan])

    assert (row["scan_min_m"], row["scan_max_m"], row["n_valid_rays"]) == (10.0, 10.0, 0)
