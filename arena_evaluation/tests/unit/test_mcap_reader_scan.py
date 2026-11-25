import pathlib
import types

import polars as pl

from arena_evaluation.processing.telemetry.messages import Envelope, RobotRef
from arena_evaluation.processing.telemetry.sensors import (
    SCAN_SUMMARY_TABLE,
    ScanRecorder,
)
from arena_evaluation.processing.telemetry.writer import TableSink

_REF = RobotRef(env_id="env_0", robot_id="env_0_jackal")


def _summary(
    tmp_path: pathlib.Path,
    ranges: list[float],
    range_min: float = 0.1,
    range_max: float = 10.0,
) -> dict:
    out = tmp_path / "telemetry"
    sink = TableSink(out)
    scan = types.SimpleNamespace(
        range_min=range_min,
        range_max=range_max,
        ranges=ranges,
        angle_min=0.0,
        angle_increment=0.1,
    )
    ScanRecorder(sink).record(Envelope(t_ns=1, ref=_REF, schema=None, body=scan))
    sink.close()
    return pl.read_parquet(out / f"{SCAN_SUMMARY_TABLE}.parquet").row(0, named=True)
