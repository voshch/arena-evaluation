import pathlib
from types import SimpleNamespace

import polars as pl
import pytest

pytest.importorskip("polars")

from arena_evaluation.processing.telemetry.context import ScheduleRecorder, record_semantics
from arena_evaluation.processing.telemetry.messages import Envelope, RobotRef
from arena_evaluation.processing.telemetry.writer import TableSink

REF = RobotRef(env_id="env_0", robot_id="env_0_jackal")


class _FakeEntity:
    def __init__(self, entity, kind, members):
        self.entity = entity
        self.kind = kind
        self.discrete_names = ["state"]
        self.discrete_values = ["open"]
        self.continuous_names = []
        self.continuous_values = []
        self.predicate_names = ["over_cap"]
        self.predicate_values = [True]
        self.members = members


def _semantics(tmp_path: pathlib.Path, *entities: _FakeEntity) -> pl.DataFrame:
    snapshot = SimpleNamespace(env_id=0, world="hospital_small", entities=list(entities))
    out = tmp_path / "telemetry"
    sink = TableSink(out)
    record_semantics(sink, Envelope(t_ns=1_000_000_000, ref=REF, schema=None, body=snapshot))
    sink.close()
    return pl.read_parquet(out / "world_semantics.parquet")


def _phase(name: str = "ramp_up_vx_1.00_h_1.00", kind: str = "ramp_up", **overrides: object) -> dict:
    phase = {
        "name": name,
        "kind": kind,
        "vx_target": 1.0,
        "vy_target": 0.0,
        "wz_target": 0.0,
        "duration_s": 1.0,
        "ramp_s": 1.0,
        "radius_m": 0.0,
    }
    phase.update(overrides)
    return phase


def _schedule(tmp_path: pathlib.Path, *payloads: tuple[int, str]) -> pl.DataFrame | None:
    recorder = ScheduleRecorder()
    for t_ns, payload in payloads:
        recorder.record(Envelope(t_ns=t_ns, ref=REF, schema=None, body=SimpleNamespace(data=payload)))
    out = tmp_path / "telemetry"
    sink = TableSink(out)
    recorder.write(sink)
    sink.close()
    path = out / "robot_characterization_schedule.parquet"
    return pl.read_parquet(path) if path.exists() else None


