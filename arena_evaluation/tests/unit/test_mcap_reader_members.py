import json
import pathlib
from types import SimpleNamespace

import polars as pl
import pytest

pytest.importorskip("polars")

from arena_evaluation.processing.telemetry.context import ScheduleRecorder, record_semantics
from arena_evaluation.processing.telemetry.messages import Envelope, RobotRef, env_id_in
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


def test_entity_fields_become_one_row_each_with_members_as_one_sorted_value(tmp_path: pathlib.Path):
    frame = _semantics(tmp_path, _FakeEntity("env_0/lobby", "occupancy_cap", ["env_0/robot_0", "env_0/human_1"]))

    assert frame.select("t_ns", "env_id", "world", "entity_name", "kind", "field_name", "field_kind", "value_str", "value_bool").rows() == [
        (1_000_000_000, "env_0", "hospital_small", "env_0/lobby", "occupancy_cap", "state", "discrete", "open", None),
        (1_000_000_000, "env_0", "hospital_small", "env_0/lobby", "occupancy_cap", "over_cap", "predicate", None, True),
        (1_000_000_000, "env_0", "hospital_small", "env_0/lobby", "occupancy_cap", "members", "members", "env_0/human_1,env_0/robot_0", None),
    ]


def test_an_entity_without_members_keeps_an_empty_members_value(tmp_path: pathlib.Path):
    frame = _semantics(tmp_path, _FakeEntity("env_0/door_1", "door", []))

    assert frame.filter(pl.col("field_name") == "members")["value_str"].to_list() == [""]


def test_every_entity_of_a_snapshot_is_flattened(tmp_path: pathlib.Path):
    frame = _semantics(tmp_path, _FakeEntity("env_0/lobby", "occupancy_cap", []), _FakeEntity("env_0/elevator_1", "elevator", ["env_0/robot_0"]))

    assert frame["entity_name"].to_list() == ["env_0/lobby"] * 3 + ["env_0/elevator_1"] * 3


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


def test_a_valid_schedule_becomes_one_row_per_phase_with_the_radius_as_turn_radius(tmp_path: pathlib.Path):
    payload = json.dumps(
        {
            "robot": "env_0_jackal",
            "model": "jackal",
            "envelope": {"vx_max": 2.0},
            "phases": [
                _phase(),
                _phase(name="arc_vx_0.50_r_1.00_left", kind="arc", vx_target=0.5, wz_target=0.5, radius_m=1.0, ramp_s=0.0),
            ],
        }
    )

    rows = _schedule(tmp_path, (1, payload))

    assert rows.select("phase_idx", "phase_label", "phase_kind", "vx_target", "wz_target", "turn_radius_m").rows() == [
        (0, "ramp_up_vx_1.00_h_1.00", "ramp_up", 1.0, 0.0, 0.0),
        (1, "arc_vx_0.50_r_1.00_left", "arc", 0.5, 0.5, 1.0),
    ]


def test_a_payload_without_phases_writes_no_schedule(tmp_path: pathlib.Path):
    assert _schedule(tmp_path, (1, json.dumps({"robot": "env_0_jackal", "model": "jackal", "envelope": {}}))) is None


def test_a_phase_missing_a_key_writes_no_schedule(tmp_path: pathlib.Path):
    incomplete = _phase()
    del incomplete["radius_m"]

    assert _schedule(tmp_path, (1, json.dumps({"phases": [incomplete]}))) is None


def test_non_json_writes_no_schedule(tmp_path: pathlib.Path):
    assert _schedule(tmp_path, (1, "not json")) is None


def test_the_latest_valid_schedule_wins(tmp_path: pathlib.Path):
    first = json.dumps({"phases": [_phase(name="first")]})
    second = json.dumps({"phases": [_phase(name="second")]})

    rows = _schedule(tmp_path, (1, first), (2, "not json"), (3, second), (4, "not json"))

    assert rows["phase_label"].to_list() == ["second"]


def test_a_transform_names_its_environment_by_frame_not_by_default():
    assert env_id_in("env_1/jackal/base_link") == "env_1"
    assert env_id_in("jackal/base_link") is None
