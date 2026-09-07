import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.conditions")

from arena_simulation_setup.shared.conditions import parse_atom

from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry, _reconstruct_events
from arena_evaluation.processing.metrics.ecological.condition_metrics import (
    ConditionComplianceCalculator,
    _EvalContext,
    _atom_series,
    _clause_verdict,
    _entity_atom_series,
    _entity_roster,
    _first_true,
    _operator_verdict,
    _ped_roster,
    _ped_zone_series,
    _robot_zone_series,
    _strip_env,
    _value_at,
    _values_equal,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def _calc():
    return ConditionComplianceCalculator(RobotParams(0.2, 0.0, 10.0))


def _start_pos(data):
    if data is not None and "pos_x" in data.columns and len(data) > 0:
        return [data["pos_x"][0], data["pos_y"][0], data["yaw"][0]]
    return []


def _episode(data, semantic_snapshot=None, conditions=None):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=_start_pos(data),
        goal_pos=[],
        semantic_snapshot=semantic_snapshot,
        conditions=conditions,
    )


def _square(name):
    import shapely

    return _ZoneGeometry(
        name=name,
        polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]),
        max_speed=None,
        quiet=False,
        restricted=False,
    )


def _snap(rows):
    """Synthetic long-format snapshot rows: (time_ns, entity, kind, field, field_kind, value)."""
    time_ns, entity, kind, field, field_kind = [], [], [], [], []
    value_str, value_num, value_bool = [], [], []
    for t, e, k, f, fk, v in rows:
        time_ns.append(t)
        entity.append(e)
        kind.append(k)
        field.append(f)
        field_kind.append(fk)
        value_str.append(v if fk == "discrete" else None)
        value_num.append(float(v) if fk == "continuous" else None)
        value_bool.append(bool(v) if fk == "predicate" else None)
    return pl.DataFrame(
        {
            "time_ns": time_ns,
            "entity": entity,
            "kind": kind,
            "field": field,
            "field_kind": field_kind,
            "value_str": value_str,
            "value_num": value_num,
            "value_bool": value_bool,
        }
    )


def _ctx(time_ns, pos_x=None, pos_y=None, snapshot=None, data=None, zones=None):
    n = len(time_ns)
    time_arr = np.asarray(time_ns)
    pos_x = np.zeros(n) if pos_x is None else np.asarray(pos_x, dtype=float)
    pos_y = np.zeros(n) if pos_y is None else np.asarray(pos_y, dtype=float)
    zones = zones or []
    data = data if data is not None else pl.DataFrame({"time_ns": list(time_ns)})
    return _EvalContext(
        events=_reconstruct_events(snapshot),
        time_ns=time_arr,
        pos_x=pos_x,
        pos_y=pos_y,
        data=data,
        zones_by_name={z.name: z for z in zones},
        entity_roster=_entity_roster(snapshot),
        ped_roster=_ped_roster(data),
    )


# -- small helpers --------------------------------------------------------


def test_strip_env_removes_only_leading_env_segment():
    assert _strip_env("env_0/ward_a_door") == "ward_a_door"
    assert _strip_env("env_12/ward_a_door/0") == "ward_a_door/0"
    assert _strip_env("blocker_1") == "blocker_1"


def test_values_equal_float_tolerance_and_string():
    assert _values_equal("2.0", "2") is True
    assert _values_equal("0.4000001", "0.4") is True
    assert _values_equal("1.0", "2.0") is False
    assert _values_equal("open", "open") is True
    assert _values_equal("open", "closed") is False
    assert _values_equal("true", "true") is True


def test_value_at_holds_seed_value_before_and_after_first_recorded_time():
    times = np.array([2_000_000_000, 4_000_000_000])
    values = ["open", "closed"]
    assert _value_at(times, values, 0) == "open"
    assert _value_at(times, values, 2_000_000_000) == "open"
    assert _value_at(times, values, 5_000_000_000) == "closed"


def test_first_true_returns_earliest_index_or_none():
    assert _first_true(np.array([False, False, True, True])) == 2
    assert _first_true(np.array([False, False])) is None


# -- operator verdicts ----------------------------------------------------


def test_always_true_and_false():
    assert _operator_verdict("always", np.array([True, True]), True, None, False) is True
    assert _operator_verdict("always", np.array([True, False]), True, None, False) is False


def test_never_true_and_false():
    assert _operator_verdict("never", np.array([False, False]), True, None, False) is True
    assert _operator_verdict("never", np.array([False, True]), True, None, False) is False


def test_eventually_true_and_false():
    assert _operator_verdict("eventually", np.array([False, True]), True, None, False) is True
    assert _operator_verdict("eventually", np.array([False, False]), True, None, False) is False


def test_unary_operator_unknown_when_atom_unresolvable():
    assert _operator_verdict("always", None, False, None, False) is None
    assert _operator_verdict("never", None, False, None, False) is None
    assert _operator_verdict("eventually", None, False, None, False) is None


def test_before_true_when_p_first():
    p = np.array([False, True, False])
    q = np.array([False, False, True])
    assert _operator_verdict("before", p, True, q, True) is True


def test_before_true_when_q_never_occurs():
    p = np.array([False, True])
    q = np.array([False, False])
    assert _operator_verdict("before", p, True, q, True) is True


def test_before_false_when_p_never_occurs():
    p = np.array([False, False])
    q = np.array([True, True])
    assert _operator_verdict("before", p, True, q, True) is False


def test_before_false_on_simultaneous_first_sample():
    p = np.array([False, True, False])
    q = np.array([False, True, False])
    assert _operator_verdict("before", p, True, q, True) is False


def test_before_unknown_when_either_atom_unresolvable():
    p = np.array([True, False])
    assert _operator_verdict("before", p, True, None, False) is None
    assert _operator_verdict("before", None, False, p, True) is None


def test_never_during_true_without_cooccurrence():
    p = np.array([True, False, False])
    q = np.array([False, False, True])
    assert _operator_verdict("never_during", p, True, q, True) is True


def test_never_during_false_on_cooccurrence():
    p = np.array([False, True, False])
    q = np.array([False, True, False])
    assert _operator_verdict("never_during", p, True, q, True) is False


def test_never_during_unknown_when_atom_unresolvable():
    p = np.array([True, False])
    assert _operator_verdict("never_during", p, True, None, False) is None


# -- entity atoms ---------------------------------------------------------


def test_entity_atom_true_after_recorded_change():
    snapshot = _snap(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (2_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
        ]
    )
    ctx = _ctx([0, 1_000_000_000, 3_000_000_000], snapshot=snapshot)
    series, ok = _entity_atom_series(parse_atom("door_1.open == true"), ctx)

    assert ok is True
    assert list(series) == [False, False, True]


def test_entity_atom_holds_seed_value_when_never_changed():
    snapshot = _snap(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
        ]
    )
    ctx = _ctx([0, 2_000_000_000], snapshot=snapshot)
    series, ok = _entity_atom_series(parse_atom("door_1.open == false"), ctx)

    assert ok is True
    assert list(series) == [True, True]


def test_entity_atom_unresolvable_unknown_entity():
    snapshot = _snap(
        [
            (1_000_000_000, "env_0/door_1", "door", "state", "discrete", "open"),
        ]
    )
    ctx = _ctx([0, 2_000_000_000], snapshot=snapshot)
    series, ok = _entity_atom_series(parse_atom("ghost.open == true"), ctx)

    assert series is None
    assert ok is False


def test_entity_atom_unresolvable_field_not_registered():
    snapshot = _snap(
        [
            (1_000_000_000, "env_0/door_1", "door", "state", "discrete", "open"),
        ]
    )
    ctx = _ctx([0, 2_000_000_000], snapshot=snapshot)
    series, ok = _entity_atom_series(parse_atom("door_1.nonsense == x"), ctx)

    assert series is None
    assert ok is False


def test_entity_roster_drops_ambiguous_bare_names():
    snapshot = _snap(
        [
            (0, "env_0/door_1", "door", "state", "discrete", "open"),
            (0, "env_1/door_1", "door", "state", "discrete", "open"),
        ]
    )
    roster = _entity_roster(snapshot)

    assert "door_1" not in roster


def test_entity_atom_resolves_multi_kind_entity_and_drops_env_duplicate():
    snapshot = _snap(
        [
            (0, "env_0/main_door/0", "door", "open", "predicate", False),
            (2_000_000_000, "env_0/main_door/0", "door", "open", "predicate", True),
            (0, "env_0/main_door/0", "gate", "locked", "predicate", True),
            (2_000_000_000, "env_0/main_door/0", "gate", "locked", "predicate", False),
            (0, "env_0/x/0", "door", "open", "predicate", True),
            (0, "env_1/x/0", "door", "open", "predicate", True),
        ]
    )
    ctx = _ctx([0, 3_000_000_000], snapshot=snapshot)

    open_series, open_ok = _entity_atom_series(parse_atom("main_door/0.open == true"), ctx)
    locked_series, locked_ok = _entity_atom_series(parse_atom("main_door/0.locked == false"), ctx)
    dup_series, dup_ok = _entity_atom_series(parse_atom("x/0.open == true"), ctx)

    assert open_ok is True and list(open_series) == [False, True]
    assert locked_ok is True and list(locked_series) == [False, True]
    assert dup_series is None and dup_ok is False


# -- robot zone atoms -----------------------------------------------------


def test_robot_zone_membership_inside_and_outside():
    ctx = _ctx([0, 1_000_000_000], pos_x=[1.0, 10.0], pos_y=[1.0, 10.0], zones=[_square("lobby")])
    series, ok = _robot_zone_series(parse_atom("robot in lobby"), ctx)

    assert ok is True
    assert list(series) == [True, False]


def test_robot_zone_unresolvable_when_zone_not_annotated():
    ctx = _ctx([0, 1_000_000_000], pos_x=[1.0, 1.0], pos_y=[1.0, 1.0], zones=[_square("lobby")])
    series, ok = _robot_zone_series(parse_atom("robot in pharmacy"), ctx)

    assert series is None
    assert ok is False


def test_robot_zone_unresolvable_without_pose_anchor():
    ctx = _ctx([0, 1_000_000_000], pos_x=[1.0, 1.0], pos_y=[1.0, 1.0], zones=[_square("lobby")])
    ctx.pose_valid = False
    series, ok = _robot_zone_series(parse_atom("robot in lobby"), ctx)

    assert series is None
    assert ok is False

    ped_data = _ped_data([0, 1_000_000_000], [["walker"], ["walker"]], [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ctx = _ctx([0, 1_000_000_000], data=ped_data, zones=[_square("lobby")])
    ctx.pose_valid = False
    series, ok = _ped_zone_series(parse_atom("walker in lobby"), ctx)

    assert ok is True
    assert list(series) == [True, True]


# -- ped zone atoms -------------------------------------------------------


def _ped_data(time_ns, names, positions):
    return pl.DataFrame(
        {
            "time_ns": list(time_ns),
            "peds_names": names,
            "peds_positions": positions,
        }
    )


def test_ped_zone_zero_order_hold_and_before_first_sample_false():
    data = _ped_data(
        [0, 1_000_000_000, 2_000_000_000],
        [[], ["env_0/blocker_1"], ["env_0/blocker_1"]],
        [[], [1.0, 1.0, 0.0], [10.0, 10.0, 0.0]],
    )
    ctx = _ctx([0, 1_000_000_000, 2_000_000_000], data=data, zones=[_square("lobby")])
    series, ok = _ped_zone_series(parse_atom("blocker_1 in lobby"), ctx)

    assert ok is True
    assert list(series) == [False, True, False]


def test_ped_zone_unresolvable_when_ped_not_recorded():
    data = _ped_data(
        [0, 1_000_000_000],
        [["env_0/blocker_1"], ["env_0/blocker_1"]],
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    ctx = _ctx([0, 1_000_000_000], data=data, zones=[_square("lobby")])
    series, ok = _ped_zone_series(parse_atom("intruder in lobby"), ctx)

    assert series is None
    assert ok is False


def test_ped_zone_unresolvable_without_peds_names_column():
    data = pl.DataFrame({"time_ns": [0, 1_000_000_000]})
    ctx = _ctx([0, 1_000_000_000], data=data, zones=[_square("lobby")])

    assert _ped_roster(data) == {}
    series, ok = _ped_zone_series(parse_atom("blocker_1 in lobby"), ctx)

    assert series is None
    assert ok is False


def test_atom_series_dispatches_membership_and_entity():
    snapshot = _snap(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (2_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
        ]
    )
    ctx = _ctx([0, 3_000_000_000], pos_x=[1.0, 1.0], pos_y=[1.0, 1.0], snapshot=snapshot, zones=[_square("lobby")])

    robot_series, robot_ok = _atom_series(parse_atom("robot in lobby"), ctx)
    entity_series, entity_ok = _atom_series(parse_atom("door_1.open == true"), ctx)

    assert robot_ok is True and list(robot_series) == [True, True]
    assert entity_ok is True and list(entity_series) == [False, True]


# -- clause verdicts / malformed ------------------------------------------


def test_clause_verdict_scores_valid_clause():
    ctx = _ctx([0, 1_000_000_000], pos_x=[1.0, 10.0], pos_y=[1.0, 10.0], zones=[_square("lobby")])

    assert _clause_verdict({"op": "eventually", "p": "robot in lobby"}, ctx) is True
    assert _clause_verdict({"op": "always", "p": "robot in lobby"}, ctx) is False


def test_clause_verdict_unknown_on_malformed_dict():
    ctx = _ctx([0, 1_000_000_000], zones=[_square("lobby")])

    assert _clause_verdict({"op": "bogus", "p": "robot in lobby"}, ctx) is None
    assert _clause_verdict({"op": "before", "p": "robot in lobby"}, ctx) is None
    assert _clause_verdict({"op": "always", "p": "not an atom"}, ctx) is None
    assert _clause_verdict({"p": "robot in lobby"}, ctx) is None


# -- calculator end to end ------------------------------------------------


def test_calculate_all_none_without_conditions():
    calc = _calc()
    data = pl.DataFrame({"time_ns": [0, 1_000_000_000], "pos_x": [0.0, 0.0], "pos_y": [0.0, 0.0], "yaw": [0.0, 0.0]})

    results = calc.calculate(_episode(data, conditions=None), {})

    assert all(v is None for v in results.values())
    assert set(results) == set(calc.output_keys())


def test_calculate_all_none_when_world_absent():
    calc = _calc()
    data = pl.DataFrame({"time_ns": [0, 1_000_000_000], "pos_x": [0.0, 0.0], "pos_y": [0.0, 0.0], "yaw": [0.0, 0.0]})
    conditions = [{"op": "never", "p": "robot in pharmacy"}]

    results = calc.calculate(_episode(data, conditions=conditions), {})

    assert all(v is None for v in results.values())


def test_calculate_false_dominates_unknown():
    calc = _calc()
    calc.world = "synthetic_conditions"
    calc._world_cache["synthetic_conditions"] = [_square("pharmacy")]

    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
            "pos_x": [1.0, 1.0],
            "pos_y": [1.0, 1.0],
            "yaw": [0.0, 0.0],
        }
    )
    conditions = [
        {"op": "never", "p": "robot in pharmacy"},
        {"op": "eventually", "p": "robot in ghost_zone"},
    ]

    results = calc.calculate(_episode(data, conditions=conditions), {})

    assert results["condition_success"] == 0.0
    assert results["clauses_total"] == 2
    assert results["clauses_failed"] == 1
    assert results["clauses_unknown"] == 1
    assert results["clauses_passed"] == 0


def test_calculate_unknown_when_only_unknown_and_true():
    calc = _calc()
    calc.world = "synthetic_conditions2"
    calc._world_cache["synthetic_conditions2"] = [_square("pharmacy")]

    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
            "pos_x": [10.0, 10.0],
            "pos_y": [10.0, 10.0],
            "yaw": [0.0, 0.0],
        }
    )
    conditions = [
        {"op": "never", "p": "robot in pharmacy"},
        {"op": "eventually", "p": "robot in ghost_zone"},
    ]

    results = calc.calculate(_episode(data, conditions=conditions), {})

    assert results["condition_success"] is None
    assert results["clauses_passed"] == 1
    assert results["clauses_unknown"] == 1
    assert results["clauses_failed"] == 0


def test_calculate_success_when_all_true():
    calc = _calc()
    calc.world = "synthetic_conditions3"
    calc._world_cache["synthetic_conditions3"] = [_square("pharmacy")]

    data = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
            "pos_x": [10.0, 10.0],
            "pos_y": [10.0, 10.0],
            "yaw": [0.0, 0.0],
        }
    )
    conditions = [{"op": "never", "p": "robot in pharmacy"}]

    results = calc.calculate(_episode(data, conditions=conditions), {})

    assert results["condition_success"] == 1.0
    assert results["clauses_passed"] == 1
    assert results["clauses_total"] == 1

    assert set(results) == set(calc.output_keys())
