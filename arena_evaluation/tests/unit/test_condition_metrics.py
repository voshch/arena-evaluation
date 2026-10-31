import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.conditions")

from arena_simulation_setup.shared.conditions import parse_atom, parse_node

from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry, _reconstruct_events
from arena_evaluation.processing.metrics.ecological.condition_metrics import (
    ConditionComplianceCalculator,
    _EvalContext,
    _clause_verdict,
    _entity_roster,
    _first_true,
    _level_cache,
    _node_series,
    _operator_verdict,
    _ped_roster,
    _strip_env,
    _value_at,
    _values_equal,
    _wall_cache,
    _world_cache,
    clips_on_axis,
    load_wall_grid,
    load_zone_geometry,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
from snapshot_rows import snapshot_frame


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


@pytest.mark.parametrize(
    ("operator", "samples", "verdict"),
    [
        ("always", [True, True], True),
        ("always", [True, False], False),
        ("never", [False, False], True),
        ("never", [False, True], False),
        ("eventually", [False, True], True),
        ("eventually", [False, False], False),
    ],
)
def test_unary_operator_verdicts(operator, samples, verdict):
    assert _operator_verdict(operator, np.array(samples), True, None, False) is verdict


def test_unary_operator_unknown_when_atom_unresolvable():
    assert _operator_verdict("always", None, False, None, False) is None
    assert _operator_verdict("never", None, False, None, False) is None
    assert _operator_verdict("eventually", None, False, None, False) is None


@pytest.mark.parametrize(
    ("operator", "p", "q", "verdict"),
    [
        pytest.param("before", [False, True, False], [False, False, True], True, id="before_p_first"),
        pytest.param("before", [False, True], [False, False], True, id="before_q_never_occurs"),
        pytest.param("before", [False, False], [True, True], False, id="before_p_never_occurs"),
        pytest.param("before", [False, True, False], [False, True, False], False, id="before_simultaneous_first_sample"),
        pytest.param("never_during", [True, False, False], [False, False, True], True, id="never_during_no_cooccurrence"),
        pytest.param("never_during", [False, True, False], [False, True, False], False, id="never_during_cooccurrence"),
    ],
)
def test_binary_operator_verdicts(operator, p, q, verdict):
    assert _operator_verdict(operator, np.array(p), True, np.array(q), True) is verdict


def test_before_unknown_when_either_atom_unresolvable():
    p = np.array([True, False])
    assert _operator_verdict("before", p, True, None, False) is None
    assert _operator_verdict("before", None, False, p, True) is None


def test_never_during_unknown_when_atom_unresolvable():
    p = np.array([True, False])
    assert _operator_verdict("never_during", p, True, None, False) is None


# -- entity atoms ---------------------------------------------------------


def test_entity_atom_true_after_recorded_change():
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (2_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
        ]
    )
    ctx = _ctx([0, 1_000_000_000, 3_000_000_000], snapshot=snapshot)
    series, ok = _node_series(parse_atom("door_1.open == true"), ctx)

    assert ok is True
    assert list(series) == [False, False, True]


def test_entity_atom_holds_seed_value_when_never_changed():
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
        ]
    )
    ctx = _ctx([0, 2_000_000_000], snapshot=snapshot)
    series, ok = _node_series(parse_atom("door_1.open == false"), ctx)

    assert ok is True
    assert list(series) == [True, True]


def test_entity_atom_unresolvable_unknown_entity():
    snapshot = snapshot_frame(
        [
            (1_000_000_000, "env_0/door_1", "door", "state", "discrete", "open"),
        ]
    )
    ctx = _ctx([0, 2_000_000_000], snapshot=snapshot)
    series, ok = _node_series(parse_atom("ghost.open == true"), ctx)

    assert series is None
    assert ok is False


def test_entity_atom_unresolvable_field_not_registered():
    snapshot = snapshot_frame(
        [
            (1_000_000_000, "env_0/door_1", "door", "state", "discrete", "open"),
        ]
    )
    ctx = _ctx([0, 2_000_000_000], snapshot=snapshot)
    series, ok = _node_series(parse_atom("door_1.nonsense == x"), ctx)

    assert series is None
    assert ok is False


def test_entity_roster_drops_ambiguous_bare_names():
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "state", "discrete", "open"),
            (0, "env_1/door_1", "door", "state", "discrete", "open"),
        ]
    )
    roster = _entity_roster(snapshot)

    assert "door_1" not in roster


def test_entity_atom_resolves_multi_kind_entity_and_drops_env_duplicate():
    snapshot = snapshot_frame(
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

    open_series, open_ok = _node_series(parse_atom("main_door/0.open == true"), ctx)
    locked_series, locked_ok = _node_series(parse_atom("main_door/0.locked == false"), ctx)
    dup_series, dup_ok = _node_series(parse_atom("x/0.open == true"), ctx)

    assert open_ok is True and list(open_series) == [False, True]
    assert locked_ok is True and list(locked_series) == [False, True]
    assert dup_series is None and dup_ok is False


def test_entity_atom_resolves_a_world_entity_without_its_level_suffix():
    snapshot = snapshot_frame(
        [
            (0, "env_0/ward/0", "zone", "quiet", "predicate", True),
            (2_000_000_000, "env_0/ward/0", "zone", "quiet", "predicate", False),
            (0, "env_0/lift/0", "elevator", "open", "predicate", True),
            (0, "env_0/lift/1", "elevator", "open", "predicate", True),
            (0, "env_0/jackal", "robot", "phase", "discrete", "0"),
            (0, "env_0/jackal/0", "door", "open", "predicate", True),
        ]
    )
    ctx = _ctx([0, 3_000_000_000], snapshot=snapshot)

    quiet_series, quiet_ok = _node_series(parse_atom("ward.quiet == false"), ctx)
    lift_series, lift_ok = _node_series(parse_atom("lift.open == true"), ctx)

    assert quiet_ok is True and list(quiet_series) == [False, True]
    assert lift_series is None and lift_ok is False
    assert _entity_roster(snapshot)["jackal"] == "env_0/jackal"


def test_entity_roster_names_a_world_object_without_its_world_prefix_unless_taken():
    snapshot = snapshot_frame(
        [
            (0, "env_0/world_wardrobe", "articulated", "openness.door", "continuous", 0.0),
            (0, "env_0/world_table", "articulated", "openness.leaf", "continuous", 0.0),
            (0, "env_0/table", "articulated", "openness.leaf", "continuous", 1.0),
        ]
    )
    roster = _entity_roster(snapshot)

    assert roster["wardrobe"] == "env_0/world_wardrobe"
    assert roster["world_wardrobe"] == "env_0/world_wardrobe"
    assert roster["table"] == "env_0/table"


# -- robot zone atoms -----------------------------------------------------


def test_robot_zone_membership_inside_and_outside():
    ctx = _ctx([0, 1_000_000_000], pos_x=[1.0, 10.0], pos_y=[1.0, 10.0], zones=[_square("lobby")])
    series, ok = _node_series(parse_atom("robot in lobby"), ctx)

    assert ok is True
    assert list(series) == [True, False]


def test_robot_zone_unresolvable_when_zone_not_annotated():
    ctx = _ctx([0, 1_000_000_000], pos_x=[1.0, 1.0], pos_y=[1.0, 1.0], zones=[_square("lobby")])
    series, ok = _node_series(parse_atom("robot in pharmacy"), ctx)

    assert series is None
    assert ok is False


def test_robot_zone_unresolvable_without_pose_anchor():
    ctx = _ctx([0, 1_000_000_000], pos_x=[1.0, 1.0], pos_y=[1.0, 1.0], zones=[_square("lobby")])
    ctx.pose_valid = False
    series, ok = _node_series(parse_atom("robot in lobby"), ctx)

    assert series is None
    assert ok is False

    ped_data = _ped_data([0, 1_000_000_000], [["walker"], ["walker"]], [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ctx = _ctx([0, 1_000_000_000], data=ped_data, zones=[_square("lobby")])
    ctx.pose_valid = False
    series, ok = _node_series(parse_atom("walker in lobby"), ctx)

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
    series, ok = _node_series(parse_atom("blocker_1 in lobby"), ctx)

    assert ok is True
    assert list(series) == [False, True, False]


def test_ped_zone_resolves_level_suffixed_names_by_authored_name():
    data = _ped_data(
        [0, 1_000_000_000],
        [["env_1/queuer_1/0", "env_1/queuer_2/0"], ["env_1/queuer_1/0", "env_1/queuer_2/0"]],
        [[10.0, 10.0, 0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 10.0, 10.0, 0.0]],
    )
    ctx = _ctx([0, 1_000_000_000], data=data, zones=[_square("lobby")])

    assert _ped_roster(data) == {"queuer_1/0": "env_1/queuer_1/0", "queuer_1": "env_1/queuer_1/0", "queuer_2/0": "env_1/queuer_2/0", "queuer_2": "env_1/queuer_2/0"}
    series, ok = _node_series(parse_atom("queuer_1 in lobby"), ctx)
    assert ok is True
    assert list(series) == [False, True]


def test_ped_roster_drops_authored_names_shared_across_levels():
    data = _ped_data([0], [["env_0/rider/0", "env_0/rider/1"]], [[1.0, 1.0, 0.0, 2.0, 2.0, 0.0]])
    assert _ped_roster(data) == {"rider/0": "env_0/rider/0", "rider/1": "env_0/rider/1"}


def test_ped_zone_unresolvable_when_ped_not_recorded():
    data = _ped_data(
        [0, 1_000_000_000],
        [["env_0/blocker_1"], ["env_0/blocker_1"]],
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )
    ctx = _ctx([0, 1_000_000_000], data=data, zones=[_square("lobby")])
    series, ok = _node_series(parse_atom("intruder in lobby"), ctx)

    assert series is None
    assert ok is False


def test_ped_zone_unresolvable_without_peds_names_column():
    data = pl.DataFrame({"time_ns": [0, 1_000_000_000]})
    ctx = _ctx([0, 1_000_000_000], data=data, zones=[_square("lobby")])

    assert _ped_roster(data) == {}
    series, ok = _node_series(parse_atom("blocker_1 in lobby"), ctx)

    assert series is None
    assert ok is False


def test_node_series_dispatches_membership_and_entity():
    snapshot = snapshot_frame(
        [
            (0, "env_0/door_1", "door", "open", "predicate", False),
            (2_000_000_000, "env_0/door_1", "door", "open", "predicate", True),
        ]
    )
    ctx = _ctx([0, 3_000_000_000], pos_x=[1.0, 1.0], pos_y=[1.0, 1.0], snapshot=snapshot, zones=[_square("lobby")])

    robot_series, robot_ok = _node_series(parse_atom("robot in lobby"), ctx)
    entity_series, entity_ok = _node_series(parse_atom("door_1.open == true"), ctx)

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
    _wall_cache["synthetic_conditions"] = None

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
    _wall_cache["synthetic_conditions2"] = None

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
    _wall_cache["synthetic_conditions3"] = None

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


# -- gesture clips --------------------------------------------------------


def _gestures(rows):
    return pl.DataFrame(
        {
            "time_ns": [t for t, _, _ in rows],
            "ped_id": [pid for _, pid, _ in rows],
            "slot": ["body"] * len(rows),
            "clip": [clip for _, _, clip in rows],
            "hand": [""] * len(rows),
            "at_x": [0.0] * len(rows),
            "at_y": [0.0] * len(rows),
            "at_z": [0.0] * len(rows),
            "render_pose_override": [False] * len(rows),
        }
    )


def test_clips_on_axis_holds_the_peds_message_and_names_every_pedestrian_in_it():
    peds = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 3_000_000_000],
            "peds_names": [["env_0/alice", "env_0/bob"], ["env_0/alice"]],
            "peds_ids": [[1, 2], [1]],
            "peds_positions": [[1.0, 1.0, 0.0, 2.0, 2.0, 0.0], [1.0, 1.0, 0.0]],
        }
    )
    gestures = _gestures([(1_000_000_000, 2, "wave"), (3_000_000_000, 1, "beckon"), (3_000_000_000, 1, "gaze")])
    axis = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000])

    assert clips_on_axis(peds, gestures, axis) == [
        {},
        {"alice": frozenset(), "bob": frozenset({"wave"})},
        {"alice": frozenset(), "bob": frozenset({"wave"})},
        {"alice": frozenset({"beckon", "gaze"})},
        {"alice": frozenset({"beckon", "gaze"})},
    ]
    assert clips_on_axis(peds, gestures.clear(), axis) == [{}, {"alice": frozenset(), "bob": frozenset()}, {"alice": frozenset(), "bob": frozenset()}, {"alice": frozenset()}, {"alice": frozenset()}]
    assert clips_on_axis(peds, None, axis) is None
    assert clips_on_axis(peds.drop("peds_ids"), gestures, axis) is None
    assert clips_on_axis(None, gestures, axis) is None


def test_calculate_plays_atom_from_the_recorded_gestures():
    calc = _calc()
    calc.world = "synthetic_conditions4"
    calc._world_cache["synthetic_conditions4"] = [_square("pharmacy")]
    _wall_cache["synthetic_conditions4"] = None
    times = [0, 1_000_000_000, 2_000_000_000]
    data = pl.DataFrame({"time_ns": times, "pos_x": [1.0] * 3, "pos_y": [1.0] * 3, "yaw": [0.0] * 3, "peds_names": [["env_0/alice"]] * 3, "peds_positions": [[2.0, 2.0, 0.0]] * 3})
    peds = pl.DataFrame({"time_ns": times, "peds_names": [["env_0/alice"]] * 3, "peds_ids": [[1]] * 3, "peds_positions": [[2.0, 2.0, 0.0]] * 3})
    gestures = _gestures([(2_000_000_000, 1, "beckon")])
    conditions = [{"op": "eventually", "p": "alice plays beckon"}, {"op": "always", "p": "not alice plays wave"}, {"op": "eventually", "p": "bob plays beckon"}]

    episode = _episode(data, conditions=conditions)
    episode.peds = peds
    episode.topics = {"ped_gestures": gestures}
    results = calc.calculate(episode, {})

    assert results["clauses_passed"] == 2
    assert results["clauses_unknown"] == 1
    assert results["condition_success"] is None

    episode.topics = {"ped_gestures": gestures.clear()}
    results = calc.calculate(episode, {})

    assert results["condition_success"] == 0.0
    assert results["clauses_failed"] == 1
    assert results["clauses_passed"] == 1
    assert results["clauses_unknown"] == 1

    episode.topics = {}
    results = calc.calculate(episode, {})

    assert results["clauses_unknown"] == 3


def test_zone_and_wall_loaders_answer_from_their_caches():
    _world_cache["synthetic_cached"] = [_square("pharmacy")]
    assert load_zone_geometry("synthetic_cached") == _world_cache["synthetic_cached"]
    _level_cache["synthetic_missing"] = None
    assert load_wall_grid("synthetic_missing") is None
    assert load_zone_geometry("synthetic_missing") is None
    assert _wall_cache["synthetic_missing"] is None and _world_cache["synthetic_missing"] is None


# -- articulated families and structured conditions -----------------------


def _wardrobe_snapshot():
    return snapshot_frame(
        [
            (0, "env_0/wardrobe", "articulated", "openness.door_left", "continuous", 0.1),
            (0, "env_0/wardrobe", "articulated", "openness.door_right", "continuous", 0.0),
            (0, "env_0/wardrobe", "articulated", "q.door_left", "continuous", 0.2),
            (0, "env_0/wardrobe", "articulated", "state", "discrete", "closed"),
            (2_000_000_000, "env_0/wardrobe", "articulated", "openness.door_left", "continuous", 0.9),
            (2_000_000_000, "env_0/wardrobe", "articulated", "state", "discrete", ""),
        ]
    )


def test_family_at_reads_the_members_held_at_an_instant():
    ctx = _ctx([0, 1_000_000_000, 3_000_000_000], snapshot=_wardrobe_snapshot())

    assert ctx.family_at("wardrobe", "openness", 1_000_000_000) == {"door_left": "0.1", "door_right": "0.0"}
    assert ctx.family_at("wardrobe", "openness", 3_000_000_000) == {"door_left": "0.9", "door_right": "0.0"}
    assert ctx.family_at("wardrobe", "q", 3_000_000_000) == {"door_left": "0.2"}
    assert ctx.family_at("wardrobe", "setting", 3_000_000_000) == {}
    assert ctx.family_at("ghost", "openness", 3_000_000_000) is None
    assert ctx.sample(2).family("wardrobe", "openness") == {"door_left": "0.9", "door_right": "0.0"}


def test_node_series_judges_field_and_family_tests():
    ctx = _ctx([0, 1_000_000_000, 3_000_000_000], snapshot=_wardrobe_snapshot())

    field_series, field_ok = _node_series(parse_node({"field": "wardrobe.openness.door_left", "ge": 0.8}), ctx)
    some_series, some_ok = _node_series(parse_node({"some": "wardrobe.openness", "ge": 0.5}), ctx)
    each_series, each_ok = _node_series(parse_node({"each": "wardrobe.openness", "ge": 0.5}), ctx)
    empty_series, empty_ok = _node_series(parse_node({"some": "wardrobe.setting", "eq": "off"}), ctx)
    ghost_series, ghost_ok = _node_series(parse_node({"some": "ghost.openness", "ge": 0.5}), ctx)
    text_series, text_ok = _node_series(parse_node("wardrobe.openness.door_left == 0.9"), ctx)

    assert field_ok and list(field_series) == [False, False, True]
    assert some_ok and list(some_series) == [False, False, True]
    assert each_ok and list(each_series) == [False, False, False]
    assert empty_ok and list(empty_series) == [False, False, False]
    assert ghost_series is None and not ghost_ok
    assert text_ok and list(text_series) == [False, False, True]


def test_node_series_times_holds_across_the_samples():
    ctx = _ctx([0, 2_000_000_000, 3_000_000_000, 4_000_000_000], snapshot=_wardrobe_snapshot())

    series, ok = _node_series(parse_node({"field": "wardrobe.openness.door_left", "ge": 0.8, "for": 2}), ctx)

    assert ok and list(series) == [False, False, False, True]


def test_clause_verdict_scores_structured_p_and_q():
    ctx = _ctx([0, 1_000_000_000, 3_000_000_000, 4_000_000_000], pos_x=[10.0, 1.0, 1.0, 1.0], pos_y=[10.0, 1.0, 1.0, 1.0], snapshot=_wardrobe_snapshot(), zones=[_square("lobby")])

    assert _clause_verdict({"op": "eventually", "p": {"some": "wardrobe.openness", "ge": 0.5, "for": 1}}, ctx) is True
    assert _clause_verdict({"op": "eventually", "p": {"some": "wardrobe.openness", "ge": 0.5, "for": 5}}, ctx) is False
    assert _clause_verdict({"op": "before", "p": {"subject": "robot", "in": "lobby"}, "q": {"field": "wardrobe.openness.door_left", "ge": 0.8}}, ctx) is True
    assert _clause_verdict({"op": "never_during", "p": {"subject": "robot", "in": "lobby"}, "q": {"each": "wardrobe.openness", "lt": 0.5}}, ctx) is False
    assert _clause_verdict({"op": "always", "p": {"field": "wardrobe.openness.door_left", "within": [0.0, 1.0]}}, ctx) is True
    assert _clause_verdict({"op": "always", "p": {"field": "wardrobe.state", "ge": 0.5}}, ctx) is None
    assert _clause_verdict({"op": "always", "p": {"field": "wardrobe.state", "ge": "half"}}, ctx) is None
