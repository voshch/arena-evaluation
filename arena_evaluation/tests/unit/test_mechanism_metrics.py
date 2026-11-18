import dataclasses

import numpy as np
import pytest
from robot_params_factory import make_robot_params

pl = pytest.importorskip("polars")

from arena_evaluation.processing.metrics.ecological.mechanism_common import (
    DoorGeom,
    EgressZone,
    ElevatorGeom,
    FieldSeries,
    MechanismWorld,
    RobotIdentity,
    SemanticView,
    entity_env,
    entity_level,
    entity_name,
    in_polygon,
    mechanism_world_from_description,
    parse_members,
    ped_tracks,
    robot_env_snapshot,
    robot_identity,
    robot_track,
    segment_distance,
    side_of,
    true_runs,
)
from arena_evaluation.processing.metrics.ecological.mechanism_metrics import (
    MechanismMetricsCalculator,
    _boardings,
    _cabin_masks,
    _Ctx,
    in_call_area,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

T = 1_000_000_000
ROBOT = "env_0_jackal"
ROBOT_MEMBER = "env_0/jackal"
PED = "env_0/Pedestrian_1"
ELEV = "env_0/elev_a/0"
DOOR = "env_0/door_1/0"


def _calc(world=None):
    calc = MechanismMetricsCalculator(make_robot_params())
    if world is not None:
        calc.world = "synthetic_mechanism_world"
        calc._world_cache["synthetic_mechanism_world"] = world
    return calc


@dataclasses.dataclass
class Scene:
    """An episode's aligned frame with the flat pedestrian table recorded alongside it."""

    data: pl.DataFrame
    peds: pl.DataFrame | None


_PED_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "ped_id": pl.String,
        "ped_name": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
    }
)


def _flat_peds(frames):
    """Pedestrian rows of frames given as (t_ns, pedestrians), one row per pedestrian per frame."""
    rows = []
    for t_ns, pedestrians in frames:
        for index, ped in enumerate(pedestrians):
            rows.append(
                {
                    "t_ns": t_ns,
                    "env_id": "env_0",
                    "ped_id": str(ped.get("ped_id", f"Pedestrian_{index}")),
                    "ped_name": str(ped.get("ped_name", "")),
                    "x": float(ped["x"]),
                    "y": float(ped["y"]),
                    "vx": None if ped.get("vx") is None else float(ped["vx"]),
                    "vy": None if ped.get("vy") is None else float(ped["vy"]),
                }
            )
    return pl.DataFrame(rows, schema=_PED_SCHEMA)


def _data(points, step_s=1.0, peds=None, **columns):
    """Robot samples on a 1 Hz clock: odometry and ground truth at the given points, with optional pedestrians."""
    n = len(points)
    frame = {
        "t_ns": [int(i * step_s * T) for i in range(n)],
        "pos_x": [float(p[0]) for p in points],
        "pos_y": [float(p[1]) for p in points],
        "yaw": [0.0] * n,
        "pos_x_gt": [float(p[0]) for p in points],
        "pos_y_gt": [float(p[1]) for p in points],
        "yaw_gt": [0.0] * n,
    }
    frame.update(columns)
    return Scene(pl.DataFrame(frame), peds)


def _peds(per_sample):
    """The pedestrian table of per-sample lists of (name, x, y, vx, vy), one frame per 1 Hz sample."""
    return _flat_peds([(i * T, [{"ped_id": name, "ped_name": name, "x": x, "y": y, "vx": vx, "vy": vy} for name, x, y, vx, vy in peds]) for i, peds in enumerate(per_sample)])


_SNAPSHOT_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "world": pl.String,
        "entity_name": pl.String,
        "kind": pl.String,
        "field_name": pl.String,
        "field_kind": pl.String,
        "value_str": pl.String,
        "value_num": pl.Float64,
        "value_bool": pl.Boolean,
    }
)


def _snap(rows):
    """World-semantics rows: (time_s, entity, kind, field, field_kind, value), members given as a list of names."""
    values = {key: [] for key in _SNAPSHOT_SCHEMA}
    for t, entity, kind, field, field_kind, value in rows:
        values["t_ns"].append(int(t * T))
        values["env_id"].append("env_0")
        values["world"].append("w")
        values["entity_name"].append(entity)
        values["kind"].append(kind)
        values["field_name"].append(field)
        values["field_kind"].append(field_kind)
        values["value_str"].append(",".join(sorted(value)) if field_kind == "members" else (value if field_kind == "discrete" else None))
        values["value_num"].append(float(value) if field_kind == "continuous" else None)
        values["value_bool"].append(bool(value) if field_kind == "predicate" else None)
    return pl.DataFrame(values, schema=_SNAPSHOT_SCHEMA)


def _episode(scene, snapshot=None, robot_name=ROBOT):
    data, peds = (scene.data, scene.peds) if isinstance(scene, Scene) else (scene, None)
    start = [data["pos_x"][0], data["pos_y"][0], 0.0] if len(data) else []
    topics = {} if peds is None else {"pedestrian_tracks": peds}
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=start, goal_pos=[], robot_name=robot_name, semantic_snapshot=snapshot, peds=peds, topics=topics)


def _ctx(episode, world=None):
    calc = _calc()
    track = robot_track(calc, episode)
    return _Ctx(
        track=track,
        view=SemanticView.from_episode(episode),
        world=world,
        identity=robot_identity(episode),
        peds=ped_tracks(episode.peds, track.t_ns),
        robot_radius=0.25,
    )


def _elevator(name="elev_a", level="0", capacity=None):
    """2x2 cabin at x,y in [0, 2], door on the +x side, call area radius 3 around (2, 1)."""
    return ElevatorGeom(name, level, [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)], (2.0, 0.0), (2.0, 2.0), 3.0, capacity)


def _door(name="door_1", level="0"):
    """Vertical door segment x = 0, y in [0, 2], activation radius 3."""
    return DoorGeom(name, level, (0.0, 0.0), (0.0, 2.0), 3.0)


# -- naming ---------------------------------------------------------------------


def test_entity_name_level_env():
    assert entity_name("env_6/south_door/0") == "south_door"
    assert entity_level("env_6/south_door/0") == "0"
    assert entity_env("env_6/south_door/0") == "env_6"
    assert entity_name("env_0/door_1") == "door_1"
    assert entity_level("env_0/door_1") is None
    assert entity_name("/env_12/elev/ground/") == "elev"
    assert entity_env("door_1") is None


def test_robot_identity_matches_recorder_and_sim_path_spellings():
    ident = RobotIdentity("env_0_jackal")
    assert ident.matches("env_0/jackal")
    assert ident.matches("/env_0/jackal/")
    assert ident.matches("env_0_jackal")
    assert ident.matches("jackal")
    assert not ident.matches("env_1/jackal")
    assert not ident.matches("env_0/Pedestrian_5")
    assert not ident.matches("")
    assert ident.any_of(["env_0/Pedestrian_5", "env_0/jackal"])


def test_robot_identity_without_env_prefix():
    ident = RobotIdentity("robot_0")
    assert ident.matches("env_3/robot_0")
    assert not ident.matches("env_3/robot_1")


def test_parse_members_round_trip():
    assert parse_members("") == []
    assert parse_members(None) == []
    assert parse_members("env_0/a,env_0/b") == ["env_0/a", "env_0/b"]


def test_robot_env_snapshot_filters_multi_env_only():
    snap = _snap([(0, "env_0/door_1/0", "door", "state", "discrete", "open"), (0, "env_1/door_1/0", "door", "state", "discrete", "closed")])
    assert robot_env_snapshot(snap, "env_1_jackal")["entity_name"].to_list() == ["env_1/door_1/0"]
    assert len(robot_env_snapshot(snap, None)) == 2
    single = snap.filter(pl.col("entity_name").str.starts_with("env_0"))
    assert len(robot_env_snapshot(single, "env_5_jackal")) == 1
    assert robot_env_snapshot(None, ROBOT) is None


# -- series and geometry helpers -----------------------------------------------


def test_field_series_holds_seed_and_steps():
    s = FieldSeries(np.array([2 * T, 5 * T]), ["closed", "open"])
    assert s.at(0) == "closed"
    assert s.at(5 * T) == "open"
    assert s.sample(np.array([0, 3 * T, 6 * T])) == ["closed", "closed", "open"]
    assert s.changes_in(0, 5 * T) == [(2 * T, "closed")]
    assert s.changes_in(3 * T, 6 * T) == [(5 * T, "open")]


def test_semantic_view_members_seen_and_fields():
    view = SemanticView.from_episode(
        _episode(
            _data([(0, 0)] * 3),
            _snap([(0, ELEV, "elevator", "members", "members", []), (1, ELEV, "elevator", "members", "members", [PED]), (0, DOOR, "door", "members", "members", [])]),
        )
    )
    assert view.members_seen("elevator")
    assert not view.members_seen("door")
    assert view.has_field("elevator", "members")
    assert not view.has_kind("schedule")


def test_geometry_helpers():
    assert list(in_polygon(np.array([1.0, 3.0]), np.array([1.0, 1.0]), [(0, 0), (2, 0), (2, 2), (0, 2)])) == [True, False]
    assert segment_distance(np.array([1.0]), np.array([1.0]), (0.0, 0.0), (0.0, 2.0))[0] == pytest.approx(1.0)
    assert segment_distance(np.array([0.0]), np.array([3.0]), (0.0, 0.0), (0.0, 2.0))[0] == pytest.approx(1.0)
    assert list(side_of(np.array([-1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0]), (0.0, 0.0), (0.0, 2.0))) == [1.0, -1.0, 0.0]
    assert true_runs(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]
    elev = _elevator()
    assert list(in_call_area(np.array([4.0, 1.0, 9.0]), np.array([1.0, 1.0, 1.0]), elev)) == [True, False, False]


def test_ped_tracks_holds_each_named_pedestrian_on_the_robot_axis():
    scene = _data([(0, 0)] * 2, peds=_peds([[("env_0/p1", 0, 0, 0.3, 0)], [("env_0/p1", 1, 0, 0.3, 0)]]))
    tracks = ped_tracks(scene.peds, np.array([0, T]))
    assert tracks.named and tracks.keys[0] == ["env_0/p1"]
    assert tracks.vel[1][0, 0] == pytest.approx(0.3)
    assert ped_tracks(None, np.array([0, T])) is None


def test_mechanism_world_from_description_shifts_levels_and_reads_params():
    from arena_simulation_setup.shared.world import Door, Elevator, Signal
    from arena_simulation_setup.tree.World import Level, World, Zone
    from arena_simulation_setup.utils.geometry import Position

    zone = Zone(name="hall", polygon=[Position(0, 0, 0), Position(4, 0, 0), Position(4, 4, 0), Position(0, 4, 0)], semantics=[{"predicate": "egress", "value": True}])
    level = Level(
        id="1",
        zones=[zone],
        doors=[Door(name="d1", start=Position(0, 0, 0), end=Position(0, 2, 0), activation_distance=[1.0, 2.5])],
        elevators=[Elevator(name="e1", position=Position(10, 10, 0), size=[2.0, 2.0, 2.0], door_side="-y", destinations=["e2"])],
    )
    signals = [Signal(name="sig", semantics=[{"preset": "signal", "params": {"position": [1.0, 2.0], "radius": 0.5}}]), Signal(name="unplaced", semantics=[{"preset": "signal"}])]
    world = World(levels={"1": level}, signals=signals)
    mech = mechanism_world_from_description(world, {"1": (100.0, 0.0)})

    door = mech.door("env_0/d1/1")
    assert door.start == (100.0, 0.0) and door.radius == 2.5 and door.level == "1"
    elev = mech.elevator("env_0/e1/1")
    assert elev.door_start == (109.0, 9.0) and elev.door_end == (111.0, 9.0)
    assert elev.capacity is None and elev.destinations == ["e2"]
    sig = mech.signal("env_0/sig")
    assert (sig.x, sig.y, sig.radius, sig.level) == (101.0, 2.0, 0.5, "1")
    assert mech.signal("env_0/unplaced") is None
    assert [z.name for z in mech.egress_zones] == ["hall"]
    assert mech.egress_zones[0].polygon[0] == (100.0, 0.0)


# -- calculator: None / empty edges ------------------------------------------------


def test_no_snapshot_all_none():
    calc = _calc()
    results = calc.calculate(_episode(_data([(0, 0), (1, 0), (2, 0)])), {})
    assert set(results) == set(calc.output_keys())
    assert all(v is None for v in results.values())


def test_empty_episode_all_none():
    calc = _calc()
    empty = pl.DataFrame({"t_ns": [], "pos_x": [], "pos_y": [], "yaw": []}, schema={"t_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64, "yaw": pl.Float64})
    results = calc.calculate(_episode(empty, _snap([(0, DOOR, "door", "state", "discrete", "closed")])), {})
    assert all(v is None for v in results.values())


def test_absent_kinds_are_none_present_kind_computed():
    calc = _calc(MechanismWorld(doors=[_door()]))
    snap = _snap([(0, DOOR, "door", "state", "discrete", "closed"), (0, DOOR, "door", "members", "members", [])])
    results = calc.calculate(_episode(_data([(10, 10)] * 3), snap), {})
    for key in ("boarded_before_exit", "elevator_wait_to_board_s", "elevator_missed_cycles", "elevator_unnecessary_calls", "entered_full_cabin", "cabin_door_blocking_s", "elevator_queue_jumps", "alarm_egress_blocking_s", "blocked_closing_door", "blocked_closing_door_s"):
        assert results[key] is None, key
    assert results["door_freeze_s"] == 0.0
    assert results["door_oscillations"] == 0
    assert results["door_yield_violations"] is None
    assert results["ped_mechanism_delay_seconds"] is None


def test_robot_never_near_mechanisms():
    world = MechanismWorld(doors=[_door()], elevators=[_elevator()])
    snap = _snap(
        [
            (0, DOOR, "door", "state", "discrete", "closed"),
            (0, DOOR, "door", "blocked_close", "predicate", False),
            (0, DOOR, "door", "members", "members", []),
            (0, ELEV, "elevator", "cabin_door", "discrete", "closed"),
            (1, ELEV, "elevator", "cabin_door", "discrete", "open"),
            (0, ELEV, "elevator", "just_arrived", "predicate", False),
            (0, ELEV, "elevator", "capacity", "continuous", 2.0),
            (0, ELEV, "elevator", "members", "members", []),
        ]
    )
    data = _data([(30, 30), (31, 30), (32, 30)], peds=_peds([[("env_0/far", 50.0, 50.0, 0.0, 0.0)]] * 3))
    results = _calc(world).calculate(_episode(data, snap), {})
    assert results["boarded_before_exit"] == 0
    assert results["entered_full_cabin"] == 0
    assert results["elevator_wait_to_board_s"] is None
    assert results["elevator_missed_cycles"] == 0
    assert results["elevator_unnecessary_calls"] == 0
    assert results["elevator_queue_jumps"] == 0
    assert results["cabin_door_blocking_s"] == 0.0
    assert results["blocked_closing_door"] == 0
    assert results["blocked_closing_door_s"] == 0.0
    assert results["door_freeze_s"] == 0.0
    assert results["door_oscillations"] == 0
    assert results["door_yield_violations"] == 0
    assert results["ped_mechanism_delay_seconds"] == 0.0


# -- elevator ---------------------------------------------------------------------------


def _ride_snapshot(ped_exits_at=None, capacity=None):
    rows = [
        (0, ELEV, "elevator", "members", "members", []),
        (0, ELEV, "elevator", "just_arrived", "predicate", False),
        (0, ELEV, "elevator", "occupants", "continuous", 0.0),
        (3, ELEV, "elevator", "just_arrived", "predicate", True),
        (3, ELEV, "elevator", "members", "members", [PED]),
        (5, ELEV, "elevator", "members", "members", [PED, ROBOT_MEMBER]),
        (7, ELEV, "elevator", "members", "members", [ROBOT_MEMBER]),
        (7, ELEV, "elevator", "just_arrived", "predicate", False),
    ]
    if ped_exits_at is not None:
        rows.append((ped_exits_at, ELEV, "elevator", "members", "members", []))
    if capacity is not None:
        rows.append((0, ELEV, "elevator", "capacity", "continuous", capacity))
    return _snap(rows)


def test_boarded_before_exit_counts_boarding_while_arrivals_inside():
    results = _calc().calculate(_episode(_data([(9, 9)] * 9), _ride_snapshot()), {})
    assert results["boarded_before_exit"] == 1
    assert results["elevator_wait_to_board_s"] is None


def test_boarded_before_exit_zero_after_arrivals_left():
    snap = _ride_snapshot(ped_exits_at=4)
    snap = snap.filter(~((pl.col("t_ns") == 5 * T) & (pl.col("field_name") == "members")))
    snap = pl.concat([snap, _snap([(5, ELEV, "elevator", "members", "members", [ROBOT_MEMBER])])])
    results = _calc().calculate(_episode(_data([(9, 9)] * 9), snap), {})
    assert results["boarded_before_exit"] == 0


def test_boarded_before_exit_none_without_just_arrived_field():
    snap = _snap([(0, ELEV, "elevator", "members", "members", []), (2, ELEV, "elevator", "members", "members", [ROBOT_MEMBER])])
    results = _calc().calculate(_episode(_data([(9, 9)] * 4), snap), {})
    assert results["boarded_before_exit"] is None
    assert results["entered_full_cabin"] is None


def test_entered_full_cabin_with_capacity():
    assert _calc().calculate(_episode(_data([(9, 9)] * 9), _ride_snapshot(capacity=1.0)), {})["entered_full_cabin"] == 1
    assert _calc().calculate(_episode(_data([(9, 9)] * 9), _ride_snapshot(capacity=2.0)), {})["entered_full_cabin"] == 0
    assert _calc().calculate(_episode(_data([(9, 9)] * 9), _ride_snapshot(capacity=-1.0)), {})["entered_full_cabin"] == 0


def test_entered_full_cabin_from_full_predicate():
    snap = pl.concat([_ride_snapshot(), _snap([(0, ELEV, "elevator", "full", "predicate", False), (4, ELEV, "elevator", "full", "predicate", True)])])
    assert _calc().calculate(_episode(_data([(9, 9)] * 9), snap), {})["entered_full_cabin"] == 1


def test_boardings_exclude_ride_arrivals():
    other = "env_0/elev_b/1"
    snap = _snap(
        [
            (0, ELEV, "elevator", "members", "members", []),
            (0, other, "elevator", "members", "members", []),
            (2, ELEV, "elevator", "members", "members", [ROBOT_MEMBER]),
            (4, ELEV, "elevator", "members", "members", []),
            (5, other, "elevator", "members", "members", [ROBOT_MEMBER]),
        ]
    )
    ctx = _ctx(_episode(_data([(9, 9)] * 8), snap))
    boardings = _boardings(ctx, _cabin_masks(ctx))
    assert boardings[ELEV] == [2]
    assert boardings[other] == []


def test_boarding_falls_back_to_geometry_without_robot_name():
    world = MechanismWorld(elevators=[_elevator()])
    snap = _snap([(0, ELEV, "elevator", "cabin_door", "discrete", "open")])
    points = [(8, 1), (4.5, 1), (4.5, 1), (4.5, 1), (3, 1), (1, 1), (1, 1)]
    results = _calc(world).calculate(_episode(_data(points), snap, robot_name=None), {})
    assert results["elevator_wait_to_board_s"] == pytest.approx(4.0)


def test_elevator_wait_and_missed_cycles():
    world = MechanismWorld(elevators=[_elevator()])
    snap = _snap(
        [
            (0, ELEV, "elevator", "cabin_door", "discrete", "closed"),
            (1.5, ELEV, "elevator", "cabin_door", "discrete", "opening"),
            (2, ELEV, "elevator", "cabin_door", "discrete", "open"),
            (3, ELEV, "elevator", "cabin_door", "discrete", "closing"),
            (3.5, ELEV, "elevator", "cabin_door", "discrete", "closed"),
            (4.5, ELEV, "elevator", "cabin_door", "discrete", "open"),
            (0, ELEV, "elevator", "members", "members", []),
            (5, ELEV, "elevator", "members", "members", [ROBOT_MEMBER]),
        ]
    )
    points = [(8, 1), (4.5, 1), (4.5, 1), (4.5, 1), (3, 1), (1, 1), (1, 1)]
    results = _calc(world).calculate(_episode(_data(points), snap), {})
    assert results["elevator_wait_to_board_s"] == pytest.approx(4.0)
    assert results["elevator_missed_cycles"] == 1
    assert results["elevator_unnecessary_calls"] == 0


def _pass_by_snapshot():
    return _snap(
        [
            (0, ELEV, "elevator", "cabin_door", "discrete", "closed"),
            (1.5, ELEV, "elevator", "cabin_door", "discrete", "opening"),
            (2, ELEV, "elevator", "cabin_door", "discrete", "open"),
            (0, ELEV, "elevator", "members", "members", []),
        ]
    )


def test_elevator_unnecessary_call_when_robot_passes_by():
    world = MechanismWorld(elevators=[_elevator()])
    points = [(8, 1), (4.5, 1), (4.5, 2), (4.5, 3), (8, 5), (9, 5)]
    data = _data(points, peds=_peds([[("env_0/far", 50.0, 50.0, 0.0, 0.0)]] * 6))
    results = _calc(world).calculate(_episode(data, _pass_by_snapshot()), {})
    assert results["elevator_unnecessary_calls"] == 1
    assert results["elevator_wait_to_board_s"] is None
    assert results["elevator_queue_jumps"] == 0


def test_elevator_call_not_attributed_when_ped_waits():
    world = MechanismWorld(elevators=[_elevator()])
    points = [(8, 1), (4.5, 1), (4.5, 2), (4.5, 3), (8, 5), (9, 5)]
    ped = [(PED, 3.0, 2.0, 0.0, 0.0)]
    data = _data(points, peds=_peds([ped] * 6))
    results = _calc(world).calculate(_episode(data, _pass_by_snapshot()), {})
    assert results["elevator_unnecessary_calls"] == 0


def test_cabin_door_blocking_seconds():
    world = MechanismWorld(elevators=[_elevator()])
    snap = _snap([(0, ELEV, "elevator", "cabin_door", "discrete", "open"), (0, ELEV, "elevator", "members", "members", [])])
    points = [(4, 1), (2.1, 1), (2.1, 1), (2.1, 1), (4, 1)]
    results = _calc(world).calculate(_episode(_data(points), snap), {})
    assert results["cabin_door_blocking_s"] == pytest.approx(2.0)

    closed = _snap([(0, ELEV, "elevator", "cabin_door", "discrete", "closed"), (0, ELEV, "elevator", "members", "members", [])])
    assert _calc(world).calculate(_episode(_data(points), closed), {})["cabin_door_blocking_s"] == 0.0


def test_cabin_door_blocking_none_without_world():
    snap = _snap([(0, ELEV, "elevator", "cabin_door", "discrete", "open")])
    assert _calc().calculate(_episode(_data([(2.1, 1)] * 3), snap), {})["cabin_door_blocking_s"] is None


def _queue_episode(ped_leaves):
    points = [(8, 1), (4.5, 1), (3, 1), (1, 1), (1, 1)]
    waiting = (PED, 3.5, 1.5, 0.0, 0.0)
    gone = (PED, 9.0, 9.0, 0.0, 0.0)
    peds = [[waiting], [waiting], [gone if ped_leaves else waiting], [gone if ped_leaves else waiting], [waiting]]
    snap = _snap([(0, ELEV, "elevator", "members", "members", []), (3, ELEV, "elevator", "members", "members", [ROBOT_MEMBER])])
    return _episode(_data(points, peds=_peds(peds)), snap)


def test_elevator_queue_jump_over_waiting_ped():
    world = MechanismWorld(elevators=[_elevator()])
    assert _calc(world).calculate(_queue_episode(ped_leaves=False), {})["elevator_queue_jumps"] == 1


def test_elevator_no_queue_jump_when_ped_left():
    world = MechanismWorld(elevators=[_elevator()])
    assert _calc(world).calculate(_queue_episode(ped_leaves=True), {})["elevator_queue_jumps"] == 0


def test_elevator_queue_jumps_none_without_peds():
    world = MechanismWorld(elevators=[_elevator()])
    snap = _snap([(0, ELEV, "elevator", "members", "members", []), (3, ELEV, "elevator", "members", "members", [ROBOT_MEMBER])])
    results = _calc(world).calculate(_episode(_data([(8, 1), (4.5, 1), (3, 1), (1, 1)]), snap), {})
    assert results["elevator_queue_jumps"] is None
    assert results["elevator_wait_to_board_s"] == pytest.approx(2.0)


# -- doors ------------------------------------------------------------------------------


def test_blocked_closing_door_count_and_seconds():
    world = MechanismWorld(doors=[_door()])
    snap = _snap(
        [
            (0, DOOR, "door", "blocked_close", "predicate", False),
            (2, DOOR, "door", "blocked_close", "predicate", True),
            (4, DOOR, "door", "blocked_close", "predicate", False),
            (6, DOOR, "door", "blocked_close", "predicate", True),
            (7, DOOR, "door", "blocked_close", "predicate", False),
            (0, DOOR, "door", "state", "discrete", "closing"),
        ]
    )
    points = [(5, 1), (0.2, 1), (0.2, 1), (0.2, 1), (0.2, 1), (5, 1), (5, 1), (5, 1), (5, 1)]
    results = _calc(world).calculate(_episode(_data(points), snap), {})
    assert results["blocked_closing_door"] == 1
    assert results["blocked_closing_door_s"] == pytest.approx(2.0)


def test_blocked_closing_door_members_fallback_without_geometry():
    snap = _snap(
        [
            (0, DOOR, "door", "blocked_close", "predicate", False),
            (1, DOOR, "door", "blocked_close", "predicate", True),
            (3, DOOR, "door", "blocked_close", "predicate", False),
            (0, DOOR, "door", "members", "members", []),
            (1, DOOR, "door", "members", "members", [ROBOT_MEMBER]),
            (3, DOOR, "door", "members", "members", []),
        ]
    )
    results = _calc().calculate(_episode(_data([(0, 0)] * 5), snap), {})
    assert results["blocked_closing_door"] == 1
    assert results["blocked_closing_door_s"] == pytest.approx(2.0)


def test_blocked_closing_door_none_when_unattributable():
    snap = _snap([(0, DOOR, "door", "blocked_close", "predicate", True)])
    results = _calc().calculate(_episode(_data([(0, 0)] * 3), snap, robot_name=None), {})
    assert results["blocked_closing_door"] is None
    assert results["blocked_closing_door_s"] is None


def test_door_freeze_and_oscillations():
    world = MechanismWorld(doors=[_door()])
    snap = _snap(
        [
            (0, DOOR, "door", "state", "discrete", "closed"),
            (1, DOOR, "door", "state", "discrete", "opening"),
            (2, DOOR, "door", "state", "discrete", "open"),
            (3, DOOR, "door", "state", "discrete", "closing"),
            (4, DOOR, "door", "state", "discrete", "opening"),
            (5, DOOR, "door", "state", "discrete", "open"),
        ]
    )
    points = [(1, 1)] * 6 + [(4.5, 1)]
    results = _calc(world).calculate(_episode(_data(points), snap), {})
    assert results["door_freeze_s"] == pytest.approx(2.0)
    assert results["door_oscillations"] == 1


def test_door_freeze_uses_open_predicate_when_state_missing():
    world = MechanismWorld(doors=[_door()])
    snap = _snap([(0, DOOR, "door", "open", "predicate", True)])
    results = _calc(world).calculate(_episode(_data([(1, 1)] * 3), snap), {})
    assert results["door_freeze_s"] == pytest.approx(1.0)
    assert results["door_oscillations"] == 0


def _crossing(ped):
    points = [(-2, 1), (-1, 1), (-0.2, 1), (0.3, 1), (1.5, 1)]
    snap = _snap([(0, DOOR, "door", "state", "discrete", "open")])
    return _episode(_data(points, peds=_peds([[ped]] * 5)), snap)


def test_door_yield_violation_ped_approaching_from_far_side():
    world = MechanismWorld(doors=[_door()])
    assert _calc(world).calculate(_crossing((PED, 1.5, 1.2, -0.5, 0.0)), {})["door_yield_violations"] == 1


def test_door_yield_no_violation_ped_moving_away_or_same_side():
    world = MechanismWorld(doors=[_door()])
    assert _calc(world).calculate(_crossing((PED, 1.5, 1.2, 0.5, 0.0)), {})["door_yield_violations"] == 0
    assert _calc(world).calculate(_crossing((PED, -1.5, 1.2, 0.5, 0.0)), {})["door_yield_violations"] == 0


def test_ped_mechanism_delay_seconds():
    world = MechanismWorld(doors=[_door()])
    ped = [(PED, 1.0, 1.0, 0.0, 0.0)]
    closed = _snap([(0, DOOR, "door", "state", "discrete", "closed")])
    data = _data([(10, 10)] * 4, peds=_peds([ped] * 4))
    assert _calc(world).calculate(_episode(data, closed), {})["ped_mechanism_delay_seconds"] == pytest.approx(3.0)

    opened = _snap([(0, DOOR, "door", "state", "discrete", "open")])
    assert _calc(world).calculate(_episode(data, opened), {})["ped_mechanism_delay_seconds"] == 0.0

    walking = _data([(10, 10)] * 4, peds=_peds([[(PED, 1.0, 1.0, 1.0, 0.0)]] * 4))
    assert _calc(world).calculate(_episode(walking, closed), {})["ped_mechanism_delay_seconds"] == 0.0


def test_ped_mechanism_delay_at_closed_elevator():
    world = MechanismWorld(elevators=[_elevator()])
    snap = _snap([(0, ELEV, "elevator", "cabin_door", "discrete", "closed"), (2, ELEV, "elevator", "cabin_door", "discrete", "open")])
    data = _data([(10, 10)] * 4, peds=_peds([[(PED, 3.5, 1.0, 0.0, 0.0)]] * 4))
    assert _calc(world).calculate(_episode(data, snap), {})["ped_mechanism_delay_seconds"] == pytest.approx(2.0)


# -- alarms -------------------------------------------------------------------------------


def test_alarm_egress_blocking_seconds():
    world = MechanismWorld(egress_zones=[EgressZone("exit_corridor", "0", [(0, 0), (4, 0), (4, 4), (0, 4)])])
    snap = _snap([(0, "env_0/fire_alarm/0", "schedule", "active", "predicate", False), (2, "env_0/fire_alarm/0", "schedule", "active", "predicate", True)])
    results = _calc(world).calculate(_episode(_data([(1, 1)] * 5), snap), {})
    assert results["alarm_egress_blocking_s"] == pytest.approx(2.0)

    outside = _calc(world).calculate(_episode(_data([(9, 9)] * 5), snap), {})
    assert outside["alarm_egress_blocking_s"] == 0.0


def test_alarm_egress_blocking_none_without_egress_zones_or_schedule():
    snap = _snap([(0, "env_0/fire_alarm/0", "schedule", "active", "predicate", True)])
    assert _calc(MechanismWorld()).calculate(_episode(_data([(1, 1)] * 3), snap), {})["alarm_egress_blocking_s"] is None
    world = MechanismWorld(egress_zones=[EgressZone("exit", "0", [(0, 0), (4, 0), (4, 4), (0, 4)])])
    door_only = _snap([(0, DOOR, "door", "state", "discrete", "open")])
    assert _calc(world).calculate(_episode(_data([(1, 1)] * 3), door_only), {})["alarm_egress_blocking_s"] is None


def test_multi_env_snapshot_scoped_to_robot_env():
    world = MechanismWorld(doors=[_door()])
    snap = _snap(
        [
            (0, "env_0/door_1/0", "door", "state", "discrete", "closing"),
            (0, "env_1/door_1/0", "door", "state", "discrete", "open"),
        ]
    )
    results = _calc(world).calculate(_episode(_data([(1, 1)] * 3), snap), {})
    assert results["door_freeze_s"] == 0.0
    other_env = _calc(world).calculate(_episode(_data([(1, 1)] * 3), snap, robot_name="env_1_jackal"), {})
    assert other_env["door_freeze_s"] == pytest.approx(1.0)


def test_elevator_ride_teleport_keeps_full_track():
    """A ride teleport landing at an elevator arrival must not cut the track to one floor."""
    snap = _snap(
        [
            (0, ELEV, "elevator", "just_arrived", "predicate", False),
            (3, ELEV, "elevator", "just_arrived", "predicate", True),
        ]
    )
    points = [(0, 0), (0.5, 0), (1, 0), (101, 0), (101.5, 0), (102, 0), (102.5, 0)]
    episode = _episode(_data(points), snap)
    track = robot_track(_calc(), episode)
    assert len(track) == len(points)

    no_arrival = _episode(_data(points), _snap([(0, ELEV, "elevator", "just_arrived", "predicate", False)]))
    assert len(robot_track(_calc(), no_arrival)) == 4
