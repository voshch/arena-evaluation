
import numpy as np
import pytest
from robot_params_factory import make_robot_params

pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.conditions")

import polars as pl

from arena_simulation_setup.shared.conditions import parse_atom

from arena_evaluation.processing.metrics.common.events import (
    reconstruct_events as _reconstruct_events,
)
from arena_evaluation.processing.metrics.common.zones import (
    ZoneGeometry as _ZoneGeometry,
)
from arena_evaluation.processing.metrics.common.caching import (
    world_cache,
)
from arena_evaluation.processing.metrics.ecological.condition_metrics import (
    ConditionComplianceCalculator,
    _EvalContext,
    _entity_roster,
    _node_series,
    _ped_roster,
    _wall_cache,
    clips_on_axis,
)
from arena_evaluation.processing.pedestrian_frames import frames_of
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from snapshot_rows import snapshot_frame


def _calc():
    return ConditionComplianceCalculator(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )


def _start_pos(data):
    if data is not None and "x" in data.columns and len(data) > 0:
        return [data["x"][0], data["y"][0], data["yaw"][0]]
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


def _ctx(t_ns, pos_x=None, pos_y=None, snapshot=None, data=None, peds=None, zones=None):
    n = len(t_ns)
    time_arr = np.asarray(t_ns, dtype=np.int64)
    pos_x = np.zeros(n) if pos_x is None else np.asarray(pos_x, dtype=float)
    pos_y = np.zeros(n) if pos_y is None else np.asarray(pos_y, dtype=float)
    zones = zones or []
    data_df = data if data is not None else pl.DataFrame({"t_ns": list(t_ns)})
    peds_df = peds
    if peds_df is None and data is not None and "ped_name" in data.columns:
        peds_df = data
    peds_frames = frames_of(peds_df) if peds_df is not None and len(peds_df) > 0 else []
    peds_times = np.array(
        [int(frame["t_ns"][0]) for frame in peds_frames], dtype=np.int64
    )
    return _EvalContext(
        events=_reconstruct_events(snapshot),
        t_ns=time_arr,
        pos_x=pos_x,
        pos_y=pos_y,
        data=data_df,
        zones_by_name={z.name: z for z in zones},
        entity_roster=_entity_roster(snapshot),
        ped_roster=_ped_roster(peds_df),
        peds_frames=peds_frames,
        peds_times=peds_times,
    )








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






def _ped_data(t_ns, names, positions):
    rows = []
    for t, n_list, p_list in zip(t_ns, names, positions):
        pts = []
        if p_list and isinstance(p_list[0], (list, tuple)):
            pts = [(p[0], p[1]) for p in p_list]
        elif p_list:
            for j in range(0, len(p_list), 3):
                if j + 1 < len(p_list):
                    pts.append((p_list[j], p_list[j + 1]))
        for idx, (name, (x, y)) in enumerate(zip(n_list, pts)):
            rows.append(
                {
                    "t_ns": int(t),
                    "ped_id": f"ped_{idx}",
                    "ped_name": str(name),
                    "x": float(x),
                    "y": float(y),
                }
            )
    if rows:
        return pl.DataFrame(
            rows,
            schema={
                "t_ns": pl.Int64,
                "ped_id": pl.Utf8,
                "ped_name": pl.Utf8,
                "x": pl.Float64,
                "y": pl.Float64,
            },
        )
    return pl.DataFrame(
        schema={
            "t_ns": pl.Int64,
            "ped_id": pl.Utf8,
            "ped_name": pl.Utf8,
            "x": pl.Float64,
            "y": pl.Float64,
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






def test_calculate_all_none_when_world_absent():
    calc = _calc()
    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )
    conditions = [{"op": "never", "p": "robot in pharmacy"}]

    results = calc.calculate(_episode(data, conditions=conditions), {})

    assert all(v is None for v in results.values())


def test_calculate_success_when_all_true():
    calc = _calc()
    calc.world = "synthetic_conditions3"
    world_cache["synthetic_conditions3"] = [_square("pharmacy")]
    _wall_cache["synthetic_conditions3"] = None

    data = pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [10.0, 10.0],
            "y": [10.0, 10.0],
            "yaw": [0.0, 0.0],
        }
    )
    conditions = [{"op": "never", "p": "robot in pharmacy"}]

    results = calc.calculate(_episode(data, conditions=conditions), {})

    assert results["condition_success"] is True
    assert results["clauses_passed"] == 1
    assert results["clauses_total"] == 1

    assert set(results) == set(calc.output_keys())




def _gestures(rows):
    return pl.DataFrame(
        {
            "t_ns": [t for t, _, _ in rows],
            "ped_id": [str(pid) for _, pid, _ in rows],
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
            "t_ns": [1_000_000_000, 1_000_000_000, 3_000_000_000],
            "ped_name": ["env_0/alice", "env_0/bob", "env_0/alice"],
            "ped_id": ["1", "2", "1"],
            "x": [1.0, 2.0, 1.0],
            "y": [1.0, 2.0, 1.0],
        }
    )
    gestures = _gestures(
        [
            (1_000_000_000, "2", "wave"),
            (3_000_000_000, "1", "beckon"),
            (3_000_000_000, "1", "gaze"),
        ]
    )
    axis = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000])

    assert clips_on_axis(peds, gestures, axis) == [
        {},
        {"alice": frozenset(), "bob": frozenset({"wave"})},
        {"alice": frozenset(), "bob": frozenset({"wave"})},
        {"alice": frozenset({"beckon", "gaze"})},
        {"alice": frozenset({"beckon", "gaze"})},
    ]
    assert clips_on_axis(peds, gestures.clear(), axis) == [
        {},
        {"alice": frozenset(), "bob": frozenset()},
        {"alice": frozenset(), "bob": frozenset()},
        {"alice": frozenset()},
        {"alice": frozenset()},
    ]
    assert clips_on_axis(peds, None, axis) is None
    assert clips_on_axis(peds.drop("ped_id"), gestures, axis) is None
    assert clips_on_axis(None, gestures, axis) is None


def test_calculate_plays_atom_from_the_recorded_gestures():
    calc = _calc()
    calc.world = "synthetic_conditions4"
    world_cache["synthetic_conditions4"] = [_square("pharmacy")]
    _wall_cache["synthetic_conditions4"] = None
    times = [0, 1_000_000_000, 2_000_000_000]
    data = pl.DataFrame(
        {"t_ns": times, "x": [1.0] * 3, "y": [1.0] * 3, "yaw": [0.0] * 3}
    )
    peds = pl.DataFrame(
        {
            "t_ns": times,
            "ped_name": ["env_0/alice"] * 3,
            "ped_id": ["1"] * 3,
            "x": [2.0] * 3,
            "y": [2.0] * 3,
        }
    )
    gestures = _gestures([(2_000_000_000, "1", "beckon")])
    conditions = [
        {"op": "eventually", "p": "alice plays beckon"},
        {"op": "always", "p": "not alice plays wave"},
        {"op": "eventually", "p": "bob plays beckon"},
    ]

    episode = _episode(data, conditions=conditions)
    episode.topics = {"pedestrian_arena_peds": peds, "pedestrian_gestures": gestures}
    results = calc.calculate(episode, {})

    assert results["clauses_passed"] == 2
    assert results["clauses_unknown"] == 1
    assert results["condition_success"] is None

    episode.topics = {
        "pedestrian_arena_peds": peds,
        "pedestrian_gestures": gestures.clear(),
    }
    results = calc.calculate(episode, {})

    assert results["condition_success"] is False
    assert results["clauses_failed"] == 1
    assert results["clauses_passed"] == 1
    assert results["clauses_unknown"] == 1

    episode.topics = {}
    results = calc.calculate(episode, {})

    assert results["clauses_unknown"] == 3




def _wardrobe_snapshot():
    return snapshot_frame(
        [
            (
                0,
                "env_0/wardrobe",
                "articulated",
                "openness.door_left",
                "continuous",
                0.1,
            ),
            (
                0,
                "env_0/wardrobe",
                "articulated",
                "openness.door_right",
                "continuous",
                0.0,
            ),
            (0, "env_0/wardrobe", "articulated", "q.door_left", "continuous", 0.2),
            (0, "env_0/wardrobe", "articulated", "state", "discrete", "closed"),
            (
                2_000_000_000,
                "env_0/wardrobe",
                "articulated",
                "openness.door_left",
                "continuous",
                0.9,
            ),
            (2_000_000_000, "env_0/wardrobe", "articulated", "state", "discrete", ""),
        ]
    )




def _fleet_pose_frame(x_m: float) -> pl.DataFrame:
    """Two judge-pose samples of one other robot standing at `x_m`."""
    return pl.DataFrame(
        {
            "t_ns": np.array([0, 1_000_000_000], dtype=np.int64),
            "pos_x": np.array([x_m, x_m], dtype=np.float64),
            "pos_y": np.zeros(2, dtype=np.float64),
            "yaw": np.zeros(2, dtype=np.float64),
        }
    )


