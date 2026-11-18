import numpy as np
import pytest
from robot_params_factory import make_robot_params

pytest.importorskip("polars")
pytest.importorskip("shapely")

import polars as pl

from arena_evaluation.processing.metrics.common.zones import (
    DoorGeometry as _DoorGeometry,
    ZoneGeometry as _ZoneGeometry,
    extract_zone_geometry as _extract_zone_geometry,
)
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    ComplianceMetricsCalculator,
    _door_open_series,
    _speed_zone_metrics,
)
from arena_evaluation.processing.metrics.base import NULL_REASON_SUFFIX
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from flat_peds import flat_peds
from snapshot_rows import snapshot_frame

SEC = 1_000_000_000
LEFT = 0.5
RIGHT = -0.5


def _calc(convention: str = "right") -> ComplianceMetricsCalculator:
    calc = ComplianceMetricsCalculator(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )
    calc.passing_convention = convention
    return calc


def _robot(xs: list[float]) -> pl.DataFrame:
    """The robot's ground-truth track on a 1 Hz clock, straight along x."""
    return pl.DataFrame(
        {
            "t_ns": [i * SEC for i in range(len(xs))],
            "x": [float(x) for x in xs],
            "y": [0.0] * len(xs),
            "yaw": [0.0] * len(xs),
        }
    )


def _peds(
    positions: list[list[list[float]]], twists: list[list[list[float]]] | None = None
) -> pl.DataFrame:
    """Per 1 Hz sample the pedestrians as [x, y] pairs, with optional [vx, vy, vz] twists per pedestrian."""
    frames = []
    for i, sample in enumerate(positions):
        pedestrians = []
        for j, (x, y) in enumerate(sample):
            ped = {"ped_id": f"Pedestrian_{j}", "x": x, "y": y}
            if twists is not None:
                ped["vx"], ped["vy"] = twists[i][j][0], twists[i][j][1]
            pedestrians.append(ped)
        frames.append((i * SEC, pedestrians))
    return flat_peds(frames).with_columns(pl.lit(0.4).alias("radius_m"))


def _episode(
    peds: pl.DataFrame,
    robot: pl.DataFrame | None,
    semantic_snapshot: pl.DataFrame | None = None,
) -> AlignedEpisodeBundle:
    topics = {"pedestrian_arena_peds": peds}
    data = robot if robot is not None else pl.DataFrame()
    if robot is not None:
        topics["robot_tracks"] = pl.DataFrame(
            {
                "t_ns": robot["t_ns"],
                "x": robot["x"],
                "y": robot["y"],
                "yaw": robot["yaw"],
                "pose_source": ["tf_gt"] * len(robot),
            }
        )
    start_pos = (
        [float(robot["x"][0]), float(robot["y"][0]), 0.0] if robot is not None else []
    )
    return AlignedEpisodeBundle(
        1,
        data,
        start_pos,
        [],
        peds=peds,
        topics=topics,
        semantic_snapshot=semantic_snapshot,
    )


def _head_on(side: float, closed: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    positions = [
        [[4.0, side]],
        [[3.0, side], [3.5, -side]],
        [[2.5, side]],
        [[1.5, side]],
    ]
    twists = [[[-1.0, 0.0]], [[-1.0, 0.0], [-1.0, 0.0]], [[-1.0, 0.0]], [[-1.0, 0.0]]]
    if closed:
        positions.append([[1.0, side]])
        twists.append([[0.0, 0.0]])
    return _peds(positions, twists), _robot(list(range(len(positions))))


@pytest.mark.parametrize(
    ("side", "convention", "closed", "expected"),
    [
        (LEFT, "right", True, 1.0),
        (RIGHT, "right", True, 0.0),
        (RIGHT, "left", True, 1.0),
        (LEFT, "left", True, 0.0),
        (LEFT, "right", False, 1.0),
        (RIGHT, "right", False, 0.0),
        (RIGHT, "left", False, 1.0),
        (LEFT, "left", False, 0.0),
    ],
)
def test_head_on_encounter_is_judged_by_the_side_at_closest_approach(
    side: float, convention: str, closed: bool, expected: float
):
    peds, robot = _head_on(side, closed)

    assert (
        _calc(convention)._compute_passing_compliance(_episode(peds, robot)) == expected
    )


def test_two_encounters_on_opposite_sides_score_one_half():
    positions = [
        [[3.0, LEFT]],
        [[2.0, LEFT]],
        [[2.0, LEFT]],
        [[5.0, RIGHT]],
        [[4.0, RIGHT]],
        [[4.0, RIGHT]],
    ]
    twists = [
        [[-1.0, 0.0]],
        [[-1.0, 0.0]],
        [[0.0, 0.0]],
        [[-1.0, 0.0]],
        [[-1.0, 0.0]],
        [[0.0, 0.0]],
    ]

    assert (
        _calc()._compute_passing_compliance(
            _episode(_peds(positions, twists), _robot([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        )
        == 0.5
    )


def test_passing_compliance_needs_a_robot_pose():
    assert (
        _calc()._compute_passing_compliance(_episode(_peds([[[1.0, 0.0]]]), None))
        is None
    )


def test_a_missing_pedestrian_radius_leaves_the_zone_metrics_intact():
    from arena_simulation_setup.shared.semantics import SemanticCfg
    from arena_simulation_setup.tree.World import Level, Zone
    from arena_simulation_setup.utils.geometry import Position

    zone = Zone(
        name="lobby",
        polygon=[
            Position(-1.0, -1.0),
            Position(-1.0, 5.0),
            Position(5.0, 5.0),
            Position(5.0, -1.0),
        ],
        semantics=[SemanticCfg(role="predicate", name="quiet", value=True)],
    )
    calc = _calc()
    calc.world = "synthetic_passing_radius"
    calc._world_cache["synthetic_passing_radius"] = (
        _extract_zone_geometry(Level(zones=[zone])),
        [],
    )
    peds, robot = _head_on(LEFT, True)
    peds = peds.with_columns(pl.lit(None, dtype=pl.Float64).alias("radius_m"))
    robot = robot.with_columns(pl.lit(0.0).alias("speed_mps"))

    results = calc.calculate(
        _episode(
            peds,
            robot,
            semantic_snapshot=_snapshot("env_0/door_1", "open"),
        ),
        {},
    )

    assert results["passing_rule_compliance"] is None
    assert results[f"passing_rule_compliance{NULL_REASON_SUFFIX}"] == "not_recorded"
    assert results["quiet_zone_dwell_seconds"] == pytest.approx(4.0)


def test_speed_violations_ignore_samples_outside_every_zone():
    zones = [
        _ZoneGeometry(
            name="corridor", polygon=None, max_speed=1.0, quiet=False, restricted=False
        )
    ]

    violations, seconds = _speed_zone_metrics(
        np.array([-1, 0, -1]), np.array([5.0, 2.0, 5.0]), np.array([1.0, 1.0]), zones
    )

    assert (violations, seconds) == (1, pytest.approx(1.0))


def test_door_series_is_empty_without_door_state_events():
    events = pl.DataFrame(
        {
            "t_ns": [0],
            "entity": ["env_0/elevator_1"],
            "kind": ["elevator"],
            "field": ["state"],
            "current": ["open"],
        }
    )

    assert _door_open_series(events) == {}


def _snapshot(entity: str, state: str) -> pl.DataFrame:
    return snapshot_frame([(0, entity, "door", "state", "discrete", state)])


def _door_episode(
    xs: list[float], ys: list[float], vel: list[float], snapshot: pl.DataFrame | None
) -> AlignedEpisodeBundle:
    n = len(xs)
    data = pl.DataFrame(
        {
            "t_ns": [i * SEC for i in range(n)],
            "x": xs,
            "y": ys,
            "yaw": [0.0] * n,
            "speed_mps": vel,
        }
    )
    return AlignedEpisodeBundle(
        1, data, [xs[0], ys[0], 0.0], [], semantic_snapshot=snapshot
    )


def _blocking(episode: AlignedEpisodeBundle, doors: list[_DoorGeometry]) -> float:
    calc = _calc()
    pos_x, pos_y, _, _, _, _ = calc.resolve_robot_pose(episode)
    return calc._doorway_blocking_time(
        episode,
        pos_x,
        pos_y,
        np.abs(episode.data["speed_mps"].to_numpy()),
        episode.data["t_ns"].to_numpy(),
        doors,
    )


def test_doorway_blocking_needs_two_samples():
    episode = _door_episode([0.0], [0.0], [0.0], _snapshot("env_0/door_1", "open"))

    assert _blocking(episode, [_DoorGeometry("door_1", 0.0, 0.0, 1.0)]) is None


