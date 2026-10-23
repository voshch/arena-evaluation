"""Trajectory plot: teleport breaks, door geometry and states, faceted time-synced GIF."""

import math
import pathlib

import numpy as np
import polars as pl
import pytest
from PIL import Image

from arena_evaluation.presentation.plot_types.trajectory import GIF_FPS, GIF_FRAMES, TrajectoryRenderer, _break_at_teleports, _door_color, _door_leaf, _first_moving_index
from arena_evaluation.processing.metrics.ecological.mechanism_common import DoorGeom, ElevatorGeom, MechanismWorld, mechanism_world_from_description
from arena_evaluation.processing.metrics.ecological.mechanism_states import MechanismStatesCalculator
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, PlotSpec, RobotParams

T = 1_000_000_000
MAP = "synthetic_trajectory_world"


def _snapshot(rows):
    cols = {k: [] for k in ("time_ns", "entity", "kind", "field", "field_kind", "value_str", "value_num", "value_bool", "value_list")}
    for t, entity, kind, field, field_kind, value in rows:
        cols["time_ns"].append(int(t * T))
        cols["entity"].append(entity)
        cols["kind"].append(kind)
        cols["field"].append(field)
        cols["field_kind"].append(field_kind)
        cols["value_str"].append(value if field_kind == "discrete" else None)
        cols["value_num"].append(float(value) if field_kind == "continuous" else None)
        cols["value_bool"].append(bool(value) if field_kind == "predicate" else None)
        cols["value_list"].append(None)
    schema = {"time_ns": pl.Int64, "entity": pl.Utf8, "kind": pl.Utf8, "field": pl.Utf8, "field_kind": pl.Utf8, "value_str": pl.Utf8, "value_num": pl.Float64, "value_bool": pl.Boolean, "value_list": pl.List(pl.Utf8)}
    return pl.DataFrame(cols, schema=schema)


def test_teleport_breaks_fast_jumps_only():
    x = np.array([0.0, 0.5, 1.0, 20.0, 20.5, 21.0])
    y = np.zeros(6)
    t = np.arange(6) * T
    bx, by = _break_at_teleports(x, y, t)
    assert np.isnan(bx[3]) and np.isnan(by[3])
    assert np.isnan(bx).sum() == 1
    slow_t = np.array([0, 1, 2, 30, 31, 32]) * T
    assert not np.isnan(_break_at_teleports(x, y, slow_t)[0]).any()
    assert np.isnan(_break_at_teleports(x, y, None)[0]).sum() == 1


def test_first_moving_index_skips_spawn_jump():
    x = np.array([50.0, 0.0, 0.5, 1.0, 1.5])
    y = np.zeros(5)
    assert _first_moving_index(x, y, np.arange(5) * T) == 1


def test_door_leaf_sliding_retracts_towards_start():
    door = DoorGeom("d", "0", (8.0, 0.0), (8.0, 2.0), 1.0)
    xs, ys = _door_leaf(door, 0.0)
    assert ys == [0.0, 2.0]
    xs, ys = _door_leaf(door, 0.75)
    assert ys == pytest.approx([0.0, 0.5])


@pytest.mark.parametrize(("swing", "open_end"), [(1, (6.0, 0.0)), (-1, (10.0, 0.0))])
def test_door_leaf_hinged_swings_to_the_configured_side(swing, open_end):
    door = DoorGeom("d", "0", (8.0, 0.0), (8.0, 2.0), 1.0, kind="hinged", swing=swing)
    (_, _), end = door.leaf(1.0)
    assert end == pytest.approx(open_end)
    _, half = door.leaf(0.5)
    assert half == pytest.approx((8.0 - swing * 2.0 * math.sin(math.pi / 4), 2.0 * math.cos(math.pi / 4)))


def test_elevator_leaf_uses_the_cabin_door_segment():
    cabin = ElevatorGeom("e", "0", [(2, 10), (4, 10), (4, 12), (2, 12)], (2.0, 10.0), (4.0, 10.0), 3.0)
    xs, ys = _door_leaf(cabin, 0.5)
    assert xs == pytest.approx([2.0, 3.0]) and ys == [10.0, 10.0]


def test_door_color_states():
    assert _door_color(0.0, False) != _door_color(0.5, False)
    assert _door_color(0.0, True) == _door_color(1.0, True)


def test_world_description_carries_kind_and_swing():
    from arena_simulation_setup.shared.world import Door
    from arena_simulation_setup.tree.World import Level, World, Zone
    from arena_simulation_setup.utils.geometry import Position

    zone = Zone(name="hall", polygon=[Position(0, 0, 0), Position(4, 0, 0), Position(4, 4, 0)])
    level = Level(id="0", zones=[zone], doors=[Door(name="h", start=Position(0, 0, 0), end=Position(0, 2, 0), kind="hinged", swing_direction=-1)])
    mech = mechanism_world_from_description(World(levels={"0": level}), {"0": (0.0, 0.0)})
    door = mech.door("env_0/h/0")
    assert (door.kind, door.swing) == ("hinged", -1)
    assert door.leaf(1.0)[1] == pytest.approx((2.0, 0.0))


def test_mechanism_states_sampled_on_the_time_grid():
    snapshot = _snapshot(
        [
            (0, "env_0/door_a/0", "door", "progress", "continuous", 0.0),
            (0, "env_0/door_a/0", "door", "locked", "predicate", False),
            (2, "env_0/door_a/0", "door", "progress", "continuous", 1.0),
            (3, "env_0/door_a/0", "door", "locked", "predicate", True),
            (0, "env_0/elev_a/0", "elevator", "cabin_door_progress", "continuous", 0.0),
            (4, "env_0/elev_a/0", "elevator", "cabin_door_progress", "continuous", 0.5),
            (0, "env_0/plate/0", "pressure_plate", "pressed", "predicate", False),
        ]
    )
    data = pl.DataFrame({"time_ns": [i * T for i in range(6)], "pos_x": [0.0] * 6, "pos_y": [0.0] * 6, "yaw": [0.0] * 6})
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0, 0, 0], goal_pos=[], robot_name="env_0_jackal", semantic_snapshot=snapshot)
    out = MechanismStatesCalculator(RobotParams()).calculate(episode, {"time": [i * T for i in range(6)]})["mechanism_states"]
    by_name = {s["entity"]: s for s in out}
    assert set(by_name) == {"door_a", "elev_a"}
    assert by_name["door_a"]["kind"] == "door" and by_name["door_a"]["level"] == "0"
    assert by_name["door_a"]["progress"] == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert by_name["door_a"]["locked"] == [False, False, False, True, True, True]
    assert by_name["elev_a"]["progress"] == [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
    assert by_name["elev_a"]["locked"] == [False] * 6
    assert MechanismStatesCalculator(RobotParams()).calculate(episode, {})["mechanism_states"] is None


def _episode_row(planner: str, episode: int, xs: list[float], result: str, states: list[dict] | None) -> dict:
    n = len(xs)
    return {
        "planner": planner,
        "local_planner": planner,
        "episode": episode,
        "map": MAP,
        "stage": "s1",
        "result": result,
        "success": result == "GOAL_REACHED",
        "path": [[x, 1.0, 0.0] for x in xs],
        "time": [i * T for i in range(n)],
        "start": [xs[0], 1.0, 0.0],
        "goal": [xs[-1], 1.0, 0.0],
        "pedestrian_path": [[[x + 0.5, 2.0, 0.0] for x in xs]],
        "mechanism_states": states,
    }


def _frames(path: pathlib.Path) -> tuple[list[Image.Image], float]:
    """Decoded frames and the total play time in ms (pillow merges identical consecutive frames)."""
    im = Image.open(path)
    out, total = [], 0.0
    for i in range(im.n_frames):
        im.seek(i)
        out.append(im.convert("RGB").copy())
        total += im.info["duration"]
    return out, total


def test_faceted_gif_animates_each_planner_with_its_own_doors(tmp_path):
    TrajectoryRenderer._worlds[MAP] = MechanismWorld(doors=[DoorGeom("door_a", "0", (4.0, 0.0), (4.0, 2.0), 1.0)])
    xs = [float(i) * 0.5 for i in range(16)]
    open_at_2 = [{"entity": "door_a", "level": "0", "kind": "door", "progress": [0.0, 0.0] + [1.0] * 14, "locked": [False] * 16}]
    stays_closed = [{"entity": "door_a", "level": "0", "kind": "door", "progress": [0.0] * 16, "locked": [True] * 16}]
    df = pl.DataFrame([_episode_row("a", 1, xs, "GOAL_REACHED", open_at_2), _episode_row("b", 1, xs[:6], "TIMEOUT", stays_closed)])
    spec = PlotSpec(id="t", type="trajectory", title="Traces", data_key="path", differentiate="local_planner", options={"facet": "local_planner", "overlay_pedestrians": True, "show_map": False})
    renderer = TrajectoryRenderer(spec)
    renderer.generate_gifs = True
    out = tmp_path / "traj.png"
    renderer.render_seaborn(df, out)

    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    frames, total_ms = _frames(out.with_suffix(".gif"))
    assert total_ms == pytest.approx((GIF_FRAMES + 1) * 1000 / GIF_FPS, rel=0.02)
    assert len(frames) > 10
    assert frames[0].tobytes() != frames[-1].tobytes()
    assert frames[len(frames) // 2].tobytes() != frames[-1].tobytes()


def test_single_panel_without_states_keeps_static_doors(tmp_path):
    TrajectoryRenderer._worlds[MAP] = MechanismWorld(doors=[DoorGeom("door_a", "0", (4.0, 0.0), (4.0, 2.0), 1.0)])
    xs = [float(i) for i in range(8)]
    df = pl.DataFrame([_episode_row("a", 1, xs, "GOAL_REACHED", None), _episode_row("a", 2, xs, "COLLISION", None)])
    spec = PlotSpec(id="t", type="trajectory", title="Traces", data_key="path", options={"show_map": False})
    renderer = TrajectoryRenderer(spec)
    renderer.generate_gifs = True
    out = tmp_path / "single.png"
    renderer.render_seaborn(df, out)
    assert out.exists()
    frames, total_ms = _frames(out.with_suffix(".gif"))
    assert total_ms == pytest.approx((GIF_FRAMES + 1) * 1000 / GIF_FPS, rel=0.02)
    assert frames[0].tobytes() != frames[-1].tobytes()


def test_describe_episode_from_a_single_directory(tmp_path):
    ep_dir = tmp_path / "bench" / "episodes" / "episode_007"
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode_007.yaml").write_text("benchmark_id: bench\nplanner: dwb\nmap: m\nstage: s\nepisode_id: 7\nrecording_started_at: '2026-10-06T00:00:00+00:00'\npython_version: 3.12.3\nros_distro: jazzy\n")
    ep = FolderManager.describe_episode(ep_dir)
    assert (ep.benchmark_id, ep.episode_id, ep.planner, ep.stage, ep.map) == ("bench", 7, "dwb", "s", "m")
    assert FolderManager.describe_episode(tmp_path / "bench") is None
    (tmp_path / "bench" / "episodes" / "episode_008").mkdir()
    assert FolderManager.describe_episode(tmp_path / "bench" / "episodes" / "episode_008") is None
