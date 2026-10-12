import logging
import math
import pathlib
import uuid

import numpy as np
import polars as pl
import pytest
import yaml
from PIL import Image

from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations
from arena_evaluation.processing.metrics.ecological.acoustic_exposure import AcousticExposureCalculator
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, EpisodeDescriptor, RobotParams


def test_acoustics_free_space():
    # 10x10 empty grid
    grid = np.zeros((10, 10), dtype=np.uint8)
    resolution = 1.0  # 1 meter per pixel for simplicity

    # Start at (0, 0)
    sx, sy = 0.0, 0.0

    # Target at (3, 4) -> true Euclidean distance = sqrt(3^2+4^2) = 5.0 m
    # Theta* recovers exact Euclidean geometry in open space (no staircase bias).
    tx = np.array([3.0], dtype=np.float32)
    ty = np.array([4.0], dtype=np.float32)

    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

    expected_dist = np.sqrt(3.0**2 + 4.0**2)  # = 5.0 m (true Euclidean, not staircase)
    expected = 20.0 * np.log10(expected_dist)
    assert np.isclose(att[0], expected, atol=0.1)


def test_acoustics_one_wall():
    # 10x10 grid with a vertical wall at x=2
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 2] = 255  # wall

    resolution = 1.0
    sx, sy = 0.0, 0.0

    tx = np.array([3.0], dtype=np.float32)
    ty = np.array([4.0], dtype=np.float32)

    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

    # The shortest path must cross the wall once.
    # With Theta* the path through the wall uses true Euclidean distance
    # (sqrt(3^2+4^2) = 5.0 m) rather than the old staircase distance.
    expected_dist = np.sqrt(3.0**2 + 4.0**2)  # = 5.0 m
    expected = 20.0 * np.log10(expected_dist) + 47.0
    assert np.isclose(att[0], expected, atol=0.5)  # slightly wider: wall forces grid step


def test_acoustics_pruning_dominance():
    # Grid where going around the wall is cheaper than going through it
    grid = np.zeros((10, 10), dtype=np.uint8)
    # Wall from y=0 to y=6 at x=2
    grid[0:7, 2] = 255

    resolution = 1.0
    sx, sy = 0.0, 3.0
    tx = np.array([4.0], dtype=np.float32)
    ty = np.array([3.0], dtype=np.float32)

    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

    # Path 1: through the wall: distance = 4m, walls = 1. Cost = 20log10(5) + 47 = 60.97
    # Path 2: around the wall: dist is roughly 4 + 4 + 4 = 12m. Cost = 20log10(13) = 22.2
    # So going around is much cheaper! The solver should return the path around.

    # Manually calculate roughly the around distance: (0,3) -> (2,7) -> (4,3)
    # dist = sqrt(2^2 + 4^2) * 2 = 2 * sqrt(20) = 8.94

    # The actual solver might find a slightly different path on the 8-connected grid
    # But it must be < 40 dB
    assert att[0] < 40.0


def _field(grid, resolution, sx, sy, mic_distance, wall_tl=47.0):
    h, w = grid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tx = np.ascontiguousarray(xx.ravel().astype(np.float32))
    ty = np.ascontiguousarray(yy.ravel().astype(np.float32))
    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=wall_tl, mic_distance=mic_distance)
    return att.reshape(h, w)


def test_receiver_inside_mic_distance_costs_the_floor():
    grid = np.zeros((40, 40), dtype=np.uint8)
    tx = np.array([25.0, 23.0, 20.0], dtype=np.float32)
    ty = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    att = compute_attenuations(grid, 0.1, 20.0, 20.0, tx, ty, wall_tl=47.0, mic_distance=2.0)
    assert np.allclose(att, 20.0 * np.log10(2.0), atol=1e-4)


def test_wall_inside_mic_distance_adds_only_its_tl():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[:, 20] = 1
    tx = np.array([23.0], dtype=np.float32)
    ty = np.array([20.0], dtype=np.float32)
    att = compute_attenuations(grid, 0.1, 17.0, 20.0, tx, ty, wall_tl=47.0, mic_distance=1.0)
    assert np.isclose(att[0], 20.0 * np.log10(1.0) + 47.0, atol=1e-4)


def test_tied_costs_inside_floor_keep_shortest_paths_beyond_it():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[18:23, 21] = 1
    floored = _field(grid, 0.1, 20.0, 20.0, mic_distance=1.0, wall_tl=1000.0)
    reference = _field(grid, 0.1, 20.0, 20.0, mic_distance=1e-3, wall_tl=1000.0)
    beyond = reference > 20.0 * np.log10(1.05)
    beyond &= reference < 100.0
    assert beyond.sum() > 1000
    assert np.abs(floored - reference)[beyond].max() < 0.25


SECOND_NS = 1_000_000_000
NULLS = dict.fromkeys(AcousticExposureCalculator.output_keys())


def _att(distance_m: float) -> float:
    return 20.0 * math.log10(max(distance_m, 1.0))


def _leq(values: list[float]) -> float:
    return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in values) / len(values))


def _white(size: int = 50) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def _walled(size: int = 50, column: int = 25) -> np.ndarray:
    pixels = _white(size)
    pixels[:, column] = 0
    return pixels


@pytest.fixture
def calc(tmp_path: pathlib.Path) -> AcousticExposureCalculator:
    calculator = AcousticExposureCalculator(RobotParams())
    calculator.maps_cache = tmp_path / "maps_cache"
    calculator.maps_cache.mkdir()
    return calculator


@pytest.fixture
def cached_map(calc: AcousticExposureCalculator):
    cache = calc.maps_cache

    def make(pixels: np.ndarray | None = None, *, resolution: float = 0.1, origin: tuple[float, float, float] = (0.0, 0.0, 0.0), png_bytes: bytes | None = None, with_png_path: bool = True) -> str:
        name = f"acx-{uuid.uuid4().hex}"
        png = cache / f"{name}.png"
        meta_path = cache / f"{name}.yaml"
        if png_bytes is not None:
            png.write_bytes(png_bytes)
        else:
            Image.fromarray(_white() if pixels is None else pixels).save(png)
        meta = {"resolution": resolution, "origin": list(origin)}
        if with_png_path:
            meta["png_path"] = str(png)
        meta_path.write_text(yaml.dump(meta))
        return name

    return make


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def module_log():
    logger = logging.getLogger("arena_evaluation.processing.metrics.ecological.acoustic_exposure")
    handler = _Records()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.messages
    logger.removeHandler(handler)
    logger.setLevel(level)


def _door_run(tmp_path: pathlib.Path, map_name: str) -> tuple[FolderManager, EpisodeDescriptor]:
    world_dir = tmp_path / "bench" / "worlds" / map_name
    world_dir.mkdir(parents=True)
    door = {"name": "front_door", "span": [[2.5, 2.0], [2.5, 3.0]], "width": 0.1}
    (world_dir / "world.yaml").write_text(yaml.dump({"format": "arena-world/2", "levels": [{"id": "0", "doors": [door]}]}))
    run = EpisodeDescriptor(episode_dir=str(tmp_path / "bench" / "ep"), benchmark_id="bench", episode_id=7, planner="p", stage="s", map=map_name)
    return FolderManager(tmp_path), run


def _semantic(rows: list[tuple[int, str, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [r[0] for r in rows],
            "kind": ["door"] * len(rows),
            "entity": [r[1] for r in rows],
            "field": ["open"] * len(rows),
            "value_bool": [r[2] for r in rows],
            "value_str": [None] * len(rows),
            "value_num": [None] * len(rows),
        },
        schema={"time_ns": pl.Int64, "kind": pl.Utf8, "entity": pl.Utf8, "field": pl.Utf8, "value_bool": pl.Boolean, "value_str": pl.Utf8, "value_num": pl.Float64},
    )


def _frames(robot: list[tuple[float | None, float | None]], peds: list[list[tuple[float, float]]], *, source: list[float | None] | None = None, times: list[int] | None = None) -> pl.DataFrame:
    columns = {
        "time_ns": pl.Series(times if times is not None else [i * SECOND_NS for i in range(len(robot))], dtype=pl.Int64),
        "pos_x_gt": pl.Series([r[0] for r in robot], dtype=pl.Float64),
        "pos_y_gt": pl.Series([r[1] for r in robot], dtype=pl.Float64),
        "peds_positions": pl.Series([[list(p) for p in frame] for frame in peds], dtype=pl.List(pl.List(pl.Float64))),
    }
    if source is not None:
        columns["total_level_af_dba"] = pl.Series(source, dtype=pl.Float64)
    return pl.DataFrame(columns)


def _episode(data: pl.DataFrame | None, map_name: str | None, **kwargs) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(episode_id=7, data=data, start_pos=[0.0, 0.0], goal_pos=[0.0, 0.0], map=map_name, **kwargs)


def _calc() -> AcousticExposureCalculator:
    return AcousticExposureCalculator(RobotParams())


def test_parse_reads_dict_items_and_skips_incomplete_ones():
    row = [{"x": 1, "y": 2}, {"x": 3}, "junk", {"x": 4.5, "y": -1}]
    assert AcousticExposureCalculator._parse_pedestrian_positions(row) == [(1.0, 2.0), (4.5, -1.0)]


def test_parse_reads_nested_pairs_and_drops_short_or_nan_entries():
    row = [[1.0, 2.0, 0.0], [3.0], [math.nan, 1.0], [1.0, math.nan], [5.0, 6.0]]
    assert AcousticExposureCalculator._parse_pedestrian_positions(row) == [(1.0, 2.0), (5.0, 6.0)]


def test_parse_reads_flat_triples_and_drops_nan_and_dangling_values():
    row = [1.0, 2.0, 0.0, math.nan, 1.0, 0.0, 3.0, math.nan, 0.0, 7.0, 8.0, 0.0, 9.0]
    assert AcousticExposureCalculator._parse_pedestrian_positions(row) == [(1.0, 2.0), (7.0, 8.0)]


def test_parse_decodes_json_strings():
    assert AcousticExposureCalculator._parse_pedestrian_positions('[{"x": 1, "y": 2}]') == [(1.0, 2.0)]


def test_parse_accepts_numpy_arrays():
    assert AcousticExposureCalculator._parse_pedestrian_positions(np.array([[1.0, 2.0], [3.0, 4.0]])) == [(1.0, 2.0), (3.0, 4.0)]
    assert AcousticExposureCalculator._parse_pedestrian_positions(np.array([1.0, 2.0, 0.0])) == [(1.0, 2.0)]


@pytest.mark.parametrize("row", ["not json", None, 3.0, [], np.array([])], ids=["bad_json", "none", "scalar", "empty_list", "empty_array"])
def test_parse_returns_nothing_for_unusable_rows(row):
    assert AcousticExposureCalculator._parse_pedestrian_positions(row) == []


def test_map_occupancy_marks_dark_pixels_and_flips_rows(calc, cached_map):
    pixels = np.full((3, 4), 255, dtype=np.uint8)
    pixels[0, 0] = 0
    pixels[1, 2] = 199
    pixels[2, 1] = 200
    name = cached_map(pixels, resolution=0.25, origin=(-1.0, 2.0, 0.0))
    grid, resolution, origin = calc._get_map_occupancy(name)
    assert grid.dtype == np.uint8
    assert grid.flags["C_CONTIGUOUS"]
    assert grid.tolist() == [[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    assert resolution == 0.25
    assert list(origin) == [-1.0, 2.0, 0.0]


def test_map_occupancy_is_none_for_unknown_map():
    assert _calc()._get_map_occupancy("unknown") is None


def test_map_occupancy_is_none_without_png_path(calc, cached_map):
    assert calc._get_map_occupancy(cached_map(with_png_path=False)) is None


def test_map_occupancy_is_none_for_unreadable_image(calc, cached_map, module_log):
    assert calc._get_map_occupancy(cached_map(png_bytes=b"not a png")) is None
    assert any(m.startswith("Failed to load map image for acoustics") for m in module_log)


def test_reference_run_yields_nulls(calc, cached_map):
    run = EpisodeDescriptor(episode_dir="ep", benchmark_id="bench", episode_id=7, planner="p", stage="s", is_reference=True)
    episode = _episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), cached_map(), run=run)
    assert calc.calculate(episode, {}) == NULLS


@pytest.mark.parametrize("data", [None, pl.DataFrame()], ids=["none", "empty"])
def test_missing_episode_data_yields_nulls(data, calc, cached_map):
    assert calc.calculate(_episode(data, cached_map()), {}) == NULLS


def test_missing_map_name_yields_nulls():
    assert _calc().calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), None), {}) == NULLS


def test_unloadable_map_yields_nulls():
    assert _calc().calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), "unknown"), {}) == NULLS


@pytest.mark.parametrize("dropped", ["pos_x_gt", "pos_y_gt", "peds_positions"])
def test_missing_required_column_yields_nulls(dropped, calc, cached_map):
    data = _frames([(1.0, 1.0)], [[(4.0, 1.0)]]).drop(dropped)
    assert calc.calculate(_episode(data, cached_map()), {}) == NULLS


def test_world_fallback_supplies_the_map(calc, cached_map):
    calc.world = cached_map()
    result = calc.calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), None), {})
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_prior_map_overrides_episode_map(calc, cached_map):
    result = calc.calculate(_episode(_frames([(1.0, 1.0)], [[(4.0, 1.0)]]), "unknown"), {"map": cached_map()})
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)


def test_exposure_tracks_robot_and_pedestrian_motion(calc, cached_map):
    robot = [(1.0, 1.0), (None, None), (None, None), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]
    peds = [
        [(4.0, 1.0)],
        [],
        [(4.0, 1.3)],
        [(4.0, 1.0)],
        [(4.0, 1.0), (1.0, 4.0)],
        [(4.0, 1.0), (1.0, 3.4)],
        [(4.0, 1.0), (1.0, 3.4)],
    ]
    source = [None, 60.0, None, 60.0, 60.0, 60.0, 60.0]
    result = calc.calculate(_episode(_frames(robot, peds, source=source), cached_map()), {})

    a3, a10, a2, a14 = _att(3.0), _att(math.sqrt(10.0)), _att(2.0), _att(1.4)
    attenuation = [[a3], [], [a3], [a10], [a10, a2], [a10, a14], [a10, a14]]
    exposure = [[42.0 - a3], [], [60.0 - a3], [60.0 - a10], [60.0 - a10, 60.0 - a2], [60.0 - a10, 60.0 - a14], [60.0 - a10, 60.0 - a14]]
    for got, want in zip(result["timeseries_acoustic_attenuation_db"], attenuation, strict=True):
        assert got == pytest.approx(want, abs=1e-3)
    for got, want in zip(result["timeseries_acoustic_exposure_dba"], exposure, strict=True):
        assert got == pytest.approx(want, abs=1e-3)
    flat = [v for frame in exposure for v in frame]
    assert result["ped_max_exposure_dba"] == pytest.approx(60.0 - a14, abs=1e-3)
    assert result["ped_leq_exposure_dba"] == pytest.approx(_leq(flat), abs=1e-3)
    assert result["ped_max_startle_factor"] == pytest.approx(a2 - a14, abs=1e-3)
    worst = result["worst_case_acoustic_frame"]
    assert worst["robot_x"] == 1.0
    assert worst["robot_y"] == 2.0
    assert worst["source_dba"] == 60.0
    assert worst["pedestrians"] == [[4.0, 1.0], [1.0, 3.4]]
    assert worst["door_states"] == {}


def test_leading_null_pose_and_empty_first_frame_solve_from_origin(calc, cached_map):
    robot = [(None, None), (None, None), (None, None)]
    peds = [[], [(3.0, 4.0)], [(3.0, 4.0)]]
    result = calc.calculate(_episode(_frames(robot, peds), cached_map()), {})
    expected = 42.0 - _att(5.0)
    assert result["timeseries_acoustic_exposure_dba"][0] == []
    assert result["timeseries_acoustic_exposure_dba"][1:] == [pytest.approx([expected], abs=1e-3)] * 2
    assert result["ped_max_exposure_dba"] == pytest.approx(expected, abs=1e-3)
    assert result["ped_leq_exposure_dba"] == pytest.approx(expected, abs=1e-3)
    assert result["ped_max_startle_factor"] == 0.0
    assert result["worst_case_acoustic_frame"]["robot_x"] == 0.0
    assert result["worst_case_acoustic_frame"]["robot_y"] == 0.0
    assert result["worst_case_acoustic_frame"]["pedestrians"] == [[3.0, 4.0]]


def test_repeated_timestamp_contributes_no_startle(calc, cached_map):
    data = _frames([(1.0, 2.0), (1.0, 1.0)], [[(4.0, 1.0)], [(4.0, 1.0)]], times=[0, 0])
    result = calc.calculate(_episode(data, cached_map()), {})
    assert result["timeseries_acoustic_exposure_dba"] == [pytest.approx([42.0 - _att(math.sqrt(10.0))], abs=1e-3), pytest.approx([42.0 - _att(3.0)], abs=1e-3)]
    assert result["ped_max_startle_factor"] == 0.0


def test_unreachable_pedestrians_are_dropped(calc, cached_map):
    data = _frames([(1.0, 1.0), (1.0, 1.0)], [[(4.0, 1.0), (9.0, 9.0)], [(9.0, 9.0)]])
    result = calc.calculate(_episode(data, cached_map()), {})
    assert result["timeseries_acoustic_attenuation_db"] == [pytest.approx([_att(3.0)], abs=1e-3), []]
    assert result["timeseries_acoustic_exposure_dba"] == [pytest.approx([42.0 - _att(3.0)], abs=1e-3), []]
    assert result["ped_max_exposure_dba"] == pytest.approx(42.0 - _att(3.0), abs=1e-3)
    assert result["worst_case_acoustic_frame"]["pedestrians"] == [[4.0, 1.0], [9.0, 9.0]]


def test_robot_off_map_yields_only_empty_timeseries(calc, cached_map):
    data = _frames([(9.0, 9.0), (9.0, 9.0)], [[(4.0, 1.0)], []])
    result = calc.calculate(_episode(data, cached_map()), {})
    assert result == {
        "ped_max_exposure_dba": None,
        "ped_leq_exposure_dba": None,
        "ped_max_startle_factor": None,
        "timeseries_acoustic_exposure_dba": [[], []],
        "timeseries_acoustic_attenuation_db": [[], []],
        "worst_case_acoustic_frame": None,
    }


def test_static_scene_logs_progress_every_fifty_evaluations(calc, cached_map, module_log):
    data = _frames([(1.0, 1.0)] * 50, [[(4.0, 1.0)]] * 50)
    result = calc.calculate(_episode(data, cached_map()), {})
    assert [m for m in module_log if "Evaluated" in m] == ["AcousticExposureCalculator: Evaluated 50 frames... (50/50 total frames processed)"]
    assert "AcousticExposureCalculator: Finished episode 7 with 2 unique solver evaluations." in module_log
    assert result["timeseries_acoustic_exposure_dba"] == [pytest.approx([42.0 - _att(3.0)], abs=1e-3)] * 50
    assert result["ped_max_startle_factor"] == 0.0


def test_door_opening_switches_attenuation_and_marks_worst_frame_open(tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, run = _door_run(tmp_path, name)
    data = _frames([(1.0, 2.5)] * 3, [[(4.0, 2.5)]] * 3, source=[60.0] * 3)
    semantic = _semantic([(SECOND_NS, "env_0/front_door/0", True)])
    result = calc.calculate(_episode(data, name, run=run, folder_manager=folder_manager, semantic_snapshot=semantic), {})
    closed, opened = _att(3.0) + 25.0, _att(3.0)
    assert result["timeseries_acoustic_attenuation_db"] == [pytest.approx([closed], abs=1e-3), pytest.approx([opened], abs=1e-3), pytest.approx([opened], abs=1e-3)]
    assert result["ped_max_exposure_dba"] == pytest.approx(60.0 - opened, abs=1e-3)
    assert result["ped_leq_exposure_dba"] == pytest.approx(_leq([60.0 - closed, 60.0 - opened, 60.0 - opened]), abs=1e-3)
    assert result["ped_max_startle_factor"] == pytest.approx(25.0, abs=1e-3)
    assert result["worst_case_acoustic_frame"]["door_states"] == {"world/front_door": "open"}


def test_doors_without_semantic_timeline_stay_closed(tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, run = _door_run(tmp_path, name)
    data = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    result = calc.calculate(_episode(data, name, run=run, folder_manager=folder_manager), {})
    assert result["timeseries_acoustic_attenuation_db"] == [pytest.approx([_att(3.0) + 25.0], abs=1e-3)]
    assert result["worst_case_acoustic_frame"]["door_states"] == {"world/front_door": "closed"}


def test_semantic_timeline_for_another_door_keeps_door_closed(tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, run = _door_run(tmp_path, name)
    data = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    semantic = _semantic([(0, "env_0/back_door/0", True)])
    result = calc.calculate(_episode(data, name, run=run, folder_manager=folder_manager, semantic_snapshot=semantic), {})
    assert result["timeseries_acoustic_attenuation_db"] == [pytest.approx([_att(3.0) + 25.0], abs=1e-3)]
    assert result["worst_case_acoustic_frame"]["door_states"] == {"world/front_door": "closed"}


@pytest.mark.parametrize("benchmark_id", [None, ""], ids=["no_run", "no_benchmark_id"])
def test_run_worlds_need_a_benchmark_id(benchmark_id, tmp_path, calc, cached_map):
    name = cached_map(_walled())
    folder_manager, _ = _door_run(tmp_path, name)
    run = None if benchmark_id is None else EpisodeDescriptor(episode_dir="ep", benchmark_id=benchmark_id, episode_id=7, planner="p", stage="s")
    data = _frames([(1.0, 2.5)], [[(4.0, 2.5)]], source=[60.0])
    result = calc.calculate(_episode(data, name, run=run, folder_manager=folder_manager), {})
    assert result["timeseries_acoustic_attenuation_db"] == [pytest.approx([_att(3.0) + 47.0], abs=1e-3)]
    assert result["worst_case_acoustic_frame"]["door_states"] == {}
