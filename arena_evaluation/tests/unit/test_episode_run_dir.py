from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, EpisodeDescriptor


def _bundle(**kwargs: object) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(episode_id=0, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[1.0, 0.0], **kwargs)


def test_run_dir_is_unset_until_the_pipeline_gives_one(tmp_path: pathlib.Path):
    assert _bundle().run_dir is None
    assert _bundle(run_dir=tmp_path / "golden").run_dir == tmp_path / "golden"


def test_a_world_snapshotted_into_the_run_resolves_through_the_run_dir(tmp_path: pathlib.Path):
    from arena_simulation_setup.tree.World import Level, World, WorldView, Zone
    from PIL import Image

    world = tmp_path / "golden" / "worlds" / "bundled_world"
    WorldView(world).save(World.from_levels(Level(zones=[Zone(name="floor", polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])])))
    (world / "map" / "0.yaml").write_text("image: 0.png\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n")
    Image.new("L", (4, 3), 255).save(world / "map" / "0.png")
    run = EpisodeDescriptor(episode_dir=str(tmp_path / "golden" / "episodes" / "episode_000"), benchmark_id="golden", episode_id=0, planner="dwb", stage="s")
    episode = _bundle(run=run, run_dir=tmp_path / "golden", map="bundled_world")
    meta = MapRegistry.get_map(episode.map, cache_dir=tmp_path / "cache", run_dir=episode.run_dir)
    assert meta is not None
    assert (meta["width"], meta["height"], meta["resolution"]) == (4, 3, 0.05)
