import yaml

from arena_evaluation.processing.map_registry import MapRegistry


def test_a_world_copied_into_the_run_wins_over_the_workspace(tmp_path) -> None:
    run_dir = tmp_path / "run"
    world_dir = run_dir / "worlds" / "office"
    (world_dir / "assets").mkdir(parents=True)
    world_spec = {
        "format": "arena-world/2",
        "levels": [
            {
                "id": "0",
                "zones": [
                    {
                        "name": "run_override_zone",
                        "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                    }
                ],
            }
        ],
    }
    (world_dir / "world.yaml").write_text(yaml.safe_dump(world_spec))

    compacted, _ = MapRegistry.compacted_world("office", run_dir=run_dir)

    assert compacted is not None
    assert [zone.name for zone in compacted.zones] == ["run_override_zone"]
