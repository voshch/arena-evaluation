import contextlib
import pathlib

import arena_simulation_setup.tree.World as world_module

from arena_evaluation.processing.map_registry import MapRegistry


class _FakeLoaded:
    level_ids: list[str] = []

    def compact_world(self, origins):
        return None


def test_a_world_copied_into_the_run_wins_over_the_workspace(tmp_path, monkeypatch) -> None:
    opened: list[str] = []

    class FakeView:
        def __init__(self, source) -> None:
            self.source = str(source)
            self.path = pathlib.Path(source)

        def load(self, level_filter=None) -> _FakeLoaded:
            opened.append(self.source)
            return _FakeLoaded()

        def level_origins(self):
            return None

    class FakeIdentifier:
        def __init__(self, bare: str) -> None:
            self.bare = bare

        @staticmethod
        def split_filter(raw: str):
            return raw, None

        def resolve_sync(self) -> FakeView:
            return FakeView("workspace")

    monkeypatch.setattr(world_module, "WorldView", FakeView)
    monkeypatch.setattr(world_module, "WorldIdentifier", FakeIdentifier)
    monkeypatch.setattr(world_module, "world_assets", lambda view, world=None: contextlib.nullcontext())
    run_dir = tmp_path / "run"
    (run_dir / "worlds" / "office").mkdir(parents=True)

    MapRegistry.compacted_world("office", run_dir=run_dir)
    MapRegistry.compacted_world("office")

    assert opened == [str(run_dir / "worlds" / "office"), "workspace"]
