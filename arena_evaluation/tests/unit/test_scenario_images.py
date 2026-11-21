"""Regenerate test-scenario images on-the-fly (visual inspection aid)."""

import pathlib
import sys

import pytest

# Allow direct import of the sibling helper scripts (not a package)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


@pytest.mark.slow
class TestScenarioImages:
    """Generate occupancy-grid PNGs and acoustic-field PNGs for all 8 scenarios."""

    def test_generate_occupancy_grids(self, tmp_path: pathlib.Path) -> None:
        """Walls = white, doors = gray, free = black."""
        from acoustic_scenarios import export_pngs

        export_pngs(tmp_path)
        pngs = sorted(tmp_path.glob("*.png"))
        assert len(pngs) == 8, f"expected 8 occupancy PNGs, got {len(pngs)}"

