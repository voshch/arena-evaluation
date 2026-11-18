"""Acoustic field renderer: door masks under downsampling and embedded video links."""

from __future__ import annotations

import numpy as np
import polars as pl

from arena_evaluation.presentation.plot_types.acoustic_field import (
    AcousticFieldAnimationRenderer,
    AcousticFieldRenderer,
)
from arena_evaluation.presentation.viz_manifest import PlotSpec
from arena_evaluation.processing.acoustics.impedance_grid import (
    downsample_mask,
    downsample_occupancy,
)


def _spec(ptype: str, **options: object) -> PlotSpec:
    return PlotSpec(id="anim", type=ptype, title="Field", options=dict(options))


def test_downsample_mask_keeps_a_band_between_strided_rows() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[21:24, 16:32] = True
    pooled = downsample_mask(mask, 4)
    assert (
        pooled.shape == downsample_occupancy(mask.astype(np.uint8), 4).shape == (12, 12)
    )
    assert pooled[5, 4:8].all()
    assert pooled.sum() == 4


def test_downsampled_field_keeps_closed_door_in_doorway() -> None:
    grid = np.zeros((48, 48), dtype=np.uint8)
    grid[22, :] = 1
    grid[22, 16:32] = 0
    door = np.zeros_like(grid, dtype=bool)
    door[21:24, 16:32] = True

    df = pl.DataFrame(
        {
            "t_ns": [0],
            "x": [2.4],
            "y": [1.0],
            "source_dba": [100.0],
        }
    )
    renderer = AcousticFieldRenderer(_spec("acoustic_field"))
    frames = renderer.compute_field_timeseries(
        df, grid, 0.1, 0.0, 0.0, {"world/d": (door, 25.0)}, downsample=4
    )
    assert frames[0] is not None
    field_dba, eff_res, (h, w), _, _ = frames[0]
    assert (h, w) == (12, 12)
    tx, ty = round(2.4 / eff_res), round(4.0 / eff_res)
    los = 20.0 * np.log10(3.0)
    assert field_dba[ty, tx] < 100.0 - los - 20.0


def test_mp4_per_episode_animations_embed_video_links() -> None:
    df = pl.DataFrame({"episode_id": [1, 2], "ped_max_exposure_dba": [60.0, 70.0]})
    chunks = AcousticFieldAnimationRenderer(
        _spec("acoustic_field_animation", format="mp4", per_episode=True)
    ).render_plotly(df)
    assert isinstance(chunks, list)
    assert ['<video src="plots/anim_episode_001.mp4"' in c for c in chunks] == [
        True,
        False,
    ]
    assert ['<video src="plots/anim_episode_002.mp4"' in c for c in chunks] == [
        False,
        True,
    ]
