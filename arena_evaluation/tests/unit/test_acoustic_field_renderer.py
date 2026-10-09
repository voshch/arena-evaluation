"""Acoustic field renderer: door masks under downsampling and embedded video links."""

from __future__ import annotations

import numpy as np
import polars as pl

from arena_evaluation.presentation.plot_types.acoustic_field import (
    AcousticFieldAnimationRenderer,
    AcousticFieldRenderer,
)
from arena_evaluation.processing.acoustics.impedance_grid import downsample_mask, downsample_occupancy
from arena_evaluation.storage.schemas import PlotSpec


def _spec(ptype: str, **options) -> PlotSpec:
    return PlotSpec(id="anim", type=ptype, title="Field", data_key="ped_max_exposure_dba", options=options)


def test_downsample_mask_keeps_a_band_between_strided_rows():
    mask = np.zeros((50, 50), dtype=bool)
    mask[21:24, 16:32] = True
    pooled = downsample_mask(mask, 4)
    assert pooled.shape == downsample_occupancy(mask.astype(np.uint8), 4).shape == (12, 12)
    assert pooled[5, 4:8].all()
    assert pooled.sum() == 4


def test_downsampled_field_keeps_closed_door_in_doorway():
    grid = np.zeros((48, 48), dtype=np.uint8)
    grid[22, :] = 1
    grid[22, 16:32] = 0
    door = np.zeros_like(grid, dtype=bool)
    door[21:24, 16:32] = True

    df = pl.DataFrame(
        {
            "time_ns": [0],
            "pos_x_gt": [2.4],
            "pos_y_gt": [1.0],
            "total_level_af_dba": [100.0],
        }
    )
    renderer = AcousticFieldRenderer(_spec("acoustic_field"))
    frames = renderer.compute_field_timeseries(df, grid, 0.1, 0.0, 0.0, {"world/d": (door, 25.0)}, downsample=4)
    field_dba, eff_res, (h, w), _, _ = frames[0]
    assert (h, w) == (12, 12)
    tx, ty = round(2.4 / eff_res), round(4.0 / eff_res)
    los = 20.0 * np.log10(3.0)
    assert field_dba[ty, tx] < 100.0 - los - 20.0


def test_mp4_per_episode_animations_embed_video_links():
    df = pl.DataFrame({"episode": [1, 2], "ped_max_exposure_dba": [60.0, 70.0]})
    chunks = AcousticFieldAnimationRenderer(_spec("acoustic_field_animation", format="mp4", per_episode=True)).render_plotly(df)
    assert ['<video src="plots/anim_episode_001.mp4"' in c for c in chunks] == [True, False]
    assert ['<video src="plots/anim_episode_002.mp4"' in c for c in chunks] == [False, True]
