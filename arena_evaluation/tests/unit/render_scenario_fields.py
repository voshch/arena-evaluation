#!/usr/bin/env python3
"""Render acoustic field overlays for each test scenario (visual inspection).

Shows the RECEIVED level in dBA (source - attenuation), not raw attenuation,
with a 3 dB (double-energy) colorbar. Source defaults to 60 dBA.

Rendering matches the main AcousticFieldRenderer style:
- Black background for walls (NaN in the field array)
- Cyan contour outlines for all doors
- Bright green contour outlines for open doors

Usage:
    python render_scenario_fields.py [out_dir] [--source-dba 60]
"""

import argparse
import pathlib

import numpy as np

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from acoustic_scenarios import scenarios, build, door_pixels, RES, WALL_TL, MIC_DIST  # noqa: E402

DOOR_TL = 25.0

from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations  # noqa: E402


def _door_mask(grid, spec) -> np.ndarray:
    """Build a boolean mask covering all door pixels in the scenario."""
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    for y, x in door_pixels(spec):
        if 0 <= y < h and 0 <= x < w:
            mask[y, x] = True
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", nargs="?", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent / "test_images" / "fields_dba")
    ap.add_argument("--source-dba", type=float, default=60.0, help="source level at 1 m (default 60)")
    args = ap.parse_args()
    OUT = args.out_dir
    SOURCE = args.source_dba

    for name, spec in scenarios().items():
        grid = build({**spec, "_name": name})
        h, w = grid.shape
        sx, sy = spec["start"]
        sxp, syp = int(round(sx / RES)), int(round(sy / RES))

        # per-pixel TL: doors get their own TL (open 0, closed DOOR_TL, walls 47)
        open_door = "open" in name
        tl = np.where(grid == 1, WALL_TL, 0.0).astype(np.float32)
        door_px = door_pixels(spec)
        for y, x in door_px:
            if 0 <= y < h and 0 <= x < w:
                tl[y, x] = 0.0 if open_door else DOOR_TL
        tl = np.ascontiguousarray(tl)

        yy, xx = np.mgrid[0:h, 0:w]
        att = compute_attenuations(
            grid,
            RES,
            sxp,
            syp,
            np.ascontiguousarray(xx.flatten().astype(np.float32)),
            np.ascontiguousarray(yy.flatten().astype(np.float32)),
            wall_tl=WALL_TL,
            mic_distance=MIC_DIST,
            pixel_tl=tl,
        ).reshape((h, w))

        # RECEIVED level: NaN walls (black background) and unreachable pixels
        field_dba = np.where((grid == 1) | np.isinf(att), np.nan, SOURCE - att)

        tx, ty = spec["target"]
        recv_target = float(SOURCE - att[int(round(ty / RES)), int(round(tx / RES))])

        # Build door masks for contour rendering
        door_mask = _door_mask(grid, spec)
        open_door_mask = door_mask & (tl == 0.0) if open_door else np.zeros_like(door_mask)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, w * 0.03), max(5, h * 0.03)))
        ax.set_facecolor("black")
        ax.grid(False)

        # Main heatmap: inferno, walls NaN -> black background
        im = ax.imshow(np.flipud(field_dba), cmap="inferno", origin="upper", vmin=0, vmax=SOURCE)
        # 3 dB ticks = double-energy steps
        cb = plt.colorbar(im, ax=ax, label="Received level (dBA)", ticks=np.arange(0, SOURCE + 1, 3))
        cb.set_ticklabels([f"{t:.0f}" for t in np.arange(0, SOURCE + 1, 3)])

        # Door contours (matching AcousticFieldRenderer._render_cell_png style)
        if door_mask.any():
            cty = np.linspace(0, h * RES, h)
            ctx = np.linspace(0, w * RES, w)
            # Cyan outline for ALL doors
            ax.contour(ctx, cty, np.flipud(door_mask.astype(np.uint8)), levels=[0.5], colors=["#00ffd5"], linewidths=1.5, alpha=0.85)
            # Bright green outline for OPEN doors
            if open_door_mask.any():
                ax.contour(ctx, cty, np.flipud(open_door_mask.astype(np.uint8)), levels=[0.5], colors=["#00ff00"], linewidths=2.5, alpha=0.9)

        ax.plot(sxp, syp, "g*", markersize=12, label="Source")
        ax.plot(tx / RES, ty / RES, "r^", markersize=10, label="Target")
        ax.set_title(f"{name}  |  received@target = {recv_target:.1f} dBA")
        ax.legend(loc="upper right")
        ax.axis("off")
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}.png", dpi=120)
        plt.close(fig)
        print(f"  {name:32s} received@target = {recv_target:6.1f} dBA  -> {OUT / name}.png")

    print(f"\nFields (dBA) written to {OUT}")


if __name__ == "__main__":
    main()
