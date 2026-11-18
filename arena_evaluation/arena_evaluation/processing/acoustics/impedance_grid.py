"""C++ acoustic attenuation solver wrapper."""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path

import numpy as np

from arena_evaluation.storage.exceptions import MetricInputError

_SRC_DIR = Path(__file__).parent
_DEFAULT_SO_FILE = _SRC_DIR / "solver.so"
_ENV_SO_VAR = "ARENA_ACOUSTIC_SOLVER_SO"


@functools.cache
def _solver_library() -> ctypes.CDLL:
    """Load the pre-built C++17 acoustic solver shared library; never compiles at import."""
    env_path = os.environ.get(_ENV_SO_VAR)
    candidates = [Path(env_path)] if env_path else [_DEFAULT_SO_FILE]
    so_path = next((path for path in candidates if path.is_file()), None)
    if so_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise MetricInputError(
            f"AcousticExposureCalculator: compiled acoustic solver library not found (searched {searched}); "
            "run `arena build` (requires a C++17 compiler) to compile processing/acoustics/solver.cpp"
        )
    lib = ctypes.CDLL(str(so_path))
    lib.solve_acoustic_field.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),  # grid
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_float,  # resolution
        ctypes.c_float,  # start_x
        ctypes.c_float,  # start_y
        np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"
        ),  # target_xs
        np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"
        ),  # target_ys
        ctypes.c_int,  # num_targets
        ctypes.c_float,  # wall_tl
        ctypes.c_float,  # mic_distance
        ctypes.c_void_p,  # pixel_tl (NULL when None)
        np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"
        ),  # out_attenuations
    ]
    lib.solve_acoustic_field.restype = None

    lib.solve_acoustic_grid.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),  # grid
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_float,  # resolution
        ctypes.c_float,  # start_x
        ctypes.c_float,  # start_y
        ctypes.c_float,  # wall_tl
        ctypes.c_float,  # mic_distance
        ctypes.c_void_p,  # pixel_tl (NULL when None)
        np.ctypeslib.ndpointer(
            dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
        ),  # out_field
    ]
    lib.solve_acoustic_grid.restype = None
    return lib


def compute_attenuations(
    occupancy_grid: np.ndarray,
    resolution: float,
    start_x_px: float,
    start_y_px: float,
    target_xs_px: np.ndarray,
    target_ys_px: np.ndarray,
    wall_tl: float = 47.0,
    mic_distance: float = 1.0,
    pixel_tl: np.ndarray | None = None,
) -> np.ndarray:
    """Compute acoustic attenuation from start pixel to target pixels, with mic_distance as the near-field floor (m)."""
    if not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.dtype != np.uint8:
        occupancy_grid = np.ascontiguousarray(occupancy_grid, dtype=np.uint8)

    if not isinstance(target_xs_px, np.ndarray) or target_xs_px.dtype != np.float32:
        target_xs_px = np.ascontiguousarray(target_xs_px, dtype=np.float32)

    if not isinstance(target_ys_px, np.ndarray) or target_ys_px.dtype != np.float32:
        target_ys_px = np.ascontiguousarray(target_ys_px, dtype=np.float32)

    num_targets = len(target_xs_px)
    if len(target_ys_px) != num_targets:
        raise ValueError("target_xs_px and target_ys_px must have the same length.")

    out_attenuations = np.empty(num_targets, dtype=np.float32)

    height, width = occupancy_grid.shape

    # pixel_tl: None -> pass a NULL pointer; otherwise C-contiguous float32
    tl_ptr = None
    if pixel_tl is not None:
        if pixel_tl.shape != (height, width):
            raise ValueError(
                f"pixel_tl shape {pixel_tl.shape} != grid shape {(height, width)}"
            )
        pixel_tl = np.ascontiguousarray(pixel_tl, dtype=np.float32)
        tl_ptr = pixel_tl.ctypes.data_as(ctypes.c_void_p)

    _solver_library().solve_acoustic_field(
        occupancy_grid,
        width,
        height,
        float(resolution),
        float(start_x_px),
        float(start_y_px),
        target_xs_px,
        target_ys_px,
        num_targets,
        float(wall_tl),
        float(mic_distance),
        tl_ptr,
        out_attenuations,
    )

    return out_attenuations


def compute_acoustic_field(
    occupancy_grid: np.ndarray,
    resolution: float,
    start_x_px: float,
    start_y_px: float,
    wall_tl: float = 47.0,
    mic_distance: float = 1.0,
    pixel_tl: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the full 2D acoustic attenuation field (shape H, W) from start pixel."""
    occupancy_grid = np.ascontiguousarray(occupancy_grid, dtype=np.uint8)

    height, width = occupancy_grid.shape
    out_field = np.empty((height, width), dtype=np.float32)

    tl_ptr = None
    if pixel_tl is not None:
        if pixel_tl.shape != (height, width):
            raise ValueError(
                f"pixel_tl shape {pixel_tl.shape} != grid shape {(height, width)}"
            )
        pixel_tl = np.ascontiguousarray(pixel_tl, dtype=np.float32)
        tl_ptr = pixel_tl.ctypes.data_as(ctypes.c_void_p)

    _solver_library().solve_acoustic_grid(
        occupancy_grid,
        width,
        height,
        float(resolution),
        float(start_x_px),
        float(start_y_px),
        float(wall_tl),
        float(mic_distance),
        tl_ptr,
        out_field,
    )

    return out_field


def downsample_occupancy(grid: np.ndarray, ds: int) -> np.ndarray:
    """Downsample an occupancy grid keeping walls (max-pool over ds x ds windows)."""
    if ds <= 1:
        return grid
    h, w = grid.shape
    h2, w2 = h // ds, w // ds
    g = grid[: h2 * ds, : w2 * ds].reshape(h2, ds, w2, ds)
    pooled = g.max(axis=(1, 3)).astype(np.uint8)
    return np.ascontiguousarray(pooled)


def downsample_mask(mask: np.ndarray, ds: int) -> np.ndarray:
    """Downsample a boolean mask with the same max-pool windows as downsample_occupancy."""
    if ds <= 1:
        return mask
    return downsample_occupancy(mask.astype(np.uint8), ds).astype(bool)
