"""C++ acoustic attenuation solver wrapper.

The solver is a C++17 shared library built from `solver.cpp` by `arena build` (the
`arena_acoustics_solver` CMake target) and loaded here through ctypes. The library's ABI
version is verified before any symbol is bound, so a library built from an older model
raises instead of yielding a different number.
"""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path
from typing import Final

import numpy as np

from arena_evaluation.processing.acoustics.constants import (
    ACOUSTIC_SOLVER_ABI,
    ACOUSTIC_SOLVER_ABI_SYMBOL,
    ACOUSTIC_SOLVER_SYMBOLS,
    MIC_DISTANCE_M,
    WALL_TRANSMISSION_LOSS_DB,
)
from arena_evaluation.storage.exceptions import MetricInputError

_SRC_DIR = Path(__file__).parent
_DEFAULT_SO_FILE = _SRC_DIR / "solver.so"
_ENV_SO_VAR: Final[str] = "ARENA_ACOUSTIC_SOLVER_SO"
_BUILD_HINT: Final[str] = (
    "run `arena build` (the arena_acoustics_solver CMake target), or "
    "`cmake -S arena_evaluation -B build -DARENA_SOLVERS_ONLY=ON && cmake --build build`, "
    "or set ARENA_ACOUSTIC_SOLVER_SO to a library built elsewhere (needs a C++17 compiler)"
)


def solver_library_path() -> Path:
    """The library that is loaded: $ARENA_ACOUSTIC_SOLVER_SO when set, else `solver.so` next to this module."""
    override = os.environ.get(_ENV_SO_VAR)
    return Path(override) if override else _DEFAULT_SO_FILE


def solver_abi(lib: ctypes.CDLL) -> int:
    """The ABI version the loaded library exports, verified against the wrapper's ABI."""
    try:
        exported = lib.arena_acoustic_solver_abi
    except AttributeError as error:
        raise MetricInputError(
            f"acoustic solver {solver_library_path()} exports no {ACOUSTIC_SOLVER_ABI_SYMBOL}, "
            f"so it predates ABI {ACOUSTIC_SOLVER_ABI}; {_BUILD_HINT}"
        ) from error
    exported.restype = ctypes.c_int
    exported.argtypes = []
    return int(exported())


def _verify_library(lib: ctypes.CDLL) -> None:
    """Raise MetricInputError unless the library exports the wrapper's ABI and solver symbols."""
    so_path = solver_library_path()
    abi = solver_abi(lib)
    if abi != ACOUSTIC_SOLVER_ABI:
        raise MetricInputError(
            f"acoustic solver {so_path} exports ABI {abi}, this package needs ABI {ACOUSTIC_SOLVER_ABI}; {_BUILD_HINT}"
        )
    missing = [
        symbol for symbol in ACOUSTIC_SOLVER_SYMBOLS if not _exports(lib, symbol)
    ]
    if missing:
        raise MetricInputError(
            f"acoustic solver {so_path} exports ABI {abi} but no {', '.join(missing)}; {_BUILD_HINT}"
        )


def _exports(lib: ctypes.CDLL, symbol: str) -> bool:
    """Whether the loaded library defines `symbol`; ctypes raises AttributeError when it does not."""
    try:
        getattr(lib, symbol)
    except AttributeError:
        return False
    return True


@functools.cache
def _solver_library() -> ctypes.CDLL:
    """Load the built C++17 acoustic solver library and bind its C ABI; never compiles."""
    so_path = solver_library_path()
    if not so_path.is_file():
        raise MetricInputError(
            f"acoustic solver library not found at {so_path}; {_BUILD_HINT}"
        )
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError as error:
        raise MetricInputError(
            f"acoustic solver library {so_path} could not be loaded: {error}; {_BUILD_HINT}"
        ) from error
    _verify_library(lib)

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
    wall_tl: float = WALL_TRANSMISSION_LOSS_DB,
    mic_distance: float = MIC_DISTANCE_M,
    pixel_tl: np.ndarray | None = None,
) -> np.ndarray:
    """Attenuation from start pixel to target pixels; `wall_tl` applies only without a `pixel_tl` map."""
    if not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.dtype != np.uint8:
        occupancy_grid = np.ascontiguousarray(occupancy_grid, dtype=np.uint8)

    if not isinstance(target_xs_px, np.ndarray) or target_xs_px.dtype != np.float32:
        target_xs_px = np.ascontiguousarray(target_xs_px, dtype=np.float32)

    if not isinstance(target_ys_px, np.ndarray) or target_ys_px.dtype != np.float32:
        target_ys_px = np.ascontiguousarray(target_ys_px, dtype=np.float32)

    num_targets = len(target_xs_px)
    if len(target_ys_px) != num_targets:
        raise MetricInputError(
            f"target_xs_px and target_ys_px must have the same length, got {num_targets} and {len(target_ys_px)}"
        )

    out_attenuations = np.empty(num_targets, dtype=np.float32)

    height, width = occupancy_grid.shape
    tl_ptr = _pixel_tl_pointer(pixel_tl, height, width)

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
    wall_tl: float = WALL_TRANSMISSION_LOSS_DB,
    mic_distance: float = MIC_DISTANCE_M,
    pixel_tl: np.ndarray | None = None,
) -> np.ndarray:
    """Full 2D attenuation field (rows, columns) from start pixel; `wall_tl` applies only without a `pixel_tl` map."""
    occupancy_grid = np.ascontiguousarray(occupancy_grid, dtype=np.uint8)

    height, width = occupancy_grid.shape
    out_field = np.empty((height, width), dtype=np.float32)
    tl_ptr = _pixel_tl_pointer(pixel_tl, height, width)

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


def _pixel_tl_pointer(
    pixel_tl: np.ndarray | None, height: int, width: int
) -> ctypes.c_void_p | None:
    """The C pointer of a per-cell transmission-loss map, NULL when there is none."""
    if pixel_tl is None:
        return None
    if pixel_tl.shape != (height, width):
        raise MetricInputError(
            f"pixel_tl shape {pixel_tl.shape} != grid shape {(height, width)}"
        )
    contiguous = np.ascontiguousarray(pixel_tl, dtype=np.float32)
    return contiguous.ctypes.data_as(ctypes.c_void_p)


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
