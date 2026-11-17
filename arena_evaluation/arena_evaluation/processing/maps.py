"""Reconstruct a dense occupancy grid from the sparse recorded environment-map tables, offline."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Final

import numpy as np
import polars as pl
from scipy.ndimage import distance_transform_edt

from arena_evaluation.processing.path.theta_star import StaticGrid

MAP_TABLE: Final[str] = "environment_map"
CELLS_TABLE: Final[str] = "environment_map_cells"
DEFAULT_KIND: Final[str] = "map"
_UNROTATED_TOLERANCE_RAD: Final[float] = 1e-9


@dataclasses.dataclass(frozen=True, slots=True)
class DenseGrid:
    """One recorded occupancy-grid snapshot, reconstructed dense; row 0 is the lowest y, as in ROS."""

    kind: str
    frame_id: str | None
    resolution_m: float
    width_cells: int
    height_cells: int
    origin_x: float
    origin_y: float
    origin_yaw: float
    cells: np.ndarray

    def static_grid(self, robot_radius_m: float = 0.0) -> StaticGrid:
        """The lethal mask inflated by a robot radius, as theta* consumes it; a rotated snapshot is refused."""
        if abs(self.origin_yaw) > _UNROTATED_TOLERANCE_RAD:
            raise ValueError(
                f"the recorded {self.kind} snapshot is rotated by {self.origin_yaw} rad; "
                "a StaticGrid is axis-aligned"
            )
        # Free is 0; occupied and unknown cells both block a route.
        lethal = self.cells != 0
        if robot_radius_m > 0.0:
            clearance_m = distance_transform_edt(~lethal) * self.resolution_m
            lethal = clearance_m <= robot_radius_m
        return StaticGrid(
            origin_xy=(self.origin_x, self.origin_y),
            resolution_m=self.resolution_m,
            lethal=lethal,
        )


def load_snapshot(
    source_dir: pathlib.Path, *, kind: str = DEFAULT_KIND, t_ns: int | None = None
) -> DenseGrid | None:
    """The latest dense snapshot of one kind at or before t_ns, None when the recording has none."""
    root = pathlib.Path(source_dir)
    metadata_path = root / f"{MAP_TABLE}.parquet"
    if not metadata_path.is_file():
        return None
    metadata = pl.read_parquet(metadata_path).filter(pl.col("kind") == kind)
    if t_ns is not None:
        metadata = metadata.filter(pl.col("t_ns") <= t_ns)
    if metadata.is_empty():
        return None
    row = metadata.sort("t_ns").row(-1, named=True)
    width = int(row["width_cells"])
    height = int(row["height_cells"])
    cells = np.zeros((height, width), dtype=np.int8)
    cells_path = root / f"{CELLS_TABLE}.parquet"
    if cells_path.is_file():
        sparse = pl.read_parquet(cells_path).filter(
            (pl.col("kind") == kind) & (pl.col("t_ns") == row["t_ns"])
        )
        cells[sparse["row"].to_numpy(), sparse["col"].to_numpy()] = sparse[
            "occupancy"
        ].to_numpy()
    return DenseGrid(
        kind=kind,
        frame_id=row["frame_id"],
        resolution_m=float(row["resolution_m"]),
        width_cells=width,
        height_cells=height,
        origin_x=float(row["origin_x"]),
        origin_y=float(row["origin_y"]),
        origin_yaw=float(row["origin_yaw"]),
        cells=cells,
    )
