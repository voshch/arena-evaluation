from __future__ import annotations

import dataclasses
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import shapely
    from arena_simulation_setup.tree.World import Level


@dataclasses.dataclass(frozen=True, slots=True)
class ZoneGeometry:
    """Zone polygon with the speed, quiet and restricted annotations recorded for it."""

    name: str
    polygon: shapely.Polygon | shapely.MultiPolygon
    max_speed: float | None
    quiet: bool
    restricted: bool


@dataclasses.dataclass(frozen=True, slots=True)
class DoorGeometry:
    """Door centre and activation radius in the flattened map frame."""

    name: str
    center_x: float
    center_y: float
    radius: float


def extract_zone_geometry(flattened: Level, require_annotation: bool = True) -> list[ZoneGeometry]:
    """Zone polygons with their recorded semantics, skipping degenerate polygons."""
    import shapely

    zones: list[ZoneGeometry] = []
    for zone in flattened.zones:
        if len(zone.polygon) < 3:
            continue

        max_speed = None
        quiet = False
        restricted = False
        has_annotation = False
        for cfg in zone.semantics:
            if cfg.role == "state" and cfg.name == "max_speed":
                has_annotation = True
                if cfg.value is not None:
                    max_speed = float(cfg.value)
            elif cfg.role == "predicate" and cfg.name == "quiet":
                has_annotation = True
                quiet = bool(cfg.value)
            elif cfg.role == "predicate" and cfg.name == "restricted":
                has_annotation = True
                restricted = bool(cfg.value)

        if require_annotation and not has_annotation:
            continue

        polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.polygon])
        zones.append(
            ZoneGeometry(
                name=zone.name,
                polygon=polygon,
                max_speed=max_speed,
                quiet=quiet,
                restricted=restricted,
            )
        )
    return zones


def extract_door_geometry(flattened: Level) -> list[DoorGeometry]:
    """Door centres and activation radii in the flattened map frame."""
    doors: list[DoorGeometry] = []
    for door in flattened.all_doors:
        center_x = (door.start.x + door.end.x) / 2.0
        center_y = (door.start.y + door.end.y) / 2.0
        radius = max(door.activation_distance)
        doors.append(DoorGeometry(name=door.name, center_x=center_x, center_y=center_y, radius=radius))
    return doors


class _PolygonZone(typing.Protocol):
    @property
    def polygon(self) -> shapely.Polygon | shapely.MultiPolygon: ...


def zone_membership(pos_x: np.ndarray, pos_y: np.ndarray, zones: Sequence[_PolygonZone]) -> np.ndarray:
    """Index of the first zone covering each point, -1 outside every zone."""
    import shapely

    idx = np.full(len(pos_x), -1, dtype=np.int64)
    for i in range(len(pos_x)):
        point = shapely.Point(pos_x[i], pos_y[i])
        for j, zone in enumerate(zones):
            if zone.polygon.covers(point):
                idx[i] = j
                break
    return idx
