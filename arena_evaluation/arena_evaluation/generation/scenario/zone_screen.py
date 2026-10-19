"""Which zones a pedestrian may start in by name: every point the runtime resolver could return there clears the robot's route."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

if typing.TYPE_CHECKING:
    import numpy as np
    from task_generator.manager.world_manager.utils import WorldMap

from ..formats import is_world
from .occupancy import static_polygon

# mirrors `Obstacles.SAFE_DIST` in task_generator/constants/runtime.py; a mismatch clears zones the resolver then fails in
RUNTIME_SAFE_DIST = 0.35


@attrs.frozen(kw_only=True)
class Verdict:
    zone: str
    ref_safe: bool
    reason: str = ""
    admissible: int = 0
    rejected: int = 0

    def __bool__(self) -> bool:
        return self.ref_safe


def runtime_map(root: Path) -> WorldMap:
    """The map the runtime plans the robot's route on: the world rendered as `WorldManagerROS` renders it."""
    from arena_rclpy_mixins.Time import Time
    from arena_simulation_setup.tree.World.view import WorldView
    from task_generator.manager.world_manager.utils import WorldMap
    from task_generator.manager.world_manager.world_manager_ros import _DEFAULT_RESOLUTION

    view = WorldView(root.resolve())
    world = view.load()
    level_origins = view.level_origins()
    origins = level_origins if level_origins is not None else {fid: (0.0, 0.0) for fid in world.level_ids}
    return WorldMap.from_world_description(world.compact_world(origins=origins), resolution=_DEFAULT_RESOLUTION, time=Time(-1, 0), _level_origins=level_origins)


def _load_grid(world_root: str | Path) -> tuple[np.ndarray, float, float, float] | None:
    """The runtime map's walls as (occupied, resolution, origin x, origin y); None for a directory that is no world."""
    from task_generator.manager.world_manager.utils import WorldOccupancy

    root = Path(world_root)
    if not is_world(root):
        return None
    world_map = runtime_map(root)
    return WorldOccupancy.not_empty(world_map.occupancy.walls), world_map.resolution, world_map.origin.x, world_map.origin.y


def _stamp_statics(occupied: np.ndarray, statics: Sequence[Mapping[str, Any]], res: float, ox: float, oy: float) -> None:
    height, width = occupied.shape
    for obj in statics:
        pose = obj.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 2:
            continue
        poly = static_polygon(obj)
        # the box, not a disc: a disc around the reception desk's origin left a goal on its edge looking free (hospital_1, 2026-08-29)
        if poly is not None:
            occupied |= _rasterize(poly, occupied.shape, res, ox, oy)
            continue
        radius = float(obj.get("radius") or obj.get("interaction_radius") or 0.5)
        x, y = float(pose[0]), float(pose[1])
        c0 = max(0, int((x - radius - ox) / res))
        c1 = min(width, int((x + radius - ox) / res) + 1)
        r0 = max(0, height - 1 - int((y + radius - oy) / res))
        r1 = min(height, height - int((y - radius - oy) / res))
        if r0 < r1 and c0 < c1:
            occupied[r0:r1, c0:c1] = True


def _rasterize(poly: Sequence[tuple[float, float]], shape: tuple[int, int], res: float, ox: float, oy: float) -> np.ndarray:
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = shape
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(
        [(int((x - ox) / res), height - 1 - int((y - oy) / res)) for x, y in poly],
        fill=1,
    )
    return np.asarray(canvas, dtype=bool)


def _cell_centres(mask: np.ndarray, res: float, ox: float, oy: float) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    rows, cols = np.nonzero(mask)
    height = mask.shape[0]
    return cols * res + ox, (height - 1 - rows) * res + oy


def _route_clearance(xs: np.ndarray, ys: np.ndarray, route: Sequence[tuple[float, float]]) -> np.ndarray:
    import numpy as np

    if not route:
        return np.full(xs.shape, np.inf)
    if len(route) == 1:
        return np.hypot(xs - route[0][0], ys - route[0][1])

    best = np.full(xs.shape, np.inf)
    for (ax, ay), (bx, by) in zip(route, route[1:], strict=False):
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom == 0:
            best = np.minimum(best, np.hypot(xs - ax, ys - ay))
            continue
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / denom, 0.0, 1.0)
        best = np.minimum(best, np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)))
    return best


def screen_zones(
    zones: Sequence[Mapping[str, Any]],
    *,
    world_root: str | Path,
    route: Sequence[tuple[float, float]],
    statics: Sequence[Mapping[str, Any]] = (),
    min_route_clearance: float,
    safe_dist: float = RUNTIME_SAFE_DIST,
    robot_zones: frozenset[str] = frozenset(),
) -> dict[str, Verdict]:
    """A verdict per zone; an unscreenable world gets verdicts that all refuse, and `robot_zones` are refused whole."""
    verdicts: dict[str, Verdict] = {}
    names = [str(z["name"]) for z in zones if isinstance(z, Mapping) and z.get("name")]

    grid = _load_grid(world_root)
    if grid is None:
        return {n: Verdict(zone=n, ref_safe=False, reason="no occupancy grid to screen against") for n in names}
    occupied, res, ox, oy = grid

    import scipy.ndimage

    _stamp_statics(occupied, statics, res, ox, oy)
    dist = scipy.ndimage.distance_transform_edt(~occupied) * res

    for zone in zones:
        if not isinstance(zone, Mapping) or not zone.get("name"):
            continue
        name = str(zone["name"])
        poly = [(float(c["x"]), float(c["y"])) for c in zone.get("corners", []) if isinstance(c, Mapping)]
        if len(poly) < 3:
            verdicts[name] = Verdict(zone=name, ref_safe=False, reason="fewer than three corners")
            continue

        mask = _rasterize(poly, dist.shape, res, ox, oy)
        if not mask.any():
            verdicts[name] = Verdict(zone=name, ref_safe=False, reason="polygon covers no grid cell")
            continue

        clearance = dist[mask]
        admissible = clearance > safe_dist
        n_admissible = int(admissible.sum())
        if n_admissible == 0:
            verdicts[name] = Verdict(
                zone=name,
                ref_safe=False,
                reason="no point in the zone clears SAFE_DIST; the resolver would warn and return a blocked point",
            )
            continue

        if name in robot_zones:
            verdicts[name] = Verdict(
                zone=name,
                ref_safe=False,
                admissible=n_admissible,
                rejected=n_admissible,
                reason="the robot's own spec names this zone as an endpoint, so it may start or finish anywhere inside it",
            )
            continue

        xs, ys = _cell_centres(mask, res, ox, oy)
        xs, ys, clearance = xs[admissible], ys[admissible], clearance[admissible]

        route_bad = _route_clearance(xs, ys, route) < min_route_clearance
        n_rejected = int(route_bad.sum())

        if n_rejected == 0:
            verdicts[name] = Verdict(zone=name, ref_safe=True, admissible=n_admissible)
            continue

        verdicts[name] = Verdict(
            zone=name,
            ref_safe=False,
            admissible=n_admissible,
            rejected=n_rejected,
            reason=f"{n_rejected}/{n_admissible} resolvable points sit within {min_route_clearance} m of the robot's route",
        )

    return verdicts


def ref_safe_zones(verdicts: Mapping[str, Verdict]) -> frozenset[str]:
    return frozenset(name for name, v in verdicts.items() if v.ref_safe)
