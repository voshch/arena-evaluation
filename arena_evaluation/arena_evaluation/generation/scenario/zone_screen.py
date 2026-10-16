"""Which zones a pedestrian may start in by name: every point the runtime resolver could return there clears the robot's route.

The resolver already keeps a body clear of walls and furniture; the robot's route is the one thing it cannot know.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml

if typing.TYPE_CHECKING:
    import numpy as np

from ..formats import level_map
from .occupancy import static_polygon

#: The clearance `TM_Scenario.reset` validates resolved poses at: `Obstacles.SAFE_DIST` (0.15) plus
#: `Obstacles.PEDESTRIAN_BODY_RADIUS` (0.25), mirrored from `task_generator/constants/runtime.py`.
#: Keep it mirrored: a kinder value clears zones the resolver then fails in, a stricter one rejects good ones.
RUNTIME_SAFE_DIST = 0.15 + 0.25


@attrs.frozen(kw_only=True)
class Verdict:
    zone: str
    ref_safe: bool
    reason: str = ""
    #: grid cells the resolver could return: inside the polygon and past `safe_dist`
    admissible: int = 0
    #: of those, the cells that fail the screen
    rejected: int = 0

    def __bool__(self) -> bool:
        return self.ref_safe


def _load_grid(world_root: str | Path) -> tuple[np.ndarray, float, float, float] | None:
    """Wall occupancy as a boolean mask, plus resolution and origin x, y. None when unreadable."""
    root = Path(world_root)
    meta_path = level_map(root)
    if meta_path is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415 - optional, only needed to screen
        from PIL import Image  # noqa: PLC0415

        meta = yaml.safe_load(meta_path.read_text()) or {}
        image = Image.open(meta_path.parent / meta.get("image", "map.png")).convert("L")
    except Exception:
        return None

    res = float(meta.get("resolution", 0.05))
    origin = meta.get("origin") or [0.0, 0.0, 0.0]
    thresh = float(meta.get("occupied_thresh", 0.65))
    negate = bool(meta.get("negate", 0))

    pixels = np.asarray(image, dtype=float) / 255.0
    shade = pixels if negate else 1.0 - pixels
    return shade > thresh, res, float(origin[0]), float(origin[1])


def _stamp_statics(occupied: np.ndarray, statics: Sequence[Mapping[str, Any]], res: float, ox: float, oy: float) -> None:
    """Stamp furniture into the wall mask: a static with a `bbox` as its rotated rectangle, otherwise
    the axis-aligned square around its radius, as `WorldManager`'s `obstacle_occupy` does."""
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
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    height, width = shape
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(
        [(int((x - ox) / res), height - 1 - int((y - oy) / res)) for x, y in poly],
        fill=1,
    )
    return np.asarray(canvas, dtype=bool)


def _cell_centres(mask: np.ndarray, res: float, ox: float, oy: float) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np  # noqa: PLC0415

    rows, cols = np.nonzero(mask)
    height = mask.shape[0]
    return cols * res + ox, (height - 1 - rows) * res + oy


def _route_clearance(xs: np.ndarray, ys: np.ndarray, route: Sequence[tuple[float, float]]) -> np.ndarray:
    import numpy as np  # noqa: PLC0415

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
    """A verdict per zone; an unscreenable world gets verdicts that all refuse. `robot_zones` (robot
    endpoints given as zones) are refused whole, since the robot may resolve anywhere inside them."""
    verdicts: dict[str, Verdict] = {}
    names = [str(z["name"]) for z in zones if isinstance(z, Mapping) and z.get("name")]

    grid = _load_grid(world_root)
    if grid is None:
        return {n: Verdict(zone=n, ref_safe=False, reason="no occupancy grid to screen against") for n in names}
    occupied, res, ox, oy = grid

    import scipy.ndimage  # noqa: PLC0415

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
        # only a cell the resolver could return can make a zone unsafe
        admissible = clearance > safe_dist
        n_admissible = int(admissible.sum())
        if n_admissible == 0:
            verdicts[name] = Verdict(
                zone=name, ref_safe=False,
                reason="no point in the zone clears SAFE_DIST; the resolver would warn and return a blocked point",
            )
            continue

        if name in robot_zones:
            # admissible is still reported: the zone resolves fine as a destination, just not as a start
            verdicts[name] = Verdict(
                zone=name, ref_safe=False, admissible=n_admissible, rejected=n_admissible,
                reason="the robot's own spec names this zone as an endpoint, so it may start or "
                       "finish anywhere inside it",
            )
            continue

        xs, ys = _cell_centres(mask, res, ox, oy)
        xs, ys, clearance = xs[admissible], ys[admissible], clearance[admissible]

        # walls and furniture are already in `admissible`; the route is all that is left
        route_bad = _route_clearance(xs, ys, route) < min_route_clearance
        n_rejected = int(route_bad.sum())

        if n_rejected == 0:
            verdicts[name] = Verdict(zone=name, ref_safe=True, admissible=n_admissible)
            continue

        verdicts[name] = Verdict(
            zone=name, ref_safe=False, admissible=n_admissible, rejected=n_rejected,
            reason=f"{n_rejected}/{n_admissible} resolvable points sit within "
                   f"{min_route_clearance} m of the robot's route",
        )

    return verdicts


def ref_safe_zones(verdicts: Mapping[str, Verdict]) -> frozenset[str]:
    return frozenset(name for name, v in verdicts.items() if v.ref_safe)
