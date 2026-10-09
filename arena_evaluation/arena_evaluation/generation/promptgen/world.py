"""What a world looks like to the generator: zones, furniture, the grid, the robot's planned route.

The route is planned offline over the runtime's occupancy grid (walls from `map.png`, furniture stamped in the way `WorldManager` does), so a placement made against it holds at reset."""

from __future__ import annotations

import heapq
import math
import typing
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import attrs

if typing.TYPE_CHECKING:
    import numpy as np

from ..formats import first_level
from ..scenario.occupancy import world_statics
from ..scenario.populate import MIN_ROUTE_CLEARANCE, _centroid, _polygon, clearance, robot_zones, route_points, work_zones, world_kind
from ..scenario.zone_screen import _load_grid, _rasterize, _route_clearance, _stamp_statics, ref_safe_zones, screen_zones
from ..scenario.zones import Role, zones_by_role

Point = tuple[float, float]

#: Metres the robot needs on either side to drive a cell; also the pedestrian body clearance.
ROBOT_CLEAR = 0.35
#: Preferred planning clearance: routes down the middle of a corridor, the way nav2's inflation
#: keeps the real robot, so placements around the route are not up against a wall.
PLAN_CLEAR = 0.6
#: Clearance the robot's start and goal get: nav2 will not plan into the inscribed radius plus inflation (mpo700: 0.48 m + 0.35 m),
#: and a goal 0.35 m from a desk stalled the robot 0.23 m short (2026-08-29). Falls back to `ROBOT_CLEAR` when no such cell is within 3 m.
ENDPOINT_CLEAR = 0.7
#: Planning cell size, metres. Coarser than the grid so A* over a hospital takes a second.
PLAN_STEP = 0.2
#: A planned leg shorter than this is replaced by the longest leg the zones offer, so a case has time to play out:
#: 30 s at 0.5 m/s, the base scenarios' own floor (`scenario.route.MIN_EPISODE_S`).
MIN_LEG_M = 15.0
#: The 60 s episode budget minus the case's reserve, at 0.5 m/s (mirrors `scenario.route.MAX_ROUTE_M`); a longer route is truncated there.
MAX_LEG_M = 20.0
#: Name fragments marking a zone a building empties through, best first.
EGRESS_HINTS = ("exit", "entrance", "entry", "lobby", "reception", "hallway", "corridor", "foyer", "living")


def _endpoint(grid: Grid, pt: Point) -> Point:
    if grid.free(pt, ENDPOINT_CLEAR):
        return pt
    return grid.nearest_free(pt, ENDPOINT_CLEAR, within=3.0) or grid.nearest_free(pt, ROBOT_CLEAR, within=3.0) or pt


def _plan(grid: Grid, start: Point, goal: Point) -> list[Point]:
    try:
        return grid.plan(start, goal, radius=PLAN_CLEAR)
    except WorldError:
        return grid.plan(start, goal)


def _length(route: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(route, route[1:], strict=False))


def _truncate(route: Sequence[Point], cap_m: float) -> list[Point]:
    """The prefix of `route` that is `cap_m` long, ending on an interpolated point of it."""
    out: list[Point] = [route[0]]
    left = cap_m
    for a, b in zip(route, route[1:], strict=False):
        leg = math.dist(a, b)
        if leg >= left:
            f = left / leg if leg > 0 else 0.0
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            return out
        out.append(b)
        left -= leg
    return list(route)


def _longest_leg(grid: Grid, start: Point, candidates: Sequence[Point], *, beat: float = 0.0) -> tuple[Point, list[Point]] | None:
    """The candidate goal (snapped into free space) with the longest drivable route from `start` up to `MAX_LEG_M`;
    None when none beats `beat` by a metre. A small world may never reach `MIN_LEG_M`; its longest leg is still the right one."""
    best: tuple[float, Point, list[Point]] | None = None
    for pt in candidates:
        goal = _endpoint(grid, pt)
        if math.dist(goal, start) < 1.0:
            continue
        try:
            route = _plan(grid, start, goal)
        except WorldError:
            continue
        length = _length(route)
        if length > MAX_LEG_M or length < beat + 1.0:
            continue
        if best is None or length > best[0]:
            best = (length, goal, route)
    return None if best is None else (best[1], best[2])


class WorldError(RuntimeError):
    pass


def _area(poly: Sequence[Point]) -> float:
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2.0


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    x, y = pt
    inside = False
    for (ax, ay), (bx, by) in zip(poly, [*poly[1:], poly[0]], strict=False):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / ((by - ay) or 1e-12) + ax:
            inside = not inside
    return inside


@attrs.define
class Grid:
    occupied: Any  # bool ndarray [rows, cols]
    res: float
    ox: float
    oy: float
    _labels_cache: dict[float, Any] = attrs.Factory(dict)

    @cached_property
    def dist(self) -> np.ndarray:
        import scipy.ndimage

        return scipy.ndimage.distance_transform_edt(~self.occupied) * self.res

    def labels_at(self, radius: float) -> np.ndarray:
        import scipy.ndimage

        key = round(radius, 2)
        if key not in self._labels_cache:
            self._labels_cache[key], _n = scipy.ndimage.label(self.dist >= key)
        return self._labels_cache[key]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.occupied.shape)  # type: ignore[return-value]

    def cell(self, pt: Point) -> tuple[int, int]:
        rows = self.shape[0]
        return rows - 1 - int((pt[1] - self.oy) / self.res), int((pt[0] - self.ox) / self.res)

    def xy(self, cell: tuple[int, int]) -> Point:
        """The centre of a cell, so `cell(xy(c)) == c` regardless of float rounding."""
        rows = self.shape[0]
        return ((cell[1] + 0.5) * self.res + self.ox, (rows - 1 - cell[0] + 0.5) * self.res + self.oy)

    def inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def clearance(self, pt: Point) -> float:
        cell = self.cell(pt)
        return float(self.dist[cell]) if self.inside(cell) else 0.0

    def free(self, pt: Point, radius: float = ROBOT_CLEAR) -> bool:
        return self.clearance(pt) >= radius

    def component(self, pt: Point, radius: float = ROBOT_CLEAR) -> int:
        cell = self.cell(pt)
        return int(self.labels_at(min(radius, ROBOT_CLEAR))[cell]) if self.inside(cell) else 0

    def nearest_free(self, pt: Point, radius: float, within: float) -> Point | None:
        import numpy as np

        r0, c0 = self.cell(pt)
        k = int(math.ceil(within / self.res))
        rows, cols = self.shape
        rs, re = max(0, r0 - k), min(rows, r0 + k + 1)
        cs, ce = max(0, c0 - k), min(cols, c0 + k + 1)
        if rs >= re or cs >= ce:
            return None
        window = self.dist[rs:re, cs:ce] >= radius
        if not window.any():
            return None
        rr, cc = np.nonzero(window)
        d = np.hypot(rr + rs - r0, cc + cs - c0)
        i = int(np.argmin(d))
        if d[i] * self.res > within:
            return None
        return self.xy((int(rr[i] + rs), int(cc[i] + cs)))

    def line_free(self, a: Point, b: Point, radius: float = ROBOT_CLEAR, step: float = 0.1) -> bool:
        n = max(1, int(math.dist(a, b) / step))
        return all(self.free((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n), radius) for i in range(n + 1))

    def plan(self, start: Point, goal: Point, radius: float = ROBOT_CLEAR, *, fine: bool = False) -> list[Point]:
        """A* on a `PLAN_STEP` lattice, string-pulled into a polyline; raises WorldError when unreachable.
        A failed search is retried at full map resolution: the lattice straddles passages barely wider than the robot (2026-08-30)."""
        k = 1 if fine else max(1, int(round(PLAN_STEP / self.res)))
        s = self._snap(start, radius, k)
        g = self._snap(goal, radius, k)
        if s is None or g is None:
            raise WorldError(f"no free cell near {'start' if s is None else 'goal'} {start if s is None else goal}")
        if s == g:
            return [start, goal]

        def h(c: tuple[int, int]) -> float:
            return math.hypot(c[0] - g[0], c[1] - g[1])

        rows, cols = self.shape
        best: dict[tuple[int, int], float] = {s: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        heap: list[tuple[float, tuple[int, int]]] = [(h(s), s)]
        while heap:
            _f, cur = heapq.heappop(heap)
            if cur == g:
                break
            cg = best[cur]
            for dr in (-k, 0, k):
                for dc in (-k, 0, k):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (cur[0] + dr, cur[1] + dc)
                    if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or self.dist[nxt] < radius:
                        continue
                    if dr and dc and (self.dist[cur[0] + dr, cur[1]] < radius or self.dist[cur[0], cur[1] + dc] < radius):
                        continue  # no corner cutting through a wall
                    ng = cg + math.hypot(dr, dc)
                    if ng < best.get(nxt, math.inf):
                        best[nxt] = ng
                        came[nxt] = cur
                        heapq.heappush(heap, (ng + h(nxt), nxt))
        if g not in came and g != s:
            if k > 1:
                return self.plan(start, goal, radius, fine=True)
            raise WorldError(f"the robot cannot reach {goal} from {start}")
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        pts = [self.xy(c) for c in reversed(cells)]
        pts[0], pts[-1] = start, goal
        return self._pull(pts, radius)

    def _snap(self, pt: Point, radius: float, k: int) -> tuple[int, int] | None:
        near = self.nearest_free(pt, radius, within=2.0)
        if near is None:
            return None
        r, c = self.cell(near)
        r0, c0 = (r // k) * k, (c // k) * k
        rows, cols = self.shape
        for ring in range(0, 12):
            best: tuple[int, int] | None = None
            best_d = math.inf
            for dr in range(-ring, ring + 1):
                for dc in range(-ring, ring + 1):
                    if max(abs(dr), abs(dc)) != ring:
                        continue
                    cand = (r0 + dr * k, c0 + dc * k)
                    if 0 <= cand[0] < rows and 0 <= cand[1] < cols and self.dist[cand] >= radius:
                        d = math.hypot(cand[0] - r, cand[1] - c)
                        if d < best_d:
                            best, best_d = cand, d
            if best is not None:
                return best
        return None

    def _pull(self, pts: list[Point], radius: float) -> list[Point]:
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j], radius):
                j -= 1
            out.append(pts[j])
            i = j
        return out


@attrs.define
class WorldContext:
    name: str
    root: Path
    zones: list[dict[str, Any]]
    roles: dict[str, Role]
    robot: dict[str, Any]
    grid: Grid | None
    start: Point
    goal: Point
    route: list[Point]
    homes: frozenset[str]
    kind: str

    @classmethod
    def load(cls, worlds_root: str | Path, world: str, base_doc: Mapping[str, Any], overrides: Mapping[str, str] | None = None) -> WorldContext:
        root = Path(worlds_root) / world
        doc = first_level(root)
        if doc is None:
            raise WorldError(f"{world}: no world.yaml")
        zones = [z for z in doc.get("zones") or [] if isinstance(z, dict) and z.get("name")]
        robots = base_doc.get("robots") or []
        if not robots or not isinstance(robots[0], dict) or robots[0].get("start") is None:
            raise WorldError(f"{world}: the base scenario declares no robot")
        robot = dict(robots[0])
        pts = route_points(robot, zones)
        if len(pts) < 2:
            raise WorldError(f"{world}: the robot's route has fewer than two resolvable endpoints")
        start, goal = pts[0], pts[1]

        statics = world_statics(root)
        loaded = _load_grid(root)
        grid: Grid | None = None
        if loaded is not None:
            occupied, res, ox, oy = loaded
            occupied = occupied.copy()
            _stamp_statics(occupied, statics, res, ox, oy)
            grid = Grid(occupied=occupied, res=res, ox=ox, oy=oy)
        route = [start, goal]
        if grid is not None:
            # a zone centroid may sit in a table; snap so the route and the reachability reference are real
            start = _endpoint(grid, start)
            goal = _endpoint(grid, goal)
            route = _plan(grid, start, goal)
            if _length(route) > MAX_LEG_M:
                # over the episode budget: keep the start, cut where the budget runs out
                cut = _truncate(route, MAX_LEG_M)
                goal = _endpoint(grid, cut[-1])
                replanned = _plan(grid, start, goal)
                route = replanned if _length(replanned) <= MAX_LEG_M * 1.2 else cut
            elif _length(route) < MIN_LEG_M:
                longer = _longest_leg(grid, start, [_centroid(_polygon(z)) for z in zones if len(_polygon(z)) >= 3], beat=_length(route))
                if longer is not None:
                    goal, route = longer

        roles = {}
        for role, names in zones_by_role(zones, overrides).items():
            for n in names:
                roles[n] = role
        try:
            verdicts = screen_zones(zones, world_root=root, route=pts, statics=statics, min_route_clearance=MIN_ROUTE_CLEARANCE, robot_zones=robot_zones(robot))
            homes = ref_safe_zones(verdicts)
        except Exception:  # screening is an optimisation; placement is still validated
            homes = frozenset()
        return cls(name=world, root=root, zones=zones, roles=roles, robot=robot, grid=grid,
                   start=start, goal=goal, route=route, homes=homes, kind=world_kind(zones, overrides))

    @cached_property
    def polygons(self) -> dict[str, list[Point]]:
        return {str(z["name"]): _polygon(z) for z in self.zones if len(_polygon(z)) >= 3}

    @cached_property
    def centroids(self) -> dict[str, Point]:
        return {name: _centroid(poly) for name, poly in self.polygons.items()}

    def zone_names(self) -> list[str]:
        return sorted(self.polygons)

    def zone_of(self, pt: Point) -> str | None:
        for name, poly in self.polygons.items():
            if point_in_polygon(pt, poly):
                return name
        return None

    def zone_point(self, name: str, *, need: float = 0.4) -> Point:
        """A point in the zone where something of radius `need` may stand: nearest the centroid among reachable cells clear of walls
        and away from the robot's endpoints, preferring cells clear of its route. The bare centroid when the world has no grid."""
        if name not in self.centroids:
            raise WorldError(f"{name!r} is not a zone of {self.name}; known: {self.zone_names()}")
        c = self.centroids[name]
        grid = self.grid
        if grid is None:
            return c
        import numpy as np

        mask = _rasterize(self.polygons[name], grid.shape, grid.res, grid.ox, grid.oy)
        reachable = grid.labels_at(ROBOT_CLEAR) == grid.component(self.start)
        # Relax the clearance in steps: a room full of furniture may hold a formation only
        # where its members stand between the desks.
        for clear in (need + 0.3, need, max(0.5, need / 2)):
            ok = mask & (grid.dist >= clear) & reachable
            rows, cols = np.nonzero(ok)
            if rows.size == 0:
                continue
            xs = (cols + 0.5) * grid.res + grid.ox
            ys = (grid.shape[0] - 1 - rows + 0.5) * grid.res + grid.oy
            far = (np.hypot(xs - self.start[0], ys - self.start[1]) >= 1.8 + need) & (np.hypot(xs - self.goal[0], ys - self.goal[1]) >= 1.3 + need)
            keep = far & (_route_clearance(xs, ys, self.route) >= 1.5 + need)
            if not keep.any():
                keep = far
            if not keep.any():
                continue
            d = np.hypot(xs - c[0], ys - c[1])
            d[~keep] = np.inf
            i = int(np.argmin(d))
            return (float(xs[i]), float(ys[i]))
        raise WorldError(f"zone {name!r} has no room for a formation of radius {need:.1f} m away from the robot's endpoints")

    def exit_zone(self) -> str | None:
        transit = sorted(n for n, r in self.roles.items() if r is Role.TRANSIT and n in self.polygons)
        if transit:
            return transit[0]
        for hint in EGRESS_HINTS:
            matched = sorted(n for n in self.polygons if hint in n.lower())
            if matched:
                return matched[0]
        return max(self.polygons, key=lambda n: (_area(self.polygons[n]), n)) if self.polygons else None

    def work_zone(self) -> str | None:
        wanted = [n for n in work_zones(self.zones) if n in self.polygons]
        return max(wanted, key=lambda n: (round(_area(self.polygons[n]), 1), "work" in n.lower(), n)) if wanted else None

    def other_zones(self, exclude: Sequence[str] = ()) -> list[str]:
        """Zones a walker may head for: not private, not the robot's own."""
        forbidden = set(exclude) | set(robot_zones(self.robot))
        return [n for n in self.zone_names() if self.roles.get(n) is not Role.PRIVATE and n not in forbidden]

    @property
    def route_length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.route, self.route[1:], strict=False))

    def point_at(self, fraction: float) -> tuple[Point, Point]:
        """Position and unit heading `fraction` of the way along the planned route."""
        remaining = max(0.0, min(1.0, fraction)) * self.route_length
        heading: Point = (1.0, 0.0)
        for a, b in zip(self.route, self.route[1:], strict=False):
            seg = math.dist(a, b)
            if seg <= 1e-9:
                continue
            heading = ((b[0] - a[0]) / seg, (b[1] - a[1]) / seg)
            if remaining <= seg:
                f = remaining / seg
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), heading
            remaining -= seg
        return self.route[-1], heading

    def fraction_of(self, pt: Point) -> float:
        """Arc-length fraction of the route point nearest `pt`."""
        best, best_d, acc = 0.0, math.inf, 0.0
        for a, b in zip(self.route, self.route[1:], strict=False):
            seg = math.dist(a, b)
            if seg <= 1e-9:
                continue
            t = max(0.0, min(1.0, ((pt[0] - a[0]) * (b[0] - a[0]) + (pt[1] - a[1]) * (b[1] - a[1])) / (seg * seg)))
            d = math.dist(pt, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if d < best_d:
                best, best_d = (acc + t * seg) / max(self.route_length, 1e-9), d
            acc += seg
        return best

    def route_clearance(self, pt: Point) -> float:
        return clearance(pt, self.route)

    def passage_width(self, pt: Point, heading: Point, limit: float = 8.0) -> tuple[float, Point, Point]:
        """Width of free space through `pt` perpendicular to `heading`, and the two wall points."""
        if self.grid is None:
            return limit, (pt[0] - heading[1] * limit / 2, pt[1] + heading[0] * limit / 2), (pt[0] + heading[1] * limit / 2, pt[1] - heading[0] * limit / 2)
        normal = (-heading[1], heading[0])
        ends = []
        for sign in (1.0, -1.0):
            d = 0.0
            while d < limit and self.grid.clearance((pt[0] + sign * normal[0] * d, pt[1] + sign * normal[1] * d)) > 0.05:
                d += 0.05
            ends.append((pt[0] + sign * normal[0] * d, pt[1] + sign * normal[1] * d))
        return math.dist(ends[0], ends[1]), ends[0], ends[1]

    def narrowest_passage(self, fraction: float, span: float = 3.0, step: float = 0.25) -> tuple[Point, Point, float]:
        """The narrowest cross-section of the route within `span` metres of `fraction`.
        Returns (point, heading, width)."""
        total = max(self.route_length, 1e-9)
        samples: list[tuple[float, Point, Point, float]] = []
        n = int(span / step)
        for i in range(-n, n + 1):
            f = fraction + i * step / total
            if not 0.05 < f < 0.95:
                continue
            pt, heading = self.point_at(f)
            if math.dist(pt, self.start) < 2.1 or math.dist(pt, self.goal) < 1.6:
                continue
            width, _l, _r = self.passage_width(pt, heading)
            samples.append((abs(i) * step, pt, heading, width))
        if not samples:
            pt, heading = self.point_at(fraction)
            return pt, heading, self.passage_width(pt, heading)[0]
        narrowest = min(w for _d, _p, _h, w in samples)
        # Among cross-sections within 0.2 m of the narrowest, the one nearest the request.
        _d, pt, heading, width = min((s for s in samples if s[3] <= narrowest + 0.2), key=lambda s: s[0])
        return pt, heading, width

    def fit_along_route(self, fraction: float, need: float, *, span: float = 0.3, step: float = 0.25) -> tuple[Point, Point, float] | None:
        """The nearest point to `fraction` on the route where a formation of radius `need` fits:
        the passage is wide enough, the point is clear of walls, and it keeps the start/goal
        distances. Returns (point, heading, fraction), or None when nothing on the route fits."""
        total = max(self.route_length, 1e-9)
        n = int(span * total / step)
        for i in range(0, n + 1):
            for sign in ((1, -1) if i else (1,)):
                f = fraction + sign * i * step / total
                if not 0.05 < f < 0.95:
                    continue
                pt, heading = self.point_at(f)
                if self.grid is not None:
                    if self.grid.clearance(pt) < need + 0.3:
                        continue
                    width, _l, _r = self.passage_width(pt, heading)
                    if width < 2 * need + 0.8:
                        continue
                if math.dist(pt, self.start) < 1.8 + need or math.dist(pt, self.goal) < 1.3 + need:
                    continue
                return pt, heading, f
        return None

    def valid(self, pt: Point, *, radius: float = 0.4, on_route: bool = False) -> str | None:
        """None when a pedestrian may stand at `pt`; otherwise why not."""
        if self.grid is not None:
            if not self.grid.inside(self.grid.cell(pt)):
                return "outside the map"
            if not self.grid.free(pt, radius):
                return f"{self.grid.clearance(pt):.2f} m from a wall or furniture"
            if self.grid.component(pt, radius) != self.grid.component(self.start, radius):
                return "not reachable from the robot's start"
        if math.dist(pt, self.start) < 1.5:
            return "within 1.5 m of the robot's start"
        if math.dist(pt, self.goal) < 1.0:
            return "within 1.0 m of the robot's goal"
        if not on_route and self.route_clearance(pt) < 1.5:
            return "within 1.5 m of the robot's route"
        return None

    def resolve(self, pt: Point, *, radius: float = 0.4, on_route: bool = False, within: float = 1.5) -> tuple[Point, str]:
        """`pt` if valid, else the nearest valid point within `within`. Raises when none."""
        why = self.valid(pt, radius=radius, on_route=on_route)
        if why is None:
            return pt, ""
        if self.grid is not None:
            for r in (0.5, 1.0, within):
                near = self.grid.nearest_free(pt, max(radius, ROBOT_CLEAR), within=r)
                if near is not None and self.valid(near, radius=radius, on_route=on_route) is None:
                    return near, f"moved {math.dist(pt, near):.2f} m ({why})"
        raise WorldError(f"no valid point within {within} m of ({pt[0]:.2f}, {pt[1]:.2f}): {why}")

    def context_for_model(self) -> str:
        """The world as the model sees it: zones with roles and centroids, the route, doors."""
        lines = [f"world: {self.name} ({self.kind})", "zones (name, role, centroid):"]
        for name in self.zone_names():
            c = self.centroids[name]
            lines.append(f"  - {name}: {self.roles.get(name, Role.UNKNOWN).value} at ({c[0]:.1f}, {c[1]:.1f})")
        lines.append(f"robot: start ({self.start[0]:.1f}, {self.start[1]:.1f}) in {self.zone_of(self.start) or '?'} -> goal ({self.goal[0]:.1f}, {self.goal[1]:.1f}) in {self.zone_of(self.goal) or '?'}, route {self.route_length:.1f} m")
        marks = []
        for f in (0.25, 0.5, 0.75):
            pt, _h = self.point_at(f)
            marks.append(f"{f:.2f} -> ({pt[0]:.1f}, {pt[1]:.1f}) in {self.zone_of(pt) or '?'}")
        lines.append("route fractions: " + "; ".join(marks))
        return "\n".join(lines)
