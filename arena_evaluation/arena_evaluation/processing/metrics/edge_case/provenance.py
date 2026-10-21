"""The generator's provenance.yaml beside the scenario an episode ran, found through the world's scenario resolver."""

from __future__ import annotations

import logging
import typing
from collections.abc import Mapping
from pathlib import Path

import attrs
import yaml

from .criticality import Point

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World import WorldView

    from arena_evaluation.storage.schemas import RunMetadata

logger = logging.getLogger(__name__)

PROVENANCE_FILE = "provenance.yaml"
OWN_STEPS = ("C", "D", "injected")
FORMATION_RADIUS_M = 1.5


@attrs.frozen
class Formation:
    center: Point | None
    radius: float
    variant: str


@attrs.frozen
class Encounter:
    agent: str
    point: Point


@attrs.frozen
class Provenance:
    id: str
    steps: tuple[str, ...]
    owned: dict[str, tuple[str, ...]]
    formation: Formation | None
    encounters: tuple[Encounter, ...] = ()
    hold_s: float = 0.0

    @property
    def agents(self) -> list[str]:
        """The case's own agents, the ones its C and D steps own and the ones it injected."""
        return sorted({name for step in OWN_STEPS for name in self.owned.get(step, ())})


def parse_provenance(raw: Mapping[str, typing.Any], origin: Point = (0.0, 0.0)) -> Provenance:
    """A provenance mapping with the formation center and encounter points moved from their level frame by `origin` into the map frame."""
    owned = {str(step): tuple(str(name) for name in names or ()) for step, names in (raw.get("owned") or {}).items()}
    spec = raw.get("formation")
    formation = None
    if spec:
        center = spec.get("center")
        formation = Formation(
            center=None if center is None else (float(center[0]) + origin[0], float(center[1]) + origin[1]),
            radius=float(spec.get("radius", FORMATION_RADIUS_M)),
            variant=str(spec.get("variant", "")),
        )
    encounters = tuple(Encounter(agent=str(e["agent"]), point=(float(e["point"][0]) + origin[0], float(e["point"][1]) + origin[1])) for e in raw.get("encounters") or ())
    hold_s = float((raw.get("robot") or {}).get("hold_s") or 0.0)
    return Provenance(id=str(raw.get("id", "")), steps=tuple(str(s) for s in raw.get("steps") or ()), owned=owned, formation=formation, encounters=encounters, hold_s=hold_s)


def scenario_name(run: RunMetadata | None) -> str | None:
    """The scenario file the episode's `scenario` task mode ran, None when neither axis ran one."""
    if run is None:
        return None
    for mode, params in ((run.tm_obstacles, run.obstacles_params), (run.tm_robots, run.robots_params)):
        if mode == "scenario" and params and params.get("file"):
            return str(params["file"])
    return None


_world_views: dict[str, WorldView | None] = {}
_provenance_cache: dict[tuple[str, str], Provenance | None] = {}


def world_view(world: str) -> WorldView | None:
    """The resolved world, None when it is not available locally."""
    if world not in _world_views:
        from arena_simulation_setup.tree.World import WorldIdentifier

        try:
            _world_views[world] = WorldIdentifier(world).resolve_sync()
        except FileNotFoundError as e:
            logger.warning("edge_case: world '%s' not available locally: %s", world, e)
            _world_views[world] = None
    return _world_views[world]


def _level_origin(view: WorldView, scenario: Mapping[str, typing.Any]) -> Point:
    origins = view.level_origins()
    if not origins:
        return (0.0, 0.0)
    robots = scenario.get("robots") or []
    level = str(robots[0].get("level")) if robots and robots[0].get("level") is not None else next(iter(origins))
    return origins.get(level, (0.0, 0.0))


def load_provenance(world: str, scenario: str) -> Provenance | None:
    """The provenance of `scenario` in `world`, None when the scenario or its provenance.yaml is not available."""
    key = (world, scenario)
    if key in _provenance_cache:
        return _provenance_cache[key]
    view = world_view(world)
    result = None
    if view is not None:
        try:
            scenario_view = view.scenario(scenario).resolve_sync()
        except FileNotFoundError as e:
            logger.warning("edge_case: scenario '%s' of world '%s' not available locally: %s", scenario, world, e)
        else:
            path = Path(scenario_view.path) / PROVENANCE_FILE
            if path.is_file():
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                result = parse_provenance(raw, _level_origin(view, scenario_view.read()))
    _provenance_cache[key] = result
    return result
