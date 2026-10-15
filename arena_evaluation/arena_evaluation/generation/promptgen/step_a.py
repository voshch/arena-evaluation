"""Step 1 - Type A, the whole crowd: `blackout`, `evacuation` and `stampede` change it at the onset; `busy` injects pedestrians.
HumanSim cannot change a running agent from outside, so every agent the change reaches becomes a behaviour tree: a walker's
route is baked into its own tree type (`walk_as_tree`). A `stimulus` raises a need at the onset, on which the type either
switches to a sequence that walks to the target holding the new parameters as its HumanSim `tune:`, or holds them over its
own sequences through a type-level `tunes:` entry. Agents created by later steps are not under A: the effects name the agents present now."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from ..scenario.populate import BUSY_MIX, extra_agents
from .agents import TypeFiles, agent_type_of, profile_of_agent
from .profiles import TIERS, is_behaviour_tree, tune_block
from .spec import ADecision
from .world import WorldContext

#: Population floor a stampede needs; below it, `busy` runs first.
STAMPEDE_FLOOR = 6

_SPEED = {"low": TIERS["speed"][0] / TIERS["speed"][1], "high": TIERS["speed"][2] / TIERS["speed"][1]}
#: Everyone who speeds up also gets the engine's speed cap lifted, or 1.5x is a brisk walk.
_RUN = {"max_speed": "high"}
#: The need an evacuation or a stampede raises on behaviour-tree agents; their derived type switches on it.
ALARM = "alarm"
#: The sequence a behaviour tree switches to, per rallying case.
SWITCH_TO = {"evacuation": "evacuate", "stampede": "rush"}


class StepError(RuntimeError):
    pass


def _agents(doc: dict[str, Any]) -> list[dict[str, Any]]:
    doc.setdefault("dynamic", [])
    return [a for a in doc["dynamic"] if isinstance(a, dict)]


def retune_effects(doc: dict[str, Any], params: Mapping[str, str], files: TypeFiles, *, label: str, when: dict[str, Any], speed_scale: float = 1.0, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """One `retune` effect per distinct agent type: at `when` its agents respawn where they stand on a derived profile with `params` pinned
    (speed scaled). Until then the population behaves as the base scenario wrote it."""
    groups: dict[str, list[str]] = {}
    derived_for: dict[str, str] = {}
    for agent in _agents(doc):
        name = str(agent["name"])
        if names is not None and name not in names:
            continue
        ref = agent_type_of(agent)
        if ref not in derived_for:
            try:
                derived_for[ref] = files.derived(ref, params, prof=profile_of_agent(agent)) if params else ref
            except Exception as exc:  # noqa: BLE001 - one odd type must not lose the crowd
                raise StepError(f"cannot derive {ref}: {exc}") from None
        groups.setdefault(ref, []).append(name)
    out = []
    for ref, members in groups.items():
        effect: dict[str, Any] = {"type": "retune", "label": label, "scope": members, "when": dict(when)}
        if derived_for[ref] != ref:
            effect["profile"] = derived_for[ref]
        if abs(speed_scale - 1.0) > 1e-9:
            effect["speed_scale"] = round(speed_scale, 3)
        out.append(effect)
    return out


def is_tree(files: TypeFiles, agent: Mapping[str, Any]) -> bool:
    return is_behaviour_tree(files.base_doc(agent_type_of(agent)))


def set_type(agent: dict[str, Any], ref: str) -> None:
    if isinstance(agent.get("agent"), dict):
        agent["agent"]["agent_type"] = ref
    else:
        agent["agent"] = {"agent_type": ref}


def walk_route(agent: Mapping[str, Any], world: WorldContext) -> list[tuple[float, float]]:
    """The walker's waypoints as points, drawn once where the runtime would draw them per episode: a zone stop is a random clear
    point in that zone (`WorldContext.zone_sample`), a point stop stays. `reverse` walks back the way it came; `random` is a fixed
    draw of stops with no stop twice in a row. Seeded by world, agent and route, so a scenario regenerates identically."""
    stops = list(agent.get("waypoints") or [])
    if not stops:
        raise StepError(f"{agent['name']}: a walker without waypoints has no route to bake")
    rng = random.Random(f"{world.name}:{agent['name']}:{stops}")
    mode = str(agent.get("waypoint_mode") or "repeat")
    if mode == "random" and len(stops) > 2:
        order = [rng.randrange(len(stops))]
        while len(order) < max(6, 2 * len(stops)):
            order.append(rng.choice([i for i in range(len(stops)) if i != order[-1]]))
        stops = [stops[i] for i in order]
    elif mode == "reverse":
        stops = [*stops, *stops[-2:0:-1]]
    return [world.zone_sample(s, rng) if isinstance(s, str) else (float(s[0]), float(s[1])) for s in stops]


def walk_as_tree(doc: dict[str, Any], files: TypeFiles, world: WorldContext, names: Sequence[str]) -> None:
    """The plain walkers among `names` on their own walking-tree types (`TypeFiles.walking`); trees are left as they are."""
    for agent in _agents(doc):
        if str(agent["name"]) in names and not is_tree(files, agent):
            set_type(agent, files.walking(agent, walk_route(agent, world)))


def tune_trees(doc: dict[str, Any], params: Mapping[str, str], files: TypeFiles, *, need: str, label: str, when: dict[str, Any], names: Sequence[str], speed_scale: float = 1.0) -> list[dict[str, Any]]:
    """The behaviour trees in `names` take a type that holds `params` once one `stimulus` raises `need` at `when`."""
    tuned_for: dict[str, str] = {}
    scope: list[str] = []
    for agent in _agents(doc):
        if str(agent["name"]) not in names:
            continue
        ref = agent_type_of(agent)
        if ref not in tuned_for:
            tune = tune_block(params, prof=profile_of_agent(agent), speed_scale=speed_scale)
            tuned_for[ref] = files.tuned(ref, need=need, tune=tune, params=params)
        set_type(agent, tuned_for[ref])
        scope.append(str(agent["name"]))
    return [{"type": "stimulus", "label": label, "scope": scope, "need": need, "when": dict(when)}] if scope else []


def inject_busy(doc: dict[str, Any], world: WorldContext, k: float, *, seed: int, prefix: str = "busy") -> list[str]:
    """Add pedestrians until the count is `k` times the current one (at least two more)."""
    present = len(_agents(doc))
    n = max(2, round(present * (k - 1.0))) if present else max(2, round(k * 3))
    added = extra_agents(n, zones=world.zones, robot=world.robot, homes=world.homes, seed=seed, world=world.name, mix=BUSY_MIX, prefix=prefix, existing=doc)
    if not added:
        raise StepError("busy: no zone to inject pedestrians into")
    doc["dynamic"] = [*doc.get("dynamic", []), *added]
    return [str(a["name"]) for a in added]


def apply(doc: dict[str, Any], decision: ADecision, world: WorldContext, files: TypeFiles, *, seed: int = 0) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Apply a Type A decision in place. Returns (effects, owned agent names, notes)."""
    effects: list[dict[str, Any]] = []
    notes: list[str] = []
    owned: list[str] = []
    case = decision.case

    if case == "busy":
        owned = inject_busy(doc, world, decision.k, seed=seed)
        notes.append(f"busy: injected {len(owned)} pedestrian(s) (k={decision.k:g})")
        return effects, owned, notes

    if not _agents(doc):
        raise StepError(f"{case}: the base population is empty - nothing to apply to")

    if case == "stampede" and len(_agents(doc)) < STAMPEDE_FLOOR:
        owned = inject_busy(doc, world, max(decision.k, 2.0), seed=seed)
        notes.append(f"stampede: population below {STAMPEDE_FLOOR}, injected {len(owned)} first")

    names = [str(a["name"]) for a in _agents(doc)]

    when = decision.when.as_effect()
    if case == "blackout":
        walk_as_tree(doc, files, world, names)
        params, scale = {"vision_range": "low"}, _SPEED["low"]
        effects.extend(tune_trees(doc, params, files, need=case, label="A:blackout", when=when, names=names, speed_scale=scale))
        notes.append(f"blackout: {len(names)} agent(s) slow down and go short-sighted at {decision.when.as_dict()}")
        return effects, names, notes

    target = world.exit_zone() if case == "evacuation" else world.work_zone()
    if target is None:
        raise StepError(f"{case}: this world declares no {'exit' if case == 'evacuation' else 'work'} zone to rally on")
    params = {**_RUN, "repulsion": "low"} if case == "stampede" else dict(_RUN)
    walk_as_tree(doc, files, world, names)
    # every agent switches itself on the alarm and walks to its own point in the target zone: a crowd sent to one point
    # jams on it (the demo's stampede gave each agent its own target too)
    for agent in _agents(doc):
        ref = agent_type_of(agent)
        x, y = world.zone_sample(target, random.Random(f"{world.name}:{agent['name']}:{case}:{target}"))
        steps = {"leave": {"kind": "go_to", "target_pose": {"x": round(x, 2), "y": round(y, 2), "theta": 0.0}, "patience": {"mean": 300.0}},
                 "stay": {"duration": {"mean": 600.0}, "patience": {"mean": 900.0}}}
        tune = tune_block(params, prof=profile_of_agent(agent), speed_scale=_SPEED["high"])
        shared = ref not in files.transient  # a walking tree is the agent's own already; a talker type is shared
        set_type(agent, files.switching(ref, need=ALARM, to=SWITCH_TO[case], steps=steps, tune=tune, tag=str(agent["name"]) if shared else ""))
    effects.append({"type": "stimulus", "label": f"A:{case}", "scope": names, "need": ALARM, "when": when})
    notes.append(f"{case}: {len(names)} agent(s) speed up{' and stop yielding' if case == 'stampede' else ''} and head for {target} at {decision.when.as_dict()}")
    return effects, names, notes
