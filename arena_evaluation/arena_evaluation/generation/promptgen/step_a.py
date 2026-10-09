"""Step 1 - Type A, the whole crowd: `blackout`, `evacuation` and `stampede` become `retune` (and `rally`) effects at the onset; `busy` injects pedestrians.
Agents created by later steps are not under A: the effects name the agents present now."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..scenario.populate import BUSY_MIX, extra_agents
from .agents import TypeFiles, agent_type_of, profile_of_agent
from .profiles import TIERS
from .spec import ADecision
from .world import WorldContext

#: Population floor a stampede needs; below it, `busy` runs first.
STAMPEDE_FLOOR = 6

_SPEED = {"low": TIERS["speed"][0] / TIERS["speed"][1], "high": TIERS["speed"][2] / TIERS["speed"][1]}
#: Everyone who speeds up also gets the engine's speed cap lifted, or 1.5x is a brisk walk.
_RUN = {"max_speed": "high"}


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
        effects.extend(retune_effects(doc, {"vision_range": "low"}, files, label="A:blackout", when=when, speed_scale=_SPEED["low"]))
        notes.append(f"blackout: {len(names)} agent(s) slow down and go short-sighted at {decision.when.as_dict()}")
        return effects, names, notes

    target = world.exit_zone() if case == "evacuation" else world.work_zone()
    if target is None:
        raise StepError(f"{case}: this world declares no {'exit' if case == 'evacuation' else 'work'} zone to rally on")
    params = {**_RUN, "repulsion": "low"} if case == "stampede" else dict(_RUN)
    effects.extend(retune_effects(doc, params, files, label=f"A:{case}", when=when, speed_scale=_SPEED["high"]))
    effects.append({"type": "rally", "label": f"A:{case}", "scope": names, "target": target, "when": when})
    notes.append(f"{case}: {len(names)} agent(s) speed up{' and stop yielding' if params else ''} and rally on {target} at {decision.when.as_dict()}")
    return effects, names, notes
