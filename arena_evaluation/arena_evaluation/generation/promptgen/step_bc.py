"""Step 3 - Type B (tune some pedestrians) and Type C (one pedestrian against the robot)."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

from ..scenario.populate import for_people
from .agents import TypeFiles, agent_type_of, mesh_for, profile_of_agent
from .lexicon import when_in
from .llm import AnswerError, Oracle, call
from .profiles import BT_ONLY, ProfileError, is_behaviour_tree, profile, scaled_speed, tier_value
from .spec import A_CASES, BDecision, CDecision, Record, SchemaError, Selection, conflict_m, parse_b, parse_c
from .step_a import StepError, retune_effects
from .step_d import population_summary
from .world import WorldContext

_SYSTEM = Path(__file__).with_name("system_bc.md")


def system_prompt() -> str:
    return _SYSTEM.read_text()


def user_prompt(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, taken_fractions: Sequence[float], who: str = "") -> str:
    # Stable-prefix ordering (see step_d.user_prompt): world first, variable material last.
    parts = [world.context_for_model(), population_summary(doc, world, record.owned)]
    if who:
        parts.append(who)
    parts.append(f"prompt: {prompt}")
    decided = {k: v for k, v in record.decisions().items() if v and k in ("A", "D")}
    if decided:
        parts.append(f"already decided: {decided}")
    if taken_fractions:
        parts.append("route fractions already used by a placement (keep C at least 2 m away): " + ", ".join(f"{f:.2f}" for f in taken_fractions))
    return "\n\n".join(parts)


def decide(prompt: str, doc: Mapping[str, Any], world: WorldContext, record: Record, *, taken_fractions: Sequence[float], cache_dir: Path | None, model: str | None, oracle: Oracle | None, who: str = "") -> tuple[BDecision | None, CDecision | None]:
    system = system_prompt()
    user = user_prompt(prompt, doc, world, record, taken_fractions=taken_fractions, who=who)
    complaint = ""
    for attempt in range(2):
        answer = call(system, user + (f"\n\nYour previous answer was rejected: {complaint}. Answer again." if complaint else ""), cache_dir=cache_dir, model=model, oracle=oracle)
        record.provenance.setdefault("BC", []).append({"model": answer.model, "response_sha256": answer.sha256, "cached": answer.cached, "attempt": attempt})
        try:
            raw = answer.json()
            b = parse_b(raw.get("B"))
            if b is not None and not (isinstance(raw.get("B"), Mapping) and raw["B"].get("when")):
                # The model gave no onset: the prompt's own time phrase is the onset (the A
                # sentence keeps its own; C durations are "for ..." phrases, not onsets).
                spoken = when_in(prompt, exclude_cases=tuple(A_CASES), lexicon=None)
                if spoken is not None:
                    b = attrs.evolve(b, when=spoken)
                    record.warnings.append(f"B onset read from the prompt: {spoken.as_dict()}")
            c = parse_c(raw.get("C"))
            if c is not None:
                gap = conflict_m(world.route_length)
                for f in taken_fractions:
                    if abs(f - c.route_fraction) * world.route_length < gap:
                        if oracle is None:
                            raise SchemaError(f"conflict: C.route_fraction {c.route_fraction:g} is within {gap:g} m of a Type D placement at {f:.2f}")
                        # The oracle cannot answer a complaint: move C away from the placement along the route.
                        shift = (gap + 0.3) / max(world.route_length, 1e-9)
                        nudged = f + shift if c.route_fraction >= f else f - shift
                        if not 0.1 <= nudged <= 0.9:
                            nudged = f - shift if c.route_fraction >= f else f + shift
                        c = attrs.evolve(c, route_fraction=round(min(0.9, max(0.1, nudged)), 3))
                        record.warnings.append(f"C moved to route fraction {c.route_fraction:g} to clear the Type D placement at {f:.2f}")
            return b, c
        except (AnswerError, SchemaError) as exc:
            complaint = str(exc)
    raise StepError(f"schema: {complaint}")


def _agents(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [a for a in (doc.get("dynamic") or []) if isinstance(a, dict)]


def _zone_of_agent(agent: Mapping[str, Any], world: WorldContext) -> str:
    pose = agent.get("pose")
    if isinstance(pose, str):
        return pose
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return world.zone_of((float(pose[0]), float(pose[1]))) or ""
    return ""


def select(doc: Mapping[str, Any], sel: Selection, world: WorldContext, owned: set[str]) -> list[dict[str, Any]]:
    """The agents a Type B selection names, in scenario order, never one another step owns."""
    pool = [a for a in _agents(doc) if str(a.get("name")) not in owned]
    if sel.names:
        by_name = {str(a.get("name")): a for a in pool}
        missing = [n for n in sel.names if n not in by_name]
        if missing:
            raise StepError(f"applicability: {missing} are not unowned agents of this scenario")
        return [by_name[n] for n in sel.names]
    if sel.profile:
        pool = [a for a in pool if (p := profile_of_agent(a)) is not None and p.name == sel.profile]
    if sel.zone:
        pool = [a for a in pool if _zone_of_agent(a, world) == sel.zone]
    if sel.amount == "half":
        pool = pool[: max(1, math.ceil(len(pool) / 2))]
    elif sel.amount == "all_but_one":
        pool = pool[:-1]
    if sel.n:
        pool = pool[: sel.n]
    if not pool:
        raise StepError(f"applicability: selection {sel.as_dict()} matched nobody among the unowned agents")
    return pool


def apply_b(doc: dict[str, Any], decision: BDecision, world: WorldContext, files: TypeFiles, *, owned: set[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    chosen = select(doc, decision.select, world, owned)
    names = [str(a["name"]) for a in chosen]
    notes: list[str] = []
    effects: list[dict[str, Any]] = []
    label = f"B:{decision.case}"

    if decision.case == "erratic_route":
        effects.append({"type": "shuffle", "label": label, "scope": names, "period": decision.period, "radius": decision.radius, "when": decision.when.as_effect()})
        notes.append(f"erratic_route: {len(names)} agent(s) rerouted every {decision.period:g} s")
        return effects, names, notes

    if decision.case == "erratic_behaviour":
        bt, plain = [], []
        for agent in chosen:
            (bt if is_behaviour_tree(files.base_doc(agent_type_of(agent))) else plain).append(agent)
        if bt:
            effects.extend(retune_effects(doc, {"need_decay": "high"}, files, label=label, when=decision.when.as_effect(), names=[str(a["name"]) for a in bt]))
        if plain:
            effects.append({"type": "shuffle", "label": label, "scope": [str(a["name"]) for a in plain], "period": decision.period, "radius": decision.radius, "when": decision.when.as_effect()})
            notes.append(f"erratic_behaviour: {len(plain)} plain walker(s) fell back to erratic_route")
        if bt:
            notes.append(f"erratic_behaviour: {len(bt)} behaviour-tree agent(s) with need_decay high")
        return effects, names, notes

    params = dict(decision.params)
    # One retune per agent type at the onset: until then the selected agents walk as the base
    # scenario wrote them. A behaviour-only parameter is dropped for plain walkers.
    by_usable: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for agent in chosen:
        ref = agent_type_of(agent)
        try:
            base = files.base_doc(ref)
        except ProfileError as exc:
            raise StepError(f"intrinsic: {exc}") from None
        usable = {k: v for k, v in params.items() if k not in BT_ONLY or is_behaviour_tree(base)}
        dropped = sorted(set(params) - set(usable))
        if dropped:
            notes.append(f"intrinsic: {dropped} skipped on {agent['name']} (not a behaviour-tree type)")
        if usable:
            by_usable.setdefault(tuple(sorted(usable.items())), []).append(str(agent["name"]))
    for key, members in by_usable.items():
        usable = dict(key)
        typed = {k: v for k, v in usable.items() if k != "speed"}
        if usable.get("speed") == "high":
            typed["max_speed"] = "high"  # the engine's cap, or a high speed is a brisk walk
        try:
            group = retune_effects(doc, typed, files, label=label, when=decision.when.as_effect(), names=members)
        except StepError as exc:
            raise StepError(f"intrinsic: {exc}") from None
        if "speed" in usable:
            # The scalar speed in the agent block wins over the type, so a speed tier is set on
            # the effect, per agent profile.
            for effect in group:
                first = next(a for a in chosen if str(a["name"]) == effect["scope"][0])
                prof = profile_of_agent(first) or profile("adult")
                effect["speed"] = tier_value("speed", usable["speed"], prof=prof)
        effects.extend(group)
    notes.append(f"intrinsic: {params} on {len(names)} agent(s) at {decision.when.as_dict()}")
    return effects, names, notes


def apply_c(decision: CDecision, world: WorldContext, files: TypeFiles, *, seed: int, intercept_index: int, taken_fractions: Sequence[float]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rng = random.Random(f"{world.name}:C:{seed}")
    gap = conflict_m(world.route_length)
    for f in taken_fractions:
        if abs(f - decision.route_fraction) * world.route_length < gap:
            raise StepError(f"conflict: route fraction {decision.route_fraction:g} is within {gap:g} m of a Type D placement at {f:.2f}")
    prof = profile(decision.profile)
    ref = files.derived(prof.base, {}, prof=prof)
    label = f"C:{decision.case}"
    effect: dict[str, Any] = {"type": "intercept", "label": label, "profile": ref, "model": mesh_for(prof, rng), "route_fraction": decision.route_fraction}
    if decision.case == "blocking":
        if decision.mode == "stand":
            effect.update({"after": "stand", "duration": decision.duration, "speed": scaled_speed(prof, "mid")})
            note = f"blocking/stand: a {prof.name} stands on the route at {decision.route_fraction:g} for {decision.duration:g} s"
        else:
            effect.update({"angle": 0.0, "speed": tier_value("speed", "low", prof=prof), "after": "continue", "lead_out": 6.0, "waypoint_mode": "once"})
            note = f"blocking/lead: a {prof.name} crawls ahead of the robot from {decision.route_fraction:g}"
    else:
        effect.update({"angle": decision.angle, "speed": scaled_speed(prof, decision.speed), "after": decision.after, "duration": decision.duration, "offset": decision.offset, "waypoint_mode": "once"})
        if decision.after == "veer":
            if decision.veer_to not in world.polygons:
                raise StepError(f"approach/veer: {decision.veer_to!r} is not a zone of this world")
            if not for_people([decision.veer_to]):
                raise StepError(f"approach/veer: {decision.veer_to!r} is a room nobody walks into (wc/storage)")
            effect["veer_to"] = decision.veer_to
        note = f"approach: a {prof.name} at {decision.angle:g} deg, fraction {decision.route_fraction:g}, then {decision.after}"
    return [effect], [f"edge_{intercept_index}"], [note]
