"""prompt + world + base scenario -> `scenarios/<base>__<prompt_id>/`.

Steps A, D and B+C run in that order over one decision record, so no step re-reads a sentence another realised; an agent belongs to at most one step."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs

from ..formats import write_scenario
from ..scenario.materialize import GENERATED_MARKER
from .agents import TypeFiles
from .lexicon import match as lexicon_match
from .llm import Oracle, Unavailable
from .prompts import Prompt
from .spec import Failure, Record, SchemaError
from .step_a import StepError
from .step_a import apply as apply_a
from .step_bc import apply_b, apply_c
from .step_bc import decide as decide_bc
from .step_d import apply as apply_d
from .step_d import decide as decide_d
from .world import WorldContext, WorldError

#: Type A cases that run on an empty base; the others need a crowd.
A_ON_EMPTY = frozenset({"busy"})
GENERATOR_VERSION = "promptgen-1"


def denied_on_empty(targets: tuple[str, ...]) -> list[str]:
    """Targets that cannot apply to an empty base: Type B and every Type A but `busy` act on the crowd; C and D generate their own pedestrians."""
    return [t for t in targets if t.startswith("B:") or (t.startswith("A:") and t.split(":", 1)[1] not in A_ON_EMPTY)]


@attrs.define
class Result:
    record: Record
    doc: dict[str, Any]
    files: TypeFiles
    effects: list[dict[str, Any]]
    scenario_id: str
    prompt: Prompt | None = None
    written: Path | None = None

    @property
    def block(self) -> dict[str, Any]:
        r = self.record
        block: dict[str, Any] = {
            "id": self.scenario_id,
            "base": r.base,
            "prompt_id": r.prompt_id,
            "steps": r.steps,
            "note": r.prompt,
            "effects": self.effects,
            "decisions": r.decisions(),
            "owned": {k: list(v) for k, v in r.owned.items()},
            **({"formation": dict(r.formation)} if r.formation else {}),
            "provenance": {**r.provenance, "generator": GENERATOR_VERSION, "status": r.status, "dedup_key": r.dedup_key()},
        }
        return block


def _fail(record: Record, step: str, exc: Exception) -> None:
    text = str(exc)
    kind = (
        text.split(":", 1)[0] if ":" in text and text.split(":", 1)[0] in ("schema", "placement", "conflict", "applicability", "infeasible", "unavailable") else "schema" if isinstance(exc, SchemaError) else "placement" if isinstance(exc, WorldError) else "unavailable" if isinstance(exc, Unavailable) else "infeasible"
    )
    record.failures.append(Failure(step=step, kind=kind, reason=text))


def _oracle_for(prompt: Prompt, key: str) -> Oracle:
    """The prompt's own slots as the model's answer."""
    if key == "D":
        return Oracle(prompt.oracle.get("D") or {"case": "none"})
    return Oracle({"B": prompt.oracle.get("B") or "none", "C": prompt.oracle.get("C") or "none"})


def pin_robot_route(doc: dict[str, Any], world: WorldContext, record: Record) -> None:
    """Pin the robot's start and goal to the planned route's endpoints, the start facing along it.
    Zone endpoints resolve to random wall-checked points at run time and would move the route every placement here was made against."""
    robots = doc.get("robots") or []
    if not robots or not isinstance(robots[0], dict) or world.grid is None or len(world.route) < 2:
        return
    (sx, sy), (gx, gy) = world.start, world.goal
    nx, ny = world.route[1]
    yaw = math.atan2(ny - sy, nx - sx)
    robots[0] = {**robots[0], "start": [round(sx, 2), round(sy, 2), round(yaw, 3)], "phases": [{"goto": [round(gx, 2), round(gy, 2)]}]}
    doc["robots"] = robots
    record.warnings.append(f"robot route pinned to the planned one: ({sx:.1f}, {sy:.1f}) -> ({gx:.1f}, {gy:.1f})")


def generate(
    prompt: Prompt,
    *,
    world: WorldContext,
    base_id: str,
    base_doc: Mapping[str, Any],
    base_dir: Path,
    seed: int = 0,
    cache_dir: Path | None = None,
    model: str | None = None,
    oracle: bool = False,
) -> Result:
    doc = copy.deepcopy(dict(base_doc))
    doc.pop("edge_case", None)
    files = TypeFiles(base_dir=Path(base_dir))
    record = Record(prompt_id=prompt.id, prompt=prompt.text, world=world.name, base=base_id, targets=tuple(prompt.targets))
    pin_robot_route(doc, world, record)
    effects: list[dict[str, Any]] = []
    owned: set[str] = set()
    intercepts = 0
    taken_fractions: list[float] = []
    populated = bool(doc.get("dynamic"))

    # nothing targeted can apply to an empty base: a non-experiment, denied before any model call
    denied = denied_on_empty(prompt.targets)
    if not populated and denied and len(denied) == len(prompt.targets):
        record.failures.append(Failure(step="base", kind="denied", reason=f"empty base: {', '.join(denied)} need a crowd and nothing else is targeted"))
        return Result(record=record, doc=doc, files=files, effects=[], scenario_id=f"{base_id}__{prompt.id}", prompt=prompt)

    # in live mode a cheap model also reads A, through typos and paraphrases, and extracts the `who` the later steps share; every disagreement with the lexicon is recorded
    a = lexicon_match(prompt.text)
    who = ""
    if oracle is False:
        from . import step_a_llm  # live mode only

        try:
            a_llm, who_block, notes = step_a_llm.decide(prompt.text, record, cache_dir=cache_dir)
            record.warnings.extend(notes)
            if who_block:
                record.provenance["who"] = who_block
                who = step_a_llm.who_line(who_block)
            if a is None and a_llm is not None:
                record.warnings.append(f"A read by the model (lexicon silent): {a_llm.case}")
                a = a_llm
            elif a is not None and a_llm is None:
                record.warnings.append(f"lexicon reads A:{a.case}, the model reads none; keeping the lexicon")
            elif a is not None and a_llm is not None and a_llm.case != a.case:
                record.warnings.append(f"A disagreement: lexicon {a.case}, model {a_llm.case}; taking the model")
                a = a_llm
        except (StepError, Unavailable) as exc:
            record.warnings.append(f"A model unavailable, lexicon only: {exc}")
    wanted_a = [t for t in prompt.targets if t.startswith("A:")]
    if a is None and wanted_a:
        record.failures.append(Failure(step="A", kind="schema", reason=f"prompt targets {wanted_a[0]} but the lexicon matched nothing"))
    if a is not None and wanted_a and f"A:{a.case}" != wanted_a[0]:
        record.warnings.append(f"lexicon matched {a.case}, prompt targets {wanted_a[0]}")
    if a is not None and not populated and a.case not in A_ON_EMPTY:
        record.failures.append(Failure(step="A", kind="denied", reason=f"{a.case} needs a crowd and the base is empty"))
        a = None
    if a is not None:
        try:
            new, names, notes = apply_a(doc, a, world, files, seed=seed)
            effects.extend(new)
            owned.update(names)
            record.owned["A"] = list(names)
            record.a = a
            record.warnings.extend(notes)
        except StepError as exc:
            _fail(record, "A", exc)

    try:
        d = decide_d(prompt.text, doc, world, record, cache_dir=cache_dir, model=model, oracle=_oracle_for(prompt, "D") if oracle else None, who=who)
    except (StepError, Unavailable) as exc:
        _fail(record, "D", exc)
        d = None
    if d is not None:
        try:
            new, names, notes, info = apply_d(doc, d, world, files, seed=seed, owned=owned, intercept_index=intercepts)
            intercepts += sum(1 for e in new if e["type"] == "intercept")
            effects.extend(new)
            owned.update(names)
            record.owned["D"] = list(names)
            record.d = d
            record.warnings.extend(notes)
            if info.get("formation"):
                record.formation = dict(info["formation"])
            if d.fraction is not None:
                taken_fractions.append(d.fraction)
        except StepError as exc:
            _fail(record, "D", exc)

    try:
        b, c = decide_bc(prompt.text, doc, world, record, taken_fractions=taken_fractions, cache_dir=cache_dir, model=model, oracle=_oracle_for(prompt, "BC") if oracle else None, who=who)
    except (StepError, Unavailable) as exc:
        _fail(record, "BC", exc)
        b = c = None
    if b is not None:
        # on an empty base every agent present was generated by A or D, so B is denied, not inapplicable
        if not populated:
            record.failures.append(Failure(step="B", kind="denied", reason="empty base: nobody to tune"))
        elif not [x for x in (doc.get("dynamic") or []) if isinstance(x, dict) and str(x.get("name")) not in owned]:
            record.failures.append(Failure(step="B", kind="applicability", reason="no unowned pedestrian to tune"))
        else:
            try:
                new, names, notes = apply_b(doc, b, world, files, owned=owned)
                effects.extend(new)
                owned.update(names)
                record.owned["B"] = list(names)
                record.b = b
                record.warnings.extend(notes)
            except StepError as exc:
                _fail(record, "B", exc)
    if c is not None:
        try:
            new, names, notes = apply_c(c, world, files, seed=seed, intercept_index=intercepts, taken_fractions=taken_fractions)
            intercepts += 1
            effects.extend(new)
            owned.update(names)
            record.owned["C"] = list(names)
            record.c = c
            record.warnings.extend(notes)
        except StepError as exc:
            _fail(record, "C", exc)

    return Result(record=record, doc=doc, files=files, effects=effects, scenario_id=f"{base_id}__{prompt.id}", prompt=prompt)


def _header(result: Result) -> str:
    r = result.record
    return "\n".join(
        (
            f"# Prompt scenario: {result.scenario_id}",
            f"# prompt: {r.prompt}",
            f"# base: {r.base} · steps: {', '.join(r.steps) or 'none'} · status: {r.status}",
            "#",
            GENERATED_MARKER,
            "# edge_case.yaml lists the runtime effects; everything else is baked into the population.",
            "",
            "",
        )
    )


def write_result(result: Result, world_root: Path, *, keep_partial: bool = True) -> Path | None:
    """Write `scenarios/<base>__<prompt_id>/`. None when the record is infeasible, denied, or partial without `keep_partial`."""
    if result.record.status in ("infeasible", "denied") or (result.record.status == "partial" and not keep_partial):
        return None
    out = Path(world_root) / "scenarios" / result.scenario_id
    write_scenario(out, {**{k: v for k, v in result.doc.items() if k != "edge_case"}, "edge_case": result.block}, _header(result))
    result.files.write(out)
    (out / "record.json").write_text(json.dumps(result.record.as_dict(), indent=1, sort_keys=True, default=str))
    result.written = out
    return out
