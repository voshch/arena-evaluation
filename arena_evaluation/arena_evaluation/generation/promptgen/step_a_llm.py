"""Step 1b - Type A by a small model from the prompt text alone (the lexicon misses a misspelled "evacation"), plus the `who` block of actors shared with Steps D and B+C.
The lexicon stays the oracle for tests and `--oracle` runs; `pipeline.generate` arbitrates between the two and records every disagreement."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .lexicon import ONSETS
from .llm import AnswerError, Oracle, call
from .profiles import ProfileError, profile
from .spec import A_CASES, ADecision, Record, SchemaError, When, parse_when
from .step_a import StepError

_SYSTEM = Path(__file__).with_name("system_a.md")

#: The Type A classifier runs on a cheaper tier - the call is text-only and tiny.
A_MODEL = os.environ.get("ARENA_PROMPT_LLM_A", "gemini-3.5-flash-lite")


def system_prompt() -> str:
    return _SYSTEM.read_text()


def user_prompt(prompt: str) -> str:
    return f"prompt: {prompt}"


def who_line(who: list[dict[str, Any]]) -> str:
    """The shared cast reading handed to Steps D and B+C."""
    if not who:
        return ""
    parts = [f"{w['count'] or 'some'} x {w['profile']} (\"{w['mention']}\")" for w in who]
    return "who (actors read from the prompt): " + "; ".join(parts)


def _parse(raw: dict[str, Any]) -> tuple[ADecision | None, list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    who: list[dict[str, Any]] = []
    for entry in raw.get("who") or []:
        if not isinstance(entry, dict) or not entry.get("mention"):
            warnings.append(f"who: dropped malformed entry {entry!r}")
            continue
        try:
            prof = profile(str(entry.get("profile", "adult"))).name
        except ProfileError:
            warnings.append(f"who: dropped {entry.get('mention')!r} (unknown profile {entry.get('profile')!r})")
            continue
        try:
            count = max(0, int(entry.get("count") or 0))
        except (TypeError, ValueError):
            count = 0
        who.append({"mention": str(entry["mention"])[:60], "profile": prof, "count": count})
    case = str(raw.get("case") or "none").strip().lower()
    if case in ("none", ""):
        return None, who, warnings
    if case not in A_CASES:
        raise SchemaError(f"A.case: {case!r} is not one of {list(A_CASES)} or 'none'")
    try:
        k = min(3.0, max(1.0, float(raw.get("k", 2.0))))
    except (TypeError, ValueError):
        raise SchemaError(f"A.k: {raw.get('k')!r} is not a number") from None
    when = parse_when(raw.get("when"), default=ONSETS[case], where="A")
    return ADecision(case=case, k=k, when=when if isinstance(when, When) else ONSETS[case]), who, warnings


def decide(prompt: str, record: Record, *, cache_dir: Path | None, model: str | None = None, oracle: Oracle | None = None) -> tuple[ADecision | None, list[dict[str, Any]], list[str]]:
    """(decision-or-None, who block, warnings). Raises `StepError` on a twice-rejected answer."""
    system = system_prompt()
    user = user_prompt(prompt)
    complaint = ""
    for attempt in range(2):
        answer = call(system, user + (f"\n\nYour previous answer was rejected: {complaint}. Answer again." if complaint else ""), cache_dir=cache_dir, model=model or A_MODEL, oracle=oracle)
        record.provenance.setdefault("A", []).append({"model": answer.model, "response_sha256": answer.sha256, "cached": answer.cached, "attempt": attempt})
        try:
            return _parse(answer.json())
        except (AnswerError, SchemaError) as exc:
            complaint = str(exc)
    raise StepError(f"schema: {complaint}")
