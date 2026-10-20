"""Step 1 - Type A by lexicon. Deterministic, offline, and round-trip tested by the prompt sets."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .spec import A_CASES, PHRASES, ADecision, When, parse_when

_LEXICON = Path(__file__).with_name("lexicon.yaml")

#: The score a case must reach to be applied.
THRESHOLD = 1.0

ONSETS: dict[str, When] = {
    "blackout": When(at=5.0),
    "evacuation": When(at=4.0),
    "stampede": When(at=2.0),
    "busy": When(at=0.0),
}


@cache
def load(path: str | None = None) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path or _LEXICON).read_text()) or {}
    unknown = sorted(set(doc) - set(A_CASES))
    if unknown:
        raise ValueError(f"lexicon: unknown case(s) {unknown}")
    return doc


def score(prompt: str, lexicon: dict[str, Any] | None = None) -> dict[str, tuple[float, int]]:
    """Case -> (total score, position of its earliest hit)."""
    text = " ".join(prompt.lower().split())
    lex = lexicon or load()
    out: dict[str, tuple[float, int]] = {}
    for case, entry in lex.items():
        total, first = 0.0, len(text) + 1
        for phrase, weight in (entry.get("phrases") or {}).items():
            pos = text.find(str(phrase).lower())
            if pos >= 0:
                total += float(weight)
                first = min(first, pos)
        if total > 0:
            out[case] = (total, first)
    return out


def _onset(prompt: str, default: When, case: str, lexicon: dict[str, Any]) -> When:
    """An onset phrase counts only in the sentence that names the case: "halfway" in a later
    sentence about a crossing pedestrian is that pedestrian's, not the evacuation's."""
    phrases = [str(p).lower() for p in (lexicon.get(case, {}).get("phrases") or {})]
    for sentence in re.split(r"(?<=[.!?])\s+", prompt.lower()):
        if not any(p in sentence for p in phrases):
            continue
        # Longest phrase first: "once the robot is halfway" must not be read as "halfway".
        for phrase, _when in sorted(PHRASES, key=lambda pw: -len(pw[0])):
            if phrase in sentence:
                return parse_when(phrase, default=default, where="A")
    return default


def when_in(text: str, *, exclude_cases: Sequence[str] = (), lexicon: dict[str, Any] | None = None) -> When | None:
    """The first time phrase in a sentence of `text` that does not name one of `exclude_cases`
    (their sentences own their phrases). Longest phrase first. None when there is none."""
    lex = lexicon or load()
    owned = [str(p).lower() for case in exclude_cases for p in (lex.get(case, {}).get("phrases") or {})]
    for sentence in re.split(r"(?<=[.!?])\s+", text.lower()):
        if any(p in sentence for p in owned):
            continue
        for phrase, when in sorted(PHRASES, key=lambda pw: -len(pw[0])):
            if phrase in sentence:
                return when
    return None


def match(prompt: str, lexicon: dict[str, Any] | None = None) -> ADecision | None:
    """The Type A case a prompt asks for, or None."""
    lex = lexicon or load()
    scores = score(prompt, lex)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))
    if not ranked or ranked[0][1][0] < THRESHOLD:
        return None
    case = ranked[0][0]
    k = 2.0
    if case == "busy":
        text = prompt.lower()
        for word, value in (lex["busy"].get("k") or {}).items():
            if str(word).lower() in text:
                k = float(value)
                break
    return ADecision(
        case=case,
        k=k,
        when=_onset(prompt, ONSETS[case], case, lex),
        dropped=tuple(c for c, _ in ranked[1:] if scores[c][0] >= THRESHOLD),
        scores={c: s for c, (s, _pos) in scores.items()},
    )
