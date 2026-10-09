"""One model call per step at temperature 0, cached as one JSON file per `prompt_sha256(model, system, user)`.
The cache doubles as the audit trail: rejected answers stay beside the accepted one."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import attrs

from ..propose import client as _client
from ..propose.client import DEFAULT_MODEL, Unavailable, ask, prompt_sha256

DEFAULT_CACHE = Path(os.environ.get("ARENA_DATA_DIR", "data")) / "promptgen" / "llm_cache"

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class AnswerError(ValueError):
    pass


@attrs.frozen
class Answer:
    text: str
    sha256: str
    model: str
    cached: bool

    def json(self) -> dict[str, Any]:
        text = _FENCE.sub("", self.text.strip()).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise AnswerError(f"no JSON object in the answer: {text[:200]!r}")
        try:
            doc = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnswerError(f"answer is not valid JSON ({exc}): {text[:200]!r}") from None
        if not isinstance(doc, dict):
            raise AnswerError("answer must be a JSON object")
        return doc


class Oracle:
    """Stand-in for the model in tests and `--oracle` runs: a fixed answer or a `(system, user)` callable; non-strings are JSON-encoded. Never cached."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    def __call__(self, system: str, user: str) -> str:
        value = self._answer(system, user) if callable(self._answer) else self._answer
        return value if isinstance(value, str) else json.dumps(value)


def call(system: str, user: str, *, cache_dir: Path | None = None, model: str | None = None, oracle: Oracle | None = None) -> Answer:
    """One completion, cached. Raises `Unavailable` when no model is configured and no oracle given."""
    name = model or DEFAULT_MODEL
    key = prompt_sha256(name, system, user)
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    path = root / f"{key}.json"
    if oracle is None and path.is_file():
        doc = json.loads(path.read_text())
        return Answer(text=str(doc["answer"]), sha256=hashlib.sha256(str(doc["answer"]).encode()).hexdigest()[:16], model=str(doc.get("model", name)), cached=True)
    if oracle is not None:
        text = oracle(system, user)
        return Answer(text=text, sha256=hashlib.sha256(text.encode()).hexdigest()[:16], model="oracle", cached=False)
    text = ask(system, user, model=name, temperature=0.0)
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": name, "system_sha256": prompt_sha256(system), "user": user, "answer": text, "usage": _client.LAST_USAGE}, indent=1))
    return Answer(text=text, sha256=hashlib.sha256(text.encode()).hexdigest()[:16], model=name, cached=False)


__all__ = ["DEFAULT_CACHE", "Answer", "AnswerError", "Oracle", "Unavailable", "call"]
