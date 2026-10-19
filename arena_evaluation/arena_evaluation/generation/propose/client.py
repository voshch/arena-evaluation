"""Synchronous calls into `task_generator.utils.gpt`'s genai client (the one `TM_Prompt` uses), for the generation CLI."""

from __future__ import annotations

import hashlib
import os

# 3.7-flash: same price as 3.6, fewest thinking tokens and fastest on the probe call (2026-08-31)
DEFAULT_MODEL = os.environ.get("ARENA_PROMPT_LLM", "gemini-3.7-flash")


class Unavailable(RuntimeError):
    pass


def prompt_sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def available() -> bool:
    try:
        _client()
    except Unavailable:
        return False
    return True


def _client() -> object:
    if not os.environ.get("LLM_API_ENDPOINT") and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise Unavailable("no LLM configured: set GEMINI_API_KEY (or GOOGLE_API_KEY) for the hosted API, or LLM_API_ENDPOINT to point at a local one")
    try:
        from task_generator.utils.gpt import genai
    except Exception as exc:
        raise Unavailable(f"the genai client is not importable: {exc}") from exc
    return genai


# a `Client()` that goes out of scope closes its httpx pool under the in-flight request
_CLIENT: object | None = None

TIMEOUT_MS = 120_000


# thinking tokens are billed but absent from the answer text, so the caller's cache records these
LAST_USAGE: dict[str, int | None] | None = None


def ask(system: str, user: str, *, model: str | None = None, temperature: float = 0.0) -> str:
    """One completion as raw text; raises `Unavailable` rather than returning a guess."""
    global _CLIENT
    genai = _client()
    response = None
    for attempt in range(2):
        if _CLIENT is None:
            # without a timeout a probe hung 26 minutes on a dropped connection (2026-08-31)
            _CLIENT = genai.Client(http_options=genai.types.HttpOptions(timeout=TIMEOUT_MS))
        try:
            response = _CLIENT.models.generate_content(
                model=model or DEFAULT_MODEL,
                contents=[user],
                config=genai.types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            break
        except Exception as exc:
            _CLIENT = None
            if attempt:
                raise Unavailable(f"the model call failed: {type(exc).__name__}: {exc}") from exc
    text = response.text
    global LAST_USAGE
    meta = response.usage_metadata
    LAST_USAGE = (
        {
            "input_tokens": meta.prompt_token_count,
            "output_tokens": meta.candidates_token_count,
            "thoughts_tokens": meta.thoughts_token_count,
            "cached_tokens": meta.cached_content_token_count,
            "total_tokens": meta.total_token_count,
        }
        if meta is not None
        else None
    )
    if not text:
        raise Unavailable("the model returned an empty response")
    return text.strip()


def strip_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    return text[:-3].strip() if text.endswith("```") else text.strip()
