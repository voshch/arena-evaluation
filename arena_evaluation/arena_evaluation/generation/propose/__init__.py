"""Model-proposed roles for the zone names `scenario.zones` rules leave unknown, reviewed into `scenario/zone_roles.yaml`."""

from __future__ import annotations

from .client import DEFAULT_MODEL, Unavailable, available, prompt_sha256
from .zone_roles import (
    STORE,
    Proposal,
    build_prompt,
    classify_tail,
    load_overrides,
    merge,
    parse,
    unknown_names,
)

__all__ = [
    "DEFAULT_MODEL",
    "STORE",
    "Proposal",
    "Unavailable",
    "available",
    "build_prompt",
    "classify_tail",
    "load_overrides",
    "merge",
    "parse",
    "prompt_sha256",
    "unknown_names",
]
