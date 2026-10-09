"""Prompt-driven edge-case generation: Type A by lexicon or model, then Type D, then Type B+C by model.
Static changes go into the scenario file; anything that depends on the live robot becomes an `effects:` entry for the `edge_case` task mode."""

from __future__ import annotations

from .pipeline import Result, generate, write_result
from .profiles import PROFILES, TIERS, Profile, derive, profile, tier_value
from .spec import Record, SchemaError

__all__ = ["PROFILES", "TIERS", "Profile", "Record", "Result", "SchemaError", "derive", "generate", "profile", "tier_value", "write_result"]
