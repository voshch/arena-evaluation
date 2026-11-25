from __future__ import annotations

from arena_evaluation.storage.exceptions import ContractViolationError


def parse_bool(token: str | None) -> bool:
    """Recorded boolean token, accepting true/false/1/0 case-insensitively; anything else raises ContractViolationError."""
    normalized = (token or "").strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ContractViolationError(f"not a boolean token: {token!r}")


def parse_float(token: str) -> float:
    """Recorded float token; malformed input raises ContractViolationError."""
    try:
        return float(token)
    except (TypeError, ValueError) as exc:
        raise ContractViolationError(f"not a float token: {token!r}") from exc
