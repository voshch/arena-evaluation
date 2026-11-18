"""The key columns of the fact tables."""

from __future__ import annotations

from typing import Final

from arena_evaluation.storage.registry.facts import FACT_TABLES

FACT_KEY_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    table.table_name: table.key for table in FACT_TABLES.values()
}
