"""Shared validator and status computation from common.py (no deps)."""

import pytest

pytest.importorskip("arena_evaluation_mcp")

from arena_evaluation_mcp.common import run_status, validate_path_component


class TestValidatePathComponent:
    @pytest.mark.parametrize(
        "name",
        [
            "basic",
            "20260811-031709-hospital_complex-basic",
            "my_suite.v2",
            "standard",
            "a",
        ],
    )
    def test_accepts_plain_names(self, name):
        assert validate_path_component(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            ".",
            "..",
            ".hidden",
            "../escape",
            "a/b",
            "a\\b",
            "/etc/passwd",
            "name with space",
            "semi;colon",
            123,
        ],
    )
    def test_rejects_traversal_and_junk(self, name):
        with pytest.raises(ValueError):
            validate_path_component(name)


class TestRunStatus:
    def test_empty_is_unknown(self):
        assert run_status([]) == "unknown"

    def test_all_terminal_is_completed(self):
        assert run_status(["ok", "failed", "partial", "skipped"]) == "completed"

    def test_any_nonterminal_is_in_progress(self):
        assert run_status(["ok", "in_progress"]) == "in_progress"
        assert run_status([None]) == "in_progress"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
