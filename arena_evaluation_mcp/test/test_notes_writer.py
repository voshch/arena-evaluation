"""notes.yaml read/write/append/merge via the shipped tools.py helpers."""

import pathlib
import tempfile

import pytest

pytest.importorskip("mcp")
pytest.importorskip("polars")
pytest.importorskip("arena_evaluation_mcp")


def _helpers():
    from arena_evaluation_mcp.tools import (
        _load_notes_file,
        _save_notes_file,
        _write_notes_file,
    )

    return _load_notes_file, _save_notes_file, _write_notes_file


@pytest.fixture
def notes_dir():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


class TestNotesWriter:
    def test_write_new_file(self, notes_dir):
        load, _save, write = _helpers()
        path = notes_dir / "notes.yaml"
        result = write(
            path,
            [
                {"label": "Key 1", "value": "Value 1"},
                {"label": "Key 2", "value": "Value 2"},
            ],
            "replace",
        )
        assert result["n_notes"] == 2
        assert path.exists()

        loaded = load(path)
        assert len(loaded) == 2
        assert loaded[0]["label"] == "Key 1"
        assert loaded[0]["value"] == "Value 1"

    def test_replace_overwrites(self, notes_dir):
        load, _save, write = _helpers()
        path = notes_dir / "notes.yaml"
        write(path, [{"label": "Old", "value": "old"}], "replace")
        write(path, [{"label": "New", "value": "new"}], "replace")

        loaded = load(path)
        assert len(loaded) == 1
        assert loaded[0]["label"] == "New"

    def test_append_adds_rows(self, notes_dir):
        load, _save, write = _helpers()
        path = notes_dir / "notes.yaml"
        write(path, [{"label": "First", "value": "1"}], "replace")
        write(path, [{"label": "Second", "value": "2"}], "append")

        loaded = load(path)
        assert len(loaded) == 2
        assert loaded[0]["label"] == "First"
        assert loaded[1]["label"] == "Second"

    def test_merge_updates_existing(self, notes_dir):
        load, _save, write = _helpers()
        path = notes_dir / "notes.yaml"
        write(
            path,
            [
                {"label": "A", "value": "old_a"},
                {"label": "B", "value": "old_b"},
            ],
            "replace",
        )
        write(
            path,
            [
                {"label": "A", "value": "new_a"},
                {"label": "C", "value": "new_c"},
            ],
            "merge",
        )

        loaded = load(path)
        assert len(loaded) == 3  # A (updated), B (unchanged), C (new)
        a_row = next(r for r in loaded if r["label"] == "A")
        assert a_row["value"] == "new_a"

    def test_unknown_mode_errors(self, notes_dir):
        _load, _save, write = _helpers()
        path = notes_dir / "notes.yaml"
        result = write(path, [{"label": "A", "value": "a"}], "zz_mode")
        assert "error" in result
        assert not path.exists()

    def test_read_empty_file(self, notes_dir):
        load, _save, _write = _helpers()
        path = notes_dir / "nonexistent.yaml"
        assert load(path) == []

    def test_read_dict_format(self, notes_dir):
        load, _save, _write = _helpers()
        path = notes_dir / "notes.yaml"
        path.write_text("Key1: Value1\nKey2: Value2\n")
        loaded = load(path)
        assert len(loaded) == 2
        labels = {r["label"] for r in loaded}
        assert "Key1" in labels
        assert "Key2" in labels

    def test_read_list_format(self, notes_dir):
        load, save, _write = _helpers()
        path = notes_dir / "notes.yaml"
        rows = [{"label": "L1", "value": "V1"}, {"label": "L2", "value": "V2"}]
        save(path, rows)
        assert load(path) == rows

    def test_unicode_content(self, notes_dir):
        load, _save, write = _helpers()
        path = notes_dir / "notes.yaml"
        label = "\u00dcnicode"
        value = "Test \u2713"
        write(path, [{"label": label, "value": value}], "replace")
        loaded = load(path)
        assert loaded[0]["value"] == value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
