from __future__ import annotations

import threading

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_evaluation.mcp.base import atomic_write_text, file_lock
from arena_evaluation.mcp.findings import (
    evaluation_finding_attach,
    evaluation_finding_create,
    evaluation_finding_status,
    evaluation_finding_verify,
)
from arena_evaluation.mcp.runs import Note, evaluation_write_notes, load_notes

from mcp_fixtures import RUN_ID, arena_data, build_run


def test_atomic_write_never_exposes_a_partial_document(tmp_path):
    path = tmp_path / "document.txt"
    atomic_write_text(path, "first\n")
    contents = [f"value {index}\n" * 500 for index in range(25)]
    seen: list[str] = []
    stop = threading.Event()

    def read_while_writing() -> None:
        while not stop.is_set():
            seen.append(path.read_text(encoding="utf-8"))

    reader = threading.Thread(target=read_while_writing)
    reader.start()
    try:
        atomic_write_text(path, "second\n")
        for content in contents:
            atomic_write_text(path, content)
    finally:
        stop.set()
        reader.join()

    allowed = {"first\n", "second\n", *contents}
    assert seen
    assert all(text in allowed for text in seen)
    assert not list(tmp_path.glob("*.tmp"))


def test_file_lock_serializes_a_read_modify_write(tmp_path):
    path = tmp_path / "counter.txt"
    atomic_write_text(path, "0")

    def bump() -> None:
        for _ in range(25):
            with file_lock(path):
                atomic_write_text(path, str(int(path.read_text(encoding="utf-8")) + 1))

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert path.read_text(encoding="utf-8") == "100"


def test_notes_replace_keeps_the_finding_rows(tmp_path):
    run_dir = build_run(tmp_path / "data" / "benchmarks", RUN_ID)
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")

    with arena_data(tmp_path):
        evaluation_write_notes(RUN_ID, [Note(label="old", value="1")], "append")
        finding = evaluation_finding_create(RUN_ID, "claim")
        evaluation_finding_verify(RUN_ID, finding.finding_id)
        evaluation_finding_status(RUN_ID, finding.finding_id, "verified", by="verifier")
        evaluation_finding_attach(RUN_ID, finding.finding_id)
        evaluation_write_notes(RUN_ID, [Note(label="summary", value="2")], "replace")
        rows = load_notes(run_dir / "notes.yaml")

    assert [row["label"] for row in rows] == [
        f"finding:{finding.finding_id}",
        "summary",
    ]
    assert "old" not in [row["label"] for row in rows]
