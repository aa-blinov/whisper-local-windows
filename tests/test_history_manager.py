"""Tests for HistoryManager JSONL persistence.

Covers:
- add_entry appends one line (O(1), not a full rewrite)
- load_history reads all lines and restores entries correctly
- persistence across HistoryManager instances
- clear_history truncates the file
- remove_entry removes the correct entry and rewrites
- max_entries cap respected in memory and on disk after compact
- JSONL format: every line is valid standalone JSON
- atomic write: compact goes through .tmp + os.replace
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path: Path, max_entries: int = 100):
    from app.history_manager import HistoryManager

    return HistoryManager(
        max_entries=max_entries,
        history_file=str(tmp_path / "history.jsonl"),
    )


def _add(mgr, text: str = "Hello", duration: float = 1.0,
         model: str = "whisper-large-v3-turbo", language: str = "ru") -> None:
    mgr.add_entry(text, duration, model, language)


def _lines(tmp_path: Path) -> list[str]:
    """Non-empty lines in the JSONL file."""
    p = tmp_path / "history.jsonl"
    if not p.exists():
        return []
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Basic add / load
# ---------------------------------------------------------------------------

def test_add_entry_creates_file(tmp_path):
    mgr = _make_manager(tmp_path)
    _add(mgr, "First")
    assert (tmp_path / "history.jsonl").exists()


def test_add_entry_appends_one_line_per_entry(tmp_path):
    """Each add_entry must write exactly one new line — not rewrite the file."""
    mgr = _make_manager(tmp_path)
    _add(mgr, "One")
    assert len(_lines(tmp_path)) == 1
    _add(mgr, "Two")
    assert len(_lines(tmp_path)) == 2
    _add(mgr, "Three")
    assert len(_lines(tmp_path)) == 3


def test_each_line_is_valid_json(tmp_path):
    mgr = _make_manager(tmp_path)
    for i in range(5):
        _add(mgr, f"Entry {i}")
    for line in _lines(tmp_path):
        obj = json.loads(line)  # must not raise
        assert "text" in obj
        assert "timestamp" in obj


def test_load_history_restores_entries(tmp_path):
    mgr = _make_manager(tmp_path)
    _add(mgr, "Alpha", duration=1.5)
    _add(mgr, "Beta", duration=2.0)

    mgr2 = _make_manager(tmp_path)
    entries = mgr2.get_entries()
    texts = [e.text for e in entries]
    assert "Alpha" in texts
    assert "Beta" in texts


def test_newest_entry_is_first(tmp_path):
    mgr = _make_manager(tmp_path)
    _add(mgr, "Older")
    _add(mgr, "Newer")

    entries = mgr.get_entries()
    assert entries[0].text == "Newer"
    assert entries[1].text == "Older"


def test_persistence_across_instances(tmp_path):
    mgr1 = _make_manager(tmp_path)
    _add(mgr1, "Persistent text", duration=3.7, model="whisper-large-v3", language="ru")

    mgr2 = _make_manager(tmp_path)
    entries = mgr2.get_entries()
    assert len(entries) == 1
    assert entries[0].text == "Persistent text"
    assert entries[0].duration == pytest.approx(3.7)
    assert entries[0].model == "whisper-large-v3"


# ---------------------------------------------------------------------------
# clear_history
# ---------------------------------------------------------------------------

def test_clear_history_empties_in_memory(tmp_path):
    mgr = _make_manager(tmp_path)
    _add(mgr, "To be cleared")
    mgr.clear_history()
    assert mgr.get_entries() == []


def test_clear_history_truncates_file(tmp_path):
    mgr = _make_manager(tmp_path)
    for i in range(10):
        _add(mgr, f"Entry {i}")
    mgr.clear_history()
    assert _lines(tmp_path) == []


def test_clear_history_then_add_works(tmp_path):
    mgr = _make_manager(tmp_path)
    _add(mgr, "Before clear")
    mgr.clear_history()
    _add(mgr, "After clear")
    entries = mgr.get_entries()
    assert len(entries) == 1
    assert entries[0].text == "After clear"

    # Reload — make sure file is also correct
    mgr2 = _make_manager(tmp_path)
    assert len(mgr2.get_entries()) == 1


# ---------------------------------------------------------------------------
# remove_entry
# ---------------------------------------------------------------------------

def test_remove_entry_removes_correct_entry(tmp_path):
    mgr = _make_manager(tmp_path)
    _add(mgr, "Keep")
    _add(mgr, "Remove me")  # index 0 (newest)
    assert mgr.remove_entry(0) is True
    texts = [e.text for e in mgr.get_entries()]
    assert "Remove me" not in texts
    assert "Keep" in texts


def test_remove_entry_persists_after_reload(tmp_path):
    mgr = _make_manager(tmp_path)
    _add(mgr, "Keep")
    _add(mgr, "Remove me")
    mgr.remove_entry(0)

    mgr2 = _make_manager(tmp_path)
    texts = [e.text for e in mgr2.get_entries()]
    assert "Remove me" not in texts
    assert "Keep" in texts


def test_remove_entry_invalid_index_returns_false(tmp_path):
    mgr = _make_manager(tmp_path)
    assert mgr.remove_entry(0) is False
    assert mgr.remove_entry(-1) is False
    assert mgr.remove_entry(999) is False


# ---------------------------------------------------------------------------
# max_entries cap
# ---------------------------------------------------------------------------

def test_max_entries_respected_in_memory(tmp_path):
    mgr = _make_manager(tmp_path, max_entries=5)
    for i in range(10):
        _add(mgr, f"Entry {i}")
    assert len(mgr.get_entries()) == 5


def test_max_entries_newest_kept_after_cap(tmp_path):
    mgr = _make_manager(tmp_path, max_entries=3)
    _add(mgr, "Old 1")
    _add(mgr, "Old 2")
    _add(mgr, "Old 3")
    _add(mgr, "New")
    texts = [e.text for e in mgr.get_entries()]
    assert "New" in texts
    assert "Old 1" not in texts


def test_max_entries_respected_after_reload(tmp_path):
    """After compaction, reloading must not exceed max_entries."""
    mgr = _make_manager(tmp_path, max_entries=5)
    for i in range(10):
        _add(mgr, f"Entry {i}")

    mgr2 = _make_manager(tmp_path, max_entries=5)
    assert len(mgr2.get_entries()) <= 5


# ---------------------------------------------------------------------------
# Atomic write (compact goes via .tmp)
# ---------------------------------------------------------------------------

def test_compact_uses_tmp_then_replaces(tmp_path, monkeypatch):
    """compact() must write to .tmp and use os.replace — never truncate
    the real file in-place, so a mid-write crash doesn't corrupt data."""
    import app.history_manager as hm_module

    replaced: list[tuple] = []
    original_replace = os.replace

    def spy_replace(src, dst):
        replaced.append((src, dst))
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    mgr = _make_manager(tmp_path, max_entries=3)
    for i in range(5):
        _add(mgr, f"Entry {i}")

    # Force a compact (remove_entry triggers it)
    mgr.remove_entry(0)

    assert len(replaced) >= 1
    src, dst = replaced[-1]
    assert src.endswith(".tmp")
    assert dst == str(tmp_path / "history.jsonl")


# ---------------------------------------------------------------------------
# Legacy JSON migration
# ---------------------------------------------------------------------------

def test_load_history_migrates_from_legacy_json(tmp_path):
    """When the .jsonl file is absent but a same-stem .json file exists,
    load_history must import those entries and write them as JSONL so
    users upgrading from the old flat-JSON format don't silently lose
    their history."""
    from app.history_manager import HistoryManager

    legacy = tmp_path / "history.json"
    rows = [
        {"timestamp": 1000.0, "text": "first",  "duration": 1.0, "model": "whisper-large-v3-turbo", "language": "en"},
        {"timestamp": 2000.0, "text": "second", "duration": 2.0, "model": "large", "language": "ru"},
    ]
    legacy.write_text(json.dumps(rows), encoding="utf-8")

    mgr = HistoryManager(max_entries=100, history_file=str(tmp_path / "history.jsonl"))

    assert len(mgr.entries) == 2
    assert {e.text for e in mgr.entries} == {"first", "second"}
    # JSONL file must have been created by the migration
    assert (tmp_path / "history.jsonl").exists()
    # Legacy file must be gone (renamed to .json.bak) so migration
    # doesn't re-run on next launch
    assert not legacy.exists(), "legacy .json must be renamed after migration"
    assert (tmp_path / "history.json.bak").exists()


def test_load_history_migration_skips_invalid_rows(tmp_path):
    """Corrupt rows in the legacy file must be skipped, not crash the migration."""
    from app.history_manager import HistoryManager

    legacy = tmp_path / "history.json"
    rows = [
        {"timestamp": 1000.0, "text": "good", "duration": 1.0, "model": "whisper-large-v3-turbo", "language": "en"},
        {"broken": "row"},   # missing required fields
        "not a dict",
    ]
    legacy.write_text(json.dumps(rows), encoding="utf-8")

    mgr = HistoryManager(max_entries=100, history_file=str(tmp_path / "history.jsonl"))

    assert len(mgr.entries) == 1
    assert mgr.entries[0].text == "good"


def test_load_history_migration_does_not_run_when_jsonl_exists(tmp_path):
    """If the .jsonl file is already present, migration must not overwrite it
    even when a .json file also exists (e.g. user downgraded then upgraded)."""
    from app.history_manager import HistoryManager

    # Pre-existing JSONL
    jsonl = tmp_path / "history.jsonl"
    row = {"timestamp": 9000.0, "text": "jsonl-entry", "duration": 0.5,
           "model": "whisper-large-v3-turbo", "language": "en"}
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    # Legacy file alongside it — should be ignored
    legacy = tmp_path / "history.json"
    legacy.write_text(json.dumps([{
        "timestamp": 1000.0, "text": "legacy", "duration": 1.0,
        "model": "old", "language": "ru"
    }]), encoding="utf-8")

    mgr = HistoryManager(max_entries=100, history_file=str(jsonl))

    assert len(mgr.entries) == 1
    assert mgr.entries[0].text == "jsonl-entry"
