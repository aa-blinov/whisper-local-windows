import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TranscriptionEntry:
    """Entry for transcription history"""
    timestamp: float
    text: str
    duration: float
    model: str
    language: str

    @property
    def datetime_str(self) -> str:
        """Get formatted datetime string"""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S %d.%m.%Y")

    @property
    def short_text(self) -> str:
        """Get shortened text for display"""
        if len(self.text) <= 50:
            return self.text
        return self.text[:47] + "..."


# How many stale disk lines (beyond max_entries) trigger an automatic compact.
_COMPACT_THRESHOLD = 50


class HistoryManager:
    """Manages transcription history with JSONL persistence.

    File format: JSONL (one JSON object per line), oldest entry first.
    In-memory representation: newest entry first (unchanged public interface).

    add_entry  → O(1) single-line append.
    remove_entry / clear_history → rewrite via .tmp + os.replace() (atomic).
    """

    def __init__(self, max_entries: int = 1000,
                 history_file: str = "transcription_history.jsonl"):
        self.max_entries = max_entries
        self.history_file = Path(history_file)
        self.entries: List[TranscriptionEntry] = []
        # Lines currently on disk; may exceed len(self.entries) when entries
        # have been removed from memory but the file hasn't been compacted yet.
        self._file_entry_count: int = 0
        self.logger = logging.getLogger(__name__)

        self.load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, text: str, duration: float, model: str, language: str) -> None:
        """Append one JSONL line — O(1), no full rewrite."""
        if not text or not text.strip():
            return

        entry = TranscriptionEntry(
            timestamp=time.time(),
            text=text.strip(),
            duration=duration,
            model=model,
            language=language,
        )

        self.entries.insert(0, entry)           # newest-first in memory
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[:self.max_entries]

        self._append_line(entry)
        self._file_entry_count += 1

        if self._file_entry_count > self.max_entries + _COMPACT_THRESHOLD:
            self._compact()

        self.logger.debug("Added history entry: %s", entry.short_text)

    def get_entries(self, limit: Optional[int] = None) -> List[TranscriptionEntry]:
        """Return history entries (newest first)."""
        if limit is None:
            return self.entries.copy()
        return self.entries[:limit]

    def get_entries_by_date(self, days_back: int = 7) -> List[TranscriptionEntry]:
        """Return entries from the last *days_back* days."""
        cutoff = time.time() - days_back * 86_400
        return [e for e in self.entries if e.timestamp >= cutoff]

    def search_entries(self, query: str) -> List[TranscriptionEntry]:
        """Return entries whose text contains *query* (case-insensitive)."""
        q = query.lower()
        return [e for e in self.entries if q in e.text.lower()]

    def clear_history(self) -> None:
        """Clear all history in memory and truncate the JSONL file."""
        self.entries.clear()
        self._file_entry_count = 0
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text("", encoding="utf-8")
        except Exception as exc:
            self.logger.error("Failed to clear history file: %s", exc)
        self.logger.info("History cleared")

    def remove_entry(self, index: int) -> bool:
        """Remove entry at *index* (0 = newest) and rewrite the file atomically."""
        if not (0 <= index < len(self.entries)):
            return False
        removed = self.entries.pop(index)
        self._compact()
        self.logger.debug("Removed history entry: %s", removed.short_text)
        return True

    def get_entry_count(self) -> int:
        """Return the number of entries currently in memory."""
        return len(self.entries)

    def get_stats(self) -> Dict:
        """Return aggregate statistics about the history."""
        if not self.entries:
            return {
                "total_entries": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "most_used_model": "N/A",
                "most_used_language": "N/A",
                "oldest_entry": "N/A",
                "newest_entry": "N/A",
            }

        total_duration = sum(e.duration for e in self.entries)
        model_counts: Dict[str, int] = {}
        lang_counts: Dict[str, int] = {}
        for e in self.entries:
            model_counts[e.model] = model_counts.get(e.model, 0) + 1
            lang_counts[e.language] = lang_counts.get(e.language, 0) + 1

        return {
            "total_entries": len(self.entries),
            "total_duration": total_duration,
            "avg_duration": total_duration / len(self.entries),
            "most_used_model": max(model_counts, key=model_counts.get),
            "most_used_language": max(lang_counts, key=lang_counts.get),
            "oldest_entry": self.entries[-1].datetime_str,
            "newest_entry": self.entries[0].datetime_str,
        }

    def export_to_text(self, filepath: str) -> bool:
        """Export history to a human-readable text file."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("Transcription History Export\n")
                f.write("=" * 50 + "\n\n")
                for e in self.entries:
                    f.write(f"Date: {e.datetime_str}\n")
                    f.write(f"Duration: {e.duration:.1f}s\n")
                    f.write(f"Model: {e.model} ({e.language})\n")
                    f.write(f"Text: {e.text}\n")
                    f.write("-" * 30 + "\n\n")
            self.logger.info("History exported to %s", filepath)
            return True
        except Exception as exc:
            self.logger.error("Failed to export history: %s", exc)
            return False

    def load_history(self) -> None:
        """Load history from the JSONL file.

        Disk order is oldest-first; in-memory result is newest-first and
        capped to ``max_entries``.  If the file has more lines than needed,
        compact() is called immediately to trim it.
        """
        self.entries = []
        self._file_entry_count = 0

        if not self.history_file.exists():
            return

        try:
            raw_lines = [
                ln
                for ln in self.history_file.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self._file_entry_count = len(raw_lines)

            parsed: List[TranscriptionEntry] = []
            for ln in raw_lines:
                try:
                    parsed.append(TranscriptionEntry(**json.loads(ln)))
                except Exception as exc:
                    self.logger.warning("Skipping invalid history line: %s", exc)

            # File is oldest-first; reverse → newest first, then cap.
            self.entries = list(reversed(parsed))[: self.max_entries]

            if len(self.entries) < self._file_entry_count:
                # More lines on disk than we keep — compact right away.
                self._compact()

            self.logger.debug("Loaded %d history entries", len(self.entries))

        except Exception as exc:
            self.logger.error("Failed to load history: %s", exc)
            self.entries = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_line(self, entry: TranscriptionEntry) -> None:
        """Append one JSON line to the JSONL file (open in append mode)."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except Exception as exc:
            self.logger.error("Failed to append history entry: %s", exc)

    def _compact(self) -> None:
        """Rewrite the JSONL file atomically, keeping only ``self.entries``.

        Writes to *<history_file>.tmp* then calls ``os.replace()`` so that a
        crash during the write cannot corrupt the real file.  Entries are
        stored oldest-first on disk (reversed vs. the newest-first in-memory
        list).
        """
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(self.history_file) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                # self.entries is newest-first; write oldest-first on disk.
                for entry in reversed(self.entries):
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            os.replace(tmp, str(self.history_file))
            self._file_entry_count = len(self.entries)
        except Exception as exc:
            self.logger.error("Failed to compact history: %s", exc)

    # ------------------------------------------------------------------
    # Compatibility shim
    # ------------------------------------------------------------------

    def save_history(self) -> None:
        """Deprecated — triggers a full compact.  Use add_entry / remove_entry instead."""
        self._compact()
