import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
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


class HistoryManager:
    """Manages transcription history with persistence and filtering"""
    
    def __init__(self, max_entries: int = 1000, history_file: str = "transcription_history.json"):
        self.max_entries = max_entries
        self.history_file = Path(history_file)
        self.entries: List[TranscriptionEntry] = []
        self.logger = logging.getLogger(__name__)
        
        # Load existing history
        self.load_history()
        
    def add_entry(self, text: str, duration: float, model: str, language: str) -> None:
        """Add new transcription entry"""
        if not text or not text.strip():
            return
            
        entry = TranscriptionEntry(
            timestamp=time.time(),
            text=text.strip(),
            duration=duration,
            model=model,
            language=language
        )
        
        self.entries.insert(0, entry)  # Add to beginning
        
        # Limit entries count
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[:self.max_entries]
            
        self.save_history()
        self.logger.debug(f"Added history entry: {entry.short_text}")
        
    def get_entries(self, limit: Optional[int] = None) -> List[TranscriptionEntry]:
        """Get history entries (newest first)"""
        if limit is None:
            return self.entries.copy()
        return self.entries[:limit]
        
    def get_entries_by_date(self, days_back: int = 7) -> List[TranscriptionEntry]:
        """Get entries from last N days"""
        cutoff_timestamp = time.time() - (days_back * 24 * 60 * 60)
        return [entry for entry in self.entries if entry.timestamp >= cutoff_timestamp]
        
    def search_entries(self, query: str) -> List[TranscriptionEntry]:
        """Search entries by text content"""
        query_lower = query.lower()
        return [entry for entry in self.entries if query_lower in entry.text.lower()]
        
    def clear_history(self) -> None:
        """Clear all history entries"""
        self.entries.clear()
        self.save_history()
        self.logger.info("History cleared")
        
    def remove_entry(self, index: int) -> bool:
        """Remove entry by index"""
        if 0 <= index < len(self.entries):
            removed = self.entries.pop(index)
            self.save_history()
            self.logger.debug(f"Removed history entry: {removed.short_text}")
            return True
        return False
        
    def get_entry_count(self) -> int:
        """Get total number of entries"""
        return len(self.entries)
        
    def get_stats(self) -> Dict[str, any]:
        """Get history statistics"""
        if not self.entries:
            return {
                "total_entries": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "most_used_model": "N/A",
                "most_used_language": "N/A",
                "oldest_entry": "N/A",
                "newest_entry": "N/A"
            }
            
        total_duration = sum(entry.duration for entry in self.entries)
        
        # Count models and languages
        model_counts = {}
        lang_counts = {}
        for entry in self.entries:
            model_counts[entry.model] = model_counts.get(entry.model, 0) + 1
            lang_counts[entry.language] = lang_counts.get(entry.language, 0) + 1
            
        most_used_model = max(model_counts, key=model_counts.get) if model_counts else "N/A"
        most_used_language = max(lang_counts, key=lang_counts.get) if lang_counts else "N/A"
        
        return {
            "total_entries": len(self.entries),
            "total_duration": total_duration,
            "avg_duration": total_duration / len(self.entries),
            "most_used_model": most_used_model,
            "most_used_language": most_used_language,
            "oldest_entry": self.entries[-1].datetime_str if self.entries else "N/A",
            "newest_entry": self.entries[0].datetime_str if self.entries else "N/A"
        }
        
    def export_to_text(self, filepath: str) -> bool:
        """Export history to text file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("Transcription History Export\n")
                f.write("=" * 50 + "\n\n")
                
                for entry in self.entries:
                    f.write(f"Date: {entry.datetime_str}\n")
                    f.write(f"Duration: {entry.duration:.1f}s\n")
                    f.write(f"Model: {entry.model} ({entry.language})\n")
                    f.write(f"Text: {entry.text}\n")
                    f.write("-" * 30 + "\n\n")
                    
            self.logger.info(f"History exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export history: {e}")
            return False
            
    def load_history(self) -> None:
        """Load history from file"""
        if not self.history_file.exists():
            return
            
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.entries = []
            for item in data.get('entries', []):
                try:
                    entry = TranscriptionEntry(**item)
                    self.entries.append(entry)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid history entry: {e}")
                    
            self.logger.debug(f"Loaded {len(self.entries)} history entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")
            self.entries = []
            
    def save_history(self) -> None:
        """Save history to file"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'entries': [asdict(entry) for entry in self.entries],
                'saved_at': time.time()
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")