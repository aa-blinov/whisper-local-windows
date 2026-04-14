import logging
import customtkinter as ctk
import tkinter as tk
from typing import List, Optional, Dict, Any
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

from app.config_manager import ConfigManager
from app.whisper_engine import WhisperEngine
from app.clipboard_manager import ClipboardManager
from app.model_mapping import alias_for, ALIAS_TO_MODEL
from app.state_manager import StateManager
from app.hotkey_listener import HotkeyListener
from app.instance_manager import guard_against_multiple_instances
from app.audio_recorder import AudioRecorder
from app.audio_feedback import AudioFeedback
from app.logging_utils import setup_logging, setup_exception_handler
from app.utils import get_project_logs_path, resolve_asset_path
from app.system_tray import SystemTray
from app.logging_utils import EarlyBufferHandler
from app.docker_backend_manager import DockerBackendManager

MODEL_OPTIONS = list(ALIAS_TO_MODEL.keys())
LANGUAGE_OPTIONS = ['ru', 'en']
BUTTON_WIDTH = 140  # Unified width for all primary buttons

# Configure CustomTkinter with modern theme
ctk.set_appearance_mode("dark")  # Modern dark theme by default
ctk.set_default_color_theme("dark-blue")  # Professional dark-blue theme

# Modern color scheme - improved for better readability and aesthetics
COLORS = {
    "primary": "#1E1E1E",        # Darker, more elegant background
    "secondary": "#2D2D30",     # Softer secondary background
    "accent": "#007ACC",        # Softer blue, better contrast
    "accent_light": "#4FC3F7",  # Light blue for hover states
    "success": "#4CAF50",       # Material green, more pleasant
    "success_light": "#81C784", # Light green for hover
    "warning": "#FF9800",       # Material orange, warmer
    "warning_light": "#FFB74D", # Light orange for hover
    "danger": "#F44336",        # Material red, less aggressive
    "danger_light": "#E57373",  # Light red for hover
    "text_primary": "#FFFFFF",  # Pure white for main text
    "text_secondary": "#E0E0E0", # Lighter gray for better readability
    "text_muted": "#B0B0B0",    # Less muted, more readable
    "text_disabled": "#888888", # Readable color for disabled text
    "border": "#404040",        # Lighter border for subtlety
    "hover": "#3C3C3C",         # Consistent hover state
    "surface": "#252526",       # For elevated surfaces
    "disabled": "#3A3A3A"      # Background for disabled buttons
}

# Modern font system - carefully selected for Windows compatibility
FONTS = {
    # Primary fonts (widely available on Windows)
    "family_primary": "Segoe UI",      # Modern Windows system font
    "family_secondary": "Calibri",     # Clean, readable alternative
    "family_monospace": "Consolas",    # Modern monospace for logs/code
    
    # Font sizes with better scaling
    "size_title": 20,          # Main titles
    "size_heading": 16,        # Section headings  
    "size_body": 13,           # Regular text (increased from 12)
    "size_small": 11,          # Small text
    "size_button": 12,         # Button text
    "size_logs": 11,           # Log text (increased from 9)
    
    # Font weights
    "weight_normal": "normal",
    "weight_bold": "bold",
}

class ToolTip:
    """Class for creating tooltips for CustomTkinter widgets"""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        
        # Bind events
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Motion>", self.on_motion)
        
    def on_enter(self, event=None):
        """Show tooltip when mouse enters widget"""
        self.show_tooltip(event)
        
    def on_leave(self, event=None):
        """Hide tooltip when mouse leaves widget"""
        self.hide_tooltip()
        
    def on_motion(self, event=None):
        """Update tooltip position on mouse motion"""
        if self.tooltip_window:
            self.update_tooltip_position(event)
            
    def show_tooltip(self, event=None):
        """Create and show tooltip window"""
        if self.tooltip_window or not self.text:
            return
            
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        # Style the tooltip
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=(FONTS["family_primary"], FONTS["size_small"]),
            wraplength=300
        )
        label.pack()
        
    def update_tooltip_position(self, event=None):
        """Update tooltip position"""
        if self.tooltip_window and event:
            x = event.x_root + 10
            y = event.y_root + 10
            self.tooltip_window.wm_geometry(f"+{x}+{y}")
            
    def hide_tooltip(self):
        """Destroy tooltip window"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
            
    def update_text(self, new_text):
        """Update tooltip text"""
        self.text = new_text

class UILogHandler(logging.Handler):
    def __init__(self, append_fn, level=logging.INFO, max_lines=500):
        super().__init__(level)
        self.append_fn = append_fn
        self.max_lines = max_lines
        self._buffer: List[str] = []

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self._buffer.append(msg)
        if len(self._buffer) > self.max_lines:
            self._buffer = self._buffer[-self.max_lines:]
        self.append_fn("\n".join(self._buffer))

class AppContext:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.last_hotkey_error: str | None = None
        cfg = self.config_manager
        whisper_cfg = cfg.get_whisper_config()
        audio_cfg = cfg.get_audio_config()
        clipboard_cfg = cfg.get_clipboard_config()
        audio_feedback_cfg = cfg.get_audio_feedback_config()

        backend_mode = whisper_cfg.get('backend_mode', 'local')
        base_url = whisper_cfg['local_url'] if backend_mode == 'local' else whisper_cfg.get('external_url', whisper_cfg['local_url'])
        
        # Convert HTTP URL to Wyoming format (remove http:// prefix)
        wyoming_url = self._convert_url_for_wyoming(base_url)
        
        canonical_model = whisper_cfg.get('model')
        # If model in config is already an alias, use it directly; otherwise convert
        if canonical_model in ALIAS_TO_MODEL:
            alias = canonical_model  # It's already an alias
            canonical = ALIAS_TO_MODEL[canonical_model]
        else:
            # It's a canonical name, convert to alias  
            alias = alias_for(canonical_model) if canonical_model else 'large-v3'
            canonical = canonical_model
            
        self.engine = WhisperEngine(
            base_url=wyoming_url,
            model_size=alias,
            language=whisper_cfg['language'],
            beam_size=whisper_cfg['beam_size'],
            remote_model=canonical
        )
        self.backend_mode = backend_mode
        self.clipboard_manager = ClipboardManager(
            key_simulation_delay=clipboard_cfg['key_simulation_delay'],
            auto_paste=clipboard_cfg['auto_paste'],
            preserve_clipboard=clipboard_cfg['preserve_clipboard']
        )
        self.audio_feedback = AudioFeedback(
            enabled=audio_feedback_cfg['enabled'],
            start_sound=audio_feedback_cfg['start_sound'],
            stop_sound=audio_feedback_cfg['stop_sound'],
            cancel_sound=audio_feedback_cfg['cancel_sound']
        )
        recorder = AudioRecorder(
            channels=audio_cfg['channels'],
            dtype=audio_cfg['dtype'],
            max_duration=audio_cfg['max_duration'],
            on_max_duration_reached=lambda data: self._on_max_duration(data)
        )
        self.state_manager = StateManager(
            audio_recorder=recorder,
            whisper_engine=self.engine,
            clipboard_manager=self.clipboard_manager,
            config_manager=self.config_manager,
            system_tray=None,
            audio_feedback=self.audio_feedback,
            docker_backend_manager=None  # Will be set later by LazyToTextUI
        )
        self.hotkey_listener: HotkeyListener | None = None
        self._mutex_handle = None

    def _convert_url_for_wyoming(self, url: str) -> str:
        """Converts HTTP URL to Wyoming format (removes http:// prefix)."""
        if not url:
            return url
        
        # Remove protocol for Wyoming
        if url.startswith('http://'):
            return url[7:]  # remove 'http://'
        elif url.startswith('https://'):
            return url[8:]  # remove 'https://'
        
        return url

    def enable_hotkeys(self):
        if self.hotkey_listener:
            return
        try:
            self._mutex_handle = guard_against_multiple_instances("LazyToTextUIHotkeys")
        except SystemExit:
            self._mutex_handle = None
            logging.getLogger(__name__).warning("Another instance already holds hotkey mutex; hotkeys disabled in this window.")
            return
        try:
            self.hotkey_listener = HotkeyListener(
                state_manager=self.state_manager,
                start_recording_hotkey=self.config_manager.get_setting('hotkey','start_recording_hotkey'),
                stop_recording_hotkey=self.config_manager.get_setting('hotkey','stop_recording_hotkey') if 'stop_recording_hotkey' in self.config_manager.config['hotkey'] else None,
                cancel_combination=None
            )
            logging.getLogger(__name__).info("Hotkeys enabled in UI")
            self.last_hotkey_error = None
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to start hotkeys: {e}")
            self.last_hotkey_error = str(e)
            self.hotkey_listener = None

    def disable_hotkeys(self):
        if self.hotkey_listener:
            try:
                self.hotkey_listener.stop_listening()
            except Exception:
                pass
            self.hotkey_listener = None
        self._mutex_handle = None
        logging.getLogger(__name__).info("Hotkeys disabled in UI")

    def _on_max_duration(self, audio_data):
        logging.getLogger(__name__).info("Max duration reached (UI callback)")
        self.state_manager.handle_max_recording_duration_reached(audio_data)

    def reconfigure_hotkeys_if_active(self):
        if not self.hotkey_listener:
            return
        try:
            self.disable_hotkeys()
            self.enable_hotkeys()
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to reconfigure hotkeys: {e}")

    def shutdown(self):
        self.disable_hotkeys()


class HistoryWindow:
    """Separate window for transcription history"""
    
    def __init__(self, parent_ui):
        self.parent_ui = parent_ui
        self.window = None
        self.history_entries = []
        self.selected_history_index = None
        self.widgets = {}
        # Cached list of entries currently shown in the tree (index-aligned
        # with tree row iids) so copy_entry() does not re-query + re-filter.
        self._filtered_entries: List = []
        # After-id for search debouncing (see _schedule_refresh).
        self._search_after_id = None
        
    def show(self):
        """Show history window"""
        if self.window and self.window.winfo_exists():
            # Window already exists, just raise it
            self.window.lift()
            self.window.focus()
            return
        
        # Create new window
        self.window = ctk.CTkToplevel(self.parent_ui.root)
        self.window.title("Lazy to Text - History")
        self.window.geometry("1000x600")
        self.window.minsize(800, 400)
        
        # Set window icon with delay (required for CTkToplevel on Windows)
        # Must use after() with 200ms+ delay for proper icon loading
        self.window.after(200, lambda: self._set_window_icon())
        
        # Make window appear on top
        self.window.lift()
        self.window.focus_force()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        
        # Create UI
        self.create_ui()
        
        # Load history
        self.refresh_history_display()
    
    def _set_window_icon(self):
        """Set icon for history window with proper CTkToplevel method"""
        try:
            if self.parent_ui.app_ico_path and Path(self.parent_ui.app_ico_path).exists():
                self.window.iconbitmap(str(self.parent_ui.app_ico_path))
                logging.getLogger(__name__).debug("History window icon set using ICO file")
                return
            
            ico_path = Path(resolve_asset_path("assets/tray_idle.ico"))
            if ico_path.exists():
                self.window.iconbitmap(str(ico_path))
                logging.getLogger(__name__).debug("History window icon set using assets ICO file")
                return
                
            png_path = Path(resolve_asset_path("assets/tray_idle.png"))
            if png_path.exists():
                photo = tk.PhotoImage(file=str(png_path))
                self.window.iconphoto(True, photo)
                self.window._icon = photo  # Keep reference
                logging.getLogger(__name__).debug("History window icon set using PNG file")
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to set history window icon: {e}")
        
    def create_ui(self):
        """Create history window UI"""
        main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            main_container,
            text="Transcription History",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=(0, 20), anchor="w")
        
        # Controls frame
        controls_frame = ctk.CTkFrame(
            main_container,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        controls_frame.pack(fill="x", pady=(0, 12))
        
        controls_inner = ctk.CTkFrame(controls_frame, fg_color="transparent")
        controls_inner.pack(fill="x", padx=16, pady=12)
        
        # Search
        self.widgets['search'] = ctk.CTkEntry(
            controls_inner,
            placeholder_text="Search...",
            width=240,
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        self.widgets['search'].bind('<KeyRelease>', lambda e: self._schedule_refresh())
        self.widgets['search'].pack(side="left", padx=(0, 12))
        
        # Filter
        self.widgets['filter'] = ctk.CTkOptionMenu(
            controls_inner,
            values=["All", "Today", "Last 7 days", "Last 30 days"],
            command=lambda v: self.refresh_history_display(),
            width=140,
            height=34,
            corner_radius=8,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_light"],
            text_color="#FFFFFF",
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        self.widgets['filter'].set("All")
        self.widgets['filter'].pack(side="left", padx=(0, 12))
        
        # Spacer
        spacer = ctk.CTkLabel(controls_inner, text="")
        spacer.pack(side="left", fill="x", expand=True)
        
        # Buttons
        self.widgets['export_button'] = ctk.CTkButton(
            controls_inner,
            text="Export",
            command=self.export_history,
            width=90,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["success"],
            hover_color=COLORS["success_light"],
            text_color="#FFFFFF"
        )
        self.widgets['export_button'].pack(side="right", padx=(8, 0))
        
        self.widgets['clear_button'] = ctk.CTkButton(
            controls_inner,
            text="Clear All",
            command=self.clear_history,
            width=90,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["danger"],
            hover_color=COLORS["danger_light"],
            text_color="#FFFFFF"
        )
        self.widgets['clear_button'].pack(side="right", padx=(8, 0))
        
        # History list container
        list_container = ctk.CTkFrame(
            main_container,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        list_container.pack(fill="both", expand=True)

        # Use native ttk.Treeview for the entries list. CTk widgets are
        # far too slow for large histories — ttk.Treeview handles thousands
        # of rows without the per-row Canvas overhead CTk pays for rounded
        # corners. Double-click copies the row; single-click updates the
        # selected index used by _copy_selected().
        from tkinter import ttk

        # The tree_frame background is intentionally matched to the row
        # colour so the 1px gap between the tree and the rounded
        # list_container doesn't flash as a lighter strip.
        tree_frame = tk.Frame(list_container, bg=COLORS["primary"], bd=0,
                              highlightthickness=0)
        tree_frame.pack(fill="both", expand=True, padx=14, pady=14)

        style = ttk.Style(self.window)
        # "clam" is the only built-in ttk theme on Windows that honours
        # background/foreground colour configuration on Treeview and
        # Scrollbar. "vista" (the default) ignores most colour options.
        try:
            style.theme_use("clam")
        except Exception:
            pass

        row_bg = COLORS["primary"]          # #1E1E1E
        row_bg_alt = COLORS["secondary"]    # #2D2D30 — subtle zebra
        header_bg = COLORS["surface"]       # #252526 — matches surrounding card
        sel_bg = COLORS["accent"]           # #007ACC

        # --- Treeview body --------------------------------------------
        style.configure(
            "History.Treeview",
            background=row_bg,
            fieldbackground=row_bg,
            foreground=COLORS["text_primary"],
            bordercolor=row_bg,
            lightcolor=row_bg,
            darkcolor=row_bg,
            borderwidth=0,
            relief="flat",
            rowheight=30,
            font=(FONTS["family_primary"], 10),
        )
        style.map(
            "History.Treeview",
            background=[("selected", sel_bg)],
            foreground=[("selected", "#FFFFFF")],
            # Kill the dotted focus rectangle clam draws by default.
            bordercolor=[("focus", row_bg)],
            lightcolor=[("focus", row_bg)],
            darkcolor=[("focus", row_bg)],
        )

        # --- Treeview header ------------------------------------------
        style.configure(
            "History.Treeview.Heading",
            background=header_bg,
            foreground=COLORS["text_secondary"],
            bordercolor=header_bg,
            lightcolor=header_bg,
            darkcolor=header_bg,
            relief="flat",
            borderwidth=0,
            padding=(10, 8),
            font=(FONTS["family_primary"], 10, "bold"),
        )
        style.map(
            "History.Treeview.Heading",
            background=[("active", COLORS["hover"]), ("pressed", COLORS["hover"])],
            foreground=[("active", COLORS["text_primary"])],
            relief=[("pressed", "flat"), ("active", "flat")],
        )

        # Remove the chunky clam borders ttk draws around the whole tree.
        try:
            style.layout(
                "History.Treeview",
                [("History.Treeview.treearea", {"sticky": "nswe"})],
            )
        except Exception:
            pass

        # --- Scrollbar (match accent colour) --------------------------
        style.configure(
            "History.Vertical.TScrollbar",
            background=COLORS["secondary"],
            troughcolor=row_bg,
            bordercolor=row_bg,
            arrowcolor=COLORS["text_muted"],
            lightcolor=COLORS["secondary"],
            darkcolor=COLORS["secondary"],
            relief="flat",
            borderwidth=0,
            gripcount=0,
            arrowsize=14,
        )
        style.map(
            "History.Vertical.TScrollbar",
            background=[("active", COLORS["accent"]), ("pressed", COLORS["accent_light"])],
            arrowcolor=[("active", COLORS["text_primary"])],
        )

        columns = ("time", "text", "model", "language", "duration")
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            style="History.Treeview",
            selectmode="browse",
        )
        tree.heading("time", text="Time", anchor="w")
        tree.heading("text", text="Text", anchor="w")
        tree.heading("model", text="Model", anchor="w")
        tree.heading("language", text="Lang", anchor="center")
        tree.heading("duration", text="Duration", anchor="e")
        tree.column("time", width=150, anchor="w", stretch=False)
        tree.column("text", width=520, anchor="w", stretch=True)
        tree.column("model", width=90, anchor="w", stretch=False)
        tree.column("language", width=60, anchor="center", stretch=False)
        tree.column("duration", width=80, anchor="e", stretch=False)

        # Zebra striping — configure the two tags used by refresh_history_display.
        tree.tag_configure("odd", background=row_bg)
        tree.tag_configure("even", background=row_bg_alt)

        vscroll = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=tree.yview,
            style="History.Vertical.TScrollbar",
        )
        tree.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)

        tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        tree.bind("<Double-1>", lambda e: self._copy_selected())
        tree.bind("<Return>", lambda e: self._copy_selected())

        self.widgets['tree'] = tree

        # Small footer with row count + explicit Copy button so the action
        # is discoverable even without knowing about double-click.
        footer = ctk.CTkFrame(main_container, fg_color="transparent")
        footer.pack(fill="x", pady=(8, 0))

        self.widgets['count_label'] = ctk.CTkLabel(
            footer,
            text="",
            font=ctk.CTkFont(family=FONTS["family_primary"], size=11),
            text_color=COLORS["text_muted"],
        )
        self.widgets['count_label'].pack(side="left")

        self.widgets['copy_button'] = ctk.CTkButton(
            footer,
            text="Copy Selected",
            command=self._copy_selected,
            width=130,
            height=30,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_light"],
            text_color="#FFFFFF",
        )
        self.widgets['copy_button'].pack(side="right")


    def get_filtered_entries(self):
        """Get filtered history entries"""
        try:
            if not self.parent_ui.ctx.state_manager.history_manager:
                return []
            
            # Get base entries based on filter
            filter_value = self.widgets['filter'].get()
            
            if filter_value == "Today":
                entries = self.parent_ui.ctx.state_manager.history_manager.get_entries_by_date(1)
            elif filter_value == "Last 7 days":
                entries = self.parent_ui.ctx.state_manager.history_manager.get_entries_by_date(7)
            elif filter_value == "Last 30 days":
                entries = self.parent_ui.ctx.state_manager.history_manager.get_entries_by_date(30)
            else:
                entries = self.parent_ui.ctx.state_manager.history_manager.get_entries(limit=200)
            
            # Apply search
            search_text = self.widgets['search'].get().strip()
            if search_text:
                entries = [e for e in entries if search_text.lower() in e.text.lower()]
            
            return entries
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to get filtered entries: {e}")
            return []
    
    def _schedule_refresh(self, delay_ms: int = 300):
        """Debounced trigger for refresh_history_display.

        Called from the search KeyRelease binding so typing doesn't rebuild
        the whole tree on every keystroke. Cancels any previously scheduled
        refresh and reschedules one `delay_ms` in the future.
        """
        try:
            if self._search_after_id is not None:
                self.window.after_cancel(self._search_after_id)
        except Exception:
            pass
        self._search_after_id = self.window.after(delay_ms, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self):
        self._search_after_id = None
        self.refresh_history_display()

    def refresh_history_display(self):
        """Refresh history display (ttk.Treeview-based, fast rebuild)."""
        try:
            tree = self.widgets.get('tree')
            if tree is None:
                return

            # Clear tree
            tree.delete(*tree.get_children())
            self.selected_history_index = None

            # Get filtered entries and cache them for copy_entry()
            entries = self.get_filtered_entries()
            self._filtered_entries = entries

            # Bulk insert. Truncating text keeps the row height predictable
            # and the insert itself cheap. Alternating tags give the zebra
            # striping configured in create_ui().
            for i, entry in enumerate(entries):
                text = entry.text if len(entry.text) <= 200 else entry.text[:197] + "..."
                # Treeview renders newlines poorly — collapse to spaces.
                text = text.replace("\r", " ").replace("\n", " ")
                tree.insert(
                    "",
                    "end",
                    iid=str(i),
                    values=(
                        entry.datetime_str,
                        text,
                        entry.model,
                        entry.language,
                        f"{entry.duration:.1f}s",
                    ),
                    tags=("even" if i % 2 == 0 else "odd",),
                )

            # Update footer count label
            count_label = self.widgets.get('count_label')
            if count_label is not None:
                count_label.configure(text=f"{len(entries)} entries")

        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to refresh history: {e}")

    def _on_tree_select(self, _event=None):
        """Track selected row index so _copy_selected() knows what to copy."""
        try:
            tree = self.widgets.get('tree')
            if tree is None:
                return
            sel = tree.selection()
            if not sel:
                self.selected_history_index = None
                return
            try:
                self.selected_history_index = int(sel[0])
            except ValueError:
                self.selected_history_index = None
        except Exception as e:
            logging.getLogger(__name__).debug(f"Tree select failed: {e}")

    def _copy_selected(self):
        """Copy currently selected entry to clipboard."""
        idx = self.selected_history_index
        if idx is None:
            return
        try:
            if 0 <= idx < len(self._filtered_entries):
                entry = self._filtered_entries[idx]
                success = self.parent_ui.ctx.clipboard_manager.copy_text(entry.text)
                if success:
                    logging.getLogger(__name__).info("Copied to clipboard", extra={'user_message': True})
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to copy entry: {e}")
    
    def export_history(self):
        """Export history to file"""
        try:
            if not self.parent_ui.ctx.state_manager.history_manager:
                return
            
            import tkinter.filedialog as fd
            filename = fd.asksaveasfilename(
                title="Export History",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                success = self.parent_ui.ctx.state_manager.history_manager.export_to_text(filename)
                if success:
                    logging.getLogger(__name__).info(f"History exported to {filename}", extra={'user_message': True})
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to export: {e}")
    
    def clear_history(self):
        """Clear all history"""
        try:
            if not self.parent_ui.ctx.state_manager.history_manager:
                return
            
            import tkinter.messagebox as mb
            result = mb.askyesno(
                "Clear History",
                "Are you sure you want to clear all history?\nThis cannot be undone.",
                icon="warning",
                parent=self.window
            )
            
            if result:
                self.parent_ui.ctx.state_manager.history_manager.clear_history()
                self.refresh_history_display()
                logging.getLogger(__name__).info("History cleared", extra={'user_message': True})
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to clear: {e}")


class LazyToTextUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Lazy to Text")
        self.root.geometry("900x650")  # Larger, more modern proportions
        self.root.minsize(800, 600)  # Minimum size for usability
        
        # Set application icon
        self.set_app_icon()
        
        # Important: set window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize application context
        self.ctx = AppContext()
        self.docker_mgr = DockerBackendManager()
        
        # Set docker manager in state manager after initialization
        self.ctx.state_manager.docker_backend_manager = self.docker_mgr
        
        # State flags
        self.quitting_flag = False
        self.window_visible = True
        self.system_tray: Optional[SystemTray] = None
        
        # UI elements
        self.widgets: Dict[str, Any] = {}
        self.app_icon = None
        self.app_ico_path = None
        
        # Track changes for save buttons
        self.hotkey_settings_changed = False
        
        # Save original values for comparison
        self.original_hotkey_settings = {}
        
        # Threading for background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.polling_running = False
        
        # Debouncing for UI updates
        self.scheduled_history_update_id = None
        self.last_history_update_time = 0
        self.history_update_debounce_ms = 500  # 500ms debounce
        
        # Lock for backend operations to prevent button flickering
        self.backend_operation_in_progress = False
        self.scheduled_button_update_id = None
        self.last_button_update_time = 0
        
        # History window
        self.history_window = None
        
        self.setup_logging()
        self.setup_system_tray()
        self.create_widgets()
        self.start_polling()
        self.setup_hotkeys()

    def set_app_icon(self):
        """Set application icon using the same icon as system tray"""
        try:
            # Try to load the same icon as used in system tray
            png_path = Path(resolve_asset_path("assets/tray_idle.png"))
            ico_path = png_path.parent / "tray_idle.ico"

            # Reset cached references before loading
            self.app_icon = None
            self.app_ico_path = None
            
            if png_path.exists():
                # Convert PNG to ICO if ICO doesn't exist
                if not ico_path.exists():
                    try:
                        img = Image.open(png_path)
                        # Convert to RGBA if not already
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        # Save as ICO with multiple sizes
                        img.save(ico_path, format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64)])
                        logging.getLogger(__name__).debug(f"Created ICO file: {ico_path}")
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Failed to create ICO file: {e}")
                
                # Try different methods to set the icon
                success = False
                
                # Method 1: Try iconbitmap with ICO file
                if ico_path.exists():
                    try:
                        self.root.wm_iconbitmap(str(ico_path))
                        success = True
                        self.app_ico_path = ico_path
                        logging.getLogger(__name__).debug("Application icon set using iconbitmap (ICO)")
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"iconbitmap failed: {e}")
                
                # Method 2: Try iconphoto with PNG
                if not success:
                    try:
                        photo = tk.PhotoImage(file=str(png_path))
                        self.root.call("wm", "iconphoto", self.root._w, photo)
                        self.root.wm_iconphoto(True, photo)
                        self.app_icon = photo  # Keep reference
                        success = True
                        logging.getLogger(__name__).debug("Application icon set using iconphoto (PNG)")
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"iconphoto failed: {e}")
                
                if success:
                    logging.getLogger(__name__).debug("Application icon set successfully")
                else:
                    logging.getLogger(__name__).warning("All icon setting methods failed")
            else:
                logging.getLogger(__name__).warning(f"Icon file not found: {png_path}")
                
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to set application icon: {e}")

    def apply_icon_to_window(self, window: tk.Toplevel) -> None:
        """Apply application icon to another window"""
        if not window:
            return

        applied = False
        try:
            # Try ICO file first
            if self.app_ico_path and Path(self.app_ico_path).exists():
                try:
                    window.wm_iconbitmap(str(self.app_ico_path))
                    applied = True
                    logging.getLogger(__name__).debug("Applied icon to window using iconbitmap (ICO)")
                except Exception as e:
                    logging.getLogger(__name__).debug(f"iconbitmap failed for window: {e}")
                    applied = False

            # Try PNG via iconphoto with main window's PhotoImage reference
            if not applied and self.app_icon is not None:
                try:
                    window.call("wm", "iconphoto", window._w, self.app_icon)
                    window.wm_iconphoto(True, self.app_icon)
                    applied = True
                    logging.getLogger(__name__).debug("Applied icon to window using iconphoto (main PhotoImage)")
                except Exception as e:
                    logging.getLogger(__name__).debug(f"iconphoto with main icon failed for window: {e}")
                    applied = False

            # Fallback: create new PhotoImage from PNG
            if not applied:
                png_path = Path(resolve_asset_path("assets/tray_idle.png"))
                ico_path = png_path.parent / "tray_idle.ico"
                
                # Try ICO file if it exists
                if not applied and ico_path.exists():
                    try:
                        window.wm_iconbitmap(str(ico_path))
                        applied = True
                        logging.getLogger(__name__).debug("Applied icon to window using fallback iconbitmap (ICO)")
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Fallback iconbitmap failed for window: {e}")
                
                # Try PNG
                if not applied and png_path.exists():
                    try:
                        photo = tk.PhotoImage(file=str(png_path))
                        window.call("wm", "iconphoto", window._w, photo)
                        window.wm_iconphoto(True, photo)
                        window._icon = photo  # Keep reference to prevent garbage collection
                        applied = True
                        logging.getLogger(__name__).debug("Applied icon to window using fallback iconphoto (PNG)")
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Fallback iconphoto failed for window: {e}")
            
            if not applied:
                logging.getLogger(__name__).warning("Failed to apply icon to window with all methods")
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to apply icon to window: {e}")

    def setup_logging(self):
        """Setup logging system"""
        root_logger = logging.getLogger()
        early_handler: EarlyBufferHandler | None = None
        if not any(isinstance(h, EarlyBufferHandler) for h in root_logger.handlers):
            early_handler = EarlyBufferHandler()
            root_logger.addHandler(early_handler)

        setup_logging(self.ctx.config_manager)

        if early_handler:
            try:
                early_handler.replay_to(root_logger)
            except Exception:
                pass
        try:
            import os
            log_cfg = self.ctx.config_manager.get_logging_config()
            log_path = os.path.join(get_project_logs_path(), log_cfg['file']['filename'])
            logging.getLogger(__name__).info(f"UI log file path: {log_path}")
        except Exception:
            pass

    def setup_system_tray(self):
        """Setup system tray"""
        tray_cfg = self.ctx.config_manager.get_system_tray_config() if 'system_tray' in self.ctx.config_manager.config else {'enabled': False}
        
        if tray_cfg.get('enabled', False):
            try:
                self.system_tray = SystemTray(
                    state_manager=None,  # Set later
                    tray_config=tray_cfg,
                    config_manager=self.ctx.config_manager,
                    show_window_callback=self.show_window,
                    is_window_visible_callback=self.is_window_visible
                )
                self.system_tray.set_hide_window_callback(self.hide_window)
                
                # Attach to state_manager
                self.ctx.state_manager.system_tray = self.system_tray
                self.system_tray.attach_state_manager(self.ctx.state_manager)
                
                # Set quit callback
                self.system_tray.set_quit_callback(self.quit_via_tray)
                
                # Start tray
                started = self.system_tray.start()
                if not started:
                    logging.getLogger(__name__).warning("System tray not started")
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"System tray init failed: {e}")
                self.system_tray = None

    def _install_smooth_scroll(self, scrollable_frame):
        """Replace CTkScrollableFrame's instant canvas scroll with an eased
        animation.

        Why: CTkScrollableFrame's default mouse-wheel handler calls
        `canvas.yview_scroll(±3, "units")` per wheel notch. Each call
        forces Tk to repaint the entire canvas, and on Windows DWM that
        produces visible tearing/stepping for large panels.

        How: we monkey-patch `canvas.yview_scroll` so that when it's
        called with units (the mouse-wheel path), it instead sets an
        animation target and schedules small per-frame interpolation
        steps via `after(16, ...)` (~60 fps). Direct scrollbar drags
        still work because they call `canvas.yview("moveto", ...)`,
        which we don't touch.
        """
        try:
            canvas = getattr(scrollable_frame, "_parent_canvas", None)
            if canvas is None:
                return

            # Remove the focus highlight border — it flashes on scroll.
            try:
                canvas.configure(highlightthickness=0, bd=0)
            except Exception:
                pass

            state = {
                "target_top": None,  # desired top fraction
                "anim_id": None,
                "orig": canvas.yview_scroll,
            }

            def _step():
                state["anim_id"] = None
                target = state["target_top"]
                if target is None:
                    return
                try:
                    cur_top, cur_bot = canvas.yview()
                except Exception:
                    return
                view_size = max(cur_bot - cur_top, 0.0001)
                max_top = max(0.0, 1.0 - view_size)
                target = max(0.0, min(max_top, target))

                diff = target - cur_top
                # Snap when close enough — avoids infinite micro-stepping.
                if abs(diff) < 0.0015:
                    try:
                        canvas.yview_moveto(target)
                    except Exception:
                        pass
                    state["target_top"] = None
                    return

                # Ease-out: move 30% of the remaining distance per frame.
                new_top = cur_top + diff * 0.30
                try:
                    canvas.yview_moveto(new_top)
                except Exception:
                    return
                state["anim_id"] = canvas.after(16, _step)

            def smooth_yview_scroll(number, what):
                # Only intercept the "units" path (mouse wheel / arrow keys).
                # Let "pages" and anything else fall through unchanged.
                if what != "units":
                    try:
                        return state["orig"](number, what)
                    except Exception:
                        return None
                try:
                    cur_top, cur_bot = canvas.yview()
                except Exception:
                    return None
                view_size = max(cur_bot - cur_top, 0.0001)
                # Each "unit" of wheel-delta ≈ 5% of viewport height.
                step = view_size * 0.05 * float(number)
                base = state["target_top"] if state["target_top"] is not None else cur_top
                state["target_top"] = base + step
                if state["anim_id"] is None:
                    _step()
                return None

            canvas.yview_scroll = smooth_yview_scroll
        except Exception as e:
            logging.getLogger(__name__).debug(f"Smooth scroll install failed: {e}")

    def _stop_scroll_propagation(self, scrollable_widget):
        """Stop scroll events from propagating to parent scrollable frame.
        
        When mouse is over a nested scrollable widget, allow it to scroll
        but prevent events from bubbling up to parent scrollable containers.
        """
        # Store reference to widget's canvas for scroll handling
        canvas = None
        try:
            if hasattr(scrollable_widget, '_parent_canvas'):
                canvas = scrollable_widget._parent_canvas
        except Exception:
            pass
        
        def on_mousewheel(event):
            if canvas:
                # Scroll the nested widget's canvas
                if event.delta:
                    # Windows and MacOS
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif event.num == 4:
                    # Linux scroll up
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    # Linux scroll down
                    canvas.yview_scroll(1, "units")
            # Stop propagation to parent
            return "break"
        
        # Bind scroll events to the widget
        scrollable_widget.bind("<MouseWheel>", on_mousewheel)
        scrollable_widget.bind("<Button-4>", on_mousewheel)
        scrollable_widget.bind("<Button-5>", on_mousewheel)
        
        # Also bind to all child widgets recursively
        def bind_children(widget):
            try:
                widget.bind("<MouseWheel>", on_mousewheel, add="+")
                widget.bind("<Button-4>", on_mousewheel, add="+")
                widget.bind("<Button-5>", on_mousewheel, add="+")
                for child in widget.winfo_children():
                    bind_children(child)
            except Exception:
                pass
        
        try:
            # Bind to internal frame if it exists (for CTkScrollableFrame)
            if hasattr(scrollable_widget, '_parent_frame'):
                bind_children(scrollable_widget._parent_frame)
        except Exception:
            pass

    def create_widgets(self):
        """Create interface widgets"""
        # Create main container with modern padding
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=24, pady=20)
        
        # Create main scrollable frame for better UX with large content.
        # NOTE: fg_color MUST be an explicit colour (not "transparent") —
        # CTkScrollableFrame is a Canvas under the hood, and a transparent
        # fill forces CTk to composite the parent chain on every scroll
        # step, which is the primary source of the tearing/flicker users
        # see on a fast mouse-wheel scroll. An opaque background lets the
        # Canvas blit straight to screen.
        main_frame = ctk.CTkScrollableFrame(
            main_container,
            fg_color=COLORS["primary"],
            scrollbar_button_color=COLORS["accent"],
            scrollbar_button_hover_color=COLORS["hover"],
        )
        main_frame.pack(fill="both", expand=True)
        # Install smooth-scroll interpolation on the underlying canvas.
        # Default CTk mousewheel handler jumps 3 units per notch which
        # causes visible stepping/tearing through Windows DWM. We redirect
        # those calls into an eased animation.
        self._install_smooth_scroll(main_frame)
        
        # Model section
        self.create_model_section(main_frame)

        # Status section (new) placed after model for organic grouping
        self.create_status_section(main_frame)
        
        # Modern spacing instead of visible separators
        
        # Hotkeys section
        self.create_hotkeys_section(main_frame)
        
        # Modern spacing
        
        # History section (only if enabled)
        history_config = self.ctx.config_manager.get_history_config()
        if history_config.get('enabled', True):
            self.create_history_section(main_frame)
        
        # Modern spacing
        
        # Logs section
        self.create_logs_section(main_frame)
        
        # Setup UI logging
        self.setup_ui_logging()
        
        # Save original values after creating UI
        self.save_original_values()
        
        # Set history update callback
        self.ctx.state_manager.history_update_callback = self.on_history_updated

    def create_model_section(self, parent):
        """Create model settings section"""
        # Title with improved typography
        model_title = ctk.CTkLabel(
            parent, 
            text="Model Configuration", 
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_title"], weight=FONTS["weight_bold"]),
            text_color=COLORS["text_primary"]
        )
        model_title.pack(pady=(0, 20), anchor="w")  # More space below
        
        # Backend frame with improved colors
        backend_frame = ctk.CTkFrame(
            parent, 
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        backend_frame.pack(fill="x", padx=0, pady=(0, 16))  # More space
        
        # Add padding to backend frame
        backend_inner = ctk.CTkFrame(backend_frame, fg_color="transparent")
        backend_inner.pack(fill="x", padx=16, pady=12)
        
        self.widgets['backend_mode'] = ctk.CTkOptionMenu(
            backend_inner,
            values=['local', 'external'],
            command=self.on_backend_mode_change,
            width=140,
            height=34,  # Slightly taller for better touch targets
            corner_radius=8,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_light"],
            dropdown_hover_color=COLORS["hover"],
            text_color="#FFFFFF",
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_normal"])
        )
        self.widgets['backend_mode'].set(self.ctx.backend_mode)
        self.widgets['backend_mode'].pack(side="left", padx=(0, 12))
        
        self.widgets['external_url'] = ctk.CTkEntry(
            backend_inner,
            placeholder_text="Server URL (e.g., localhost:10300)",
            width=240,
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        self.widgets['external_url'].insert(0, self.ctx.config_manager.get_setting('whisper','external_url'))
        self.widgets['external_url'].pack(side="left", padx=(0, 12))
        
        # Backend buttons frame with modern styling
        self.widgets['backend_buttons_frame'] = ctk.CTkFrame(backend_inner, fg_color="transparent")
        self.widgets['backend_buttons_frame'].pack(side="left", padx=0)
        
        # Model controls with improved styling
        model_controls_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        model_controls_frame.pack(fill="x", padx=0, pady=(0, 16))
        
        # Add padding to model controls
        model_inner = ctk.CTkFrame(model_controls_frame, fg_color="transparent")
        model_inner.pack(fill="x", padx=16, pady=12)
        
        self.widgets['model_dropdown'] = ctk.CTkOptionMenu(
            model_inner,
            values=MODEL_OPTIONS,
            command=self.on_model_change,
            width=140,
            height=34,
            corner_radius=8,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_light"],
            text_color="#FFFFFF",
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_normal"])
        )
        current_model = self.ctx.engine.model_size if self.ctx.engine.model_size in MODEL_OPTIONS else MODEL_OPTIONS[0]
        self.widgets['model_dropdown'].set(current_model)
        self.widgets['model_dropdown'].pack(side="left", padx=(0, 12))
        
        self.widgets['beam_size'] = ctk.CTkEntry(
            model_inner,
            placeholder_text="Beam size",
            width=120,
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        self.widgets['beam_size'].insert(0, str(self.ctx.engine.beam_size))
        self.widgets['beam_size'].pack(side="left", padx=(0, 12))
        
        self.widgets['language_dropdown'] = ctk.CTkOptionMenu(
            model_inner,
            values=LANGUAGE_OPTIONS,
            command=self.on_language_change,
            width=100,
            height=34,
            corner_radius=8,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_light"],
            text_color="#FFFFFF",
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_normal"])
        )
        current_lang = self.ctx.engine.language if self.ctx.engine.language in LANGUAGE_OPTIONS else 'ru'
        self.widgets['language_dropdown'].set(current_lang)
        self.widgets['language_dropdown'].pack(side="left", padx=(0, 12))
        
        self.widgets['switch_button'] = ctk.CTkButton(
            model_inner,
            text="Apply Changes",
            command=self.switch_model,
            width=140,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["success"],
            hover_color=COLORS["success_light"],
            text_color="#FFFFFF"
        )
        self.widgets['switch_button'].pack(side="left", padx=0)
        
        # Progress bar (initially hidden)
        self.widgets['progress_bar'] = ctk.CTkProgressBar(model_controls_frame)
        # Don't pack, show when needed
        
        # Server status & container model labels (hidden; kept for internal updates)
        self.widgets['server_status'] = ctk.CTkLabel(parent, text="Server status: ?", text_color="gray")
        self.widgets['container_model'] = ctk.CTkLabel(parent, text="Container model: ?", text_color="gray")
        # Не вызываем pack, чтобы не дублировать с панелью Status
        
        # Create and update backend buttons
        self.create_backend_buttons()
        self.update_backend_buttons_state()
        
        # Add tooltips to model section
        self.add_model_tooltips()

    def create_status_section(self, parent):
        """Create modern status panel with visual indicators"""
        # Status title with better typography
        status_title = ctk.CTkLabel(
            parent, 
            text="System Status", 
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_title"], weight=FONTS["weight_bold"]),
            text_color=COLORS["text_primary"]
        )
        status_title.pack(pady=(20, 20), anchor="w")

        status_container = ctk.CTkFrame(
            parent, 
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        status_container.pack(fill="x", padx=0, pady=(0, 16))
        
        # Inner frame for padding
        status_inner = ctk.CTkFrame(status_container, fg_color="transparent")
        status_inner.pack(fill="x", padx=16, pady=16)

        # Configure grid layout - 2 columns, equal height rows
        status_inner.grid_columnconfigure(0, weight=1)
        status_inner.grid_columnconfigure(1, weight=1)
        status_inner.grid_rowconfigure(0, weight=1)
        
        # Docker Status Card (only for local mode)
        if self.ctx.backend_mode == 'local':
            docker_card = ctk.CTkFrame(
                status_inner,
                fg_color=COLORS["primary"],
                corner_radius=8,
                border_width=1,
                border_color=COLORS["border"]
            )
            docker_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
            
            docker_card_inner = ctk.CTkFrame(docker_card, fg_color="transparent")
            docker_card_inner.pack(fill="both", expand=True, padx=12, pady=12)
            
            # Docker status indicator (colored dot + text)
            docker_status_row = ctk.CTkFrame(docker_card_inner, fg_color="transparent")
            docker_status_row.pack(fill="x", pady=(0, 8))
            
            self.widgets['docker_status_indicator'] = ctk.CTkLabel(
                docker_status_row,
                text="●",
                font=ctk.CTkFont(size=20),
                text_color="gray",
                width=30
            )
            self.widgets['docker_status_indicator'].pack(side="left")
            
            self.widgets['docker_status_text'] = ctk.CTkLabel(
                docker_status_row,
                text="Docker",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_bold"]),
                text_color=COLORS["text_primary"],
                anchor="w"
            )
            self.widgets['docker_status_text'].pack(side="left", fill="x", expand=True)
            
            # Docker details
            self.widgets['docker_details'] = ctk.CTkLabel(
                docker_card_inner,
                text="Checking...",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_small"]),
                text_color=COLORS["text_secondary"],
                anchor="w"
            )
            self.widgets['docker_details'].pack(fill="x")
            
            # Container Status Card
            container_card = ctk.CTkFrame(
                status_inner,
                fg_color=COLORS["primary"],
                corner_radius=8,
                border_width=1,
                border_color=COLORS["border"]
            )
            container_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
            
            container_card_inner = ctk.CTkFrame(container_card, fg_color="transparent")
            container_card_inner.pack(fill="both", expand=True, padx=12, pady=12)
            
            # Container status indicator
            container_status_row = ctk.CTkFrame(container_card_inner, fg_color="transparent")
            container_status_row.pack(fill="x", pady=(0, 8))
            
            self.widgets['container_status_indicator'] = ctk.CTkLabel(
                container_status_row,
                text="●",
                font=ctk.CTkFont(size=20),
                text_color="gray",
                width=30
            )
            self.widgets['container_status_indicator'].pack(side="left")
            
            self.widgets['container_status_text'] = ctk.CTkLabel(
                container_status_row,
                text="Container",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_bold"]),
                text_color=COLORS["text_primary"],
                anchor="w"
            )
            self.widgets['container_status_text'].pack(side="left", fill="x", expand=True)
            
            # Container details (ID and Image)
            self.widgets['container_details'] = ctk.CTkLabel(
                container_card_inner,
                text="No container",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_small"]),
                text_color=COLORS["text_secondary"],
                anchor="w",
                justify="left",
                wraplength=300
            )
            self.widgets['container_details'].pack(fill="x")

            # View Logs button
            self.widgets['container_logs_button'] = ctk.CTkButton(
                container_card_inner,
                text="View Logs",
                command=self.open_container_logs,
                width=120,
                height=30,
                corner_radius=8,
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_light"],
                text_color="#FFFFFF"
            )
            self.widgets['container_logs_button'].pack(pady=(8, 0), anchor="w")
        else:
            # External mode - show Server and Model
            server_card = ctk.CTkFrame(
                status_inner,
                fg_color=COLORS["primary"],
                corner_radius=8,
                border_width=1,
                border_color=COLORS["border"]
            )
            server_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
            
            server_card_inner = ctk.CTkFrame(server_card, fg_color="transparent")
            server_card_inner.pack(fill="both", expand=True, padx=12, pady=12)
            
            server_status_row = ctk.CTkFrame(server_card_inner, fg_color="transparent")
            server_status_row.pack(fill="x", pady=(0, 8))
            
            self.widgets['server_status_indicator'] = ctk.CTkLabel(
                server_status_row,
                text="●",
                font=ctk.CTkFont(size=20),
                text_color="gray",
                width=30
            )
            self.widgets['server_status_indicator'].pack(side="left")
            
            self.widgets['server_status_text'] = ctk.CTkLabel(
                server_status_row,
                text="External Server",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_bold"]),
                text_color=COLORS["text_primary"],
                anchor="w"
            )
            self.widgets['server_status_text'].pack(side="left", fill="x", expand=True)
            
            self.widgets['server_details'] = ctk.CTkLabel(
                server_card_inner,
                text="Checking...",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_small"]),
                text_color=COLORS["text_secondary"],
                anchor="w",
                justify="left"
            )
            self.widgets['server_details'].pack(fill="x")
            
            # Model configuration card
            model_card = ctk.CTkFrame(
                status_inner,
                fg_color=COLORS["primary"],
                corner_radius=8,
                border_width=1,
                border_color=COLORS["border"]
            )
            model_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew")
            
            model_card_inner = ctk.CTkFrame(model_card, fg_color="transparent")
            model_card_inner.pack(fill="both", expand=True, padx=12, pady=12)
            
            model_status_row = ctk.CTkFrame(model_card_inner, fg_color="transparent")
            model_status_row.pack(fill="x", pady=(0, 8))
            
            self.widgets['model_status_indicator'] = ctk.CTkLabel(
                model_status_row,
                text="●",
                font=ctk.CTkFont(size=20),
                text_color="gray",
                width=30
            )
            self.widgets['model_status_indicator'].pack(side="left")
            
            self.widgets['model_status_text'] = ctk.CTkLabel(
                model_status_row,
                text="Model Config",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_bold"]),
                text_color=COLORS["text_primary"],
                anchor="w"
            )
            self.widgets['model_status_text'].pack(side="left", fill="x", expand=True)
            
            self.widgets['model_details'] = ctk.CTkLabel(
                model_card_inner,
                text=f"{self.ctx.engine.model_size} • {self.ctx.engine.language} • beam {self.ctx.engine.beam_size}",
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_small"]),
                text_color=COLORS["text_secondary"],
                anchor="w",
                justify="left"
            )
            self.widgets['model_details'].pack(fill="x")
        
        # Initial fill
        self.refresh_status_panel()

    def refresh_status_panel(self):
        """Update status panel with current Docker and Container information"""
        try:
            if self.ctx.backend_mode == 'local':
                # Check Docker availability
                docker_available = self.docker_mgr.is_available()
                
                # Update Docker status
                if self.widgets.get('docker_status_indicator'):
                    if docker_available:
                        self.widgets['docker_status_indicator'].configure(text_color='green')
                        if self.widgets.get('docker_details'):
                            self.widgets['docker_details'].configure(text="Docker Desktop available")
                    else:
                        self.widgets['docker_status_indicator'].configure(text_color='red')
                        if self.widgets.get('docker_details'):
                            self.widgets['docker_details'].configure(text="Docker Desktop not running")
                
                # Get container information
                if docker_available:
                    container_status, health_ok = self.docker_mgr.get_health_and_status(self.ctx.engine.health_check)
                    details = self.docker_mgr.get_container_details()
                    container_id = details.get('short_id') if details else None
                    container_name = details.get('name') if details else None
                    container_image = details.get('image') if details else None
                    
                    # Refine 'running' state by checking readiness log line
                    ready = False
                    if container_status == 'running':
                        try:
                            ready = self.docker_mgr.has_ready_log_line(r"Connection to .*10300 port .* succeeded!")
                        except Exception:
                            ready = False

                    # Update container status indicator color
                    if self.widgets.get('container_status_indicator'):
                        if container_status == 'running' and ready:
                            indicator_color = 'green'
                        elif container_status == 'running':
                            indicator_color = 'orange'
                        elif container_status in ('stopped', 'not_found'):
                            indicator_color = 'gray'
                        else:
                            indicator_color = 'red'
                        self.widgets['container_status_indicator'].configure(text_color=indicator_color)

                    # Always show ID/Name/Image if container exists, independent of health
                    if self.widgets.get('container_details'):
                        if details:
                            # Build details text lines
                            lines = []
                            if container_id:
                                lines.append(f"ID: {container_id}")
                            if container_name:
                                lines.append(f"Name: {container_name}")
                            if container_image:
                                lines.append(f"Image: {container_image}")
                            # Add status line based on readiness
                            if container_status == 'running':
                                if ready:
                                    lines.append("Status: ready")
                                else:
                                    lines.append("Status: waiting")
                            else:
                                lines.append(f"Status: {container_status}")
                            detail_text = "\n".join(lines) if lines else f"Status: {container_status}"
                        else:
                            detail_text = "No container"
                        self.widgets['container_details'].configure(text=detail_text)
                else:
                    # Docker not available - container can't run
                    if self.widgets.get('container_status_indicator'):
                        self.widgets['container_status_indicator'].configure(text_color='gray')
                    if self.widgets.get('container_details'):
                        self.widgets['container_details'].configure(text="Docker required")
            else:
                # External mode - check server availability
                try:
                    server_ok = self.ctx.engine.health_check()
                except Exception:
                    server_ok = False
                
                if self.widgets.get('server_status_indicator'):
                    if server_ok:
                        self.widgets['server_status_indicator'].configure(text_color='green')
                        if self.widgets.get('server_details'):
                            external_url = self.ctx.config_manager.get_setting('whisper', 'external_url')
                            self.widgets['server_details'].configure(text=f"Connected to {external_url}")
                    else:
                        self.widgets['server_status_indicator'].configure(text_color='red')
                        if self.widgets.get('server_details'):
                            self.widgets['server_details'].configure(text="Server unreachable")
                
                # Model indicator follows server status
                if self.widgets.get('model_status_indicator'):
                    self.widgets['model_status_indicator'].configure(text_color='green' if server_ok else 'gray')
                    
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to refresh status panel: {e}")

    def create_backend_buttons(self):
        """Create backend control buttons"""
        # Clear frame
        for widget in self.widgets['backend_buttons_frame'].winfo_children():
            widget.destroy()
            
        if self.ctx.backend_mode == 'local':
            self.widgets['start_backend_button'] = ctk.CTkButton(
                self.widgets['backend_buttons_frame'],
                text="Start Server",
                command=self.start_backend,
                width=120,
                height=34,
                corner_radius=8,
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
                fg_color=COLORS["success"],
                hover_color=COLORS["success_light"],
                text_color="#FFFFFF"
            )
            self.widgets['start_backend_button'].pack(side="left", padx=(0, 8))
            
            self.widgets['stop_backend_button'] = ctk.CTkButton(
                self.widgets['backend_buttons_frame'],
                text="Stop Server",
                command=self.stop_backend,
                width=120,
                height=34,
                corner_radius=8,
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
                fg_color=COLORS["danger"],
                hover_color=COLORS["danger_light"],
                text_color="#FFFFFF"
            )
            self.widgets['stop_backend_button'].pack(side="left", padx=0)
            
            # Add tooltips for backend buttons
            try:
                ToolTip(self.widgets['start_backend_button'], 
                       "Start Docker container")
                ToolTip(self.widgets['stop_backend_button'], 
                       "Stop Docker container")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to add backend button tooltips: {e}")

    def open_container_logs(self):
        """Open a simple window to display recent container logs"""
        try:
            # Build window
            log_win = ctk.CTkToplevel(self.root)
            log_win.title("Container Logs")
            log_win.geometry("800x480")
            try:
                log_win.lift()
                log_win.focus_force()
                log_win.attributes('-topmost', True)
                log_win.after(100, lambda: log_win.attributes('-topmost', False))
            except Exception:
                pass

            frame = ctk.CTkFrame(log_win, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            frame.pack(fill="both", expand=True, padx=12, pady=12)

            text = ctk.CTkTextbox(frame, corner_radius=8, border_width=0, fg_color=COLORS["primary"], text_color=COLORS["text_secondary"], font=ctk.CTkFont(family=FONTS["family_monospace"], size=FONTS["size_logs"]))
            text.pack(fill="both", expand=True, padx=8, pady=8)
            try:
                text.configure(state="disabled")
            except Exception:
                pass

            def load_logs():
                logs = None
                try:
                    logs = self.docker_mgr.get_recent_logs(tail=400)
                except Exception:
                    logs = None
                if logs is None:
                    logs = "No logs available or Docker not running."
                try:
                    text.configure(state="normal")
                    text.delete("1.0", "end")
                    text.insert("1.0", logs)
                    text.configure(state="disabled")
                except Exception:
                    pass

            btn_row = ctk.CTkFrame(frame, fg_color="transparent")
            btn_row.pack(fill="x", padx=8, pady=(0,8))
            refresh_btn = ctk.CTkButton(btn_row, text="Refresh", command=load_logs, width=100, height=28, corner_radius=8,
                                        font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"]),
                                        fg_color=COLORS["accent"], hover_color=COLORS["accent_light"], text_color="#FFFFFF")
            refresh_btn.pack(side="right")

            load_logs()
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to open logs window: {e}")

    def create_hotkeys_section(self, parent):
        """Create hotkeys settings section"""
        # Hotkeys title with better typography
        hotkeys_title = ctk.CTkLabel(
            parent, 
            text="Keyboard Shortcuts", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        hotkeys_title.pack(pady=(20, 20), anchor="w")
        
        # Hotkey controls with improved styling
        hotkey_controls_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        hotkey_controls_frame.pack(fill="x", padx=0, pady=(0, 16))
        
        # Add padding to hotkey controls
        hotkey_inner = ctk.CTkFrame(hotkey_controls_frame, fg_color="transparent")
        hotkey_inner.pack(fill="x", padx=16, pady=12)
        
        self.widgets['start_hotkey'] = ctk.CTkEntry(
            hotkey_inner,
            placeholder_text="Start recording hotkey",
            width=180,
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        self.widgets['start_hotkey'].insert(0, self.ctx.config_manager.get_setting('hotkey','start_recording_hotkey'))
        self.widgets['start_hotkey'].bind('<KeyRelease>', self.on_hotkey_settings_change)
        self.widgets['start_hotkey'].pack(side="left", padx=(0, 12))
        
        self.widgets['stop_hotkey'] = ctk.CTkEntry(
            hotkey_inner,
            placeholder_text="Stop recording hotkey",
            width=180,
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        stop_value = self.ctx.config_manager.get_setting('hotkey','stop_recording_hotkey') if 'stop_recording_hotkey' in self.ctx.config_manager.config['hotkey'] else ''
        self.widgets['stop_hotkey'].insert(0, stop_value)
        self.widgets['stop_hotkey'].bind('<KeyRelease>', self.on_hotkey_settings_change)
        self.widgets['stop_hotkey'].pack(side="left", padx=(0, 12))
        
        self.widgets['auto_paste_checkbox'] = ctk.CTkCheckBox(
            hotkey_inner,
            text="Auto paste",
            command=self.on_hotkey_settings_change,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_normal"]),
            text_color=COLORS["text_secondary"],
            checkbox_width=20,
            checkbox_height=20,
            corner_radius=4
        )
        if self.ctx.clipboard_manager.auto_paste:
            self.widgets['auto_paste_checkbox'].select()
        self.widgets['auto_paste_checkbox'].pack(side="left", padx=(0, 12))
        
        # Spacer to push Apply button to the right
        spacer = ctk.CTkLabel(hotkey_inner, text="", width=1)
        spacer.pack(side="left", fill="x", expand=True)

        # Improved Apply button
        self.widgets['save_hotkeys_button'] = ctk.CTkButton(
            hotkey_inner,
            text="Apply",
            command=self.save_hotkeys,
            state="disabled",
            width=120,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["disabled"],
            hover_color=COLORS["warning_light"],
            text_color=COLORS["text_disabled"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.widgets['save_hotkeys_button'].pack(side="right", padx=0)
        
        # Initialize disabled state for save hotkeys button
        self._disable_button('save_hotkeys_button')
        
        # Add tooltips to hotkeys section
        self.add_hotkeys_tooltips()

    def create_history_section(self, parent):
        """Create history section (button to open separate window)"""
        # History title
        history_title = ctk.CTkLabel(
            parent, 
            text="Transcription History", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        history_title.pack(pady=(20, 20), anchor="w")
        
        # Open History button
        open_history_button = ctk.CTkButton(
            parent,
            text="Open Transcription History",
            command=self.open_history_window,
            height=44,
            corner_radius=10,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_light"],
            text_color="#FFFFFF"
        )
        open_history_button.pack(fill="x", pady=(0, 16))

    def open_history_window(self):
        """Open history window"""
        if not self.history_window:
            self.history_window = HistoryWindow(self)
        self.history_window.show()
    
    def create_logs_section(self, parent):
        """Create logs section"""
        # Logs title with better typography
        logs_title = ctk.CTkLabel(
            parent, 
            text="Application Logs", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        logs_title.pack(pady=(20, 20), anchor="w")
        
        # Improved text field for logs (removed control buttons frame)
        logs_container = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        logs_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        self.widgets['log_output'] = ctk.CTkTextbox(
            logs_container,
            height=180,
            corner_radius=8,
            border_width=0,
            fg_color=COLORS["primary"],
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(family=FONTS["family_monospace"], size=FONTS["size_logs"])  # Modern monospace font for logs
        )
        self.widgets['log_output'].pack(fill="both", expand=True, padx=12, pady=12)
        
        # Add tooltips to logs section
        self.add_logs_tooltips()

    def setup_ui_logging(self):
        """Setup UI logging"""
        def update_log(text: str):
            # Safe UI update from another thread
            self.root.after(0, lambda: self._update_log_safe(text))
        
        handler = UILogHandler(update_log, level=logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        root_logger = logging.getLogger()
        # Remove old early handlers
        for h in list(root_logger.handlers):
            if isinstance(h, EarlyBufferHandler):
                root_logger.removeHandler(h)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    def _update_log_safe(self, text: str):
        """Safe log update in UI thread"""
        try:
            self.widgets['log_output'].delete("1.0", "end")
            self.widgets['log_output'].insert("1.0", text)
        except Exception:
            # If we can't update UI, just ignore
            pass

    def setup_hotkeys(self):
        """Setup hotkeys"""
        self.ctx.enable_hotkeys()
        if self.ctx.hotkey_listener:
            self.update_status("Hotkeys enabled")
        else:
            if self.ctx.last_hotkey_error and 'already registered' in self.ctx.last_hotkey_error.lower():
                self.update_status("Hotkey conflict: adjust Start/Stop hotkeys and Apply.")
            elif self.ctx.last_hotkey_error:
                self.update_status(f"Hotkeys error: {self.ctx.last_hotkey_error}"[:160])
            else:
                self.update_status("Hotkeys NOT enabled (mutex busy)")

    def add_model_tooltips(self):
        """Add tooltips to model section widgets"""
        try:
            # Backend mode tooltip
            ToolTip(self.widgets['backend_mode'], 
                   "local: Docker container\nexternal: Remote server")
            
            # External URL tooltip
            ToolTip(self.widgets['external_url'], 
                   "External Whisper server URL\nExample: localhost:10300")
            
            # Model dropdown tooltip
            ToolTip(self.widgets['model_dropdown'], 
                   "Whisper model size")
            
            # Beam size tooltip
            ToolTip(self.widgets['beam_size'], 
                   "Beam search size (1-20)\nLower=faster, Higher=better")
            
            # Language dropdown tooltip
            ToolTip(self.widgets['language_dropdown'], 
                   "Speech recognition language")
            
            # Switch button tooltip
            ToolTip(self.widgets['switch_button'], 
                   "Apply model settings")
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to add model tooltips: {e}")

    def add_hotkeys_tooltips(self):
        """Add tooltips to hotkeys section widgets"""
        try:
            ToolTip(self.widgets['start_hotkey'], "Hotkey to start recording")
            ToolTip(self.widgets['stop_hotkey'], "Hotkey to stop recording")
            ToolTip(self.widgets['auto_paste_checkbox'], "Auto-paste recognized text")
            ToolTip(self.widgets['save_hotkeys_button'], "Apply hotkey changes")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to add hotkeys tooltips: {e}")



    def refresh_history_display(self, force_rebuild=False):
        """Refresh history display in separate window if open"""
        try:
            if self.history_window and self.history_window.window and self.history_window.window.winfo_exists():
                self.history_window.refresh_history_display()
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to refresh history display: {e}")
    
    def refresh_history_display_debounced(self, force_rebuild=False):
        """Debounced version of refresh_history_display"""
        # Cancel any pending update
        if self.scheduled_history_update_id:
            self.root.after_cancel(self.scheduled_history_update_id)
        
        # Schedule new update
        self.scheduled_history_update_id = self.root.after(
            self.history_update_debounce_ms,
            lambda: self.refresh_history_display(force_rebuild)
        )

    def on_history_entry_click(self, index):
        """Handle history entry click (deprecated - history now in separate window)"""
        pass



    def on_history_updated(self):
        """Callback when history is updated with rate limiting"""
        try:
            # Rate limit updates to avoid UI thrashing
            current_time = time.time()
            if current_time - self.last_history_update_time < 1.0:  # Min 1 second between updates
                # Use debounced update if called too frequently
                self.root.after(0, lambda: self.refresh_history_display_debounced(force_rebuild=True))
            else:
                # Update immediately if enough time has passed
                self.last_history_update_time = current_time
                self.root.after(0, lambda: self.refresh_history_display(force_rebuild=True))
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to schedule history update: {e}")

    def add_logs_tooltips(self):
        """Add tooltips to logs section widgets"""
        try:
            # Log output tooltip
            ToolTip(self.widgets['log_output'], 
                   "Application logs and status messages")
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to add logs tooltips: {e}")

    def start_polling(self):
        """Start background status polling"""
        self.polling_running = True
        self.executor.submit(self._polling_loop)

    def _polling_loop(self):
        """Background status polling loop"""
        counter = 0
        status_check_counter = 0
        while self.polling_running and not self.quitting_flag:
            try:
                counter += 1
                status_check_counter += 1
                
                # Poll less frequently - reduced UI load
                # Check backend status: visible=10s, hidden=40s
                check_interval = 5 if self.window_visible else 20  # multiplied by 2s sleep = 10s/40s
                
                if status_check_counter >= check_interval:
                    status_check_counter = 0
                    # _update_backend_status() now collects all Docker data
                    # once and schedules a single _apply_backend_state() on
                    # the UI thread — no separate refresh_status_panel()
                    # round-trip needed here.
                    self._update_backend_status()
                    
                # Update UI elements only when visible and less frequently
                if self.window_visible:
                    # Update switch button every 4 seconds instead of every 2
                    if counter % 2 == 0:
                        self.root.after(0, self.update_switch_button_state)
                    
                    # Update history display very rarely (every 60 seconds)
                    if counter % 30 == 0:
                        self.root.after(0, self.refresh_history_display)
                    
            except Exception as e:
                logging.getLogger(__name__).debug(f"Polling error: {e}")
                
            time.sleep(2.0)  # Increased from 0.5s to 2s - 4x reduction in polling frequency

    def _update_backend_status(self):
        """Collect backend status on the worker thread, then apply to UI.

        All Docker / health-check I/O happens here exactly once per tick.
        The result is packed into a plain dict and dispatched via a single
        root.after(0, ...) to _apply_backend_state(), which touches only
        Tk widgets — no Docker calls from the UI thread.

        This replaces the old path that fanned out 3–5 after(0) callbacks
        and performed ~16 redundant Docker queries per tick (is_available,
        get_health_and_status, has_ready_log_line, get_container_details
        were each re-run multiple times via refresh_status_panel()).
        """
        try:
            state: Dict[str, Any] = {'mode': self.ctx.backend_mode}

            if self.ctx.backend_mode == 'local':
                # Gather everything up-front — each call at most once.
                docker_available = self.docker_mgr.is_available()
                container_status = None
                ready = False
                container_details = None
                container_model = None

                if docker_available:
                    try:
                        container_status, _health_ok = self.docker_mgr.get_health_and_status(
                            self.ctx.engine.health_check
                        )
                    except Exception:
                        container_status = 'error'

                    try:
                        container_details = self.docker_mgr.get_container_details()
                    except Exception:
                        container_details = None

                    if container_status == 'running':
                        try:
                            ready = self.docker_mgr.has_ready_log_line(
                                r"Connection to .*10300 port .* succeeded!"
                            )
                        except Exception:
                            ready = False

                        try:
                            container_model = self.docker_mgr.get_container_model_info(self.ctx.engine)
                        except Exception:
                            container_model = None

                state.update({
                    'docker_available': docker_available,
                    'container_status': container_status,
                    'ready': ready,
                    'container_details': container_details,
                    'container_model': container_model,
                })
            else:
                # External mode — single health probe.
                try:
                    server_ok = self.ctx.engine.health_check()
                except Exception:
                    server_ok = False
                try:
                    external_url = self.ctx.config_manager.get_setting('whisper', 'external_url')
                except Exception:
                    external_url = ""
                state.update({
                    'server_ok': server_ok,
                    'external_url': external_url,
                })

            # Single UI hop — one after(0), one widget-update pass.
            self.root.after(0, self._apply_backend_state, state)

        except Exception as e:
            logging.getLogger(__name__).debug(f"Backend status update error: {e}")

    def _apply_backend_state(self, state: Dict[str, Any]):
        """Apply a pre-collected backend state dict to all UI widgets.

        Runs on the Tk main thread. Performs NO Docker or network I/O —
        every value it needs is already in `state`. This method is the
        single point that reconciles server_status label, container_model
        label, docker/container indicators, details panels, and backend
        buttons in one pass.
        """
        try:
            if state.get('mode') == 'local':
                docker_available = state.get('docker_available', False)
                container_status = state.get('container_status')
                ready = state.get('ready', False)
                details = state.get('container_details')
                container_model = state.get('container_model')

                # Reuse the already-fetched Docker state for button update —
                # no extra Docker calls from the UI thread.
                self.update_backend_buttons_state(
                    container_status=container_status,
                    docker_available=docker_available,
                )

                # --- Server status label -------------------------------
                if not docker_available:
                    self._set_label('server_status',
                                    "Server status: not running (docker unavailable)", "red")
                elif container_status == 'running':
                    if ready:
                        self._set_label('server_status', "Server status: running", "green")
                    else:
                        self._set_label('server_status', "Server status: waiting", "orange")
                elif container_status in ('stopped', 'not_found'):
                    self._set_label('server_status', "Server status: not running", "gray")
                else:
                    self._set_label('server_status', "Server status: error", "red")

                # --- Container model label -----------------------------
                if docker_available and container_status == 'running':
                    if container_model:
                        canonical_name = ALIAS_TO_MODEL.get(container_model, container_model)
                        display_text = f"Container model: {container_model} ({canonical_name})"
                        self._set_label('container_model', display_text,
                                        "green" if ready else "orange")
                    else:
                        self._set_label('container_model', "Container model: unknown", "gray")
                elif container_status in ('stopped', 'not_found'):
                    self._set_label('container_model', "Container model: -", "gray")
                elif docker_available:
                    self._set_label('container_model', "Container model: error", "red")
                else:
                    self._set_label('container_model', "Container model: -", "gray")

                # --- Docker indicator + details ------------------------
                if self.widgets.get('docker_status_indicator'):
                    self.widgets['docker_status_indicator'].configure(
                        text_color='green' if docker_available else 'red'
                    )
                if self.widgets.get('docker_details'):
                    self.widgets['docker_details'].configure(
                        text="Docker Desktop available" if docker_available
                        else "Docker Desktop not running"
                    )

                # --- Container indicator + details ---------------------
                if self.widgets.get('container_status_indicator'):
                    if not docker_available:
                        indicator_color = 'gray'
                    elif container_status == 'running' and ready:
                        indicator_color = 'green'
                    elif container_status == 'running':
                        indicator_color = 'orange'
                    elif container_status in ('stopped', 'not_found'):
                        indicator_color = 'gray'
                    else:
                        indicator_color = 'red'
                    self.widgets['container_status_indicator'].configure(text_color=indicator_color)

                if self.widgets.get('container_details'):
                    if not docker_available:
                        detail_text = "Docker required"
                    elif details:
                        lines = []
                        cid = details.get('short_id')
                        cname = details.get('name')
                        cimage = details.get('image')
                        if cid:
                            lines.append(f"ID: {cid}")
                        if cname:
                            lines.append(f"Name: {cname}")
                        if cimage:
                            lines.append(f"Image: {cimage}")
                        if container_status == 'running':
                            lines.append("Status: ready" if ready else "Status: waiting")
                        else:
                            lines.append(f"Status: {container_status}")
                        detail_text = "\n".join(lines) if lines else f"Status: {container_status}"
                    else:
                        detail_text = "No container"
                    self.widgets['container_details'].configure(text=detail_text)

            else:
                # External mode
                server_ok = state.get('server_ok', False)
                external_url = state.get('external_url', "")

                if server_ok:
                    self._set_label('server_status', "Server status: running", "green")
                else:
                    self._set_label('server_status', "Server status: error", "red")

                if self.widgets.get('server_status_indicator'):
                    self.widgets['server_status_indicator'].configure(
                        text_color='green' if server_ok else 'red'
                    )
                if self.widgets.get('server_details'):
                    self.widgets['server_details'].configure(
                        text=f"Connected to {external_url}" if server_ok else "Server unreachable"
                    )
                if self.widgets.get('model_status_indicator'):
                    self.widgets['model_status_indicator'].configure(
                        text_color='green' if server_ok else 'gray'
                    )

        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to apply backend state: {e}")

    def _set_label(self, key: str, text: str, color: str):
        """Helper: configure a named label widget if it exists."""
        try:
            w = self.widgets.get(key)
            if w is not None:
                w.configure(text=text, text_color=color)
        except Exception:
            pass

    def _update_server_status(self, text: str, color: str):
        """Update server status label (kept for any external callers)."""
        self._set_label('server_status', text, color)

    def _update_container_model(self, text: str, color: str):
        """Update container model label (kept for any external callers)."""
        self._set_label('container_model', text, color)

    # Event handlers
    def on_backend_mode_change(self, value):
        """Backend mode change handler"""
        # Update external URL state
        self.widgets['external_url'].configure(state="normal" if value == 'external' else "disabled")
        # Recreate backend buttons
        self.create_backend_buttons()
        self.update_backend_buttons_state()

    def on_model_change(self, value):
        """Model change handler"""
        self.update_switch_button_state()

    def on_beam_change(self, event):
        """Beam size change handler"""
        self.update_switch_button_state()

    def on_language_change(self, value):
        """Language change handler"""
        self.update_switch_button_state()

    def on_model_settings_change(self, event=None):
        """Model settings change handler (for text fields)"""
        pass  # Leave empty as Save model button is removed

    def on_hotkey_settings_change(self, event=None):
        """Hotkey settings change handler"""
        # Check for changes
        self.check_hotkey_settings_changed()

    def save_original_values(self):
        """Save original values for change tracking"""
        # Hotkey settings
        self.original_hotkey_settings = {
            'start_hotkey': self.widgets['start_hotkey'].get(),
            'stop_hotkey': self.widgets['stop_hotkey'].get(),
            'auto_paste_checkbox': self.widgets['auto_paste_checkbox'].get()
        }

    def _enable_button(self, button_name, button_type="primary"):
        """Enable button with proper colors based on type"""
        button = self.widgets.get(button_name)
        if not button:
            return
            
        colors_map = {
            "primary": (COLORS["accent"], COLORS["accent_light"]),
            "success": (COLORS["success"], COLORS["success_light"]),
            "warning": (COLORS["warning"], COLORS["warning_light"]),
            "danger": (COLORS["danger"], COLORS["danger_light"])
        }
        
        fg_color, hover_color = colors_map.get(button_type, colors_map["primary"])
        button.configure(
            state="normal",
            fg_color=fg_color,
            hover_color=hover_color,
            text_color="#FFFFFF"
        )
    
    def _disable_button(self, button_name):
        """Disable button with proper colors"""
        button = self.widgets.get(button_name)
        if not button:
            return
            
        button.configure(
            state="disabled",
            fg_color=COLORS["disabled"],
            text_color=COLORS["text_disabled"]
        )

    def check_hotkey_settings_changed(self):
        """Check for changes in hotkey settings"""
        current_values = {
            'start_hotkey': self.widgets['start_hotkey'].get(),
            'stop_hotkey': self.widgets['stop_hotkey'].get(),
            'auto_paste_checkbox': self.widgets['auto_paste_checkbox'].get()
        }
        
        changed = current_values != self.original_hotkey_settings
        self.hotkey_settings_changed = changed
        
        if changed:
            self._enable_button('save_hotkeys_button', 'warning')
        else:
            self._disable_button('save_hotkeys_button')

    def update_switch_button_state(self):
        """Update Switch button state"""
        try:
            selected_model = self.widgets['model_dropdown'].get()
            current_model = self.ctx.engine.model_size
            
            selected_language = self.widgets['language_dropdown'].get()
            current_language = self.ctx.engine.language
            
            # Check beam size
            try:
                selected_beam = int(self.widgets['beam_size'].get().strip())
                current_beam = self.ctx.engine.beam_size
                is_beam_different = selected_beam != current_beam
            except (ValueError, AttributeError):
                is_beam_different = True
            
            if self.ctx.backend_mode == 'local':
                # For local mode also check container
                container_model = self.docker_mgr.get_container_model_info(self.ctx.engine)
                container_beam = self.docker_mgr.get_container_beam_info()
                container_lang = self.docker_mgr.get_container_lang_info()
                
                is_model_different = not ((selected_model == current_model) or (container_model == selected_model))
                is_container_beam_different = container_beam != selected_beam if container_beam is not None else True
                is_lang_different = not ((selected_language == current_language) or (container_lang == selected_language))
                
                should_enable = is_model_different or is_beam_different or is_container_beam_different or is_lang_different
            else:
                # External mode
                is_model_different = selected_model != current_model
                is_lang_different = selected_language != current_language
                should_enable = is_model_different or is_beam_different or is_lang_different
            
            if should_enable:
                self._enable_button('switch_button', 'success')
            else:
                self._disable_button('switch_button')
            
        except Exception as e:
            logging.getLogger(__name__).debug(f"Error updating switch button state: {e}")

    def update_backend_buttons_state(self, container_status=None, docker_available=None):
        """Update backend buttons state.

        When called from the hot polling path (_apply_backend_state), the
        caller already has fresh Docker status in hand and passes it in —
        no extra Docker round-trips on the UI thread. When called from
        button handlers / user actions, args are omitted and we fall back
        to querying Docker directly (acceptable: one-off, user-triggered).
        """
        if self.ctx.backend_mode != 'local':
            return

        # Skip update if operation in progress (prevents flickering)
        if self.backend_operation_in_progress:
            return

        try:
            if container_status is None or docker_available is None:
                container_status, _ = self.docker_mgr.get_health_and_status(self.ctx.engine.health_check)
                docker_available = self.docker_mgr.is_available()

            if not docker_available:
                self._disable_button('start_backend_button')
                self._disable_button('stop_backend_button')
                return
                
            if container_status == 'running':
                self._disable_button('start_backend_button')
                self._enable_button('stop_backend_button', 'danger')
            elif container_status in ('stopped', 'not_found'):
                self._enable_button('start_backend_button', 'success')
                self._disable_button('stop_backend_button')
            else:
                self._enable_button('start_backend_button', 'success')
                self._disable_button('stop_backend_button')
                    
        except Exception as e:
            logging.getLogger(__name__).debug(f"Error updating backend buttons state: {e}")

    # Button handlers
    def save_hotkeys(self):
        """Save hotkey settings"""
        try:
            auto_paste = self.widgets['auto_paste_checkbox'].get()
            self.ctx.config_manager.update_user_setting('clipboard', 'auto_paste', auto_paste)
            self.ctx.clipboard_manager.update_auto_paste(auto_paste)
            
            start_hotkey = self.widgets['start_hotkey'].get().strip()
            self.ctx.config_manager.update_user_setting('hotkey','start_recording_hotkey', start_hotkey)
            
            stop_hotkey = self.widgets['stop_hotkey'].get().strip()
            if stop_hotkey:
                self.ctx.config_manager.update_user_setting('hotkey','stop_recording_hotkey', stop_hotkey)
            else:
                if 'stop_recording_hotkey' in self.ctx.config_manager.config['hotkey']:
                    self.ctx.config_manager.config['hotkey']['stop_recording_hotkey'] = ''
            
            self.ctx.reconfigure_hotkeys_if_active()
            self.update_status("Hotkeys saved")
            
            # Reset change flag and disable button
            self.hotkey_settings_changed = False
            self._disable_button('save_hotkeys_button')
            
            # Save new original values
            self.original_hotkey_settings = {
                'start_hotkey': self.widgets['start_hotkey'].get(),
                'stop_hotkey': self.widgets['stop_hotkey'].get(),
                'auto_paste_checkbox': self.widgets['auto_paste_checkbox'].get()
            }
            
        except Exception as e:
            self.update_status(f"Error saving hotkeys: {e}")

    def switch_model(self):
        """Switch model"""
        try:
            new_model = self.widgets['model_dropdown'].get()
            new_language = self.widgets['language_dropdown'].get()
            
            # Validate beam size
            try:
                new_beam_size = int(self.widgets['beam_size'].get().strip())
                if new_beam_size < 1 or new_beam_size > 20:
                    self.update_status("Invalid beam size (must be 1-20)")
                    return
            except ValueError:
                self.update_status("Invalid beam size (must be integer 1-20)")
                return
            
            # Check if switching is needed
            if (new_model == self.ctx.engine.model_size and
                new_beam_size == self.ctx.engine.beam_size and
                new_language == self.ctx.engine.language):
                self.update_status(f"Model '{new_model}', beam size {new_beam_size}, and language '{new_language}' already active")
                return
            
            # Show progress bar
            self.show_progress(True)
            self.update_status(f"Switching to {new_model} (beam: {new_beam_size}, lang: {new_language})...")
            
            # Run switching in background thread
            self.executor.submit(self._async_switch_model, new_model, new_beam_size, new_language)
            
        except Exception as e:
            self.update_status(f"Error switching model: {e}")
            self.show_progress(False)

    def _async_switch_model(self, new_model, new_beam_size, new_language):
        """Asynchronous model switching"""
        try:
            # Update local configuration
            old_model = self.ctx.engine.model_size
            old_beam = self.ctx.engine.beam_size
            old_language = self.ctx.engine.language
            
            self.ctx.engine.model_size = new_model
            self.ctx.engine.beam_size = new_beam_size
            self.ctx.engine.language = new_language
            canonical = ALIAS_TO_MODEL.get(new_model, new_model)
            self.ctx.engine.remote_model = canonical
            
            self.ctx.config_manager.update_user_setting('whisper','model', new_model)
            self.ctx.config_manager.update_user_setting('whisper','beam_size', new_beam_size)
            self.ctx.config_manager.update_user_setting('whisper','language', new_language)
            
            if self.ctx.backend_mode == 'local':
                # Restart Docker container
                self.root.after(0, lambda: self.update_status(f"Creating new container with {new_model} (beam: {new_beam_size}, lang: {new_language})..."))
                
                container_result = self.docker_mgr.restart_with_model_beam_and_lang(new_model, new_beam_size, new_language)
                
                if container_result == "running":
                    self.root.after(0, lambda: self.update_status(f"Switched to {new_model} (beam: {new_beam_size}, lang: {new_language}, container recreated)"))
                    self.root.after(0, lambda: self._update_server_status("Server status: running", "green"))
                    # Wait for readiness log line asynchronously (non-blocking UI)
                    def _wait_ready():
                        ok = False
                        try:
                            ok = self.docker_mgr.wait_for_log_line(r"Connection to .*10300 port .* succeeded!", timeout=40)
                        except Exception:
                            ok = False
                        if ok:
                            self.root.after(0, lambda: self.update_status("Backend ready (port 10300 reachable)"))
                        else:
                            self.root.after(0, lambda: self.update_status("Backend start: readiness log not detected (timeout)"))
                        self.root.after(0, self._update_backend_status)
                    self.executor.submit(_wait_ready)
                    
                    # Update container model information
                    container_model = self.docker_mgr.get_container_model_info(self.ctx.engine)
                    if container_model:
                        # Get canonical model name
                        canonical_name = ALIAS_TO_MODEL.get(container_model, container_model)
                        display_text = f"Container model: {container_model} ({canonical_name})"
                        self.root.after(0, lambda: self._update_container_model(display_text, "green"))
                    else:
                        self.root.after(0, lambda: self._update_container_model("Container model: unknown", "gray"))
                else:
                    self.root.after(0, lambda: self.update_status(f"Failed to restart container: {container_result}"))
                    self.root.after(0, lambda: self._update_server_status("Server status: error", "red"))
                    self.root.after(0, lambda: self._update_container_model("Container model: error", "red"))
                
                # Update backend buttons state
                self.root.after(0, self.update_backend_buttons_state)
            else:
                # External mode
                self.root.after(0, lambda: self.update_status(f"Switched to {new_model} (beam: {new_beam_size}, lang: {new_language}, external server)"))
            
            self.root.after(0, lambda: self.show_progress(False))
            self.root.after(0, self.update_switch_button_state)
            
            logging.getLogger(__name__).info(f"Model switched: {old_model} -> {new_model}, beam: {old_beam} -> {new_beam_size}, language: {old_language} -> {new_language}")
            
        except Exception as ex:
            logging.getLogger(__name__).error(f"Model switch error: {ex}")
            error_msg = str(ex)  # Save error message
            self.root.after(0, lambda: self.update_status(f"Error switching model: {error_msg}"))
            self.root.after(0, lambda: self.show_progress(False))
            self.root.after(0, self.update_switch_button_state)

    def start_backend(self):
        """Start backend"""
        try:
            # Set operation in progress flag
            self.backend_operation_in_progress = True
            
            # Disable buttons
            self._disable_button('start_backend_button')
            self._disable_button('stop_backend_button')
            
            self.update_status("Starting container...")
            
            # Run in background thread
            self.executor.submit(self._async_start_backend)
            
        except Exception as e:
            self.update_status(f"Error starting backend: {e}")

    def _async_start_backend(self):
        """Asynchronous backend start"""
        try:
            res = self.docker_mgr.start()
            logging.getLogger(__name__).info(f"Backend start: {res}")
            
            status_text = "Container running" if res == 'running' else f"Container status: {res}"
            self.root.after(0, lambda: self.update_status(status_text))
            
            # Clear operation flag
            self.backend_operation_in_progress = False
            
            # Update UI state and wait for readiness line
            self.root.after(0, self._update_backend_status)
            self.root.after(0, self.update_backend_buttons_state)
            if res == 'running':
                def _wait_ready():
                    ok = False
                    try:
                        ok = self.docker_mgr.wait_for_log_line(r"Connection to .*10300 port .* succeeded!", timeout=40)
                    except Exception:
                        ok = False
                    if ok:
                        self.root.after(0, lambda: self.update_status("Backend ready (port 10300 reachable)"))
                    else:
                        self.root.after(0, lambda: self.update_status("Backend start: readiness log not detected (timeout)"))
                    self.root.after(0, self._update_backend_status)
                self.executor.submit(_wait_ready)
            
        except Exception as ex:
            logging.getLogger(__name__).error(f"Backend start error: {ex}")
            error_msg = str(ex)
            # Clear operation flag on error
            self.backend_operation_in_progress = False
            self.root.after(0, lambda: self.update_status(f"Error starting backend: {error_msg}"))
            self.root.after(0, self.update_backend_buttons_state)

    def stop_backend(self):
        """Stop backend"""
        try:
            # Set operation in progress flag
            self.backend_operation_in_progress = True
            
            # Disable buttons
            self._disable_button('start_backend_button')
            self._disable_button('stop_backend_button')
            
            self.update_status("Stopping container...")
            
            # Run in background thread
            self.executor.submit(self._async_stop_backend)
            
        except Exception as e:
            self.update_status(f"Error stopping backend: {e}")

    def _async_stop_backend(self):
        """Asynchronous backend stop"""
        try:
            res = self.docker_mgr.stop()
            logging.getLogger(__name__).info(f"Backend stop: {res}")
            
            status_text = "Container stopped" if res in ('stopped','not_found') else f"Container status: {res}"
            self.root.after(0, lambda: self.update_status(status_text))
            
            # Clear operation flag
            self.backend_operation_in_progress = False
            
            # Update UI state
            self.root.after(0, self._update_backend_status)
            self.root.after(0, self.update_backend_buttons_state)
            
        except Exception as ex:
            logging.getLogger(__name__).error(f"Backend stop error: {ex}")
            error_msg = str(ex)
            # Clear operation flag on error
            self.backend_operation_in_progress = False
            self.root.after(0, lambda: self.update_status(f"Error stopping backend: {error_msg}"))
            self.root.after(0, self.update_backend_buttons_state)

    def hide_to_tray_manually(self):
        """Manual hide to tray"""
        try:
            if self.system_tray and self.system_tray.is_running:
                self.hide_window()
                self.update_status("Window hidden to system tray")
                logging.getLogger(__name__).info("Window manually hidden to system tray")
            else:
                self.update_status("System tray not available")
                logging.getLogger(__name__).warning("Attempted to hide to tray but system tray not active")
        except Exception as e:
            logging.getLogger(__name__).error(f"Manual hide to tray failed: {e}")
            self.update_status(f"Error hiding to tray: {e}")

    # Utility methods
    def show_progress(self, show: bool):
        """Show/hide progress bar"""
        try:
            if show:
                self.widgets['progress_bar'].pack(side="left", padx=5)
                self.widgets['progress_bar'].start()
            else:
                self.widgets['progress_bar'].stop()
                self.widgets['progress_bar'].pack_forget()
        except Exception:
            pass

    def update_status(self, text: str):
        """Update status through log"""
        try:
            # Send status to log as INFO message
            logging.getLogger(__name__).info(text, extra={'user_message': True})
        except Exception:
            pass

    def _normalize_url(self, raw: str) -> str:
        """URL normalization"""
        raw = raw.strip()
        if not raw:
            return raw
        if not raw.startswith(('http://', 'https://')):
            raw = 'http://' + raw
        while raw.endswith('/') and len(raw) > len('http://')+1:
            raw = raw[:-1]
        return raw

    def _is_valid_url(self, u: str) -> bool:
        """URL validity check"""
        if not u:
            return False
        if not (u.startswith('http://') or u.startswith('https://')):
            return False
        without_scheme = u.split('://',1)[1]
        host = without_scheme.split('/')[0]
        return host == 'localhost' or '.' in host or ':' in host

    # Window management
    def hide_window(self):
        """Hide window"""
        try:
            self.root.withdraw()  # Simple window hiding in Tkinter
            self.window_visible = False
            # Update tray menu to reflect window state change
            if self.system_tray and self.system_tray.is_running:
                self.system_tray.refresh_menu()
            logging.getLogger(__name__).info("Window hidden via callback")
        except Exception as ex:
            logging.getLogger(__name__).error(f"Hide window callback failed: {ex}")

    def show_window(self):
        """Show window"""
        try:
            # Make window transparent while CustomTkinter toggles titlebar colors.
            # This keeps the unavoidable withdraw/deiconify dance invisible to the user.
            try:
                self.root.attributes('-alpha', 0.0)
            except tk.TclError:
                pass

            # Deiconify first
            self.root.deiconify()
            
            # Force update to complete rendering before lift/focus
            # This minimizes visible flicker from CTk's titlebar color manipulation on Windows
            self.root.update_idletasks()
            
            # Small delay to let CTk finish titlebar color setting
            # CTk internally calls withdraw/deiconify which causes flicker
            self.root.after(60, lambda: self._complete_show_window())
            
            self.window_visible = True
            logging.getLogger(__name__).info("Window shown via callback")
        except Exception as ex:
            logging.getLogger(__name__).error(f"Show window failed: {ex}")
    
    def _complete_show_window(self):
        """Complete window show after titlebar color is set"""
        try:
            self.root.lift()       # Bring to front
            self.root.focus_force()  # Give focus
            # Update tray menu to reflect window state change
            if self.system_tray and self.system_tray.is_running:
                self.system_tray.refresh_menu()
            # Restore opacity once the window is fully ready.
            def _restore_opacity():
                try:
                    self.root.attributes('-alpha', 1.0)
                except tk.TclError:
                    pass

            # Give Windows a bit more time on first reveal to finish DWM tweaks.
            self.root.after(90, _restore_opacity)
        except Exception as ex:
            logging.getLogger(__name__).debug(f"Complete show window failed: {ex}")

    def is_window_visible(self) -> bool:
        """Check window visibility"""
        return self.window_visible

    def on_close(self):
        """Window close handler"""
        try:
            tray_active = bool(self.system_tray and self.system_tray.is_running)
        except Exception:
            tray_active = False
            
        logging.getLogger(__name__).info(f"on_close called: tray_active={tray_active}, quit_flag={self.quitting_flag}")
            
        if tray_active and not self.quitting_flag:
            # Tray is active and not forcing exit - hide window
            logging.getLogger(__name__).info("Attempting to hide window to system tray")
            try:
                self.hide_window()
                return  # Don't close application
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to hide window to tray: {e}")
                # If hiding failed, continue with normal closing
         
        # Complete application shutdown
        logging.getLogger(__name__).info("Proceeding with complete application shutdown")
        self.quit_application()

    def quit_via_tray(self):
        """Quit via tray"""
        try:
            self.quitting_flag = True
            logging.getLogger(__name__).info("Quit via tray called")
            
            # Stop polling
            self.polling_running = False
            
            # Stop tray
            if self.system_tray and self.system_tray.is_running:
                try:
                    self.system_tray.stop()
                except Exception as e:
                    logging.getLogger(__name__).warning(f"system_tray.stop failed: {e}")
            
            # Clean up resources
            try:
                self.ctx.shutdown()
            except Exception as e:
                logging.getLogger(__name__).warning(f"ctx.shutdown failed: {e}")
            
            logging.getLogger(__name__).info("Terminating process")
            
            # Force termination with small delay
            def delayed_terminate():
                time.sleep(0.2)
                try:
                    import os
                    import sys
                    current_pid = os.getpid()
                    if sys.platform == "win32":
                        os.system(f"taskkill /F /PID {current_pid}")
                    else:
                        os.kill(current_pid, 9)
                except Exception as e:
                    logging.getLogger(__name__).error(f"Failed to terminate process: {e}")
                    import os
                    os._exit(1)
            
            term_thread = threading.Thread(target=delayed_terminate, daemon=True)
            term_thread.start()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Quit via tray failed: {e}")
            import os
            os._exit(1)

    def quit_application(self):
        """Complete application shutdown"""
        self.quitting_flag = True
        
        # Stop polling
        self.polling_running = False
        
        # Cancel any pending UI updates
        if self.scheduled_history_update_id:
            try:
                self.root.after_cancel(self.scheduled_history_update_id)
            except Exception:
                pass
        
        # Clean up resources
        try:
            self.ctx.shutdown()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error during ctx.shutdown: {e}")
        
        try:
            if self.system_tray:
                self.system_tray.stop()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error stopping tray: {e}")
        
        # Stop executor gracefully with timeout
        try:
            self.executor.shutdown(wait=True, timeout=5.0)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error shutting down executor: {e}")
        
        logging.getLogger(__name__).info("Application shutdown complete")
        
        # Close Tkinter
        try:
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error closing Tkinter: {e}")

    def run(self):
        """Application startup"""
        # Initial backend status check
        try:
            if self.ctx.backend_mode == 'local':
                initial_status, _ = self.docker_mgr.get_health_and_status(self.ctx.engine.health_check)
                self.update_backend_buttons_state()
            # Initial status panel refresh
            self.refresh_status_panel()
        except Exception:
            pass
        
        # Start UI
        logging.getLogger(__name__).info("Starting Tkinter UI")
        self.root.mainloop()


def main():
    """Main function"""
    import threading
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        logging.getLogger(__name__).info(f"UI received signal {signum} - shutting down gracefully")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_exception_handler()
    
    try:
        app = LazyToTextUI()
        app.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("UI shutting down...")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


if __name__ == "__main__":
    main()