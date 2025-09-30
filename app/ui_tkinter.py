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
            alias = alias_for(canonical_model) if canonical_model else 'turbo'
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
            audio_feedback=self.audio_feedback
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
        
        # State flags
        self.quitting_flag = False
        self.window_visible = True
        self.system_tray: Optional[SystemTray] = None
        
        # UI elements
        self.widgets: Dict[str, Any] = {}
        
        # Track changes for save buttons
        self.hotkey_settings_changed = False
        
        # Save original values for comparison
        self.original_hotkey_settings = {}
        
        # Threading for background tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.polling_running = False
        
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

    def create_widgets(self):
        """Create interface widgets"""
        # Create main container with modern padding
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=24, pady=20)
        
        # Create main scrollable frame for better UX with large content
        main_frame = ctk.CTkScrollableFrame(
            main_container, 
            fg_color="transparent",
            scrollbar_button_color=COLORS["accent"],
            scrollbar_button_hover_color=COLORS["hover"]
        )
        main_frame.pack(fill="both", expand=True)
        
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
            width=120,
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
        """Create status panel similar to logs but compact"""
        # Status title with better typography
        status_title = ctk.CTkLabel(
            parent, 
            text="System Status", 
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_title"], weight=FONTS["weight_bold"]),
            text_color=COLORS["text_primary"]
        )
        status_title.pack(pady=(20, 20), anchor="w")

        status_frame = ctk.CTkFrame(
            parent, 
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        status_frame.pack(fill="x", padx=0, pady=(0, 16))

        # Status textbox with improved styling
        self.widgets['status_output'] = ctk.CTkTextbox(
            status_frame, 
            height=80,
            corner_radius=8,
            border_width=0,
            fg_color=COLORS["primary"],
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        self.widgets['status_output'].pack(fill="x", expand=False, padx=16, pady=12)
        try:
            self.widgets['status_output'].configure(state="disabled")
        except Exception:
            pass

        # Initial fill
        self.refresh_status_panel()

    def refresh_status_panel(self):
        """Compose and display status text in status_output"""
        box = self.widgets.get('status_output')
        if not box:
            return
        try:
            # Gather data
            server_lbl = self.widgets.get('server_status')
            container_lbl = self.widgets.get('container_model')
            server_text = server_lbl.cget('text') if server_lbl else 'Server status: ?'
            container_text = container_lbl.cget('text') if container_lbl else 'Container model: ?'
            lines: list[str] = []
            lines.append(server_text)
            if self.ctx.backend_mode == 'local':
                lines.append(container_text)
            else:
                lines.append(f"Current model: {self.ctx.engine.model_size} (lang: {self.ctx.engine.language}, beam: {self.ctx.engine.beam_size})")

            text = "\n".join(lines)

            box.configure(state="normal")
            box.delete("1.0","end")
            box.insert("1.0", text)
            box.configure(state="disabled")
        except Exception:
            pass

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
        """Create history section"""
        # History title with better typography
        history_title = ctk.CTkLabel(
            parent, 
            text="Transcription History", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        history_title.pack(pady=(20, 20), anchor="w")
        
        # History controls with improved styling
        history_controls_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        history_controls_frame.pack(fill="x", padx=0, pady=(0, 12))
        
        # Add padding to history controls
        history_inner = ctk.CTkFrame(history_controls_frame, fg_color="transparent")
        history_inner.pack(fill="x", padx=16, pady=12)
        
        # Improved search entry
        self.widgets['history_search'] = ctk.CTkEntry(
            history_inner,
            placeholder_text="Search transcription history...",
            width=240,
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            placeholder_text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"])
        )
        self.widgets['history_search'].bind('<KeyRelease>', self.on_history_search)
        self.widgets['history_search'].pack(side="left", padx=(0, 12))
        
        # Improved filter combobox
        self.widgets['history_filter'] = ctk.CTkOptionMenu(
            history_inner,
            values=["All", "Today", "Last 7 days", "Last 30 days"],
            command=self.on_history_filter_change,
            width=120,
            height=34,
            corner_radius=8,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_light"],
            text_color="#FFFFFF",
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_normal"])
        )
        self.widgets['history_filter'].set("All")
        self.widgets['history_filter'].pack(side="left", padx=(0, 12))
        
        # Spacer
        spacer = ctk.CTkLabel(history_inner, text="", width=1)
        spacer.pack(side="left", fill="x", expand=True)
        
        # Improved history buttons
        self.widgets['copy_history_button'] = ctk.CTkButton(
            history_inner,
            text="Copy",
            command=self.copy_selected_history,
            width=80,
            height=34,
            corner_radius=8,
            state="disabled",
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["disabled"],
            hover_color=COLORS["accent_light"],
            text_color=COLORS["text_disabled"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.widgets['copy_history_button'].pack(side="right", padx=(8, 0))
        
        # Initialize disabled state for copy button
        self._disable_button('copy_history_button')
        
        self.widgets['export_history_button'] = ctk.CTkButton(
            history_inner,
            text="Export",
            command=self.export_history,
            width=80,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["success"],
            hover_color=COLORS["success_light"],
            text_color="#FFFFFF"
        )
        self.widgets['export_history_button'].pack(side="right", padx=(8, 0))
        
        self.widgets['clear_history_button'] = ctk.CTkButton(
            history_inner,
            text="Clear",
            command=self.clear_history,
            width=80,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["danger"],
            hover_color=COLORS["danger_light"],
            text_color="#FFFFFF"
        )
        self.widgets['clear_history_button'].pack(side="right", padx=(8, 0))
        
        # Initialize disabled state for history buttons
        self._disable_button('export_history_button')
        self._disable_button('clear_history_button')
        
        # Improved history listbox frame
        history_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        history_frame.pack(fill="both", expand=False, padx=0, pady=(0, 16))
        
        # Configure grid for history frame
        history_frame.grid_rowconfigure(0, weight=1)
        history_frame.grid_columnconfigure(0, weight=1)
        
        # Improved history scrollable frame
        self.widgets['history_scrollable'] = ctk.CTkScrollableFrame(
            history_frame,
            height=180,
            corner_radius=8,
            fg_color=COLORS["primary"],
            scrollbar_button_color=COLORS["accent"],
            scrollbar_button_hover_color=COLORS["accent_light"]
        )
        self.widgets['history_scrollable'].grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        
        # History entries container
        self.history_entries = []
        self.selected_history_index = None
        
        # Load initial history
        self.refresh_history_display()
        
        # Add tooltips
        self.add_history_tooltips()

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
        
        # Improved log control buttons
        log_controls_frame = ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        log_controls_frame.pack(fill="x", padx=0, pady=(0, 12))
        
        log_inner = ctk.CTkFrame(log_controls_frame, fg_color="transparent")
        log_inner.pack(fill="x", padx=16, pady=12)
        
        # Spacer to push clear button to the right
        spacer = ctk.CTkLabel(log_inner, text="")
        spacer.pack(side="left", fill="x", expand=True)
        
        self.widgets['clear_logs_button'] = ctk.CTkButton(
            log_inner,
            text="Clear Logs",
            command=self.clear_logs,
            width=120,
            height=34,
            corner_radius=8,
            font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_button"], weight=FONTS["weight_bold"]),
            fg_color=COLORS["warning"],
            hover_color=COLORS["warning_light"],
            text_color="#FFFFFF"
        )
        self.widgets['clear_logs_button'].pack(side="right", padx=0)
        
        # Improved text field for logs
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

    def add_history_tooltips(self):
        """Add tooltips to history section widgets"""
        try:
            ToolTip(self.widgets['history_search'], "Search in transcription history")
            ToolTip(self.widgets['history_filter'], "Filter history by date range")
            ToolTip(self.widgets['copy_history_button'], "Copy selected entry to clipboard")
            ToolTip(self.widgets['export_history_button'], "Export history to text file")
            ToolTip(self.widgets['clear_history_button'], "Clear all history entries")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to add history tooltips: {e}")

    def refresh_history_display(self, force_rebuild=False):
        """Refresh history display with current filter and search"""
        try:
            # Get filtered entries
            entries = self.get_filtered_history_entries()
            
            # Only rebuild if the number of entries changed or force rebuild
            if force_rebuild or len(entries) != len(self.history_entries):
                # Clear existing entries
                for widget in self.history_entries:
                    widget.destroy()
                self.history_entries.clear()
                self.selected_history_index = None
                
                # Create display entries
                for i, entry in enumerate(entries):
                    self.create_history_entry_widget(entry, i)
                    
            # Update button states
            self.update_history_button_states()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to refresh history display: {e}")

    def get_filtered_history_entries(self):
        """Get history entries with current filter and search applied"""
        try:
            # Check if history is available
            if not self.ctx.state_manager.history_manager:
                return []
                
            # Get base entries based on filter
            filter_value = self.widgets['history_filter'].get()
            
            if filter_value == "Today":
                entries = self.ctx.state_manager.history_manager.get_entries_by_date(1)
            elif filter_value == "Last 7 days":
                entries = self.ctx.state_manager.history_manager.get_entries_by_date(7)
            elif filter_value == "Last 30 days":
                entries = self.ctx.state_manager.history_manager.get_entries_by_date(30)
            else:  # "All"
                entries = self.ctx.state_manager.history_manager.get_entries(limit=100)
                
            # Apply search filter
            search_text = self.widgets['history_search'].get().strip()
            if search_text:
                entries = [e for e in entries if search_text.lower() in e.text.lower()]
                
            return entries
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to get filtered history entries: {e}")
            return []

    def create_history_entry_widget(self, entry, index):
        """Create widget for single history entry"""
        try:
            # Improved entry frame with better styling
            entry_frame = ctk.CTkFrame(
                self.widgets['history_scrollable'],
                fg_color=COLORS["surface"],
                corner_radius=10,
                border_width=1,
                border_color=COLORS["border"]
            )
            entry_frame.pack(fill="x", padx=6, pady=4)
            
            # Configure grid
            entry_frame.grid_columnconfigure(1, weight=1)
            
            # Time label with improved styling
            time_label = ctk.CTkLabel(
                entry_frame,
                text=entry.datetime_str,
                width=120,
                font=ctk.CTkFont(size=10, weight="normal"),
                text_color=COLORS["text_muted"]
            )
            time_label.grid(row=0, column=0, padx=12, pady=10, sticky="w")
            
            # Text label with better typography
            text_label = ctk.CTkLabel(
                entry_frame,
                text=entry.short_text,
                font=ctk.CTkFont(family=FONTS["family_primary"], size=FONTS["size_body"], weight=FONTS["weight_normal"]),
                text_color=COLORS["text_primary"],
                anchor="w"
            )
            text_label.grid(row=0, column=1, padx=12, pady=10, sticky="ew")
            
            # Info label with better styling
            info_text = f"{entry.model} • {entry.language} • {entry.duration:.1f}s"
            info_label = ctk.CTkLabel(
                entry_frame,
                text=info_text,
                font=ctk.CTkFont(size=10, weight="normal"),
                text_color=COLORS["text_secondary"],
                width=140
            )
            info_label.grid(row=0, column=2, padx=12, pady=10, sticky="e")
            
            # Modern copy button
            def create_copy_command(entry_index):
                return lambda: self.copy_history_entry_by_index(entry_index)
            
            copy_button = ctk.CTkButton(
                entry_frame,
                text="Copy",
                width=65,
                height=30,
                corner_radius=6,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_light"],
                text_color="#FFFFFF",
                command=create_copy_command(index)
            )
            copy_button.grid(row=0, column=3, padx=12, pady=10, sticky="e")
            
            # Bind click events
            def on_click(event, idx=index):
                self.on_history_entry_click(idx)
                
            for widget in [entry_frame, time_label, text_label, info_label]:
                widget.bind("<Button-1>", on_click)
                
            # Add tooltip with full text
            ToolTip(entry_frame, f"Full text: {entry.text}")
            ToolTip(copy_button, "Copy this entry to clipboard")
            
            self.history_entries.append(entry_frame)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to create history entry widget: {e}")

    def on_history_entry_click(self, index):
        """Handle history entry click"""
        try:
            # Update selection
            old_selection = self.selected_history_index
            self.selected_history_index = index
            
            # Update visual selection
            if old_selection is not None and old_selection < len(self.history_entries):
                self.history_entries[old_selection].configure(fg_color=("gray86", "gray20"))
                
            if index < len(self.history_entries):
                self.history_entries[index].configure(fg_color=("gray78", "gray30"))
                
            # Update button states
            self.update_history_button_states()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to handle history entry click: {e}")

    def on_history_search(self, event=None):
        """Handle search text change"""
        self.refresh_history_display(force_rebuild=True)

    def on_history_filter_change(self, value):
        """Handle filter change"""
        self.refresh_history_display(force_rebuild=True)

    def copy_history_entry_by_index(self, index):
        """Copy history entry by index directly"""
        try:
            entries = self.get_filtered_history_entries()
            
            if 0 <= index < len(entries):
                entry = entries[index]
                success = self.ctx.clipboard_manager.copy_text(entry.text)
                if success:
                    self.update_status(f"Copied: {entry.short_text}")
                else:
                    self.update_status("Failed to copy to clipboard")
            else:
                self.update_status(f"Invalid entry index: {index}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to copy entry: {e}")
            self.update_status(f"Failed to copy entry: {e}")

    def copy_selected_history(self):
        """Copy selected history entry to clipboard"""
        try:
            if self.selected_history_index is None:
                return
                
            entries = self.get_filtered_history_entries()
            if self.selected_history_index < len(entries):
                entry = entries[self.selected_history_index]
                success = self.ctx.clipboard_manager.copy_text(entry.text)
                if success:
                    self.update_status("Copied history entry to clipboard")
                else:
                    self.update_status("Failed to copy to clipboard")
                
        except Exception as e:
            self.update_status(f"Failed to copy history entry: {e}")

    def export_history(self):
        """Export history to file"""
        try:
            if not self.ctx.state_manager.history_manager:
                self.update_status("History is disabled")
                return
                
            import tkinter.filedialog as fd
            filename = fd.asksaveasfilename(
                title="Export History",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                success = self.ctx.state_manager.history_manager.export_to_text(filename)
                if success:
                    self.update_status(f"History exported to {filename}")
                else:
                    self.update_status("Failed to export history")
                    
        except Exception as e:
            self.update_status(f"Failed to export history: {e}")

    def clear_history(self):
        """Clear all history"""
        try:
            if not self.ctx.state_manager.history_manager:
                self.update_status("History is disabled")
                return
                
            # Confirm dialog
            import tkinter.messagebox as mb
            result = mb.askyesno(
                "Clear History",
                "Are you sure you want to clear all transcription history?\nThis action cannot be undone.",
                icon="warning"
            )
            
            if result:
                self.ctx.state_manager.history_manager.clear_history()
                self.refresh_history_display(force_rebuild=True)
                self.update_status("History cleared")
                
        except Exception as e:
            self.update_status(f"Failed to clear history: {e}")

    def update_history_button_states(self):
        """Update history button states based on selection"""
        try:
            has_selection = self.selected_history_index is not None
            if has_selection:
                self._enable_button('copy_history_button', 'primary')
            else:
                self._disable_button('copy_history_button')
            
            has_entries = len(self.history_entries) > 0
            if has_entries:
                self._enable_button('export_history_button', 'success')
                self._enable_button('clear_history_button', 'danger')
            else:
                self._disable_button('export_history_button')
                self._disable_button('clear_history_button')
                    
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to update history button states: {e}")

    def on_history_updated(self):
        """Callback when history is updated"""
        try:
            # Schedule UI update in main thread with force rebuild for new entries
            self.root.after(0, lambda: self.refresh_history_display(force_rebuild=True))
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to schedule history update: {e}")

    def add_logs_tooltips(self):
        """Add tooltips to logs section widgets"""
        try:
            # Clear logs button tooltip
            ToolTip(self.widgets['clear_logs_button'], 
                   "Clear all log messages")
            
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
        while self.polling_running and not self.quitting_flag:
            try:
                counter += 1
                
                # Poll less frequently if window is hidden
                check_interval = 5 if self.window_visible else 20
                
                if counter >= check_interval:
                    counter = 0
                    self._update_backend_status()
                    
                # Update Switch button state
                if self.window_visible:
                    self.root.after(0, self.update_switch_button_state)
                    # Update history display occasionally (less frequent to reduce flicker)
                    if counter % 60 == 0:  # Every 30 seconds
                        self.root.after(0, self.refresh_history_display)
                    
            except Exception as e:
                logging.getLogger(__name__).debug(f"Polling error: {e}")
                
            time.sleep(0.5)

    def _update_backend_status(self):
        """Update backend status"""
        try:
            if self.ctx.backend_mode == 'local':
                container_status, health_ok = self.docker_mgr.get_health_and_status(self.ctx.engine.health_check)
                docker_available = self.docker_mgr.is_available()
                
                self.root.after(0, lambda: self.update_backend_buttons_state())
                
                if not docker_available:
                    self.root.after(0, lambda: self._update_server_status("Server status: not running (docker unavailable)", "red"))
                elif container_status == 'running' and health_ok:
                    self.root.after(0, lambda: self._update_server_status("Server status: running", "green"))
                    # Update container model information
                    container_model = self.docker_mgr.get_container_model_info(self.ctx.engine)
                    if container_model:
                        # Get canonical model name
                        canonical_name = ALIAS_TO_MODEL.get(container_model, container_model)
                        display_text = f"Container model: {container_model} ({canonical_name})"
                        self.root.after(0, lambda: self._update_container_model(display_text, "green"))
                    else:
                        self.root.after(0, lambda: self._update_container_model("Container model: unknown", "gray"))
                elif container_status in ('stopped', 'not_found'):
                    self.root.after(0, lambda: self._update_server_status("Server status: not running", "gray"))
                    self.root.after(0, lambda: self._update_container_model("Container model: -", "gray"))
                else:
                    self.root.after(0, lambda: self._update_server_status("Server status: error", "red"))
                    self.root.after(0, lambda: self._update_container_model("Container model: error", "red"))
            else:
                # External mode
                try:
                    ok = self.ctx.engine.health_check()
                except Exception:
                    ok = False
                
                if ok:
                    self.root.after(0, lambda: self._update_server_status("Server status: running", "green"))
                else:
                    self.root.after(0, lambda: self._update_server_status("Server status: error", "red"))
                    
        except Exception as e:
            logging.getLogger(__name__).debug(f"Backend status update error: {e}")

    def _update_server_status(self, text: str, color: str):
        """Update server status in UI"""
        try:
            self.widgets['server_status'].configure(text=text, text_color=color)
            # Refresh status panel if exists
            self.refresh_status_panel()
        except Exception:
            pass

    def _update_container_model(self, text: str, color: str):
        """Update container model information in UI"""
        try:
            self.widgets['container_model'].configure(text=text, text_color=color)
            self.refresh_status_panel()
        except Exception:
            pass

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

    def update_backend_buttons_state(self):
        """Update backend buttons state"""
        if self.ctx.backend_mode != 'local':
            return
            
        try:
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
            
            # Update UI state
            self.root.after(0, self._update_backend_status)
            self.root.after(0, self.update_backend_buttons_state)
            
        except Exception as ex:
            logging.getLogger(__name__).error(f"Backend start error: {ex}")
            error_msg = str(ex)
            self.root.after(0, lambda: self.update_status(f"Error starting backend: {error_msg}"))
            self.root.after(0, self.update_backend_buttons_state)

    def stop_backend(self):
        """Stop backend"""
        try:
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
            
            # Update UI state
            self.root.after(0, self._update_backend_status)
            self.root.after(0, self.update_backend_buttons_state)
            
        except Exception as ex:
            logging.getLogger(__name__).error(f"Backend stop error: {ex}")
            error_msg = str(ex)
            self.root.after(0, lambda: self.update_status(f"Error stopping backend: {error_msg}"))
            self.root.after(0, self.update_backend_buttons_state)

    def clear_logs(self):
        """Clear logs"""
        try:
            self.widgets['log_output'].delete("1.0", "end")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error clearing logs: {e}")

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
            self.root.deiconify()  # Show window
            self.root.lift()       # Bring to front
            self.root.focus_force()  # Give focus
            self.window_visible = True
            # Update tray menu to reflect window state change
            if self.system_tray and self.system_tray.is_running:
                self.system_tray.refresh_menu()
            logging.getLogger(__name__).info("Window shown via callback")
        except Exception as ex:
            logging.getLogger(__name__).error(f"Show window failed: {ex}")

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
        
        # Stop executor
        try:
            self.executor.shutdown(wait=False)
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