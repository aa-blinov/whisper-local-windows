import logging
import threading
from typing import Optional, TYPE_CHECKING
from pathlib import Path

from .utils import resolve_asset_path

try:
    import pystray
    from PIL import Image
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    pystray = None
    Image = None

if TYPE_CHECKING:
    from .state_manager import StateManager
    from .config_manager import ConfigManager

class SystemTray:
    def __init__(self,
                 state_manager: Optional['StateManager'],
                 tray_config: dict = None,
                 config_manager: Optional['ConfigManager'] = None,
                 show_window_callback: Optional[callable] = None,
                 is_window_visible_callback: Optional[callable] = None):
        """System tray initialization.

        The state_manager parameter can be None (deferred binding) so that the object
        can be created before StateManager/UI is fully ready.
        """
        self.state_manager = state_manager
        self.tray_config = tray_config or {}
        self.config_manager = config_manager
        self.show_window_callback = show_window_callback
        self.is_window_visible_callback = is_window_visible_callback
        self.logger = logging.getLogger(__name__)

        self.icon = None  # pystray Icon
        self.is_running = False
        self.current_state = "idle"
        self.thread = None
        self.available = True
        self.icons = {}
        self._quit_callback = None  # Ability to notify UI of exit intent
        self._quit_flag = False

        if self._check_tray_availability():
            self._load_icons_to_cache()

    def attach_state_manager(self, state_manager: 'StateManager'):
        """Late binding of StateManager (if initially None)."""
        self.state_manager = state_manager
        # Attach updated menu if icon already started
        if self.is_running and self.icon:
            try:
                self.icon.menu = self._create_menu()
            except Exception as e:
                self.logger.error(f"Failed to rebuild tray menu after attaching state manager: {e}")
    
    def _check_tray_availability(self) -> bool:
        if not self.tray_config.get('enabled', True):
            self.logger.warning("[tray] disabled in configuration")
            self.available = False
        elif not TRAY_AVAILABLE:
            self.logger.warning("[tray] pystray or Pillow not installed -> tray unavailable")
            self.available = False
        else:
            self.logger.debug(f"[tray] pystray backend available (TRAY_AVAILABLE={TRAY_AVAILABLE})")
        return self.available
    
    def _load_icons_to_cache(self):
        try:
            self.icons = {}
            
            icon_files = {
                "idle": "assets/tray_idle.png",
                "recording": "assets/tray_recording.png",
                "processing": "assets/tray_processing.png"
            }
            
            for state, asset_path in icon_files.items():
                icon_path = Path(resolve_asset_path(asset_path))
                
                try:
                    if icon_path.exists():
                        self.icons[state] = Image.open(str(icon_path))
                    else:
                        self.icons[state] = self._create_fallback_icon(state)
                        self.logger.warning(f"Icon file not found, using fallback: {icon_path}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to load icon {icon_path}: {e}")
                    self.icons[state] = self._create_fallback_icon(state)

        except Exception as e:
            self.logger.error(f"Failed to load system tray: {e}")
            self.available = False
        
    def _create_fallback_icon(self, state: str):
        colors = {
            'idle': (128, 128, 128),      # Gray
            'recording': (34, 139, 34),   # Green
            'processing': (255, 165, 0)   # Orange
        }
        
        color = colors.get(state, (128, 128, 128))  # Default to gray
        icon = Image.new('RGBA', (16, 16), color + (255,))

        return icon
    
    def _create_menu(self):
        try:
            if not self.state_manager:
                raise RuntimeError("State manager not attached yet")
            # Simplified menu with only Show/Hide and Exit
            menu_items = []
            
            # Add item for showing/hiding window with default action
            if self.is_window_visible_callback and self.show_window_callback:
                is_visible = self.is_window_visible_callback()
                if is_visible:
                    # Check if platform supports default action
                    has_default = getattr(pystray.Icon, 'HAS_DEFAULT', True)
                    menu_items.append(pystray.MenuItem(
                        "Hide Window", 
                        self._hide_window,
                        default=has_default
                    ))
                else:
                    # Check if platform supports default action
                    has_default = getattr(pystray.Icon, 'HAS_DEFAULT', True)
                    menu_items.append(pystray.MenuItem(
                        "Show Window", 
                        self._show_window,
                        default=has_default
                    ))
            
            menu_items.append(pystray.MenuItem("Exit", self._quit_application_from_tray))
            
            menu = pystray.Menu(*menu_items)

            return menu
                
        except Exception as e:
            self.logger.error(f"Error in _create_menu: {e}")
            raise

    def _tray_toggle_recording(self, icon=None, item=None):
        self.state_manager.toggle_recording()

    def _set_transcription_mode(self, auto_paste: bool):        
        self.state_manager.update_transcription_mode(auto_paste)
        self.icon.menu = self._create_menu()

    def _select_model(self, model_size: str):
        try:
            # Request model change through state_manager (may be a stub in new architecture)
            success = False
            if hasattr(self.state_manager, 'request_model_change'):
                success = self.state_manager.request_model_change(model_size)
            
            if success:
                # Save under new 'model' key
                self.config_manager.update_user_setting('whisper', 'model', model_size)
                self.icon.menu = self._create_menu()
            else:
                self.logger.warning(f"Request to change model to {model_size} was not accepted")
                
        except Exception as e:
            self.logger.error(f"Error selecting model {model_size}: {e}")

    def _quit_application_from_tray(self, icon=None, item=None):
        self.logger.info("[tray] Exit requested")
        self._quit_flag = True
        
        # Stop tray in background (don't block callback)
        def _stop_tray_async():
            try:
                self.stop()
            except Exception:
                pass
        
        threading.Thread(target=_stop_tray_async, daemon=True).start()
        
        try:
            if self._quit_callback:
                self._quit_callback()  # This should terminate the process
            else:
                self.logger.warning("[tray] No quit callback, using fallback")
                import os
                os._exit(0)
        except Exception as e:
            self.logger.error(f"[tray] Quit callback failed: {e}")
            # Emergency exit
            import os
            os._exit(1)

    def set_quit_callback(self, fn):
        self._quit_callback = fn

    def _show_window(self, icon=None, item=None):
        """Show application window."""
        if self.show_window_callback:
            try:
                self.show_window_callback()
                # Update menu after window state change
                if self.icon:
                    self.icon.menu = self._create_menu()
            except Exception as e:
                self.logger.error(f"Failed to show window: {e}")
    
    def _hide_window(self, icon=None, item=None):
        """Hide application window."""
        # This function will only be called from the menu,
        # hiding logic when closing window is handled in UI
        try:
            if hasattr(self, '_hide_window_callback') and self._hide_window_callback:
                self._hide_window_callback()
            # Update menu after window state change
            if self.icon:
                self.icon.menu = self._create_menu()
        except Exception as e:
            self.logger.error(f"Failed to hide window: {e}")
    
    def set_hide_window_callback(self, callback):
        """Set callback for hiding window."""
        self._hide_window_callback = callback
    
    def refresh_menu(self):
        """Refresh tray menu to reflect current window state."""
        if self.is_running and self.icon:
            try:
                self.icon.menu = self._create_menu()
                self.logger.info("Tray menu refreshed")
            except Exception as e:
                self.logger.error(f"Failed to refresh tray menu: {e}")
    
    def update_state(self, new_state: str):
        if not TRAY_AVAILABLE or not self.is_running:
            return
        
        self.current_state = new_state
        
        try:
            self.icon.icon = self.icons[new_state]
            self.icon.menu = self._create_menu()
        except Exception as e:
            self.logger.error(f"Failed to update tray icon: {e}")
    
    def start(self):        
        if not self.available:
            return False
        
        if self.is_running:
            self.logger.warning("System tray is already running")
            return True
        
        try:
            if not self.icons:
                self.logger.warning("[tray] no icons loaded; attempting to load now")
                self._load_icons_to_cache()
            idle_icon = self.icons.get("idle")
            if idle_icon is None:
                self.logger.error("[tray] idle icon missing; aborting start")
                return False
            menu = self._create_menu()

            self.icon = pystray.Icon(
                name="lazy-to-text",
                icon=idle_icon,
                title="Lazy to text",
                menu=menu
            )
            
            self.logger.debug("[tray] launching tray thread...")
            self.thread = threading.Thread(target=self._run_tray, daemon=True)
            self.thread.start()

            self.is_running = True
            self.logger.info("System tray started", extra={'user_message': True})
            return True
        except Exception as e:
            self.logger.error(f"Failed to start system tray: {e}")
            return False
    
    def _run_tray(self):
        try:
            self.logger.debug("[tray] entering icon.run() loop")
            self.icon.run()  # pystray runloop (blocking in this thread)
        except Exception as e:
            self.logger.error(f"System tray thread error: {e}")
        finally:
            self.is_running = False
            self.logger.debug("[tray] icon thread ended")
    
    def stop(self):
        if not self.is_running:
            return
        
        try:
            self.icon.stop()
                
            # Wait for thread to finish to avoid deadlock
            if self.thread and self.thread.is_alive() and self.thread != threading.current_thread():
                self.thread.join(timeout=2.0)
                
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"Error stopping system tray: {e}")