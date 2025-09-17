import logging
import time
from typing import Optional

import pyperclip
import win32gui
import win32api
import win32con

class ClipboardManager:    
    def __init__(self, key_simulation_delay, auto_paste, preserve_clipboard):
        self.logger = logging.getLogger(__name__)
        self.key_simulation_delay = key_simulation_delay
        self.auto_paste = auto_paste
        self.preserve_clipboard = preserve_clipboard
        self._test_clipboard_access()
        self._print_status()
    
    def _test_clipboard_access(self):
        try:
            pyperclip.paste()
            self.logger.info("Clipboard access test successful")
            
        except Exception as e:
            self.logger.error(f"Clipboard access test failed: {e}")
            raise
    
    def _print_status(self):
        if self.auto_paste:
            method_name = "key simulation (CTRL+V)"
            self.logger.info(f"Auto-paste is ENABLED using {method_name}", extra={'user_message': True})
        else:
            self.logger.info("Auto-paste is DISABLED - paste manually with Ctrl+V", extra={'user_message': True})
    
    def copy_text(self, text: str) -> bool:
        if not text:
            return False
        
        try:
            self.logger.info(f"Copying text to clipboard ({len(text)} chars)")
            pyperclip.copy(text)
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to copy text to clipboard: {e}")
            return False
    
    def get_clipboard_content(self) -> Optional[str]:
        try:
            clipboard_content = pyperclip.paste()
            
            if clipboard_content:
                return clipboard_content
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to paste text from clipboard: {e}")
            return None
    
    def copy_with_notification(self, text: str) -> bool:
        if not text:
            return False
        
        success = self.copy_text(text)
        
        if success:
            self.logger.info("Copied to clipboard", extra={'user_message': True})
            self.logger.info("You can now paste with Ctrl+V in any application!", extra={'user_message': True})
        
        return success
    
    def clear_clipboard(self) -> bool:
        try:
            pyperclip.copy("")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear clipboard: {e}")
            return False
    
    def get_active_window_handle(self) -> Optional[int]:
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                window_title = win32gui.GetWindowText(hwnd)
                self.logger.info(f"Active window: '{window_title}' (handle: {hwnd})")
                return hwnd
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to get active window handle: {e}")
            return None       
    
    def execute_auto_paste(self, text: str, preserve_clipboard: bool) -> bool:              
        try:
            original_content = None
            if preserve_clipboard:
                original_content = pyperclip.paste()

            if not self.copy_text(text):
                return False
            time.sleep(max(0.02, self.key_simulation_delay))

            try:
                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    self.logger.debug(f"Auto-paste target window: '{win32gui.GetWindowText(hwnd)}' ({hwnd})")
            except Exception:
                pass

            self._send_ctrl_v()
            self.logger.info("Auto-pasted via Win32 key simulation", extra={'user_message': True})

            if original_content is not None:
                restore_delay = max(0.15, self.key_simulation_delay * 3)
                self.logger.debug(f"Waiting {restore_delay:.3f}s before restoring original clipboard content")
                time.sleep(restore_delay)
                pyperclip.copy(original_content)
                time.sleep(self.key_simulation_delay)

            return True
            
        except Exception as e:
            self.logger.error(f"Failed to simulate paste keypress: {e}")
            return False
        
    def send_enter_key(self) -> bool:
        try:
            self.logger.info("Sending ENTER key to active application")
            self._send_enter()
            self.logger.info("Text submitted with ENTER (Win32)!", extra={'user_message': True})

            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send ENTER key: {e}")
            return False

    def deliver_transcription(self,
                              transcribed_text: str,
                              use_auto_enter: bool = False) -> bool:
        
        try:
            if use_auto_enter:
                self.logger.info("Auto-pasting text and SENDING with ENTER...", extra={'user_message': True})
               
                success = self.execute_auto_paste(transcribed_text, self.preserve_clipboard)
                if success:
                    success = self.send_enter_key()

            elif self.auto_paste:
                self.logger.info("Auto-pasting text...", extra={'user_message': True})
                success = self.execute_auto_paste(transcribed_text, self.preserve_clipboard)             
                    
            else:
                self.logger.info("Copying to clipboard...", extra={'user_message': True})
                success = self.copy_with_notification(transcribed_text)        

            return success

        except Exception as e:
            self.logger.error(f"Delivery workflow failed: {e}")
            return False
        
    def update_auto_paste(self, enabled: bool):
        self.auto_paste = enabled
        self._print_status()

    def _key_down(self, vk_code: int):
        try:
            win32api.keybd_event(vk_code, 0, 0, 0)
        except Exception as e:
            self.logger.error(f"key_down failed for vk={vk_code}: {e}")

    def _key_up(self, vk_code: int):
        try:
            win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        except Exception as e:
            self.logger.error(f"key_up failed for vk={vk_code}: {e}")

    def _send_ctrl_v(self):
        try:
            self._key_down(win32con.VK_CONTROL)
            time.sleep(0.01)
            self._key_down(ord('V'))
            time.sleep(0.01)
            self._key_up(ord('V'))
            time.sleep(0.005)
            self._key_up(win32con.VK_CONTROL)
            time.sleep(max(0.02, self.key_simulation_delay))
        except Exception as e:
            self.logger.error(f"Failed to send Ctrl+V: {e}")

    def _send_enter(self):
        try:
            self._key_down(win32con.VK_RETURN)
            time.sleep(0.01)
            self._key_up(win32con.VK_RETURN)
            time.sleep(max(0.02, self.key_simulation_delay))
        except Exception as e:
            self.logger.error(f"Failed to send ENTER: {e}")
