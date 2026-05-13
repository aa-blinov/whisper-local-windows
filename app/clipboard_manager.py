import logging
import sys
import time
from typing import Optional

import pyperclip

from app.macos_permissions import is_post_event_access_trusted

if sys.platform == "win32":
    import win32api
    import win32con
    import win32gui
else:
    win32api = None  # type: ignore[assignment]
    win32con = None  # type: ignore[assignment]
    win32gui = None  # type: ignore[assignment]

# macOS virtual key codes (ANSI layout — Apple keeps these stable
# across keyboard layouts via the HIToolbox key-code abstraction;
# the OS does the layout translation when the event is delivered).
# Defined as module constants so the keystroke helpers don't have
# to reach into the Carbon framework just for two integers.
_MAC_KEYCODE_V = 0x09        # ANSI_V
_MAC_KEYCODE_RETURN = 0x24   # ANSI_Return
_MAC_KEYCODE_LEFT_COMMAND = 0x37


class ClipboardManager:
    def __init__(self, key_simulation_delay, auto_paste, preserve_clipboard):
        self.logger = logging.getLogger(__name__)
        self.key_simulation_delay = key_simulation_delay
        self.auto_paste = auto_paste
        self.preserve_clipboard = preserve_clipboard
        self._test_clipboard_access()
        self._check_mac_post_event_access()
        self._print_status()
    
    def _test_clipboard_access(self):
        try:
            pyperclip.paste()
            self.logger.info("Clipboard access test successful")

        except Exception as e:
            self.logger.error(f"Clipboard access test failed: {e}")
            raise

    def _check_mac_post_event_access(self) -> None:
        """Probe whether macOS will let us drive auto-paste keystrokes.

        This is intentionally *diagnostic only*. We no longer auto-fire
        the system permission prompt from startup code — the Settings UI
        owns that interaction so the user sees a clear explanation of
        why auto-paste is unavailable.
        """
        if sys.platform != "darwin":
            return
        granted = is_post_event_access_trusted()
        if granted is None:
            self.logger.debug(
                "macOS Accessibility probe unavailable in this runtime "
                "(older macOS or older pyobjc) — skipping dedicated "
                "auto-paste permission check."
            )
            return
        if granted:
            self.logger.info(
                "macOS keyboard-control access: granted — auto-paste "
                "keystrokes can be delivered to the focused app."
            )
            return
        self.logger.warning(
            "macOS keyboard-control access: NOT granted — auto-paste "
            "Cmd+V will not reach the focused app until the user "
            "allows Lazy to Text under Accessibility."
        )
    
    def _print_status(self):
        if sys.platform == "darwin":
            paste_combo = "Cmd+V"
        else:
            paste_combo = "Ctrl+V"
        if self.auto_paste:
            self.logger.info(
                f"Auto-paste is ENABLED using key simulation ({paste_combo})",
                extra={'user_message': True},
            )
        else:
            self.logger.info(
                f"Auto-paste is DISABLED - paste manually with {paste_combo}",
                extra={'user_message': True},
            )
    
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
        if sys.platform != "win32":
            # macOS / Linux don't expose a process-friendly window
            # handle here — focus tracking would need NSWorkspace /
            # X11 calls and isn't load-bearing for paste delivery
            # (the OS routes synthetic key events to the focused
            # window directly).
            return None
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

            if sys.platform == "win32":
                try:
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd:
                        self.logger.debug(
                            f"Auto-paste target window: "
                            f"'{win32gui.GetWindowText(hwnd)}' ({hwnd})"
                        )
                except Exception:
                    pass

            self._send_paste_combo()
            self.logger.info(
                "Auto-pasted via key simulation", extra={'user_message': True}
            )

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
            self.logger.info(
                "Text submitted with ENTER!", extra={'user_message': True}
            )

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

    def _send_paste_combo(self):
        """Send the paste hotkey to the focused window.

        Windows: raw ``win32api.keybd_event`` Ctrl+V — works against
        every Win32 app, no permission prompts.

        macOS: first try ``AXUIElementPostKeyboardEvent`` against the
        active app through the Accessibility API. That targets the
        focused application directly and is more reliable for a
        background helper app than spraying a generic Quartz event
        into the session. If that path isn't available, fall back to
        the lower-level Quartz ``CGEventPost`` implementation.

        Linux: keep ``pyautogui`` (X11 / Wayland backend).
        """
        if sys.platform == "win32":
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
            return

        if sys.platform == "darwin":
            if not self._send_mac_accessibility_key_sequence(
                [
                    (0, _MAC_KEYCODE_LEFT_COMMAND, True),
                    (ord("v"), _MAC_KEYCODE_V, True),
                    (ord("v"), _MAC_KEYCODE_V, False),
                    (0, _MAC_KEYCODE_LEFT_COMMAND, False),
                ],
                "Cmd+V",
            ):
                self._send_mac_keystroke_with_cmd(_MAC_KEYCODE_V, "Cmd+V")
            return

        # Linux — pyautogui with X11 / Wayland.
        try:
            import pyautogui

            pyautogui.hotkey("ctrl", "v")
            time.sleep(max(0.02, self.key_simulation_delay))
        except Exception as e:
            self.logger.error(f"Failed to send Ctrl+V: {e}")

    def _send_enter(self):
        if sys.platform == "win32":
            try:
                self._key_down(win32con.VK_RETURN)
                time.sleep(0.01)
                self._key_up(win32con.VK_RETURN)
                time.sleep(max(0.02, self.key_simulation_delay))
            except Exception as e:
                self.logger.error(f"Failed to send ENTER: {e}")
            return

        if sys.platform == "darwin":
            if not self._send_mac_accessibility_key_sequence(
                [
                    (0x0D, _MAC_KEYCODE_RETURN, True),
                    (0x0D, _MAC_KEYCODE_RETURN, False),
                ],
                "Enter",
            ):
                self._send_mac_keystroke(_MAC_KEYCODE_RETURN, "Enter")
            return

        try:
            import pyautogui

            pyautogui.press("enter")
            time.sleep(max(0.02, self.key_simulation_delay))
        except Exception as e:
            self.logger.error(f"Failed to send ENTER: {e}")

    # ---- macOS keystroke helpers --------------------------------------------

    def _send_mac_accessibility_key_sequence(
        self,
        steps: list[tuple[int, int, bool]],
        label: str,
    ) -> bool:
        """Send keys to the active macOS app via Accessibility APIs.

        This path targets the focused application directly instead of
        posting low-level events into the generic Quartz stream. It is
        more reliable for background helper apps like ours because the
        destination stays the active app rather than the current
        process.
        """
        try:
            from ApplicationServices import (
                AXUIElementCreateSystemWide,
                AXUIElementPostKeyboardEvent,
            )
        except ImportError as exc:
            self.logger.debug(
                "Accessibility keyboard API unavailable for %s: %s",
                label,
                exc,
            )
            return False

        try:
            target = AXUIElementCreateSystemWide()
            for key_char, key_code, key_down in steps:
                result = AXUIElementPostKeyboardEvent(
                    target,
                    key_char,
                    key_code,
                    bool(key_down),
                )
                if int(result) != 0:
                    self.logger.debug(
                        "Accessibility key send failed for %s "
                        "(char=%s keycode=%s down=%s result=%s)",
                        label,
                        key_char,
                        key_code,
                        key_down,
                        result,
                    )
                    return False
                time.sleep(0.01)
            time.sleep(max(0.02, self.key_simulation_delay))
            return True
        except Exception as exc:
            self.logger.debug(
                "Accessibility key send raised for %s: %s", label, exc
            )
            return False

    def _send_mac_keystroke(
        self, key_code: int, label: str, flags: int = 0
    ) -> None:
        """Post a synthetic key-down + key-up pair via Quartz CGEvent.

        Three details that took experimentation to get right —
        documented inline because every one of them is a silent-
        failure mode that ``pynput`` and ``pyautogui`` also trip
        on, and there is no exception or log to point at:

        1. **Pass an explicit CGEventSource** built with
           ``kCGEventSourceStateCombinedSessionState``.  Passing
           ``None`` materialises a default event whose modifier
           flags don't combine with the live HID state — most
           apps see "v" without the Command bit set even though
           we called ``CGEventSetFlags(ev, kCGEventFlagMaskCommand)``.
           Telegram and Electron-based editors are the canonical
           offenders: they look at the ``flags`` field of the V
           event itself and reject the paste when the bit isn't
           there.  With a combined-session source the OS carries
           the modifier forward correctly.

        2. **Post to ``kCGAnnotatedSessionEventTap``**, not
           ``kCGHIDEventTap``.  HID is the window-server tap and
           is meant for system-wide HID input;  the session tap
           is what delivers events to the focused application.
           On macOS 14+ posting Cmd+V to HID from a non-foreground
           app frequently lands in the void.

        3. **Set the modifier flag directly on the V key event,
           not via a separate flagsChanged event.**  Real
           keyboards work that way (the OS aggregates modifier
           state across events), but synthetic CGEvents don't
           carry a separate flagsChanged forward reliably — the
           app sees the V event with empty flags.  Stamping the
           flag on the down + up events themselves is what every
           working PyObjC paste-recipe does.

        ``key_code`` is an HIToolbox virtual key code (``0x09``
        for V, ``0x24`` for Return — see ``_MAC_KEYCODE_*`` above).
        ``flags`` is an OR'd ``kCGEventFlagMask*`` value;  pass 0
        for an unmodified keystroke.

        Failures are logged but never raised — the worker thread
        that calls this has the text in clipboard already, so the
        user can fall back to a manual ``Cmd+V`` if the synthetic
        path is denied.
        """
        try:
            from Quartz import (
                CGEventCreateKeyboardEvent,
                CGEventPost,
                CGEventSetFlags,
                CGEventSourceCreate,
                kCGAnnotatedSessionEventTap,
                kCGEventSourceStateCombinedSessionState,
            )
        except ImportError as exc:
            self.logger.error(
                "Failed to import Quartz for %s send: %s", label, exc,
            )
            return

        try:
            source = CGEventSourceCreate(
                kCGEventSourceStateCombinedSessionState
            )
            event_down = CGEventCreateKeyboardEvent(source, key_code, True)
            event_up = CGEventCreateKeyboardEvent(source, key_code, False)
            if flags:
                CGEventSetFlags(event_down, flags)
                CGEventSetFlags(event_up, flags)
            CGEventPost(kCGAnnotatedSessionEventTap, event_down)
            time.sleep(0.02)
            CGEventPost(kCGAnnotatedSessionEventTap, event_up)
            time.sleep(max(0.02, self.key_simulation_delay))
        except Exception as exc:
            self.logger.error("Failed to send %s via Quartz: %s", label, exc)

    def _send_mac_keystroke_with_cmd(self, key_code: int, label: str) -> None:
        """``_send_mac_keystroke`` with the Command modifier flag set."""
        try:
            from Quartz import kCGEventFlagMaskCommand
        except ImportError as exc:
            self.logger.error(
                "Failed to import Quartz for %s send: %s", label, exc,
            )
            return
        self._send_mac_keystroke(key_code, label, flags=kCGEventFlagMaskCommand)
