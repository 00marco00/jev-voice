"""Menu bar app: jev-voice without a terminal.

A rumps status-bar icon owns the Cocoa runloop on the main thread; the voice
engine runs in a background thread and shares the floating transcription pill.
No Dock icon (accessory policy). Default mode is hold (press Caps Lock to
talk); switch to smart/always from the Mode submenu (engine restarts cleanly).

    jev-tray                      # hold mode (JEV_MODE=smart|hold|always)
    jev-tray --mode smart --device "MacBook"

Autostart at login: LaunchAgent `ai.jev.tray` (installed by scripts/setup.sh),
logs to ~/Library/Logs/jev-tray.log. Note: macOS grants Microphone /
Accessibility / Input Monitoring per identity — first launch manually from a
terminal so the prompts appear, then the agent inherits the grants.
"""
from __future__ import annotations

import argparse
import os
import threading
from types import SimpleNamespace

import rumps

from . import main as voice
from .icons import PAUSED, RUNNING, symbol_png
from .overlay import Overlay


class VoiceTray(rumps.App):
    def __init__(self, mode: str, device: str | None, no_overlay: bool,
                 icon_on: str | None, icon_off: str | None) -> None:
        if icon_on:
            super().__init__("jev-voice", title=None, icon=icon_on, template=True, quit_button=None)
        else:
            super().__init__("🎙", quit_button=None)
        self.mode = mode
        self.device = device
        self.no_overlay = no_overlay
        self.icon_on = icon_on
        self.icon_off = icon_off
        self.session = None
        self.thread = None
        self.paused = False
        self.mode_items = {}
        mode_menu = rumps.MenuItem("Mode")
        for m in ("hold", "smart", "always"):
            item = rumps.MenuItem(m.capitalize(), callback=self.switch_mode)
            item.state = 1 if m == mode else 0
            mode_menu.add(item)
            self.mode_items[m] = item
        self.menu = [
            rumps.MenuItem("Pause", callback=self.toggle),
            None,
            mode_menu,
            None,
            rumps.MenuItem("Quit", callback=self.quit),
        ]

    # ------------------------------------------------------------ engine

    def start_engine(self) -> None:
        voice.STOP.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="voice-engine")
        self.thread.start()

    def stop_engine(self) -> None:
        voice.STOP.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
        self.session = None

    def switch_mode(self, sender: rumps.MenuItem) -> None:
        mode = sender.title.lower()
        if mode == self.mode:
            return
        self._set_paused(False)
        self.stop_engine()
        self.mode = mode
        for m, item in self.mode_items.items():
            item.state = 1 if m == mode else 0
        voice.OVERLAY.set("idle", f"Mode: {mode} — starting…")
        self.start_engine()

    def _run(self) -> None:
        args = SimpleNamespace(
            device=self.device, quiet=False, dry_run=False,
            hold=False, always_on=False, ptt=False, no_overlay=self.no_overlay,
        )
        s = voice.Session(args)
        self.session = s
        try:
            if self.mode == "hold":
                voice.run_capslock(s)
            elif self.mode == "always":
                voice.run_always_on(s)
            else:
                voice.run_smart(s)
        except KeyboardInterrupt:
            pass
        finally:
            s.close()

    # ------------------------------------------------------------ menu

    def toggle(self, sender: rumps.MenuItem) -> None:
        if self.session is None:
            rumps.notification("jev-voice", "Engine not ready yet", "Model still loading…")
            return
        self._set_paused(not self.paused)
        sender.title = "Resume" if self.paused else "Pause"

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            if self.session is not None:
                self.session.listener.pause(3600.0)
                self.session.listener.drain()
            if self.icon_off:
                self.icon = self.icon_off
            else:
                self.title = "⏸"
            voice.OVERLAY.set("idle", "Paused")
        else:
            if self.session is not None:
                self.session.listener.paused_until = 0.0
                self.session.listener.drain()
            if self.icon_on:
                self.icon = self.icon_on
            else:
                self.title = "🎙"
            voice.OVERLAY.set("idle", "Listening")

    def quit(self, _sender: rumps.MenuItem) -> None:
        rumps.quit_application()


def main() -> None:
    p = argparse.ArgumentParser(prog="jev-tray", description=__doc__)
    p.add_argument("--mode", default=os.environ.get("JEV_MODE", "hold"),
                   choices=["smart", "hold", "always"])
    p.add_argument("--device", default=os.environ.get("JEV_DEVICE"))
    p.add_argument("--no-overlay", action="store_true",
                   default=os.environ.get("OVERLAY", "1") in ("0", "false", "no"))
    args = p.parse_args()

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory  # type: ignore
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    if not args.no_overlay:
        ov = Overlay()
        ov.prepare()
        voice.OVERLAY = ov

    icon_on = symbol_png(RUNNING)
    icon_off = symbol_png(PAUSED)
    app = VoiceTray(args.mode, args.device, args.no_overlay,
                    str(icon_on) if icon_on else None,
                    str(icon_off) if icon_off else None)
    app.start_engine()
    app.run()


if __name__ == "__main__":
    main()
