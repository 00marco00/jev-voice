"""Menu bar app: jev-voice without a terminal.

A rumps status-bar icon (🎙 running · ⏸ paused) owns the Cocoa runloop on the
main thread; the voice engine runs in a background thread and shares the
floating transcription pill. No Dock icon (accessory policy).

    jev-tray                      # hands-free (JEV_MODE=smart|hold|always)
    jev-tray --mode hold --device "MacBook"

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
from .overlay import Overlay


class VoiceTray(rumps.App):
    def __init__(self, mode: str, device: str | None, no_overlay: bool) -> None:
        super().__init__("🎙", quit_button=None)
        self.mode = mode
        self.device = device
        self.no_overlay = no_overlay
        self.session = None
        self.paused = False
        self.menu = [
            rumps.MenuItem("Pause", callback=self.toggle),
            None,
            rumps.MenuItem(f"Mode: {mode}", callback=None),
            None,
            rumps.MenuItem("Quit", callback=self.quit),
        ]

    # ------------------------------------------------------------ engine

    def start_engine(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="voice-engine").start()

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
        if self.paused:
            self.session.listener.paused_until = 0.0
            self.session.listener.drain()
            self.paused = False
            sender.title = "Pause"
            self.title = "🎙"
            voice.OVERLAY.set("idle", "Listening")
        else:
            self.session.listener.pause(3600.0)
            self.session.listener.drain()
            self.paused = True
            sender.title = "Resume"
            self.title = "⏸"
            voice.OVERLAY.set("idle", "Paused")

    def quit(self, _sender: rumps.MenuItem) -> None:
        rumps.quit_application()


def main() -> None:
    p = argparse.ArgumentParser(prog="jev-tray", description=__doc__)
    p.add_argument("--mode", default=os.environ.get("JEV_MODE", "smart"),
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

    app = VoiceTray(args.mode, args.device, args.no_overlay)
    app.start_engine()
    app.run()


if __name__ == "__main__":
    main()
