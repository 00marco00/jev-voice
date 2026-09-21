"""Monochrome menu-bar icons rendered from SF Symbols.

Renders `mic.fill` / `mic.slash.fill` to cached PNGs so rumps can display
them with template=True (native menu-bar look, cf. the emoji 🎙 which
sticks out). Must run on the main thread. Falls back to None when
unavailable (caller keeps the emoji title).
"""
from __future__ import annotations

from pathlib import Path

CACHE = Path.home() / ".cache" / "jev-voice" / "icons"
RUNNING = "mic.fill"
PAUSED = "mic.slash.fill"
SIZE = 22


def symbol_png(name: str, size: int = SIZE) -> Path | None:
    out = CACHE / f"{name.replace('.', '_')}_{size}.png"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        from AppKit import (  # type: ignore
            NSBitmapImageRep,
            NSColor,
            NSCompositingOperationSourceOver,
            NSDeviceRGBColorSpace,
            NSGraphicsContext,
            NSImage,
            NSMakeSize,
            NSPNGFileType,
        )

        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, name)
        if img is None:
            return None
        img.setSize_(NSMakeSize(size, size))
        rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None, size, size, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0
        )
        NSGraphicsContext.saveGraphicsState()
        try:
            NSGraphicsContext.setCurrentContext_(
                NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
            )
            NSColor.blackColor().set()
            img.drawAtPoint_fromRect_operation_fraction_(
                (0, 0), ((0, 0), (size, size)), NSCompositingOperationSourceOver, 1.0
            )
        finally:
            NSGraphicsContext.restoreGraphicsState()
        data = rep.representationUsingType_properties_(NSPNGFileType, None)
        if data is None:
            return None
        CACHE.mkdir(parents=True, exist_ok=True)
        if not data.writeToFile_atomically_(str(out), False):
            return None
        return out
    except Exception:
        return None
