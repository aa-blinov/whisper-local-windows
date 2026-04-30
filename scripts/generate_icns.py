"""Generate ``app/assets/Lazy to Text.icns`` from the runtime squircle
icon used in the Dock.

py2app picks up the .icns via ``setup.py``'s ``iconfile`` argument
when bundling. We reuse the same ``_render_dock_icon_at`` painter
that the live app uses for ``QApplication.setWindowIcon`` so the
bundle's Finder / Dock / Cmd-Tab icon is visually identical to
what the in-app code paints — no separate exported PNG to keep
in sync.

Workflow:

  1. Render the squircle at every size macOS expects in an
     ``.iconset`` (``16/32/64/128/256/512/1024`` px, with ``@2x``
     variants where applicable).
  2. Save each as a numbered PNG under a temp ``.iconset`` dir
     (Apple requires this directory layout).
  3. Hand it to ``iconutil --convert icns`` — Apple's stock CLI
     that ships with every Mac, no extra dependency.

Run on a Mac with the project venv active:

    uv run python scripts/generate_icns.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Apple's iconset specification — file names must match exactly,
# the OS uses them to pick the right pixmap for each surface
# (16 px = Finder list view, 1024 px = Mission Control, etc.).
_ICONSET_SIZES = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


def main() -> int:
    if sys.platform != "darwin":
        print("generate_icns.py only runs on macOS", file=sys.stderr)
        return 1

    # Late imports — Qt is heavy, only pay the cost when the
    # script is actually invoked (not on module import for tests
    # that scan the scripts/ directory).
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    # py2app calls iconutil before any QApplication exists; make
    # one here so QPainter / QPixmap have an event loop to run
    # against.
    app = QApplication.instance() or QApplication(sys.argv)
    del app  # silence unused-warning; just need the side effect

    from app.gui.app import _render_dock_icon_at

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "app" / "assets" / "lazy_to_text.icns"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(suffix=".iconset") as tmp:
        iconset_dir = Path(tmp) / "lazy_to_text.iconset"
        iconset_dir.mkdir()
        for filename, size in _ICONSET_SIZES:
            pixmap = _render_dock_icon_at(size)
            image: QImage = pixmap.toImage()
            target = iconset_dir / filename
            if not image.save(str(target), "PNG"):
                print(
                    f"failed to save {target}", file=sys.stderr,
                )
                return 2

        # ``iconutil`` is a stock macOS CLI shipped at
        # ``/usr/bin/iconutil``; bail if missing (sandboxed CI?).
        try:
            subprocess.run(
                [
                    "iconutil",
                    "--convert", "icns",
                    "--output", str(output_path),
                    str(iconset_dir),
                ],
                check=True,
            )
        except FileNotFoundError:
            print(
                "iconutil not found — install Xcode command-line tools "
                "(``xcode-select --install``)",
                file=sys.stderr,
            )
            return 3
        except subprocess.CalledProcessError as exc:
            print(f"iconutil failed: {exc}", file=sys.stderr)
            return 4

    # Sanity check — every iconset entry should contribute, the
    # final file shouldn't be the 1-frame fallback Apple emits on
    # error.
    size_bytes = output_path.stat().st_size
    if size_bytes < 100_000:
        print(
            f"warning: {output_path.name} is only {size_bytes} bytes — "
            "likely missing iconset frames",
            file=sys.stderr,
        )
    print(f"wrote {output_path} ({size_bytes / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
