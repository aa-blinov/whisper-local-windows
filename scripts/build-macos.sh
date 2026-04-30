#!/usr/bin/env bash
#
# Build (or rebuild) the macOS ``Lazy to Text.app`` bundle.
#
# Default mode is ``alias`` — py2app symlinks the bundle's
# ``site-packages`` and entry script back into the project's venv,
# so each build takes ~5–10 s and source edits are picked up on
# the next launch with no rebuild.  Pass ``--release`` to produce
# a self-contained bundle suitable for distribution (~700 MB,
# 5–10 minutes).
#
# Usage:
#     ./scripts/build-macos.sh             # alias / dev (fast)
#     ./scripts/build-macos.sh --release   # full bundle (slow)
#
# Prerequisites: ``uv sync`` has been run in the repo (so the venv
# has py2app installed) and the project is on a Mac with Xcode
# command-line tools (``iconutil`` ships with them).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="alias"
if [[ "${1:-}" == "--release" ]]; then
    mode="release"
fi

# Always start from a clean dist/ — py2app refuses to overwrite a
# pre-existing bundle and the residue of an earlier build can
# silently mask missing files.
rm -rf build dist

# Refresh the icon — the squircle is rendered programmatically
# from the same painter the running app uses, so any token /
# colour change picks up automatically.  Skip silently if iconutil
# is missing; the build still works without an icon (it just looks
# generic in the Dock).
if command -v iconutil >/dev/null 2>&1; then
    echo "→ regenerating .icns from the runtime squircle"
    uv run python scripts/generate_icns.py
else
    echo "  (iconutil not found — bundle will use a default icon)"
fi

# py2app 0.28 trips on PEP 621 ``[project] dependencies`` —
# setuptools auto-derives ``Distribution.install_requires`` from
# them and py2app raises ``install_requires is no longer
# supported``.  We can't drop ``dependencies`` from
# ``pyproject.toml`` permanently (every other tool — uv, pip,
# IDEs — reads them from there), so we temporarily strip the
# block, run py2app, restore the original on exit (including on
# Ctrl-C or build failure).
restore_pyproject() {
    if [[ -f pyproject.toml.bak ]]; then
        mv pyproject.toml.bak pyproject.toml
    fi
}
trap restore_pyproject EXIT INT TERM

cp pyproject.toml pyproject.toml.bak
python3 <<'PY'
import re
text = open("pyproject.toml").read()
# Remove the multi-line ``dependencies = [...]`` block so
# setuptools doesn't populate ``install_requires`` from it.
patched = re.sub(
    r"^dependencies\s*=\s*\[.*?^\]\s*$",
    "",
    text,
    flags=re.DOTALL | re.MULTILINE,
)
open("pyproject.toml", "w").write(patched)
PY

# py2app expects a literal ``-A`` flag for alias mode; setup.py
# leaves ``alias`` out of the static config so non-alias builds
# stay self-contained.
if [[ "$mode" == "alias" ]]; then
    echo "→ py2app alias build (fast iteration)"
    uv run python setup.py py2app -A
else
    echo "→ py2app release build (full bundle, slow)"
    uv run python setup.py py2app
fi

# Restore happens via the EXIT trap; keep a no-op here so the
# next ``set -e`` step doesn't see a stale exit code from py2app.
true

bundle="dist/Lazy to Text.app"
if [[ ! -d "$bundle" ]]; then
    echo "✗ expected $bundle to exist after build" >&2
    exit 1
fi

# Ad-hoc codesign so Gatekeeper doesn't quarantine the bundle on
# first launch (Apple Silicon requires a signature even for local
# unsigned binaries).  ``--deep`` ensures every embedded framework
# / dylib also gets the same identity; ``--force`` overwrites any
# stale signature from a prior build.
echo "→ ad-hoc codesigning (no Apple Developer ID required)"
codesign --sign - --deep --force "$bundle" || {
    echo "  warning: codesign failed; bundle may need a quarantine"   \
         "exemption to launch (xattr -dr com.apple.quarantine \"$bundle\")" >&2
}

echo
echo "✓ built $bundle"
echo "  drag it into /Applications, or run:"
echo "      open \"$bundle\""
