# Build (or rebuild) the Windows portable folder bundle for
# Lazy to Text via PyInstaller — **CPU-only** variant.
#
# Output: dist/LazyToText/  (drop-onto-another-PC folder, no
# install required, ~400 MB).  The folder layout is portable —
# config.yaml, app.log, history and downloaded model weights all
# land in %APPDATA%\LazyToText so the bundle itself stays read-
# only.
#
# Usage:
#     .\scripts\build-windows.ps1                # default folder build
#     .\scripts\build-windows.ps1 -Clean         # nuke build/ + dist/ first
#     .\scripts\build-windows.ps1 -OneFile       # single-file .exe (slower
#                                                #  startup, opaque to AV)
#     .\scripts\build-windows.ps1 -SkipSync      # skip ``uv sync``
#     .\scripts\build-windows.ps1 -DryRun        # validate prereqs only
#
# For a **GPU-accelerated** bundle (NVIDIA CUDA) use:
#     .\scripts\build-windows-cuda.ps1
#
# Prerequisites: ``uv sync`` has been run (so the venv has
# pyinstaller installed) and you're on Windows with the right
# pywin32 / global_hotkeys wheels available.

[CmdletBinding()]
Param(
    [switch]$Clean,
    [switch]$OneFile,
    [Alias('Python')][string]$Uv = 'uv',
    [switch]$SkipSync,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# Pin CWD to repo root regardless of where the user invoked the
# script from — the spec resolves paths relative to ``Path.cwd()``.
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    Write-Host '== Lazy to Text — Windows portable build (CPU) =='

    if ($Clean) {
        Write-Host 'Cleaning build/ and dist/ ...'
        Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
    }

    # Validate uv is on PATH before doing anything that might fail
    # noisily; ``uv --version`` is a cheap reachability check.
    Write-Host "Using uv executable: $Uv"
    & $Uv --version > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to execute '$Uv'. Install uv from https://github.com/astral-sh/uv first." `
            -ForegroundColor Red
        exit 1
    }

    if (-not $SkipSync) {
        Write-Host 'Resolving dependencies via uv sync (CPU-only)...'
        $syncArgs = @('sync')
        if (Test-Path 'uv.lock') { $syncArgs += '--frozen' }
        & $Uv @syncArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "uv sync failed (exit $LASTEXITCODE)" -ForegroundColor Red
            exit $LASTEXITCODE
        }
        # ``onnxruntime-gpu`` and ``onnxruntime`` share the same Python
        # namespace but ship as different wheels.  If a previous build
        # (or dev session) installed the GPU wheel, ``uv sync`` may
        # leave the namespace half-removed (dist-info present but no
        # module files).  Force-reinstall guarantees a clean CPU-only
        # wheel for the bundle.
        Write-Host 'Ensuring clean onnxruntime (CPU) wheel...'
        & $Uv 'pip' 'install' '--force-reinstall' 'onnxruntime'
        if ($LASTEXITCODE -ne 0) {
            Write-Host "onnxruntime reinstall failed (exit $LASTEXITCODE)" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } else {
        Write-Host 'Skipping uv sync per user request.'
    }

    if ($DryRun) {
        Write-Host 'Dry-run requested; skipping PyInstaller execution.'
        exit 0
    }

    $env:LAZYTOTEXT_BUILD_NAME = 'LazyToText'

    if ($OneFile) {
        # ``--onefile`` packs everything inside a single .exe and
        # extracts to %TEMP%\_MEIxxxxxx on every launch.  Slower
        # startup (~3-5 s extra), bigger AV-flag risk because
        # heuristics distrust self-extracting binaries.  Folder
        # mode is the recommended distribution path; one-file is
        # here for the rare case where the user really wants a
        # single artifact.
        Write-Host 'Building one-file variant (experimental)...'
        $args = @(
            'run', 'pyinstaller',
            '--noconfirm',
            '--onefile',
            '--name', 'LazyToText',
            '--icon', 'app\assets\tray_idle.ico',
            '--add-data', 'app\assets;assets',
            '--add-data', 'app\gui\styles;gui\styles',
            '--hidden-import', 'win32timezone',
            '--hidden-import', 'global_hotkeys',
            '--hidden-import', 'PySide6.QtCore',
            '--hidden-import', 'PySide6.QtGui',
            '--hidden-import', 'PySide6.QtWidgets',
            '--hidden-import', 'PySide6.QtSvg',
            '--runtime-hook', 'scripts\pyi_runtime_hook.py',
            'lazy-to-text-ui.py'
        )
        & $Uv @args
    } else {
        Write-Host 'Building folder bundle via lazy_to_text.spec ...'
        & $Uv 'run' 'pyinstaller' '--noconfirm' 'lazy_to_text.spec'
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyInstaller failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host 'Build finished successfully.' -ForegroundColor Green
    if ($OneFile) {
        $exe = Resolve-Path 'dist\LazyToText.exe'
        Write-Host "Output:  $exe"
        Write-Host "Run:     $exe"
    } else {
        $folder = Resolve-Path 'dist\LazyToText'
        Write-Host "Output:  $folder"
        Write-Host "Run:     $(Join-Path $folder 'LazyToText.exe')"
        Write-Host ''
        Write-Host 'Drop the entire dist\LazyToText\ folder onto another PC —'
        Write-Host 'no install / admin / PATH required.  User data lives in'
        Write-Host '%APPDATA%\LazyToText so the bundle stays read-only.'
    }
} finally {
    Pop-Location
}
