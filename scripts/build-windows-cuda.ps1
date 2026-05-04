# Build (or rebuild) the Windows portable folder bundle for
# Lazy to Text via PyInstaller — **GPU-accelerated (CUDA)** variant.
#
# Output: dist/LazyToText/  (drop-onto-another-PC folder, no
# install required, ~1.8 GB).  The bundle includes
# ``onnxruntime-gpu`` + NVIDIA CUDA redistributables so it works
# out-of-the-box on any Windows PC with an NVIDIA GPU.  It still
# falls back to CPU on machines without a GPU.
#
# Usage:
#     .\scripts\build-windows-cuda.ps1                # default folder build
#     .\scripts\build-windows-cuda.ps1 -Clean         # nuke build/ + dist/ first
#     .\scripts\build-windows-cuda.ps1 -OneFile       # single-file .exe
#     .\scripts\build-windows-cuda.ps1 -SkipSync      # skip dependency sync
#     .\scripts\build-windows-cuda.ps1 -DryRun        # validate prereqs only
#
# For a **CPU-only** bundle (smaller, no NVIDIA required) use:
#     .\scripts\build-windows.ps1
#
# Prerequisites: ``uv sync`` has been run (so the venv has
# pyinstaller installed) and you're on Windows.

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
    Write-Host '== Lazy to Text — Windows portable build (GPU / CUDA) =='

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
        Write-Host 'Resolving dependencies via uv sync --extra cuda...'
        $syncArgs = @('sync', '--extra', 'cuda')
        if (Test-Path 'uv.lock') { $syncArgs += '--frozen' }
        & $Uv @syncArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "uv sync failed (exit $LASTEXITCODE)" -ForegroundColor Red
            exit $LASTEXITCODE
        }

        # The extra installs CUDA runtime DLLs (cublas, cuDNN, …)
        # but *not* the GPU onnxruntime wheel — that lives in the
        # ``[gpu,hub]`` extra of ``onnx-asr``.  We install it
        # explicitly so PyInstaller bundles the GPU provider.
        #
        # ``--force-reinstall`` guards against the edge case where a
        # previous CPU-only ``uv sync`` left the ``onnxruntime``
        # namespace half-removed (dist-info present but no module
        # files).  Forcing the GPU wheel guarantees a clean install.
        Write-Host 'Installing onnxruntime-gpu wheel...'
        & $Uv 'pip' 'install' '--force-reinstall' 'onnxruntime-gpu'
        if ($LASTEXITCODE -ne 0) {
            Write-Host "onnxruntime-gpu install failed (exit $LASTEXITCODE)" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } else {
        Write-Host 'Skipping dependency sync per user request.'
    }

    if ($DryRun) {
        Write-Host 'Dry-run requested; skipping PyInstaller execution.'
        exit 0
    }

    $env:LAZYTOTEXT_BUILD_NAME = 'LazyToText-CUDA'

    if ($OneFile) {
        Write-Host 'Building one-file variant (experimental)...'
        $args = @(
            'run', 'pyinstaller',
            '--noconfirm',
            '--onefile',
            '--name', 'LazyToText-CUDA',
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
        $exe = Resolve-Path 'dist\LazyToText-CUDA.exe'
        Write-Host "Output:  $exe"
        Write-Host "Run:     $exe"
    } else {
        $folder = Resolve-Path 'dist\LazyToText-CUDA'
        Write-Host "Output:  $folder"
        Write-Host "Run:     $(Join-Path $folder 'LazyToText-CUDA.exe')"
        Write-Host ''
        Write-Host 'Drop the entire dist\LazyToText-CUDA\ folder onto another PC —'
        Write-Host 'no install / admin / PATH required.  Works on NVIDIA GPUs'
        Write-Host 'via CUDA, falls back to CPU on all other machines.'
    }
} finally {
    Pop-Location
}
