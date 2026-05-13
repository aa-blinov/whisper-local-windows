# Install NVIDIA CUDA redistributables via pip so ONNX Runtime can
# load the CUDA Execution Provider without a system-wide CUDA Toolkit.
#
# This is the recommended path for Windows users with an NVIDIA GPU
# who want GPU-accelerated transcription.  The packages add ~1 GB
# download / ~2.5 GB on disk but remove the need to install CUDA
# Toolkit + cuDNN manually.
#
# Usage:
#     .\scripts\setup-cuda-windows.ps1
#
# The script detects whether you're inside a uv-managed venv and uses
# ``uv pip install`` automatically; otherwise it falls back to plain
# ``pip install``.

[CmdletBinding()]
Param()

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    Write-Host '== Lazy to Text — CUDA redistributables setup =='

    $cudaPackages = @(
        'onnxruntime-gpu'
        'nvidia-cublas-cu12>=12.0'
        'nvidia-cuda-runtime-cu12>=12.0'
        'nvidia-cudnn-cu12>=9.0'
        'nvidia-cufft-cu12>=11.0'
    )

    # Prefer uv when available (fast, uses the project lock file).
    $useUv = $false
    try {
        uv --version > $null 2>&1
        if ($LASTEXITCODE -eq 0) { $useUv = $true }
    } catch {
        $useUv = $false
    }

    if ($useUv) {
        Write-Host "Installing via uv (project venv)..."
        uv pip install @cudaPackages
    } else {
        Write-Host "uv not found; falling back to pip..."
        pip install @cudaPackages
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installation failed (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host 'CUDA redistributables installed.' -ForegroundColor Green
    Write-Host ''
    Write-Host 'GPU acceleration is now available.  Restart the app:'
    Write-Host '    uv run lazy-to-text-ui'
} finally {
    Pop-Location
}
