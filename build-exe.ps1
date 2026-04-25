Param(
  [switch]$Clean,
  [switch]$OneFile,
  [Alias('Python')][string]$Uv = 'uv',
  [switch]$SkipSync,
  [switch]$DryRun
)

Write-Host '== Lazy to text build script (uv powered) =='

if ($Clean) {
  Write-Host 'Cleaning dist/ and build/ ...'
  Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
}

Write-Host "Using uv executable: $Uv"
try {
  & $Uv --version > $null 2>&1
} catch {
  Write-Host "Failed to execute '$Uv'. Ensure uv is installed (https://github.com/astral-sh/uv)." -ForegroundColor Red
  exit 1
}

if ($LASTEXITCODE -ne 0) {
  Write-Host "uv command returned exit code $LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

if (-not $SkipSync) {
  Write-Host 'Ensuring project dependencies via uv sync...'
  $syncArgs = @('sync')
  if (Test-Path 'uv.lock') {
    $syncArgs += '--frozen'
  }
  & $Uv @syncArgs
  if ($LASTEXITCODE -ne 0) {
    Write-Host "uv sync failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
  }
} else {
  Write-Host 'Skipping uv sync per user request.'
}

if ($DryRun) {
  Write-Host 'Dry-run requested; skipping PyInstaller execution.'
  exit 0
}

if ($OneFile) {
  Write-Host 'Building one-file variant (experimental)...'
  # One-file: assets внутри exe -> придётся доставать их из _MEIPASS
  $pyinstallerArgs = @(
    'run',
    'pyinstaller',
    '--onefile',
    '--name', 'LazyToText',
    '--icon', 'app\assets\tray_idle.ico',
    '--add-data', 'app\assets;assets',
    '--add-data', 'app\gui\styles;gui\styles',
    '--add-data', 'config.yaml;.',
    '--hidden-import', 'win32timezone',
    '--hidden-import', 'global_hotkeys',
    '--hidden-import', 'PySide6.QtCore',
    '--hidden-import', 'PySide6.QtGui',
    '--hidden-import', 'PySide6.QtWidgets',
    'lazy-to-text-ui.py'
  )
  & $Uv @pyinstallerArgs
} else {
  Write-Host 'Building using spec file (folder mode)...'
  $specArgs = @('run', 'pyinstaller', 'lazy_to_text.spec')
  & $Uv @specArgs
}

if ($LASTEXITCODE -eq 0) {
  Write-Host 'Build finished successfully.' -ForegroundColor Green
  if ($OneFile) {
    $exePath = (Resolve-Path 'dist\LazyToText.exe').Path
    Write-Host 'Output:' $exePath
    Write-Host 'Run:' $exePath
  } else {
    $folderPath = (Resolve-Path 'dist\LazyToText').Path
    Write-Host 'Output:' $folderPath
    Write-Host 'Run:' (Join-Path $folderPath 'LazyToText.exe')
  }
} else {
  Write-Host 'Build failed.' -ForegroundColor Red
  exit $LASTEXITCODE
}
