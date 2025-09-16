Param(
  [switch]$Clean,
  [switch]$OneFile,
  [string]$Python = 'python'
)

Write-Host '== Lazy to text build script =='

if ($Clean) {
  Write-Host 'Cleaning dist/ and build/ ...'
  Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
}

# Ensure pyinstaller
try {
  & $Python -m pip show pyinstaller > $null 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing PyInstaller...'
    & $Python -m pip install --upgrade pip
    & $Python -m pip install pyinstaller
  }
} catch {
  Write-Host 'Installing PyInstaller (fresh)...'
  & $Python -m pip install --upgrade pip
  & $Python -m pip install pyinstaller
}

if ($OneFile) {
  Write-Host 'Building one-file variant (experimental)...'
  # One-file: assets внутри exe -> придётся доставать их из _MEIPASS
  & $Python -m PyInstaller --onefile --name LazyToText `
    --icon src\whisper_key\assets\tray_idle.ico `
    --add-data "src\\whisper_key\\assets;whisper_key\\assets" `
    --add-data "config.yaml;." `
    --hidden-import customtkinter `
    --hidden-import PIL._tkinter_finder `
    --hidden-import pystray._win32 `
    --hidden-import win32timezone `
    lazy-to-text-ui.py
} else {
  Write-Host 'Building using spec file (folder mode)...'
  & $Python -m PyInstaller lazy_to_text.spec
}

if ($LASTEXITCODE -eq 0) {
  Write-Host 'Build finished successfully.' -ForegroundColor Green
  Write-Host 'Output:' (Resolve-Path dist\LazyToText)
  Write-Host 'Run:' (Join-Path (Resolve-Path dist\LazyToText) 'LazyToText.exe')
} else {
  Write-Host 'Build failed.' -ForegroundColor Red
  exit 1
}
