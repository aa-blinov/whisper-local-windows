; Inno Setup script for Lazy to Text — per-user Windows installer.
;
; Wraps the PyInstaller onedir output (``dist/LazyToText/``) into a
; single .exe installer that:
;   - drops the app under ``%LOCALAPPDATA%\Programs\LazyToText\``
;     so it doesn't need admin (no Program Files dance, no UAC),
;   - registers an uninstaller in Add/Remove Programs,
;   - creates Start Menu + optional Desktop shortcuts,
;   - optionally registers a Windows-startup task,
;   - blocks installation if the app is already running (via the
;     same mutex the runtime uses for single-instance).
;
; Compile from the project root:
;   iscc /DAppVersion=0.1.0 installer\lazy_to_text.iss
;
; The CI workflow passes ``/DAppVersion=…`` derived from the git
; tag. Without the override the script falls back to a dev label
; so local compiles still work.

#define AppName "Lazy to Text"
#define AppPublisher "Alexander Blinov"
#define AppURL "https://github.com/aa-blinov/lazy-to-text"
#define AppExeName "LazyToText.exe"
; Must match ``app.gui.app.try_acquire_single_instance("LazyToTextQt")`` —
; updating one without the other will let the installer overwrite
; an in-flight running instance and corrupt its state.
#define AppMutex "LazyToTextQt_SingleInstance"

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
; Stable GUID — never regenerate, otherwise existing installs
; won't be detected as the same app and upgrades will leave two
; entries in Add/Remove Programs.
AppId={{D7B4F3A2-8C5E-4D9F-9A1B-3F2E1D8C7B6A}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}/releases
DefaultDirName={localappdata}\Programs\LazyToText
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; per-user install — no admin elevation needed. The user can
; still bump to all-users via the Setup wizard if they're admin.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=LazyToText-{#AppVersion}-setup
; lzma2/ultra64 + solid compression: roughly halves a 5 GB
; onedir output's footprint. Slow to build (~5 min on a runner)
; but only done once per release.
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
AppMutex={#AppMutex}
SetupIconFile=..\app\assets\tray_idle.ico
UninstallDisplayIcon={app}\{#AppExeName}
LicenseFile=..\LICENSE
DisableDirPage=auto
DisableReadyPage=no
ShowLanguageDialog=auto
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start Lazy to Text when Windows starts"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
; The whole onedir output. ``recursesubdirs`` to descend into
; ``_internal/``; ``createallsubdirs`` to make sure empty dirs
; survive packaging (some libs probe for them).
Source: "..\dist\LazyToText\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; ``{autoprograms}`` resolves to the per-user Start Menu since
; ``PrivilegesRequired=lowest``.
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startupicon

[Run]
; The 'Launch app now' checkbox at the end of the installer.
; ``nowait`` so the installer doesn't block until the app exits;
; ``postinstall`` keeps the option visible only on a successful
; install (not on uninstall or rollback).
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
