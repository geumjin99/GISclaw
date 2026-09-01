; Inno Setup script for GISclaw. Invoked by desktop/build_windows.ps1 with
; /DMyAppVersion=2.0.0 /DMyAppFullVersion=2.0.0-beta.1
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef MyAppFullVersion
  #define MyAppFullVersion MyAppVersion
#endif

[Setup]
AppId={{9B2F6E2A-5C1D-4B7E-9A0E-3F1C2D8E7A11}
AppName=GISclaw
AppVersion={#MyAppVersion}
AppVerName=GISclaw {#MyAppFullVersion}
AppPublisher=Han Jinzhen
AppPublisherURL=https://github.com/geumjin99/GISclaw
DefaultDirName={localappdata}\Programs\GISclaw
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=..\..\build
OutputBaseFilename=GISclaw-{#MyAppFullVersion}-windows-x64-setup
SetupIconFile=..\gisclaw.ico
UninstallDisplayIcon={app}\gisclaw\desktop\gisclaw.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\..\build\GISclaw\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\GISclaw"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\gisclaw\desktop\launcher.py"""; WorkingDir: "{app}\gisclaw"; IconFilename: "{app}\gisclaw\desktop\gisclaw.ico"
Name: "{autodesktop}\GISclaw"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\gisclaw\desktop\launcher.py"""; WorkingDir: "{app}\gisclaw"; IconFilename: "{app}\gisclaw\desktop\gisclaw.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\gisclaw\desktop\launcher.py"""; WorkingDir: "{app}\gisclaw"; Description: "Launch GISclaw"; Flags: nowait postinstall skipifsilent
