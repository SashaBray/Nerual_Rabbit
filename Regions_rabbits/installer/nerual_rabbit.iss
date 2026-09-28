; Мастер установки Nerual Rabbit (Inno Setup 6).
;
; Собирается командой  python make_exe.py --installer  (она же передаёт версию и путь к сборке).
; Вручную:  ISCC.exe /DAppVersion=1.0.0 installer\nerual_rabbit.iss
;
; Всё хранится в ОДНОЙ папке: программа и рядом с ней workspace — база, модели, прогнозы.
; Удалили папку программы — удалились и данные. Поэтому папка установки должна быть доступна
; для записи обычному пользователю: установка идёт без прав администратора (по умолчанию в
; AppData\Local\Programs), а «Program Files» и папки без права записи мастер не принимает.
;
; Тихая установка (для проверки):
;   NerualRabbit-Setup-1.0.0.exe /VERYSILENT /DIR="D:\Nerual Rabbit"

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\NerualRabbit"
#endif
#ifndef MaxDirLen
  ; предел длины папки установки: 259 символов Windows минус самый длинный путь внутри программы
  ; (make_exe.py считает его по готовой сборке и передаёт /DMaxDirLen)
  #define MaxDirLen "150"
#endif
#define AppName "Nerual Rabbit"
#define AppExe "NerualRabbit.exe"

[Setup]
; AppId не менять между версиями — по нему установщик узнаёт прежнюю установку и обновляет её
AppId={{8E6A2B4C-3F1D-4C7E-9A52-6D0B1E7F4A93}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Nerual Rabbit
VersionInfoVersion={#AppVersion}
VersionInfoDescription=Прогноз урожайности Nerual Rabbit — установка
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Только для текущего пользователя и без прав администратора: программа пишет базу в свою папку.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
SetupIconFile=..\logos\NR_app_v7.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
OutputDir=..\dist
OutputBaseFilename=NerualRabbit-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Messages]
SelectDirLabel3=Программа и все её данные — база, скачанные модели и прогнозы — будут храниться в этой папке (со временем это несколько гигабайт). Выберите диск с достаточным местом.
SelectDirBrowseLabel=Нажмите «Далее», чтобы продолжить, или «Обзор», чтобы выбрать другую папку. Папка должна быть доступна для записи, поэтому «Program Files» не подходит.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function IsUnderProgramFiles(Path: String): Boolean;
var
  P: String;
begin
  P := AddBackslash(Lowercase(Path));
  Result := (Pos(AddBackslash(Lowercase(ExpandConstant('{commonpf64}'))), P) = 1) or
            (Pos(AddBackslash(Lowercase(ExpandConstant('{commonpf32}'))), P) = 1);
end;

function DirIsWritable(Dir: String): Boolean;
var
  Probe: String;
  Existed: Boolean;
begin
  Result := False;
  Existed := DirExists(Dir);
  if not ForceDirectories(Dir) then
    Exit;
  Probe := AddBackslash(Dir) + '.nr_write_test';
  if SaveStringToFile(Probe, 'ok', False) then
  begin
    DeleteFile(Probe);
    Result := True;
  end;
  if not Existed then
    RemoveDir(Dir);                                    { проверка не должна оставлять пустых папок }
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir: String;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    Dir := WizardDirValue();
    if Length(Dir) > {#MaxDirLen} then
    begin
      SuppressibleMsgBox('Слишком длинный путь к папке установки (' + IntToStr(Length(Dir)) + ' символов, ' +
             'допустимо не больше {#MaxDirLen}).' + #13#10#13#10 +
             'Внутри программы есть глубоко вложенные файлы, и Windows не сможет их создать. ' +
             'Выберите папку покороче, например D:\Nerual Rabbit.', mbError, MB_OK, IDOK);
      Result := False;
      Exit;
    end;
    if IsUnderProgramFiles(Dir) then
    begin
      SuppressibleMsgBox('Программу нельзя установить в «Program Files»: база данных и прогнозы хранятся ' +
             'в папке программы, а писать туда без прав администратора нельзя.' + #13#10#13#10 +
             'Оставьте папку по умолчанию или выберите другую, например D:\Nerual Rabbit.',
             mbError, MB_OK, IDOK);
      Result := False;
      Exit;
    end;
    if not DirIsWritable(Dir) then
    begin
      SuppressibleMsgBox('Не удалось создать папку или записать в неё:' + #13#10 + Dir + #13#10#13#10 +
             'Выберите другую папку.', mbError, MB_OK, IDOK);
      Result := False;
    end;
  end;
end;

{ ---- удаление: данные лежат в папке программы; спрашиваем, удалять ли их ---- }

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir, DataDir: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  AppDir := ExpandConstant('{app}');
  DataDir := AddBackslash(AppDir) + 'workspace';
  if DirExists(DataDir) and not UninstallSilent() then
    if MsgBox('Программа удалена.' + #13#10#13#10 +
              'Удалить также ваши данные — базу, скачанные модели и прогнозы?' + #13#10 +
              DataDir + #13#10#13#10 +
              'Их нельзя будет восстановить. Если оставить, при повторной установке в эту же ' +
              'папку программа продолжит работать с ними.',
              mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(DataDir, True, True, True);
  RemoveDir(AppDir);                                   { папка программы — только если осталась пустой }
end;
