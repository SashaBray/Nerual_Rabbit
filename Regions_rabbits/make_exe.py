"""Сборка дистрибутива программы: справочники по умолчанию + PyInstaller.

    python make_exe.py            # собрать dist/NerualRabbit/ (папка с NerualRabbit.exe)
    python make_exe.py --zip      # и сложить её в dist/NerualRabbit-<дата>.zip для раздачи
    python make_exe.py --installer   # и собрать мастер установки dist/NerualRabbit-Setup-<версия>.exe

1. В ``build/workspace_defaults/`` раскладываются справочники, без которых программа не
   стартует: культуры, задания, списки параметров, словари ЕМИСС. Берутся строго по БЕЛОМУ
   списку — ukey, аватары и пользовательские данные в дистрибутив не попадают, а новые
   секретные файлы, если появятся, не утекут «заодно». Дополнительно проверяется, что
   значение ukey автора не встречается ни в одном файле сборки.
2. PyInstaller собирает приложение по ``nerual_rabbit.spec``; справочники едут внутри и при
   первом запуске раскладываются в ``workspace`` рядом с exe (``config.ensure_workspace``) —
   там же живут база, модели и прогнозы.
3. ``--installer`` — мастер установки на Inno Setup (``installer/nerual_rabbit.iss``): выбор папки
   (с проверкой права записи), ярлыки, удаление через «Установка и удаление программ».

База, модели и датасеты в дистрибутив не входят: программа берёт их из каталога
(вкладка «7. Каталог», источник по умолчанию — ``assets.DEFAULT_SOURCE``).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(ROOT, 'workspace')
STAGE = os.path.join(ROOT, 'build', 'workspace_defaults')

# Белый список: (подкаталог workspace, файлы). Только справочники, без секретов.
ALLOWLIST = {
    'config': ['bdpmo_cultures.csv', 'bdpmo_triggers.json', 'build_defaults.json', 'cultures.csv',
               'harvest_dates.csv', 'ingest.json', 'parameter_lists.csv', 'parameters.csv'],
    'tasks': ['dataset_tasks.csv'],
    'source': ['Categories_of_farms_dictionary.json', 'Culture_Dictionary.json',
               'Region_Dictionary.json', 'Regions_and_dist_tree.json', 'Regions_uid_dict.json',
               'Cultures_masks.xlsx'],
}
SECRET_FILES = [os.path.join(WORKSPACE, 'config', 'ukey.txt'),
                os.path.join(WORKSPACE, 'config', 'ukeys.csv')]


def stage_defaults():
    """Собрать справочники по умолчанию в build/workspace_defaults. -> число файлов."""
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    n = 0
    for sub, names in ALLOWLIST.items():
        os.makedirs(os.path.join(STAGE, sub), exist_ok=True)
        for name in names:
            src = os.path.join(WORKSPACE, sub, name)
            if not os.path.exists(src):
                raise SystemExit(f'Нет обязательного справочника: {src}')
            shutil.copyfile(src, os.path.join(STAGE, sub, name))
            n += 1
    return n


def _secret_values():
    """Значения ukey автора (чтобы убедиться, что их нет в сборке)."""
    values = set()
    for path in SECRET_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8-sig', errors='ignore') as f:
            for token in f.read().replace(',', ' ').replace(';', ' ').split():
                if len(token) >= 16:                       # ключи длинные; заголовки/id отсекаем
                    values.add(token.strip())
    return values


def check_no_secrets(folder):
    """Упасть, если хоть один файл в ``folder`` содержит значение ukey автора."""
    secrets = _secret_values()
    if not secrets:
        return 0
    checked = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            try:
                with open(path, 'rb') as f:
                    data = f.read()
            except OSError:
                continue
            checked += 1
            for s in secrets:
                if s.encode('utf-8') in data:
                    raise SystemExit(f'СТОП: в файл сборки попал ukey — {path}')
    return checked


def smoke_test(dist, timeout=240):
    """Запустить собранный exe в проверочном режиме (см. smoke_hook.py). -> (ok, подробности).

    Сборка PyInstaller может завершиться успешно, а exe — падать при импорте. Проверочный режим
    импортирует всё приложение без интерфейса и сообщает результат кодом выхода и файлом.
    """
    exe = os.path.join(dist, 'NerualRabbit.exe')
    result = os.path.join(tempfile.gettempdir(), 'nr_smoke_result.txt')
    if os.path.exists(result):
        os.remove(result)
    env = dict(os.environ, NR_SMOKE_TEST=result, QT_QPA_PLATFORM='offscreen')
    try:
        proc = subprocess.run([exe], env=env, cwd=dist, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f'exe не завершился за {timeout} с (завис при запуске)'
    details = ''
    if os.path.exists(result):
        with open(result, encoding='utf-8') as f:
            details = f.read().strip()
    return proc.returncode == 0 and details == 'ok', details or f'код выхода {proc.returncode}'


INNO_CANDIDATES = [
    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Inno Setup 6', 'ISCC.exe'),
    r'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
    r'C:\Program Files\Inno Setup 6\ISCC.exe',
]


def build_installer(dist):
    """Собрать мастер установки (Inno Setup) из готовой папки dist. -> путь к setup.exe."""
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    import config                                           # версия программы — из одного места
    iscc = next((p for p in INNO_CANDIDATES if os.path.exists(p)), None)
    if iscc is None:
        raise SystemExit('Не найден Inno Setup 6 (ISCC.exe). Установите: '
                         'winget install JRSoftware.InnoSetup')
    script = os.path.join(ROOT, 'installer', 'nerual_rabbit.iss')
    # Windows не создаёт файлы с путём длиннее 259 символов: папка установки + самый длинный
    # путь внутри программы должны уложиться, иначе установка падает с «не удаётся найти путь»
    longest = max(len(os.path.relpath(os.path.join(r, f), dist)) + 1
                  for r, _, fs in os.walk(dist) for f in fs)
    max_dir = 259 - longest - 5                              # запас на разделитель и unins000.dat
    print(f'Самый длинный путь внутри программы: {longest} символов -> папка установки до {max_dir}')
    cmd = [iscc, f'/DAppVersion={config.APP_VERSION}', f'/DSourceDir={dist}',
           f'/DMaxDirLen={max_dir}', '/Q', script]
    print('Сборка установщика:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=os.path.dirname(script))
    path = os.path.join(ROOT, 'dist', f'NerualRabbit-Setup-{config.APP_VERSION}.exe')
    print(f'Установщик: {path} ({os.path.getsize(path) / 1048576:.0f} МБ)')
    return path


def main():
    ap = argparse.ArgumentParser(description='Сборка дистрибутива Nerual Rabbit (PyInstaller)')
    ap.add_argument('--zip', action='store_true', help='сложить готовую папку в zip для раздачи')
    ap.add_argument('--skip-build', action='store_true', help='только подготовить справочники')
    ap.add_argument('--installer', action='store_true',
                    help='собрать мастер установки NerualRabbit-Setup-<версия>.exe (нужен Inno Setup 6)')
    ap.add_argument('--installer-only', action='store_true',
                    help='собрать только установщик из уже готовой dist/NerualRabbit (без PyInstaller)')
    args = ap.parse_args()
    if args.installer_only:
        build_installer(os.path.join(ROOT, 'dist', 'NerualRabbit'))
        return

    n = stage_defaults()
    print(f'Справочники по умолчанию: {n} файлов -> {STAGE}')
    print(f'Проверка на секреты: просмотрено {check_no_secrets(STAGE)} файлов, ukey не найден')
    if args.skip_build:
        return

    cmd = [sys.executable, '-m', 'PyInstaller', os.path.join(ROOT, 'nerual_rabbit.spec'),
           '--noconfirm', '--distpath', os.path.join(ROOT, 'dist'),
           '--workpath', os.path.join(ROOT, 'build', 'pyinstaller')]
    print('Запуск:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    dist = os.path.join(ROOT, 'dist', 'NerualRabbit')
    print(f'Проверка на секреты в дистрибутиве: просмотрено {check_no_secrets(dist)} файлов, '
          'ukey не найден')
    print('Проверочный запуск exe (импорт всего приложения, без интерфейса)…')
    ok, details = smoke_test(dist)
    if not ok:
        raise SystemExit('СТОП: собранный exe не запускается —\n' + details)
    print('Проверочный запуск: exe стартует, torch/sklearn/pyarrow/PySide6 импортируются')
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dist) for f in fs)
    print(f'Готово: {dist} ({size / 1048576:.0f} МБ)')
    if args.zip:
        base = os.path.join(ROOT, 'dist', 'NerualRabbit-' + datetime.now().strftime('%Y-%m-%d'))
        path = shutil.make_archive(base, 'zip', root_dir=os.path.join(ROOT, 'dist'),
                                   base_dir='NerualRabbit')
        print(f'Архив для раздачи: {path} ({os.path.getsize(path) / 1048576:.0f} МБ)')
    if args.installer:
        build_installer(dist)


if __name__ == '__main__':
    main()
