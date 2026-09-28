"""Централизованные пути и настройки подпроекта Regions_rabbits.

Весь код лежит в ``code/``, все данные — в ``workspace/`` (см. план рефакторинга).
Единственный источник ukey — файл ``workspace/config/ukey.txt`` (в коде ключей нет).
"""

import os
import shutil
import sys


# Собранное приложение (PyInstaller) или запуск из исходников?
FROZEN = getattr(sys, 'frozen', False)

# Корень: в сборке — папка рядом с exe (там же появится workspace), иначе родитель code/.
ROOT = (os.path.dirname(os.path.abspath(sys.executable)) if FROZEN
        else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Неизменяемые ресурсы, приехавшие вместе с программой (логотипы, значки): в сборке
# PyInstaller распаковывает их во временный каталог _MEIPASS, в исходниках — это ROOT.
RESOURCES = getattr(sys, '_MEIPASS', ROOT)

# Справочники, без которых программа не стартует (культуры, задания, параметры, словари ЕМИСС),
# едут внутри сборки и раскладываются в workspace при первом запуске. Собирает их
# make_exe.py строго по белому списку — секреты (ukey) в дистрибутив не попадают.
DEFAULTS_DIR = os.path.join(RESOURCES, 'workspace_defaults')

# Версия программы: попадает в установщик и в список «Установка и удаление программ».
APP_VERSION = '1.0.0'

# Данные (база, модели, прогнозы) лежат в папке workspace РЯДОМ С ПРОГРАММОЙ: всё компактно
# в одном месте, удалили папку программы — удалилась и база. Поэтому ставить программу нужно
# туда, где есть права на запись (не в «Program Files») — это проверяет мастер установки.
# Переменная окружения RR_WORKSPACE — только для разработки и тестов.
WORKSPACE = os.environ.get('RR_WORKSPACE') or os.path.join(ROOT, 'workspace')

CONFIG_DIR = os.path.join(WORKSPACE, 'config')
DB_DIR = os.path.join(WORKSPACE, 'database')
TASKS_DIR = os.path.join(WORKSPACE, 'tasks')
SOURCE_DIR = os.path.join(WORKSPACE, 'source')
DATASETS_DIR = os.path.join(WORKSPACE, 'datasets')
LEGACY_DIR = os.path.join(WORKSPACE, 'legacy')
REPORTS_DIR = os.path.join(WORKSPACE, 'reports')     # датированные отчёты (сопоставление районов и т.п.)
MODELS_DIR = os.path.join(WORKSPACE, 'models')       # общий каталог моделей (по папке на модель)
EXPORTS_DIR = os.path.join(WORKSPACE, 'exports')     # выгрузки прогнозов (папка-артефакт + ZIP)

# Конфигурационные файлы
UKEY_FILE = os.path.join(CONFIG_DIR, 'ukey.txt')                  # запасной одиночный ukey
BDPMO_CULTURES_FILE = os.path.join(CONFIG_DIR, 'bdpmo_cultures.csv')
BDPMO_TRIGGERS_FILE = os.path.join(CONFIG_DIR, 'bdpmo_triggers.json')
INGEST_FILE = os.path.join(CONFIG_DIR, 'ingest.json')

# Справочники задания (всё по id)
CULTURES_FILE = os.path.join(CONFIG_DIR, 'cultures.csv')          # culture_id -> title/bdpmo/mask
UKEYS_FILE = os.path.join(CONFIG_DIR, 'ukeys.csv')               # ukey_id -> ukey (СЕКРЕТ, в .gitignore)
PARAMETERS_FILE = os.path.join(CONFIG_DIR, 'parameters.csv')      # parameter_id -> name/is_ndvi
PARAMETER_LISTS_FILE = os.path.join(CONFIG_DIR, 'parameter_lists.csv')  # param_list_id -> [parameter_id, order, historical]
BUILD_DEFAULTS_FILE = os.path.join(CONFIG_DIR, 'build_defaults.json')   # редкие настройки сборки

# Таблица заданий
DATASET_TASKS_FILE = os.path.join(TASKS_DIR, 'dataset_tasks.csv')

# Исходные справочники
REGIONS_TREE_FILE = os.path.join(SOURCE_DIR, 'Regions_and_dist_tree.json')
REGIONS_UID_FILE = os.path.join(SOURCE_DIR, 'Regions_uid_dict.json')
PMODB_DIR = os.path.join(SOURCE_DIR, 'PMODB_init')
CULTURES_MASKS_XLSX = os.path.join(SOURCE_DIR, 'Cultures_masks.xlsx')


def read_ukey():
    """Прочитать ukey из workspace/config/ukey.txt. Пусто/нет файла -> ''."""
    if not os.path.exists(UKEY_FILE):
        return ''
    with open(UKEY_FILE, 'r', encoding='utf-8-sig') as f:
        return f.read().strip()


def dataset_dir(task_id):
    """Каталог вывода для задания."""
    return os.path.join(DATASETS_DIR, str(task_id))


def ensure_workspace():
    """Первый запуск: создать каталоги workspace и разложить справочники по умолчанию.

    Существующие файлы НЕ перезаписываются — у автора и у пользователя, который уже что-то
    правил, всё остаётся как есть. Возвращает список скопированных файлов.
    """
    ensure_dirs()
    copied = []
    if not os.path.isdir(DEFAULTS_DIR):
        return copied
    for root, _dirs, files in os.walk(DEFAULTS_DIR):
        rel = os.path.relpath(root, DEFAULTS_DIR)
        target_dir = WORKSPACE if rel == '.' else os.path.join(WORKSPACE, rel)
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            dst = os.path.join(target_dir, name)
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(root, name), dst)
                copied.append(os.path.relpath(dst, WORKSPACE))
    return copied


def ensure_dirs():
    """Создать все рабочие каталоги, если их ещё нет."""
    for d in (CONFIG_DIR, DB_DIR, TASKS_DIR, SOURCE_DIR, DATASETS_DIR, LEGACY_DIR, REPORTS_DIR,
              MODELS_DIR, EXPORTS_DIR):
        os.makedirs(d, exist_ok=True)
