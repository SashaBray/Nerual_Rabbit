"""Каталог данных и моделей: что автор выложил, что установлено, как это скачать.

Программа раздаётся пустой: база и модели подтягиваются из ОДНОЙ публичной папки автора.
В папке лежат готовые архивы и файл-манифест ``catalog.json`` — он и есть контракт: без него
клиент видел бы только имена файлов и не знал ни культуры, ни версии, ни размера, ни хэша.

Источник (``catalog_source`` в таблице ``app_state``, по умолчанию :data:`DEFAULT_SOURCE`) —
либо публичная ссылка Яндекс.Диска, либо ЛОКАЛЬНАЯ ПАПКА. Второй режим нужен не только для
разработки: так же раздаются обновления по локальной сети или с флешки в закрытом контуре.

Формат манифеста (``catalog.json`` в корне папки)::

    {"catalog_version": 1, "updated_at": "2026-09-12",
     "assets": [
       {"id": "model:winter_wheat_all_nn_v4", "kind": "model", "title": "…", "culture": "Озимая пшеница",
        "version": 4, "file": "models/winter_wheat_all_nn_v4.zip", "bytes": 76500000,
        "sha256": "…", "installs_to": "models/winter_wheat_all_nn_v4", "comment": "…",
        "metrics": {"rmse": 6.46, "r2": 0.77}},
       {"id": "db:full-2026-09", "kind": "db", "title": "База: ряды и урожайности", "version": "2026-09",
        "file": "db/db-full-2026-09.zip", "bytes": …, "sha256": "…",
        "db_schema_version": 2, "tables": ["parms", "series", "time_series", "yields", …]}]}

Установка атомарна: качаем во временный файл (с докачкой по ``Range``), сверяем sha256,
распаковываем во временный каталог и только потом подменяем целевой. Снимок БД НИКОГДА не
трогает пользовательские таблицы (:data:`db.USER_TABLES`) — иначе первая же «свежая база»
стёрла бы человеку его ручные урожайности, аккаунты и историю прогнозов.

Что установлено — в таблице БД ``installed_assets`` (своих JSON-файлов модуль не заводит).
"""
import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime
import pandas as pd

import config
import db

# Публичная папка автора (только просмотр и скачивание). Пользователь может указать другой
# источник во вкладке «Каталог» — он сохранится в БД и будет важнее этого значения.
DEFAULT_SOURCE = 'https://disk.360.yandex.ru/d/R_J6tWybdT0Kxw'

SOURCE_KEY = 'catalog_source'          # app_state: ссылка/папка каталога
CACHE_KEY = 'catalog_json'             # app_state: последний успешно прочитанный манифест
CACHE_AT_KEY = 'catalog_fetched_at'    # app_state: когда он прочитан
CACHE_SRC_KEY = 'catalog_cached_from'  # app_state: из какого источника кэш (чужой не подставляем)

CATALOG_FILE = 'catalog.json'
INSTALLED_TABLE = 'installed_assets'
YANDEX_API = 'https://cloud-api.yandex.net/v1/disk/public/resources'
CHUNK = 1 << 20                        # 1 МБ — размер куска при скачивании

KIND_TITLES = {'model': 'модель', 'db': 'база данных', 'reference': 'справочники',
               'dataset': 'датасет обучения'}

# Необязательные записи каталога: нужны только тем, кто переобучает модели. Они тяжёлые
# (гигабайты), поэтому НЕ показываются в списке по умолчанию, не попадают под «выделить всё»
# и не учитываются в отметке «есть обновления». Скачать их можно только осознанно — включив
# показ и выбрав запись руками.
OPTIONAL_KINDS = {'dataset'}


# --------------------------------------------------------------------------- #
# Источник каталога
# --------------------------------------------------------------------------- #
def source():
    """Текущий источник каталога: публичная ссылка Яндекс.Диска или путь к папке."""
    return os.environ.get('RR_CATALOG_SOURCE') or db.get_state(SOURCE_KEY, '') or DEFAULT_SOURCE


def set_source(text):
    """Запомнить источник каталога в БД (пусто — вернуться к зашитому по умолчанию)."""
    db.set_state(SOURCE_KEY, (text or '').strip())


def is_yandex(src=None):
    """Источник — публичная ссылка Яндекс.Диска (а не локальная папка)?"""
    s = (src if src is not None else source()).strip().lower()
    return s.startswith('http://') or s.startswith('https://')


def _tmp_dir():
    d = os.path.join(config.WORKSPACE, '.downloads')
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Доступ к файлам источника (Яндекс.Диск или локальная папка)
# --------------------------------------------------------------------------- #
# Коды, при которых имеет смысл повторить запрос: у открытого API Яндекса есть лимиты,
# и на раздаче, которую тянут несколько человек, 429 — обычное дело.
_RETRY_CODES = (429, 500, 502, 503, 504)


def _api_get(url, params, timeout=30, attempts=3):
    """GET к API Диска с повтором при лимите/временной ошибке сервера."""
    import requests

    last = None
    for i in range(attempts):
        last = requests.get(url, params=params, timeout=timeout)
        if last.status_code not in _RETRY_CODES:
            return last
        if i < attempts - 1:
            time.sleep(1.5 * (i + 1))                      # короткая пауза и ещё попытка
    return last


def _yandex_href(public_key, rel_path, timeout=30):
    """Временная прямая ссылка на файл внутри публичной папки (открытое API, без токена)."""
    r = _api_get(YANDEX_API + '/download',
                 {'public_key': public_key, 'path': '/' + rel_path.lstrip('/')}, timeout=timeout)
    if r.status_code == 404:
        if rel_path.strip('/') == CATALOG_FILE:
            raise FileNotFoundError(
                'В публичной папке нет файла catalog.json — выложите в неё содержимое '
                'папки release (catalog.json и подпапки models/ и db/).')
        raise FileNotFoundError(f'В публичной папке нет файла {rel_path}')
    r.raise_for_status()
    return r.json()['href']


def list_source(rel_path='/', src=None, timeout=30):
    """Что лежит в источнике: [(имя, тип, размер)]. Для диагностики во вкладке «Каталог»."""
    src = src or source()
    if not src:
        return []
    if not is_yandex(src):
        base = os.path.join(src, rel_path.strip('/'))
        if not os.path.isdir(base):
            return []
        out = []
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            out.append((name, 'dir' if os.path.isdir(full) else 'file',
                        0 if os.path.isdir(full) else os.path.getsize(full)))
        return out

    r = _api_get(YANDEX_API, {'public_key': src, 'path': rel_path, 'limit': 1000}, timeout=timeout)
    r.raise_for_status()
    items = r.json().get('_embedded', {}).get('items', [])
    return [(i.get('name'), i.get('type'), int(i.get('size') or 0)) for i in items]


_base_cache = {}          # источник -> подпапка, в которой лежит catalog.json ('' = корень)


def _has_file(rel_path, src, timeout=30):
    """Есть ли такой файл в источнике (без скачивания).

    «Нет» — это ТОЛЬКО ответ 404. Ошибки сети и лимиты API пробрасываем наверх: исключения
    ``requests`` наследуются от ``OSError``, и если ловить его, недоступный сервис выглядел бы
    как «в папке нет файла» — пользователь искал бы ошибку в своей раздаче.
    """
    if not is_yandex(src):
        return os.path.exists(os.path.join(src, rel_path.replace('/', os.sep)))
    try:
        _yandex_href(src, rel_path, timeout=timeout)
        return True
    except FileNotFoundError:
        return False


def base_path(src=None, timeout=30):
    """Подпапка источника, в которой лежит ``catalog.json`` (``''`` — корень).

    Выкладывая релиз, легко залить папку ``release`` целиком, а не её содержимое — тогда
    всё оказывается уровнем ниже. Вместо того чтобы заставлять перезаливать гигабайты,
    ищем манифест в корне, а если его там нет — в папках первого уровня. Несколько
    манифестов сразу — ошибка: непонятно, какой релиз раздаётся.
    """
    src = src or source()
    if src in _base_cache:
        return _base_cache[src]
    if _has_file(CATALOG_FILE, src, timeout):
        _base_cache[src] = ''
        return ''
    found = []
    for name, kind, _size in list_source('/', src=src, timeout=timeout):
        if kind == 'dir' and _has_file(f'{name}/{CATALOG_FILE}', src, timeout):
            found.append(name)
    if len(found) > 1:
        raise FileNotFoundError('В публичной папке несколько каталогов: '
                                + ', '.join(f'{n}/{CATALOG_FILE}' for n in found)
                                + '. Оставьте один — иначе непонятно, какой релиз раздаётся.')
    _base_cache[src] = found[0] if found else ''
    return _base_cache[src]


def _full(rel_path, src, timeout=30):
    """Путь файла внутри источника с учётом подпапки релиза."""
    base = base_path(src, timeout)
    return f'{base}/{rel_path}' if base else rel_path


def _read_text(rel_path, src=None, timeout=30):
    """Прочитать небольшой текстовый файл источника (манифест)."""
    src = src or source()
    if not src:
        raise ValueError('Источник каталога не задан.')
    rel_path = _full(rel_path, src, timeout)
    if not is_yandex(src):
        path = os.path.join(src, rel_path.replace('/', os.sep))
        with open(path, encoding='utf-8-sig') as f:
            return f.read()

    import requests
    href = _yandex_href(src, rel_path, timeout=timeout)
    r = requests.get(href, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.encoding or 'utf-8'
    return r.text


# --------------------------------------------------------------------------- #
# Манифест
# --------------------------------------------------------------------------- #
def fetch_catalog(src=None, timeout=30, use_cache=True):
    """Прочитать ``catalog.json`` из источника; при сбое сети — последний удачный из БД.

    Возвращает dict манифеста; ключ ``_offline`` = True, если отдан кэш.
    """
    src = src or source()
    try:
        data = json.loads(_read_text(CATALOG_FILE, src=src, timeout=timeout))
        db.set_state(CACHE_KEY, json.dumps(data, ensure_ascii=False))
        db.set_state(CACHE_AT_KEY, datetime.now().isoformat(timespec='seconds'))
        db.set_state(CACHE_SRC_KEY, src)
        return data
    except Exception:                                      # noqa: BLE001 — сеть/файл: отдаём кэш
        if not use_cache:
            raise
        cached = cached_catalog(src)                       # только кэш ЭТОГО же источника
        if cached is None:
            raise
        cached['_offline'] = True
        return cached


def cached_catalog(src=None):
    """Последний успешно прочитанный манифест из БД (или ``None``).

    ``src`` — если задан, кэш отдаётся только когда он снят с ЭТОГО источника: иначе после
    смены ссылки программа показывала бы чужой список как «источник недоступен».
    """
    if src is not None and (db.get_state(CACHE_SRC_KEY, '') or '') != src:
        return None
    raw = db.get_state(CACHE_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def fetched_at():
    """Когда каталог читался в последний раз (строка ISO или '')."""
    return db.get_state(CACHE_AT_KEY, '') or ''


def is_optional(asset):
    """Запись не нужна для прогноза (датасет обучения или явный флаг ``optional``)."""
    return bool(asset.get('optional')) or str(asset.get('kind')) in OPTIONAL_KINDS


def assets(catalog=None, include_optional=True):
    """Список записей каталога (как есть, с проверкой обязательных полей)."""
    cat = catalog if catalog is not None else fetch_catalog()
    out = []
    for a in cat.get('assets', []):
        if not a.get('id') or not a.get('file'):
            continue
        if not include_optional and is_optional(a):
            continue
        out.append(a)
    return out


# --------------------------------------------------------------------------- #
# Что установлено
# --------------------------------------------------------------------------- #
def installed():
    """``{asset_id: запись}`` из таблицы ``installed_assets``."""
    df = db.load(INSTALLED_TABLE)
    if df.empty:
        return {}
    return {str(r['asset_id']): r.to_dict() for _, r in df.iterrows()}


def _remember(asset, path):
    """Записать факт установки (одна строка на asset_id)."""
    df = db.load(INSTALLED_TABLE)
    if not df.empty:
        df = df[df['asset_id'].astype(str) != str(asset['id'])]
    rec = {'asset_id': str(asset['id']), 'kind': str(asset.get('kind') or ''),
           'title': str(asset.get('title') or ''), 'version': str(asset.get('version') or ''),
           'sha256': str(asset.get('sha256') or ''), 'bytes': int(asset.get('bytes') or 0),
           'path': str(path or ''), 'source': source(),
           'installed_at': datetime.now().isoformat(timespec='seconds')}
    db.save(INSTALLED_TABLE, pd.concat([df, pd.DataFrame([rec])], ignore_index=True))


def _forget(asset_id):
    df = db.load(INSTALLED_TABLE)
    if df.empty:
        return
    db.save(INSTALLED_TABLE, df[df['asset_id'].astype(str) != str(asset_id)].reset_index(drop=True))


def state_of(asset, inst=None):
    """Состояние записи каталога: ``'нет'`` | ``'установлено'`` | ``'обновление'``."""
    inst = installed() if inst is None else inst
    have = inst.get(str(asset['id']))
    if not have:
        return 'нет'
    same = str(have.get('sha256') or '') == str(asset.get('sha256') or '')
    return 'установлено' if same else 'обновление'


def status_frame(catalog=None, include_optional=False):
    """Таблица для интерфейса: что доступно, что установлено, что обновилось.

    Датасеты обучения по умолчанию НЕ показываются: обычному пользователю нужны база и
    модели, а выборки — гигабайты, которые незачем качать «заодно».
    """
    cat = catalog if catalog is not None else fetch_catalog()
    inst = installed()
    rows = []
    for a in assets(cat, include_optional=include_optional):
        rows.append({
            'состояние': state_of(a, inst),
            'что': KIND_TITLES.get(str(a.get('kind')), str(a.get('kind') or '')),
            'название': a.get('title') or a.get('id'),
            'культура': a.get('culture') or '',
            'версия': a.get('version') or '',
            'размер, МБ': round(int(a.get('bytes') or 0) / 1048576, 1),
            'комментарий': a.get('comment') or '',
            'id': a['id'],
        })
    order = {'обновление': 0, 'нет': 1, 'установлено': 2}
    rows.sort(key=lambda r: (order.get(r['состояние'], 3), str(r['что']), str(r['название'])))
    return pd.DataFrame(rows)


def updates_available(catalog=None):
    """Сколько записей каталога новее установленных (для тихой проверки при старте)."""
    try:
        cat = catalog if catalog is not None else fetch_catalog(timeout=10)
    except Exception:                                      # noqa: BLE001 — нет сети: молчим
        return 0
    inst = installed()
    return sum(1 for a in assets(cat, include_optional=False)      # датасетами не дёргаем
               if state_of(a, inst) == 'обновление')


# --------------------------------------------------------------------------- #
# Скачивание
# --------------------------------------------------------------------------- #
def _sha256(path, progress=None):
    h = hashlib.sha256()
    total = os.path.getsize(path) or 1
    done = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(int(100 * done / total))
    return h.hexdigest()


def _download(asset, dest, src=None, progress=None, message=None):
    """Скачать файл записи в ``dest`` с докачкой по ``Range`` и проверкой sha256."""
    src = src or source()
    say = message or (lambda _t: None)
    step = progress or (lambda _p: None)
    total = int(asset.get('bytes') or 0)
    rel = _full(asset['file'], src)

    if not is_yandex(src):                                 # локальная папка/сеть — просто копируем
        srcfile = os.path.join(src, rel.replace('/', os.sep))
        say(f'Копирую {os.path.basename(rel)}…')
        shutil.copyfile(srcfile, dest)
        step(100)
        return dest

    import requests
    part = dest + '.part'
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if total and have > total:                             # битый огрызок от прошлого раза
        os.remove(part); have = 0
    if have:
        say(f'Продолжаю загрузку с {have / 1048576:.0f} МБ…')

    href = _yandex_href(src, rel)                          # ссылка временная — берём перед каждой попыткой
    headers = {'Range': f'bytes={have}-'} if have else {}
    with requests.get(href, headers=headers, stream=True, timeout=60) as r:
        if have and r.status_code == 200:                  # сервер не принял докачку — начинаем заново
            have = 0
            if os.path.exists(part):
                os.remove(part)
        r.raise_for_status()
        if not total:
            total = int(r.headers.get('Content-Length') or 0) + have
        done = have
        with open(part, 'ab' if have else 'wb') as f:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    step(int(100 * done / total))
    os.replace(part, dest)
    return dest


def _verify(asset, path, message=None):
    """Сверить sha256 скачанного файла с манифестом (если хэш указан)."""
    want = str(asset.get('sha256') or '')
    if not want:
        return
    if message:
        message('Проверяю контрольную сумму…')
    got = _sha256(path)
    if got.lower() != want.lower():
        os.remove(path)
        raise ValueError('Контрольная сумма не сошлась — файл скачался повреждённым. '
                         'Попробуйте ещё раз.')


# --------------------------------------------------------------------------- #
# Установка
# --------------------------------------------------------------------------- #
def _unzip_to_temp(archive, message=None):
    """Распаковать архив во временный каталог; вернуть путь к нему."""
    if message:
        message('Распаковываю…')
    tmp = os.path.join(_tmp_dir(), 'unpack_' + datetime.now().strftime('%H%M%S_%f'))
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp)
    return tmp


def _install_model(asset, tmp, message=None):
    """Поставить папку модели в ``workspace/models/<имя>`` (подмена целиком, атомарно)."""
    name = os.path.basename(str(asset.get('installs_to') or '').rstrip('/')) or str(asset['id']).split(':')[-1]
    inner = os.path.join(tmp, name)
    if not os.path.isdir(inner):                           # архив без общей папки — берём его целиком
        entries = [e for e in os.listdir(tmp)]
        inner = os.path.join(tmp, entries[0]) if len(entries) == 1 and os.path.isdir(
            os.path.join(tmp, entries[0])) else tmp
    target = os.path.join(config.MODELS_DIR, name)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    old = target + '.old'
    if os.path.exists(old):
        shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(target):
        os.replace(target, old)
    try:
        shutil.move(inner, target)
    except Exception:                                      # noqa: BLE001 — вернуть прежнюю версию
        if os.path.exists(old):
            os.replace(old, target)
        raise
    shutil.rmtree(old, ignore_errors=True)
    note = str(asset.get('comment') or '').strip()         # заметка едет манифестом, а не архивом:
    note_path = os.path.join(target, 'comment.txt')        # так правка комментария не меняет sha256
    if note and not os.path.exists(note_path):             # архива и не требует перезаливки модели
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(note)
    if message:
        message(f'Модель установлена: {name}')
    return target


def _install_db(asset, tmp, message=None):
    """Положить таблицы снимка в БД, не трогая пользовательские таблицы.

    Пользовательские таблицы (``db.USER_TABLES``) не заменяются, даже если они перечислены
    в манифесте: это данные, которых нет ни у кого другого. Заменяемые файлы уходят в бэкап
    ``workspace/legacy/db_backup_<дата>/``.
    """
    want = int(asset.get('db_schema_version') or db.SCHEMA_VERSION)
    if want != db.SCHEMA_VERSION:
        raise ValueError(f'Снимок базы сделан для схемы v{want}, а программа работает со схемой '
                         f'v{db.SCHEMA_VERSION}. Обновите программу.')
    os.makedirs(db.DB_DIR, exist_ok=True)
    backup = os.path.join(config.LEGACY_DIR, 'db_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    moved, skipped = [], []
    files = [f for f in sorted(os.listdir(tmp)) if os.path.isfile(os.path.join(tmp, f))]
    for fname in files:
        table, ext = os.path.splitext(fname)
        if ext.lower() not in ('.csv', '.parquet'):
            continue
        if table in db.USER_TABLES:                        # защита на стороне клиента
            skipped.append(table)
            continue
        for old_ext in ('.csv', '.parquet'):               # у таблицы на диске ровно один формат
            old = os.path.join(db.DB_DIR, table + old_ext)
            if os.path.exists(old):
                os.makedirs(backup, exist_ok=True)
                shutil.move(old, os.path.join(backup, table + old_ext))
        shutil.move(os.path.join(tmp, fname), os.path.join(db.DB_DIR, fname))
        moved.append(table)
    db.reload()
    if message:
        message(f'Таблиц обновлено: {len(moved)} ({", ".join(moved[:6])}'
                + ('…' if len(moved) > 6 else '') + ')')
        if skipped:
            message(f'   ваши таблицы сохранены нетронутыми: {", ".join(skipped)}')
        if os.path.isdir(backup):                          # бэкап появляется, только если было что заменять
            message(f'   прежние файлы — в {backup}')
    return db.DB_DIR


def _install_dataset(asset, tmp, message=None):
    """Положить датасет обучения в ``workspace/datasets/<задание>``.

    Матрица приезжает как ``matrix.npy`` (float32) — так раздача в 9 раз легче; здесь она
    разворачивается обратно в ``matrix.csv``, который читают все обучающие скрипты. Проверено,
    что после разворота массивы совпадают побитово с исходными (скрипты и так читают матрицу
    как float32); строчный формат при этом короче — файл получается примерно вдвое меньше
    исходного. Разворот — единственная долгая часть установки: порядка минуты на 200 МБ.
    """
    import numpy as np
    import pandas as pd

    name = os.path.basename(str(asset.get('installs_to') or '').rstrip('/'))         or str(asset['id']).split(':')[-1]
    target = os.path.join(config.DATASETS_DIR, name)
    npy = os.path.join(tmp, 'matrix.npy')
    if os.path.exists(npy):
        if message:
            message('Разворачиваю матрицу в matrix.csv (это самая долгая часть)…')
        mat = np.load(npy)
        pd.DataFrame(mat).to_csv(os.path.join(tmp, 'matrix.csv'), sep=' ', header=False,
                                 index=False, float_format='%.9g')
        del mat
        os.remove(npy)

    os.makedirs(config.DATASETS_DIR, exist_ok=True)
    old = target + '.old'
    if os.path.exists(old):
        shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(target):
        os.replace(target, old)
    try:
        shutil.move(tmp, target)
    except Exception:                                      # noqa: BLE001 — вернуть прежний датасет
        if os.path.exists(old):
            os.replace(old, target)
        raise
    shutil.rmtree(old, ignore_errors=True)
    if message:
        size = sum(os.path.getsize(os.path.join(target, f)) for f in os.listdir(target))
        message(f'Датасет установлен: {name} ({size / 1048576:.0f} МБ на диске)')
    return target


def install(asset, src=None, progress=None, message=None, keep_archive=False):
    """Скачать и установить запись каталога. -> dict с путём и тем, что сделано."""
    say = message or (lambda _t: None)
    step = progress or (lambda _p: None)
    kind = str(asset.get('kind') or '')
    say(f'{asset.get("title") or asset["id"]} ({int(asset.get("bytes") or 0) / 1048576:.1f} МБ)')

    dest = os.path.join(_tmp_dir(), os.path.basename(asset['file']))
    _download(asset, dest, src=src, progress=lambda p: step(int(p * 0.8)), message=say)
    step(82)
    _verify(asset, dest, message=say)
    step(88)

    tmp = _unzip_to_temp(dest, message=say)
    try:
        if kind == 'model':
            path = _install_model(asset, tmp, message=say)
        elif kind in ('db', 'reference'):
            path = _install_db(asset, tmp, message=say)
        elif kind == 'dataset':
            path = _install_dataset(asset, tmp, message=say)   # временный каталог СТАНОВИТСЯ датасетом
        else:
            raise ValueError(f'Неизвестный тип записи каталога: {kind!r}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if not keep_archive and os.path.exists(dest):
            os.remove(dest)
    _remember(asset, path)
    step(100)
    say('Готово')
    return {'id': asset['id'], 'kind': kind, 'path': path}


def _rmtree_stubborn(path, attempts=4, pause=0.4):
    """Удалить папку, переживая короткие блокировки файлов (антивирус, индексатор Windows).

    Возвращает True, только если папки на диске действительно не осталось: один
    ``ignore_errors`` врал бы об успехе и оставлял мусор, который каталог потом считает
    установленным.
    """
    for _ in range(attempts):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.isdir(path):
            return True
        time.sleep(pause)
    return not os.path.isdir(path)


def remove(asset_id):
    """Удалить установленную модель или датасет (папку) и забыть запись. Снимки БД не удаляются.

    Возвращает False, если файлы удалить не удалось; запись при этом остаётся в
    ``installed_assets``, чтобы каталог не показывал лежащее на диске как отсутствующее.
    """
    inst = installed().get(str(asset_id))
    if not inst:
        return False
    roots = {'model': config.MODELS_DIR, 'dataset': config.DATASETS_DIR}
    root = roots.get(str(inst.get('kind')))
    if root:                                               # сносим только папку внутри своего корня
        path = str(inst.get('path') or '')
        if path and os.path.isdir(path) and os.path.dirname(os.path.abspath(path)) == \
                os.path.abspath(root):
            if not _rmtree_stubborn(path):
                return False
    _forget(asset_id)
    return True
