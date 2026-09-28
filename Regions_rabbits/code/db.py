"""Файловая реляционная БД проекта Nerual Rabbit (без SQL-движка).

Все данные хранятся набором CSV-таблиц в каталоге ``Database/``:

* ``territories``      — справочник территорий (регион/район)
* ``yields``           — урожайности культур по годам (длинная таблица, из источников)
* ``user_yields``      — урожайности, введённые пользователем вручную (переживают
  переподгрузку источников и работают, когда источники недоступны)
* ``time_series``      — спутниковые временные ряды (длинная таблица, день года -> значение)
* ``time_series_meta`` — календарь рядов (first_date / last_date из labels)
* ``models``           — модели и их тип/культура/границы дней/файл весов
* ``model_params``     — упорядоченный список параметров модели + нормировка
* ``predictions``      — прогнозы с отметкой времени (made_at)
* ``cultures_masks``   — маски NDVI по культурам (дамп Cultures_masks.xlsx)
* ``regions``          — справочник region_name -> uid

Модуль — тонкий слой доступа на pandas, без ORM. Таблицы лениво читаются и
кэшируются в памяти; запись дописывает строки и сбрасывает CSV на диск — через временный
файл и ``os.replace`` под общим замком, потому что в приложении в БД пишут и фоновые потоки.

Единая точка кодирования ключа территории — :func:`territory_id`. Район может
отсутствовать (региональный уровень) — тогда в ключе используется токен ``nan``,
что совпадает с исторической конвенцией имён файлов временных рядов
(``<parm>_<id_region>_<id_district>_<year>.json``).
"""

import os
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config


DB_DIR = config.DB_DIR     # workspace/database (см. code/config.py)

# Имя таблицы -> столбцы, по которым она хранится/читается как строковые ключи,
# чтобы '34' и 34.0 не расходились между миграцией и рантаймом.
_STR_KEY_COLUMNS = {
    'territories': ['territory_id', 'id_country'],
    'yields': ['territory_id', 'culture'],
    'user_yields': ['territory_id', 'culture'],  # ручной ввод оператора (приоритет над источником)
    'time_series': [],                          # series_id, day, value — только числа
    'series': ['territory_id'],                  # series_id, territory_id, parm_id, year, dates
    'parms': ['parm'],                           # parm_id, parm
    'models': ['model_name', 'model_type', 'culture', 'weights_file'],
    'model_params': ['model_name', 'parm'],
    'predictions': ['territory_id', 'culture', 'model_name', 'batch_id'],
    'territory_metrics': ['territory_id', 'culture', 'batch_id'],  # уточнённые MSE/R² партии
    'model_metrics': ['culture', 'model_name'],   # точность моделей по культурам (MSE/RMSE/R²)
    'regions': ['region', 'region_uid'],
    'territory_uids': ['territory_id'],          # territory_id -> текущий vega_uid района
    'accounts': ['name', 'email', 'ukey'],       # учётные записи приложения
    'user_lists': ['territory_id', 'kind', 'label'],  # сохранённые списки районов/регионов пользователя
    'templates': ['name', 'items'],              # шаблоны территорий (items — JSON-строка), только в БД
    'app_state': ['key', 'value'],               # состояние приложения (сессия входа и т.п.)
    'installed_assets': ['asset_id', 'kind', 'title', 'version', 'sha256', 'source'],  # что скачано из каталога
    'forecast_outcomes': ['batch_id', 'territory_id', 'label', 'culture', 'model', 'planned',
                          'status', 'reason'],          # план и итог партии: что посчитано и почему нет
}

# Формат CSV таблиц: разделитель «;» и запятая в дробных числах — в таком виде Excel с
# русскими настройками сразу раскладывает файл по столбцам, а числа остаются числами
# (при разделителе «,» весь ряд попадает в одну ячейку). Так же пишутся выгрузки прогнозов
# (app_export). Файлы прежнего формата читаются по-прежнему: разделитель определяется по
# первой строке файла, см. ``_csv_dialect``.
CSV_SEP = ';'
CSV_DECIMAL = ','

# Оверлей скачанных рядов — не таблица для чтения глазами, а журнал дозаписи: его пишет и
# читает по смещениям ``overlay_index`` в своём формате (запятая), трогать нельзя.
_RAW_CSV_TABLES = {'time_series_downloaded', 'time_series_downloaded.index'}

_cache = {}
_index_cache = {}     # ленивые индексы (parms / series / points) для быстрых чтений
_array_cache = {}     # мемоизация приведённых рядов: (tid, parm, year, size, padding) -> np.array
_mean_cache = {}      # мемоизация межгодовых средних: (tid, parm, year, n, size, padding) -> np.array
_overlay = {}         # индекс оверлея скачанных рядов (НЕ сбрасывается при save базы)


def _clear_derived():
    """Сбросить производные кэши базы (индексы и мемоизацию). Оверлей не трогаем."""
    _index_cache.clear()
    _array_cache.clear()
    _mean_cache.clear()


# --------------------------------------------------------------------------- #
# Базовые операции с таблицами
#
# Тяжёлые таблицы (``_PARQUET_TABLES``) хранятся в Parquet, если установлен движок
# (pyarrow/fastparquet) — колоночный формат со словарным/RLE/delta-кодированием и
# сжатием сильно уменьшает размер и ускоряет чтение. Если движка нет — прозрачный
# фолбэк на CSV. Маленькие справочные таблицы всегда CSV (удобно смотреть глазами).
# На диске для таблицы всегда ровно один формат: при записи устаревший удаляется.
# --------------------------------------------------------------------------- #
_PARQUET_TABLES = {'time_series'}

# Запись таблиц сериализуется и идёт через временный файл: фоновые потоки приложения
# (чтение каталога, прогноз, выгрузка) пишут в БД одновременно с главным, а два
# параллельных `to_csv` в один файл рвут его так, что он перестаёт читаться.
# Замок реентрантный: под ним же выполняется целый цикл «прочитать -> изменить -> записать»
# (см. set_state), внутри которого save берёт его повторно.
_write_lock = threading.RLock()


def _replace_retry(src, dst, attempts=20, pause=0.05):
    """``os.replace`` с повторами на Windows.

    Замена файла запрещена, пока его хоть на мгновение держит другой процесс — антивирус или
    индексатор, которые открывают только что записанную таблицу. Без повторов любая запись в БД
    вскоре после предыдущей могла упасть с ``PermissionError``. Ждём нарастающими паузами
    (в сумме до ~10 с) и только потом сдаёмся.
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(pause * (i + 1))


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _csv_path(table_name):
    return os.path.join(DB_DIR, table_name + '.csv')


def _parquet_path(table_name):
    return os.path.join(DB_DIR, table_name + '.parquet')


def path_of(table_name):
    """Путь к файлу таблицы на диске (Parquet приоритетнее CSV, если есть)."""
    pq = _parquet_path(table_name)
    return pq if os.path.exists(pq) else _csv_path(table_name)


def _csv_dialect(path, table_name=None):
    """(разделитель, десятичный знак) файла: нынешний «;» или прежний «,».

    Формат определяем по первой строке: у новых файлов заголовок разделён «;», у прежних —
    запятыми. Так одна и та же программа читает и свои новые таблицы, и снимки базы,
    выложенные раньше.
    """
    if table_name in _RAW_CSV_TABLES:
        return ',', '.'
    try:
        with open(path, encoding='utf-8-sig') as f:
            head = f.readline()
    except OSError:
        return CSV_SEP, CSV_DECIMAL
    return (CSV_SEP, CSV_DECIMAL) if head.count(';') > head.count(',') else (',', '.')


def _csv_ready(df, table_name):
    """Вернуть числа к числовому типу перед записью CSV.

    Столбец, собранный из разных источников, нередко оказывается строковым (``object``).
    Тогда ``to_csv`` пишет его как есть — с точкой вместо запятой, и Excel показывает такую
    ячейку текстом, а «23.5» превращает в дату. Переводим только те столбцы, где ЧИСЛАМИ
    оказались все непустые значения, и никогда — ключевые строковые (``_STR_KEY_COLUMNS``):
    там «64» и «3_7» обязаны остаться строками.
    """
    keys = set(_STR_KEY_COLUMNS.get(table_name, []))
    out = df
    for col in df.columns:
        if col in keys or df[col].dtype != object:
            continue
        conv = pd.to_numeric(df[col], errors='coerce')
        if conv.notna().any() and int(conv.isna().sum()) == int(df[col].isna().sum()):
            if out is df:
                out = df.copy()
            out[col] = conv
    return out


def load(table_name):
    """Прочитать таблицу (с кэшированием). Возвращает копию-ссылку из кэша."""
    if table_name not in _cache:
        pq, csv = _parquet_path(table_name), _csv_path(table_name)
        if os.path.exists(pq):
            _cache[table_name] = pd.read_parquet(pq)        # типы сохранены форматом
        elif os.path.exists(csv):
            str_cols = _STR_KEY_COLUMNS.get(table_name, [])
            sep, decimal = _csv_dialect(csv, table_name)
            _cache[table_name] = pd.read_csv(
                csv, sep=sep, decimal=decimal,
                dtype={c: str for c in str_cols}, encoding='utf-8-sig')
        else:
            _cache[table_name] = pd.DataFrame()
    return _cache[table_name]


def save(table_name, df):
    """Записать таблицу на диск и обновить кэш (Parquet для тяжёлых, иначе CSV).

    Поддерживает ровно один формат на таблицу: при удачной записи в один формат
    устаревший файл другого формата удаляется (чтобы load не прочитал старьё).
    """
    os.makedirs(DB_DIR, exist_ok=True)
    pq, csv = _parquet_path(table_name), _csv_path(table_name)

    with _write_lock:                                       # см. _write_lock: пишем по одному
        if table_name in _PARQUET_TABLES and _has_parquet():
            tmp = pq + '.tmp'
            df.to_parquet(tmp, index=False)
            _replace_retry(tmp, pq)                         # атомарная подмена: битых файлов нет
            if os.path.exists(csv):
                os.remove(csv)
        else:
            tmp = csv + '.tmp'
            raw = table_name in _RAW_CSV_TABLES
            sep, decimal = (',', '.') if raw else (CSV_SEP, CSV_DECIMAL)  # см. CSV_SEP: файл для Excel
            (df if raw else _csv_ready(df, table_name)).to_csv(
                tmp, index=False, sep=sep, decimal=decimal,
                encoding='utf-8-sig')                       # BOM -> Excel читает кириллицу верно
            _replace_retry(tmp, csv)
            if os.path.exists(pq):
                os.remove(pq)

        _cache[table_name] = df
        _clear_derived()      # индексы и мемоизация устарели


# Версия схемы БД: 1 — денормализованная time_series (territory_id, parm, year, day, value),
# 2 — нормализованная (parms + series + time_series). Каталог обновлений сверяет её со снимком,
# чтобы старая программа не поставила себе базу нового формата (см. assets.py).
SCHEMA_VERSION = 2

# Крупные таблицы в прежнем формате не переписываем: время старта важнее, а глазами в Excel
# их всё равно не читают (ряды наблюдений — это сотни тысяч строк).
_CONVERT_LIMIT_BYTES = 64 << 20


def convert_csv_dialect(limit_bytes=_CONVERT_LIMIT_BYTES):
    """Перевести таблицы прежнего формата (разделитель «,») в нынешний «;». -> список таблиц.

    Нужна при обновлении программы: база, собранная прежней версией или поставленная снимком
    из каталога, открывалась в Excel одной колонкой. Файлы уже нынешнего формата и журнал
    оверлея не трогаем, так что повторные запуски ничего не делают.
    """
    done = []
    if not os.path.isdir(DB_DIR):
        return done
    for name in sorted(os.listdir(DB_DIR)):
        if not name.endswith('.csv'):
            continue
        table = name[:-4]
        path = os.path.join(DB_DIR, name)
        if table in _RAW_CSV_TABLES or os.path.getsize(path) > limit_bytes:
            continue
        if _csv_dialect(path, table)[0] == CSV_SEP:         # уже в нынешнем формате
            continue
        try:
            save(table, load(table))
            done.append(table)
        except Exception:                                  # noqa: BLE001 — обновление формата не должно мешать запуску
            _cache.pop(table, None)
    return done


def ensure_user_tables():
    """Создать пустые файлы пользовательских таблиц, которых ещё нет. -> список созданных.

    Прежде ``user_yields.csv`` появлялся только после первого сохранения, и на свежей
    установке таблицы ручного ввода в папке базы просто не было — непонятно, куда смотреть
    и что править. Теперь файл заводится сразу, с заголовком и без строк.
    """
    created = []
    os.makedirs(DB_DIR, exist_ok=True)
    for table, columns in (('user_yields', USER_YIELDS_COLUMNS),):
        if os.path.exists(_csv_path(table)) or os.path.exists(_parquet_path(table)):
            continue
        save(table, pd.DataFrame(columns=columns))
        created.append(table)
    return created

# Таблицы пользователя: их НЕЛЬЗЯ перезаписывать снимком БД из каталога — это данные,
# которых нет ни у кого, кроме владельца программы (ручные урожайности, аккаунты, прогнозы).
USER_TABLES = {'user_yields', 'accounts', 'predictions', 'territory_metrics',
               'templates', 'user_lists', 'app_state', 'installed_assets', 'forecast_outcomes'}


def get_state(key, default=None):
    """Значение из таблицы ``app_state`` (состояние приложения) либо ``default``."""
    df = load('app_state')
    if df.empty:
        return default
    sub = df[df['key'].astype(str) == str(key)]
    return str(sub.iloc[-1]['value']) if not sub.empty else default


def set_state(key, value):
    """Записать значение в ``app_state`` (одна строка на ключ).

    Перед правкой таблица перечитывается с диска, и весь цикл «прочитать -> изменить ->
    записать» идёт под замком записи: ``app_state`` пишут и фоновые потоки (тихая проверка
    каталога при старте), и без этого настройка, сохранённая в те же секунды, пропадала.
    """
    with _write_lock:                                  # чтение и запись — одним куском: иначе фоновый
        _cache.pop('app_state', None)                  # поток, прочитавший таблицу раньше, затрёт
        df = load('app_state')                         # только что записанное значение своей копией
        if not df.empty:
            df = df[df['key'].astype(str) != str(key)]
        rec = pd.DataFrame([{'key': str(key), 'value': '' if value is None else str(value)}])
        save('app_state', pd.concat([df, rec], ignore_index=True))


def reload():
    """Сбросить кэш (например, после внешней миграции)."""
    _cache.clear()
    _clear_derived()
    idx = _overlay.pop('index', None)
    if idx is not None:
        idx.close()             # индекс оверлея перечитается с диска (и достроится, если файл вырос)
    _overlay.clear()


# --------------------------------------------------------------------------- #
# Территории
# --------------------------------------------------------------------------- #
def _norm_id(value):
    """Привести id к каноничному строковому виду: 34.0 -> '34', пусто -> 'nan'."""
    if value is None:
        return 'nan'
    if isinstance(value, float) and np.isnan(value):
        return 'nan'
    s = str(value).strip().strip('[]')          # '[nan]'/'[34]' -> 'nan'/'34'
    if s == '' or s.lower() in ('nan', 'none'):
        return 'nan'
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def territory_id(id_region, id_district):
    """Единый ключ территории ``"<id_region>_<id_district>"`` (район -> ``nan``)."""
    return _norm_id(id_region) + '_' + _norm_id(id_district)


def get_territory(tid):
    """Вернуть запись территории как dict либо ``None``."""
    df = load('territories')
    if df.empty:
        return None
    hit = df[df['territory_id'] == tid]
    return hit.iloc[0].to_dict() if not hit.empty else None


def current_vega_uid(tid):
    """Текущий районный uid Веги для территории (из territory_uids) либо ``None``.

    Старые id районов перенумерованы Вегой; ряды по-прежнему хранятся под старым
    ``territory_id``, но ЗАПРОС к Веге надо слать по этому актуальному uid.
    """
    df = load('territory_uids')
    if df.empty:
        return None
    hit = df[df['territory_id'] == str(tid)]
    return int(hit.iloc[0]['vega_uid']) if not hit.empty else None


# --------------------------------------------------------------------------- #
# Урожайности
# --------------------------------------------------------------------------- #
def get_yields(tid, culture):
    """Вернуть ``{year: yield}`` по территории и культуре (источник + ручной ввод).

    Значения из ``user_yields`` (введённые оператором) НАКЛАДЫВАЮТСЯ поверх значений
    источника: ручной ввод — осознанная правка, и он остаётся доступен, даже когда
    сервисы-источники недоступны. Что именно откуда взято — см. :func:`get_yields_detailed`.
    """
    out = _source_yields(tid, culture)
    out.update(get_user_yields(tid, culture))
    return out


def _source_yields(tid, culture):
    """``{year: yield}`` только из таблицы источников ``yields`` (без ручного ввода)."""
    df = load('yields')
    if df.empty:
        return {}
    mask = (df['territory_id'] == tid) & (df['culture'] == str(culture))
    sub = df[mask]
    return {int(y): float(v) for y, v in zip(sub['year'], sub['yield'])}


def get_years(tid, culture):
    """Отсортированный список годов, по которым есть урожайность."""
    return sorted(get_yields(tid, culture).keys())


# --------------------------------------------------------------------------- #
# Урожайности от пользователя (ручной ввод) — таблица ``user_yields``
#
# Отдельная таблица, а не строки в ``yields``, потому что загрузчики источников
# (ЕМИСС, архив БДПМО) ПЕРЕПИСЫВАЮТ ``yields`` целиком. Введённое руками должно
# пережить любую переподгрузку и подставляться, когда источники недоступны.
# Столбцы: user_yield_id, territory_id, culture, year, yield, account_id, comment, updated_at.
# --------------------------------------------------------------------------- #
USER_YIELDS_COLUMNS = ['user_yield_id', 'territory_id', 'culture', 'year', 'yield',
                       'account_id', 'comment', 'updated_at']


def _yield_key(df):
    """Строковый ключ строк таблицы урожайности: ``'territory_id|culture|year'``."""
    year = pd.to_numeric(df['year'], errors='coerce').astype('Int64').astype(str)
    return df['territory_id'].astype(str) + '|' + df['culture'].astype(str) + '|' + year


def _user_yields_index():
    """``{(territory_id, culture): {year: value}}``; кэш сбрасывается в save/reload."""
    if 'user_yields' not in _index_cache:
        df = load('user_yields')
        idx = {}
        if not df.empty:
            for tid, cult, year, val in zip(df['territory_id'], df['culture'],
                                            df['year'], df['yield']):
                try:
                    v = float(val)
                    y = int(year)
                except (TypeError, ValueError):
                    continue
                if v != v:                                   # NaN — не значение
                    continue
                idx.setdefault((str(tid), str(cult)), {})[y] = v
        _index_cache['user_yields'] = idx
    return _index_cache['user_yields']


def get_user_yields(tid, culture):
    """``{year: yield}``, введённые пользователем вручную по территории и культуре."""
    return dict(_user_yields_index().get((str(tid), str(culture)), {}))


def get_yields_detailed(tid, culture):
    """``{year: (value, source)}``, где source — ``'source'`` (из БД) или ``'user'`` (руками)."""
    out = {y: (v, 'source') for y, v in _source_yields(tid, culture).items()}
    for y, v in get_user_yields(tid, culture).items():
        out[y] = (v, 'user')
    return out


def upsert_user_yields(records, account_id=None, comment=None):
    """Записать урожайности пользователя, заменяя совпадающие (territory_id, culture, year).

    ``records`` — список dict с ключами ``territory_id, culture, year, yield``
    (опционально ``account_id``, ``comment``). Одна запись файла на весь пакет.
    """
    if not records:
        return 0
    now = datetime.now().isoformat(timespec='seconds')
    rows = []
    for r in records:
        rows.append({
            'territory_id': str(r['territory_id']),
            'culture': str(r['culture']),
            'year': int(r['year']),
            'yield': float(r['yield']),
            'account_id': r.get('account_id', account_id),
            'comment': r.get('comment', comment),
            'updated_at': now,
        })
    new = pd.DataFrame(rows)
    df = load('user_yields')
    if not df.empty:
        nkey = set(_yield_key(new))
        df = df[~_yield_key(df).isin(nkey)]
    df = pd.concat([df, new], ignore_index=True)
    df['user_yield_id'] = range(1, len(df) + 1)
    save('user_yields', df.reindex(columns=USER_YIELDS_COLUMNS))
    return len(new)


def delete_user_yields(keys):
    """Удалить значения пользователя по ключам ``[(territory_id, culture, year), ...]``."""
    keys = [(str(t), str(c), int(y)) for t, c, y in (keys or [])]
    if not keys:
        return 0
    df = load('user_yields')
    if df.empty:
        return 0
    drop = _yield_key(df).isin({f'{t}|{c}|{y}' for t, c, y in keys})
    removed = int(drop.sum())
    if removed:
        df = df[~drop].reset_index(drop=True)
        df['user_yield_id'] = range(1, len(df) + 1)
        save('user_yields', df.reindex(columns=USER_YIELDS_COLUMNS))
    return removed


def yields_frame():
    """Длинная таблица урожайностей с наложенным ручным вводом (столбцы как у ``yields``).

    Строки пользователя заменяют совпадающие (territory_id, culture, year) и добавляют
    отсутствующие. Нужна там, где урожайности читаются целиком (план сборки датасета).
    """
    base = load('yields')
    user = load('user_yields')
    if user.empty:
        return base
    u = user[['territory_id', 'culture', 'year', 'yield']].copy()
    u['yield'] = pd.to_numeric(u['yield'], errors='coerce')
    u = u[u['yield'].notna()]
    if u.empty:
        return base
    u['territory_id'] = u['territory_id'].astype(str)
    u['culture'] = u['culture'].astype(str)
    u['year'] = u['year'].astype(int)
    if base.empty:
        return u.reset_index(drop=True)
    nkey = set(_yield_key(u))
    return pd.concat([base[~_yield_key(base).isin(nkey)], u], ignore_index=True)


# --------------------------------------------------------------------------- #
# Временные ряды (нормализованная схема ради минимума памяти)
#
#   parms.csv        : parm_id, parm                         — справочник параметров
#   series.csv       : series_id, territory_id, parm_id, year, first_date, last_date
#   time_series.csv  : series_id, day, value                 — только числа (19.7M строк)
#
# Повторяющийся текст (parm) и повторяющиеся ключи (territory_id, year) вынесены
# в маленькие справочники; в гигантской таблице остаются лишь целочисленный
# series_id и числовые day/value.
# --------------------------------------------------------------------------- #
def _parm_maps():
    """(parm -> parm_id, parm_id -> parm). Кэшируется, сбрасывается в save/reload."""
    if 'parms' not in _index_cache:
        df = load('parms')
        to_id, to_name = {}, {}
        if not df.empty:
            for pid, name in zip(df['parm_id'], df['parm']):
                to_id[str(name)] = int(pid)
                to_name[int(pid)] = str(name)
        _index_cache['parms'] = (to_id, to_name)
    return _index_cache['parms']


def _series_index():
    """((tid, parm_id, year) -> series_id, series_id -> (first_date, last_date))."""
    if 'series' not in _index_cache:
        df = load('series')
        idx, dates = {}, {}
        if not df.empty:
            for sid, tid, pid, year, fd, ld in zip(
                    df['series_id'], df['territory_id'], df['parm_id'], df['year'],
                    df.get('first_date', ['']*len(df)), df.get('last_date', ['']*len(df))):
                idx[(str(tid), int(pid), int(year))] = int(sid)
                dates[int(sid)] = (str(fd), str(ld))
        _index_cache['series'] = (idx, dates)
    return _index_cache['series']


def _points_groups():
    """series_id -> позиции строк в time_series (groupby.indices, C-level)."""
    if 'points' not in _index_cache:
        df = load('time_series')
        groups = df.groupby('series_id').indices if not df.empty else {}
        _index_cache['points'] = (df, groups)
    return _index_cache['points']


def _series_id_of(tid, parm, year):
    """series_id ряда (tid, parm, year) либо None."""
    pid = _parm_maps()[0].get(str(parm))
    if pid is None:
        return None
    return _series_index()[0].get((str(tid), int(pid), int(year)))


# Оверлей скачанных рядов — отдельный денормализованный append-лог в БД,
# чтобы загрузка в процессе сборки НЕ переписывала большую базовую таблицу.
OVERLAY_TABLE = 'time_series_downloaded'   # territory_id, parm, year, vega_uid, day, value


def _downloads():
    """Оверлей скачанных рядов как ленивое отображение (tid, parm, year) -> [[day, value], ...].

    Раньше здесь читался ВЕСЬ файл оверлея (``pandas.read_csv``) и строился словарь всех
    рядов — на оверлее в 3.7 ГБ это минуты и ~11 ГБ RAM при первом же чтении любого ряда.
    Теперь :mod:`overlay_index` один раз строит индекс смещений блоков и сохраняет его
    рядом с оверлеем, а точки читаются по одному ряду (``seek``+``read``) и кэшируются.

    Возвращаемый объект — ``Mapping``: ``key in ...``, ``len(...)``, ``.keys()`` отвечают
    по индексу (без чтения данных), ``.get(key)`` читает точки с диска. Провенанс
    ``vega_uid`` лежит в том же индексе (см. :func:`download_uids`).
    """
    idx = _overlay.get('index')
    if idx is None:
        import overlay_index
        idx = overlay_index.OverlayIndex(_csv_path(OVERLAY_TABLE))
        _overlay['index'] = idx
    return idx


def download_uids():
    """Провенанс оверлея: (tid, parm, year) -> vega_uid (с которого ряд скачан)."""
    return _downloads().uids()


def get_time_series(tid, parm, year):
    """Вернуть разрежённый ряд ``[[day, value], ...]`` по возрастанию дня.

    Сначала смотрит оверлей скачанных рядов, затем базовую таблицу.
    """
    over = _downloads().points((str(tid), str(parm), int(year)))
    if over is not None:
        return [[int(d), float(v)] for d, v in over]     # копия: кэш оверлея не отдаём наружу

    sid = _series_id_of(tid, parm, year)
    if sid is None:
        return []
    df, groups = _points_groups()
    pos = groups.get(sid)
    if pos is None or len(pos) == 0:
        return []
    sub = df.iloc[pos].sort_values('day')
    return [[int(d), float(v)] for d, v in zip(sub['day'], sub['value'])]


def has_time_series(tid, parm, year):
    """Есть ли непустой ряд (оверлей или база; замена ``chek_file``)."""
    if (str(tid), str(parm), int(year)) in _downloads():
        return True
    sid = _series_id_of(tid, parm, year)
    if sid is None:
        return False
    _df, groups = _points_groups()
    pos = groups.get(sid)
    return pos is not None and len(pos) > 0


def _series_dates(tid, parm, year):
    """Первая/последняя дата ряда из series (или синтез из года)."""
    sid = _series_id_of(tid, parm, year)
    if sid is not None:
        hit = _series_index()[1].get(sid)
        if hit is not None and hit[0]:
            return hit
    return f'{int(year)}-01-01', f'{int(year)}-12-31'


def get_time_series_array(tid, parm, year, size, padding='zeros'):
    """Прочитать ряд из БД и привести к плотному массиву длины ``size``.

    Переиспользует интерполяцию ``TimeRow.get_array_by_size`` (поведение
    идентично прежнему чтению из .json-файла). Возвращает ``np.array``;
    пустой массив, если ряда нет (как при ``status == False``).
    """
    from timeseries_lib import TimeRow

    key = (tid, str(parm), int(year), int(size), padding)   # мемоизация: один ряд интерполируется один раз
    cached = _array_cache.get(key)
    if cached is not None:
        return cached

    xy = get_time_series(tid, parm, year)
    if not xy:
        return np.array([])     # пустое не кэшируем — после докачки ряд пересчитается

    first_date, last_date = _series_dates(tid, parm, year)
    xy_str = [[str(int(d)), float(v)] for d, v in xy]
    data = {'data': {'1': {'xy': xy_str,
                           'labels': {xy_str[0][0]: first_date,
                                      xy_str[-1][0]: last_date}}}}
    time_data, _ = TimeRow(data=data).get_array_by_size(my_size=size, padding=padding)
    _array_cache[key] = time_data
    return time_data


def mean_over_years(tid, parm, year, average_last, size, padding='zeros'):
    """Скользящее межгодовое среднее ряда — климатология для года ``year``.

    Берёт ``average_last`` ПРЕДЫДУЩИХ лет ``[year-1 ... year-average_last]`` (БЕЗ текущего
    года, чтобы не было утечки), приводит каждый к длине ``size`` и усредняет поэлементно,
    игнорируя нули.
    """
    parm = str(parm).replace('_historical', '').replace('_hist', '')
    key = (tid, parm, int(year), int(average_last), int(size), padding)
    cached = _mean_cache.get(key)
    if cached is not None:
        return cached

    arrays = []
    for i in range(1, int(average_last) + 1):       # предыдущие average_last лет, без текущего
        arr = get_time_series_array(tid, parm, int(year) - i, size, padding)
        if arr.shape[0] != 0:
            arrays.append(arr)

    if not arrays:
        return np.array([])     # пустое не кэшируем — после докачки пересчитается

    # поэлементное среднее, игнорируя нули — векторизовано (вместо двойного Python-цикла)
    stacked = np.vstack(arrays)                 # (k, size)
    total = stacked.sum(axis=0)
    not_null = np.count_nonzero(stacked, axis=0)
    not_null[not_null == 0] = 1
    target = total / not_null
    _mean_cache[key] = target
    return target


def _invalidate_reads(tid, parm, year):
    """Сбросить мемоизацию, на которую мог повлиять новый ряд (tid, parm, year).

    Базовые «пустые» результаты не кэшируются, поэтому достаточно убрать средние,
    в окно которых попадает этот год (массивы по этому ключу не кэшировались — их не было).
    """
    pstr = str(parm)
    y = int(year)
    for k in [k for k in _mean_cache       # окно = [k_year-N .. k_year-1]; год y в нём -> y+1<=k_year<=y+N
              if k[0] == tid and k[1] == pstr and y + 1 <= k[2] <= y + int(k[3])]:
        _mean_cache.pop(k, None)


def is_accumulated(parm):
    """Накопительный ли параметр (нарастающий итог: ``mean_prec_acc``, ``mean_temp_acc``)."""
    return str(parm).replace('_historical', '').replace('_hist', '').endswith('_acc')


def to_daily(parm, pts):
    """Свести ряд к ОДНОЙ точке в сутки. ``pts`` — ``[[day, value], ...]`` по возрастанию дня.

    Вега отдаёт метеопараметры 4 раза в сутки (шаг 6 ч), а номер дня хранится целым, так что
    без свёртки на один ``day`` приходится 4 точки, и результат интерполяции зависит от их
    порядка. Модели работают с суточным разрешением (как и базовая ``time_series``), поэтому:

    * обычный параметр — СРЕДНЕЕ значений за сутки (так же сворачивает сама Вега: прирост
      суточного ``mean_temp_acc`` равен среднему 4 значений ``mean_temp``);
    * накопительный (``*_acc``) — значение на КОНЕЦ суток (``mean_prec_acc`` — точная
      нарастающая сумма шестичасовых ``mean_prec``; среднее отставало бы на полсуток).

    Сутки с единственной точкой не трогаются вовсе (NDVI, ``mean_temp_acc`` — бит-в-бит).
    NaN при усреднении пропускаются; если в сутках только NaN — остаётся NaN.
    Входной порядок внутри суток считается хронологическим (сортировка по дню устойчива).
    """
    acc = is_accumulated(parm)
    out = []
    i, n = 0, len(pts)
    while i < n:
        day = pts[i][0]
        j = i + 1
        while j < n and pts[j][0] == day:
            j += 1
        if j - i == 1:
            out.append([day, pts[i][1]])
        else:
            vals = [v for _d, v in pts[i:j]]
            good = [v for v in vals if v == v]
            if not good:
                val = float('nan')
            elif acc:
                val = good[-1]
            else:
                val = round(sum(good) / len(good), 6)    # исходные значения — 1..4 знака
            out.append([day, val])
        i = j
    return out


def put_time_series(tid, parm, year, xy, labels=None, vega_uid=None):
    """Сохранить скачанный ряд в ОВЕРЛЕЙ (память + дозапись в append-лог БД).

    Базовую таблицу ``time_series`` не трогает и кэши базы не сбрасывает — поэтому
    пригодно для массовой загрузки в процессе сборки. Ряды оверлея видны при чтении
    наравне с базой и сохраняются между запусками.

    ``xy`` — список ``[day, value]``; ``labels`` игнорируются (год даёт нужную дату).
    Ряд сводится к одной точке в сутки (:func:`to_daily`).
    ``vega_uid`` — uid Веги, с которого скачан ряд (провенанс; для района — актуальный
    районный uid, для региона — id_region). Хранится, чтобы после смены сопоставления
    можно было перекачать только изменившиеся районы.
    """
    tid, pstr, year = str(tid), str(parm), int(year)
    pts = [[int(float(d)), float(v)] for d, v in xy if v is not None]
    if not pts:
        return
    pts.sort(key=lambda p: p[0])          # устойчиво: порядок внутри суток сохраняется
    pts = to_daily(pstr, pts)

    # дозапись в append-лог (без переписывания базы) + обновление индекса и кэша оверлея;
    # ряд сразу виден при чтении, повторная загрузка ряда перекрывает прежний блок
    _downloads().append_series((tid, pstr, year), pts, vega_uid=vega_uid)
    _invalidate_reads(tid, pstr, year)                    # точечно сбросить устаревшие средние


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #
def get_model(model_name):
    """Метаданные модели как dict либо ``None``."""
    df = load('models')
    if df.empty:
        return None
    hit = df[df['model_name'] == str(model_name)]
    return hit.iloc[0].to_dict() if not hit.empty else None


def list_model_params(model_name):
    """Упорядоченный список параметров модели.

    Возвращает список dict ``{param_index, parm, normalization, scalar}``
    в порядке ``param_index`` (порядок load-bearing для матрицы признаков).
    """
    df = load('model_params')
    if df.empty:
        return []
    sub = df[df['model_name'] == str(model_name)].sort_values('param_index')
    return sub.to_dict('records')


def get_masks_df():
    """Маски культур в том же виде, что прежний ``pd.read_excel(Cultures_masks)``."""
    return load('cultures_masks')


# --------------------------------------------------------------------------- #
# Прогнозы
# --------------------------------------------------------------------------- #
def upsert_prediction(tid, culture, model_name, predict_year, value,
                      dispersion=None, average_productivity=None, made_at=None,
                      account_id=None, comment=None, batch_id=None):
    """Дописать прогноз с отметкой времени ``made_at`` (по умолчанию — сейчас).

    ``account_id`` — кто сделал прогноз; ``comment`` — комментарий; ``batch_id`` — id партии
    (один запуск прогноза по списку территорий = одна партия).
    """
    if made_at is None:
        made_at = datetime.now().isoformat(timespec='seconds')
    df = load('predictions')
    next_id = 1 if df.empty else int(df['prediction_id'].max()) + 1
    record = {
        'prediction_id': next_id,
        'territory_id': tid,
        'culture': str(culture),
        'model_name': str(model_name),
        'predict_year': int(predict_year),
        'value': None if value is None else float(value),
        'dispersion': None if dispersion is None else float(dispersion),
        'average_productivity': None if average_productivity is None else float(average_productivity),
        'made_at': made_at,
        'account_id': None if account_id is None else int(account_id),
        'comment': None if comment is None else str(comment),
        'batch_id': None if batch_id is None else str(batch_id),
    }
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save('predictions', df)
    return next_id


def upsert_yields_bulk(records):
    """Пакетно записать урожайности в таблицу ``yields``, заменяя совпадающие (territory_id, culture, year).

    records — список dict с ключами ``territory_id, culture, year, yield``. Одна запись файла на все.
    """
    if not records:
        return 0
    new = pd.DataFrame(records)
    new['year'] = new['year'].astype(int)
    df = load('yields')
    if not df.empty:
        key = (df['territory_id'].astype(str) + '|' + df['culture'].astype(str) + '|' + df['year'].astype(str))
        nkey = set(new['territory_id'].astype(str) + '|' + new['culture'].astype(str) + '|' + new['year'].astype(str))
        df = df[~key.isin(nkey)]
    df = pd.concat([df, new], ignore_index=True)
    save('yields', df)
    return len(new)


def upsert_territory_metrics(tid, culture, batch_id, mse, r2, last_k=None,
                             account_id=None, made_at=None, rmse=None, mae=None, pearson=None):
    """Сохранить уточнённые метрики территории (MSE/RMSE/MAE/R²/Пирсон) для партии — при прогнозе.

    Отчёт по партии затем просто читает их из БД (``get_batch_metrics``), без повторного бэктеста.
    """
    if made_at is None:
        made_at = datetime.now().isoformat(timespec='seconds')
    _f = lambda v: None if v is None or v != v else float(v)
    df = load('territory_metrics')
    next_id = 1 if df.empty else int(df['metric_id'].max()) + 1
    record = {
        'metric_id': next_id,
        'territory_id': tid,
        'culture': str(culture),
        'batch_id': None if batch_id is None else str(batch_id),
        'mse': _f(mse), 'rmse': _f(rmse), 'mae': _f(mae), 'r2': _f(r2), 'pearson': _f(pearson),
        'last_k': None if last_k is None else int(last_k),
        'account_id': None if account_id is None else int(account_id),
        'made_at': made_at,
    }
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save('territory_metrics', df)
    return next_id


def get_batch_metrics(batch_id):
    """Уточнённые метрики территорий партии: {territory_id: {mse,rmse,mae,r2,pearson,last_k}} (свежайшие)."""
    df = load('territory_metrics')
    if df.empty:
        return {}
    sub = df[df['batch_id'].astype(str) == str(batch_id)].sort_values('made_at')
    out = {}
    for _, r in sub.iterrows():                              # сортировка по времени -> остаётся свежайшая
        out[str(r['territory_id'])] = {k: r.get(k) for k in ('mse', 'rmse', 'mae', 'r2', 'pearson', 'last_k')}
    return out


def delete_batch(batch_id):
    """Удалить прогнозы, уточнённые метрики и план/итоги партии. -> (удалено_прогнозов, метрик)."""
    counts = {}
    for tbl in ('predictions', 'territory_metrics', 'forecast_outcomes'):
        df = load(tbl)
        if df.empty or 'batch_id' not in df:
            counts[tbl] = 0
            continue
        keep = df['batch_id'].astype(str) != str(batch_id)
        removed = int((~keep).sum())
        if removed:
            save(tbl, df[keep].reset_index(drop=True))
        counts[tbl] = removed
    return counts['predictions'], counts['territory_metrics']


def upsert_model_metrics(culture, model_name, n, mse, rmse, r2,
                         sample=None, last_k=None, made_at=None):
    """Сохранить точность модели по культуре (перезаписывает прежнюю запись для пары)."""
    if made_at is None:
        made_at = datetime.now().isoformat(timespec='seconds')
    df0 = load('model_metrics')
    next_id = 1 if df0.empty else int(df0['model_metric_id'].max()) + 1
    if not df0.empty:                                       # выкинуть прежнюю запись пары
        mask = (df0['culture'].astype(str) == str(culture)) & (df0['model_name'].astype(str) == str(model_name))
        df0 = df0[~mask]
    _f = lambda v: None if v is None or v != v else float(v)
    record = {'model_metric_id': next_id, 'culture': str(culture), 'model_name': str(model_name),
              'n': None if n is None else int(n), 'mse': _f(mse), 'rmse': _f(rmse), 'r2': _f(r2),
              'sample': None if sample is None else int(sample),
              'last_k': None if last_k is None else int(last_k), 'made_at': made_at}
    df = pd.concat([df0, pd.DataFrame([record])], ignore_index=True)
    save('model_metrics', df)
    return next_id


def get_model_metrics(culture=None):
    """Точность моделей: {(culture, model_name): {'n','mse','rmse','r2','sample','last_k','made_at'}}."""
    df = load('model_metrics')
    if df.empty:
        return {}
    if culture is not None:
        df = df[df['culture'].astype(str) == str(culture)]
    out = {}
    for _, r in df.iterrows():
        out[(str(r['culture']), str(r['model_name']))] = {
            'n': r.get('n'), 'mse': r.get('mse'), 'rmse': r.get('rmse'), 'r2': r.get('r2'),
            'sample': r.get('sample'), 'last_k': r.get('last_k'), 'made_at': r.get('made_at')}
    return out


def get_prediction(tid, culture, model_name, predict_year):
    """Свежайший прогноз по логическому ключу либо ``None``."""
    df = load('predictions')
    if df.empty:
        return None
    mask = (
        (df['territory_id'] == tid)
        & (df['culture'] == str(culture))
        & (df['model_name'] == str(model_name))
        & (df['predict_year'] == int(predict_year))
    )
    sub = df[mask].sort_values('made_at')
    return sub.iloc[-1].to_dict() if not sub.empty else None
