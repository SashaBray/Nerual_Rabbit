"""Файловая реляционная БД проекта Nerual Rabbit (без SQL-движка).

Все данные хранятся набором CSV-таблиц в каталоге ``Database/``:

* ``territories``      — справочник территорий (регион/район)
* ``yields``           — урожайности культур по годам (длинная таблица)
* ``time_series``      — спутниковые временные ряды (длинная таблица, день года -> значение)
* ``time_series_meta`` — календарь рядов (first_date / last_date из labels)
* ``models``           — модели и их тип/культура/границы дней/файл весов
* ``model_params``     — упорядоченный список параметров модели + нормировка
* ``predictions``      — прогнозы с отметкой времени (made_at)
* ``cultures_masks``   — маски NDVI по культурам (дамп Cultures_masks.xlsx)
* ``regions``          — справочник region_name -> uid

Модуль — тонкий слой доступа на pandas, без ORM. Таблицы лениво читаются и
кэшируются в памяти; запись дописывает строки и сбрасывает CSV на диск.

Единая точка кодирования ключа территории — :func:`territory_id`. Район может
отсутствовать (региональный уровень) — тогда в ключе используется токен ``nan``,
что совпадает с исторической конвенцией имён файлов временных рядов
(``<parm>_<id_region>_<id_district>_<year>.json``).
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Database')

# Имя таблицы -> столбцы, по которым она хранится/читается как строковые ключи,
# чтобы '34' и 34.0 не расходились между миграцией и рантаймом.
_STR_KEY_COLUMNS = {
    'territories': ['territory_id', 'id_country'],
    'yields': ['territory_id', 'culture'],
    'time_series': ['territory_id', 'parm'],
    'time_series_meta': ['territory_id', 'parm'],
    'models': ['model_name', 'model_type', 'culture', 'weights_file'],
    'model_params': ['model_name', 'parm'],
    'predictions': ['territory_id', 'culture', 'model_name'],
    'regions': ['region', 'region_uid'],
}

_cache = {}


# --------------------------------------------------------------------------- #
# Базовые операции с таблицами
# --------------------------------------------------------------------------- #
def path_of(table_name):
    """Путь к CSV-файлу таблицы."""
    return os.path.join(DB_DIR, table_name + '.csv')


def load(table_name):
    """Прочитать таблицу (с кэшированием). Возвращает копию-ссылку из кэша."""
    if table_name not in _cache:
        file_path = path_of(table_name)
        if not os.path.exists(file_path):
            _cache[table_name] = pd.DataFrame()
        else:
            str_cols = _STR_KEY_COLUMNS.get(table_name, [])
            _cache[table_name] = pd.read_csv(
                file_path,
                dtype={c: str for c in str_cols},
                encoding='utf-8',
            )
    return _cache[table_name]


def save(table_name, df):
    """Записать таблицу на диск и обновить кэш."""
    os.makedirs(DB_DIR, exist_ok=True)
    df.to_csv(path_of(table_name), index=False, encoding='utf-8')
    _cache[table_name] = df


def reload():
    """Сбросить кэш (например, после внешней миграции)."""
    _cache.clear()


# --------------------------------------------------------------------------- #
# Территории
# --------------------------------------------------------------------------- #
def _norm_id(value):
    """Привести id к каноничному строковому виду: 34.0 -> '34', пусто -> 'nan'."""
    if value is None:
        return 'nan'
    if isinstance(value, float) and np.isnan(value):
        return 'nan'
    s = str(value).strip()
    if s == '' or s.lower() == 'nan':
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


# --------------------------------------------------------------------------- #
# Урожайности
# --------------------------------------------------------------------------- #
def get_yields(tid, culture):
    """Вернуть ``{year: yield}`` по территории и культуре."""
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
# Временные ряды
# --------------------------------------------------------------------------- #
def get_time_series(tid, parm, year):
    """Вернуть разрежённый ряд ``[[day, value], ...]`` по возрастанию дня."""
    df = load('time_series')
    if df.empty:
        return []
    mask = (df['territory_id'] == tid) & (df['parm'] == str(parm)) & (df['year'] == int(year))
    sub = df[mask].sort_values('day')
    return [[int(d), float(v)] for d, v in zip(sub['day'], sub['value'])]


def has_time_series(tid, parm, year):
    """Есть ли непустой ряд (замена ``chek_file``)."""
    return len(get_time_series(tid, parm, year)) > 0


def _series_dates(tid, parm, year):
    """Первая/последняя дата ряда из meta (или синтез из года)."""
    meta = load('time_series_meta')
    if not meta.empty:
        hit = meta[(meta['territory_id'] == tid)
                   & (meta['parm'] == str(parm))
                   & (meta['year'] == int(year))]
        if not hit.empty:
            return str(hit.iloc[0]['first_date']), str(hit.iloc[0]['last_date'])
    return f'{int(year)}-01-01', f'{int(year)}-12-31'


def get_time_series_array(tid, parm, year, size, padding='zeros'):
    """Прочитать ряд из БД и привести к плотному массиву длины ``size``.

    Переиспользует интерполяцию ``TimeRow.get_array_by_size`` (поведение
    идентично прежнему чтению из .json-файла). Возвращает ``np.array``;
    пустой массив, если ряда нет (как при ``status == False``).
    """
    from Library_working_with_time_series_v1 import TimeRow

    xy = get_time_series(tid, parm, year)
    if not xy:
        return np.array([])

    first_date, last_date = _series_dates(tid, parm, year)
    xy_str = [[str(int(d)), float(v)] for d, v in xy]
    data = {'data': {'1': {'xy': xy_str,
                           'labels': {xy_str[0][0]: first_date,
                                      xy_str[-1][0]: last_date}}}}
    time_data, _ = TimeRow(data=data).get_array_by_size(my_size=size, padding=padding)
    return time_data


def mean_over_years(tid, parm, year, average_last, size, padding='zeros'):
    """Межгодовое среднее ряда (замена дискового ``Operations.mean_in_district``).

    Берёт ``average_last`` лет начиная с ``year`` назад, приводит каждый к длине
    ``size`` и усредняет поэлементно, игнорируя нули (как в исходной логике).
    """
    parm = str(parm).replace('_historical', '').replace('_hist', '')
    arrays = []
    for i in range(int(average_last)):
        arr = get_time_series_array(tid, parm, int(year) - i, size, padding)
        if arr.shape[0] != 0:
            arrays.append(arr)

    if not arrays:
        return np.array([])

    target = np.zeros(size)
    for j in range(size):
        total = 0.0
        not_null = 0
        for arr in arrays:
            total += arr[j]
            if arr[j] != 0:
                not_null += 1
        target[j] = total / (not_null if not_null else 1)
    return target


def put_time_series(tid, parm, year, xy, labels=None):
    """Записать/заменить ряд (сток загрузок Vega, замена ``write_time_row_json``).

    ``xy`` — список ``[day, value]`` (день может быть строкой или числом).
    ``labels`` — необязательный календарь ``{day_str: date_str}`` для meta.
    """
    year = int(year)
    parm = str(parm)
    df = load('time_series')
    if not df.empty:
        keep = ~(
            (df['territory_id'] == tid)
            & (df['parm'] == parm)
            & (df['year'] == year)
        )
        df = df[keep]
    rows = [
        {'territory_id': tid, 'parm': parm, 'year': year,
         'day': int(float(d)), 'value': float(v)}
        for d, v in xy
    ]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    save('time_series', df)

    if labels:
        dates = sorted(str(v) for v in labels.values())
        meta = load('time_series_meta')
        if not meta.empty:
            meta = meta[~(
                (meta['territory_id'] == tid)
                & (meta['parm'] == parm)
                & (meta['year'] == year)
            )]
        meta = pd.concat([meta, pd.DataFrame([{
            'territory_id': tid, 'parm': parm, 'year': year,
            'first_date': dates[0] if dates else '',
            'last_date': dates[-1] if dates else '',
        }])], ignore_index=True)
        save('time_series_meta', meta)


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
                      dispersion=None, average_productivity=None, made_at=None):
    """Дописать прогноз с отметкой времени ``made_at`` (по умолчанию — сейчас)."""
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
    }
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save('predictions', df)
    return next_id


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
