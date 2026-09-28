"""Чтение задания на сборку датасета по id и разворачивание всех ссылок.

Задание — строка в ``workspace/tasks/dataset_tasks.csv``; в ней только id и числа:

    task_id, culture_id, ukey_id, param_list_id,
    time_rows_size, duration_years, feature_hist_last, prod_hist_last,
    years, regions_filter, output_name

Детали разворачиваются по справочникам ``cultures.csv`` (культура -> bdpmo/mask),
``ukeys.csv`` (ukey_id -> ключ), ``parameters.csv`` + ``parameter_lists.csv``
(список признаков), ``build_defaults.json`` (редкие настройки сборки).
"""

import json

import pandas as pd

import config


def _to_bool(value, default=True):
    s = str(value).strip().lower()
    if s in ('', 'nan', 'none'):
        return default
    return s in ('1', 'true', 'yes', 'да', 't', 'y')


def _to_int(value, default=None):
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def parse_years(spec):
    """'2018-2022' -> [2018..2022]; '2018;2020' -> [2018,2020]."""
    spec = str(spec).strip()
    if '-' in spec and ';' not in spec:
        a, b = spec.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(p) for p in spec.replace(',', ';').split(';') if p.strip()]


def parse_regions_filter(spec):
    """'all'/'' -> None (без фильтра); 'id1;id2' -> {'id1','id2'}."""
    s = str(spec).strip().lower()
    if s in ('', 'all', 'nan', 'none'):
        return None
    return {p.strip() for p in str(spec).replace(',', ';').split(';') if p.strip()}


def _read_csv(path):
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig')


def resolve_culture(culture_id):
    """culture_id -> dict {title, bdpmo_culture, mask_culture}."""
    df = _read_csv(config.CULTURES_FILE)
    hit = df[df['culture_id'] == str(culture_id)]
    if hit.empty:
        raise ValueError(f'culture_id={culture_id} не найден в {config.CULTURES_FILE}')
    r = hit.iloc[0]
    return {'title': r['title'], 'bdpmo_culture': r['bdpmo_culture'],
            'mask_culture': r['mask_culture']}


def resolve_ukey(ukey_id):
    """ukey_id -> сам ukey (запасной — config.read_ukey())."""
    try:
        df = _read_csv(config.UKEYS_FILE)
        hit = df[df['ukey_id'] == str(ukey_id)]
        if not hit.empty:
            return str(hit.iloc[0]['ukey']).strip()
    except FileNotFoundError:
        pass
    return config.read_ukey()


def load_feature_list(param_list_id):
    """param_list_id -> упорядоченный список признаков.

    Каждый элемент: ``{parameter_id, name, is_ndvi, historical, order_index}``.
    """
    lists = _read_csv(config.PARAMETER_LISTS_FILE)
    params = _read_csv(config.PARAMETERS_FILE)
    name_by_id = {str(pid): (nm, is_ndvi) for pid, nm, is_ndvi
                  in zip(params['parameter_id'], params['name'], params['is_ndvi'])}

    sub = lists[lists['param_list_id'] == str(param_list_id)].copy()
    if sub.empty:
        raise ValueError(f'param_list_id={param_list_id} не найден в {config.PARAMETER_LISTS_FILE}')
    sub['order_index'] = sub['order_index'].astype(int)
    sub = sub.sort_values('order_index')

    features = []
    for _, r in sub.iterrows():
        pid = str(r['parameter_id'])
        if pid not in name_by_id:
            raise ValueError(f'parameter_id={pid} не найден в {config.PARAMETERS_FILE}')
        name, is_ndvi = name_by_id[pid]
        features.append({
            'parameter_id': _to_int(pid),
            'name': str(name),
            'is_ndvi': _to_bool(is_ndvi, default=False),
            'historical': _to_bool(r['historical'], default=False),
            'order_index': int(r['order_index']),
        })
    return features


def load_build_defaults():
    """Редкие настройки сборки из build_defaults.json (с дефолтами)."""
    defaults = {'padding': 'zeros', 'edge_fixup': True,
                'concat_newest_first': True, 'normalize': 'none',
                'checkpoint_every': 200}     # сохранять датасет каждые N примеров (защита от прерывания)
    try:
        with open(config.BUILD_DEFAULTS_FILE, 'r', encoding='utf-8-sig') as f:
            defaults.update(json.load(f))
    except FileNotFoundError:
        pass
    return defaults


def load_task(task_id):
    """Прочитать задание по id и вернуть полностью развёрнутый dict."""
    df = _read_csv(config.DATASET_TASKS_FILE)
    hit = df[df['task_id'] == str(task_id)]
    if hit.empty:
        raise ValueError(f'Задание "{task_id}" не найдено в {config.DATASET_TASKS_FILE}')
    row = hit.iloc[0]

    culture = resolve_culture(row['culture_id'])
    defaults = load_build_defaults()

    task = {
        'task_id': str(row['task_id']),
        'output_name': str(row.get('output_name') or row['task_id']),
        # культура (развёрнута из culture_id)
        'culture_id': _to_int(row['culture_id']),
        'culture_title': culture['title'],
        'bdpmo_culture': culture['bdpmo_culture'],
        'mask_culture': culture['mask_culture'],
        # ukey (развёрнут из ukey_id)
        'ukey_id': _to_int(row['ukey_id']),
        'ukey': resolve_ukey(row['ukey_id']),
        # список признаков (развёрнут из param_list_id)
        'param_list_id': _to_int(row['param_list_id']),
        'features': load_feature_list(row['param_list_id']),
        # числовые параметры сборки
        'time_rows_size': _to_int(row.get('time_rows_size'), 365),
        'duration_years': _to_int(row.get('duration_years'), 2),
        'feature_hist_last': _to_int(row.get('feature_hist_last'), 5),   # n — окно усреднения рядов
        'prod_hist_last': _to_int(row.get('prod_hist_last'), 4),         # m — окно скольз. среднего урожайности
        'years': parse_years(row['years']),
        'regions_filter': parse_regions_filter(row.get('regions_filter')),
        # редкие настройки из build_defaults.json
        'padding': defaults['padding'],
        'edge_fixup': bool(defaults['edge_fixup']),
        'concat_newest_first': bool(defaults['concat_newest_first']),
        'normalize': defaults['normalize'],
        'checkpoint_every': int(defaults['checkpoint_every']),
    }
    return task
