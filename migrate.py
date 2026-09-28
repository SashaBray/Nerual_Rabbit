"""Миграция исторических данных Nerual Rabbit в CSV-таблицы каталога ``Database/``.

Импортирует:

* временные ряды  ``Temporary_information/TimeRows/*.json``  -> time_series, time_series_meta
* расчётные листы ``Destination_files/**/*.xlsx``            -> territories, yields
* модели          ``Service_files/Models/<name>/parms.json`` -> models, model_params
* справочники     ``Cultures_masks.xlsx`` / ``Regions_uid_dict.json`` -> cultures_masks, regions

Скрипт идемпотентен: каждая таблица перезаписывается целиком. Чтение ``.xlsx``
работает как через ``pandas.read_excel`` (если установлен openpyxl), так и через
встроенный stdlib-фолбэк (zipfile + xml), поэтому миграция не требует доп.
зависимостей.

Запуск:  ``python migrate.py``
"""

import glob
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import db


ROOT = os.path.dirname(os.path.abspath(__file__))
TIMEROWS_DIR = os.path.join(ROOT, 'Temporary_information', 'TimeRows')
DESTINATION_DIR = os.path.join(ROOT, 'Destination_files')
MODELS_DIR = os.path.join(ROOT, 'Service_files', 'Models')
COMMON_DIR = os.path.join(ROOT, 'Service_files', 'Common_information')

_SS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


# --------------------------------------------------------------------------- #
# Чтение .xlsx (pandas -> фолбэк на stdlib)
# --------------------------------------------------------------------------- #
def _col_letters(ref):
    return ''.join(ch for ch in ref if ch.isalpha())


def _read_xlsx_fallback(path):
    """Минимальный читатель первого листа .xlsx через zip/xml (без openpyxl)."""
    z = zipfile.ZipFile(path)
    names = z.namelist()

    shared = []
    if 'xl/sharedStrings.xml' in names:
        ss = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in ss.findall(f'{_SS}si'):
            shared.append(''.join(t.text or '' for t in si.iter(f'{_SS}t')))

    sheet_files = sorted(n for n in names if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
    ws = ET.fromstring(z.read(sheet_files[0]))

    grid = []
    for r in ws.iter(f'{_SS}row'):
        cells = {}
        for c in r.findall(f'{_SS}c'):
            col = _col_letters(c.get('r', ''))
            t = c.get('t')
            text = None
            if t == 'inlineStr':
                is_ = c.find(f'{_SS}is')
                if is_ is not None:
                    text = ''.join(x.text or '' for x in is_.iter(f'{_SS}t'))
            else:
                v = c.find(f'{_SS}v')
                if v is not None:
                    text = shared[int(v.text)] if t == 's' else v.text
            cells[col] = text
        grid.append(cells)

    if not grid:
        return pd.DataFrame()

    header = grid[0]
    unnamed = 0
    col_names = {}
    for col, name in sorted(header.items()):
        if name is None or str(name).strip() == '':
            col_names[col] = f'Unnamed: {unnamed}'
            unnamed += 1
        else:
            col_names[col] = str(name)

    records = []
    for row in grid[1:]:
        records.append({col_names.get(col, col): val for col, val in row.items()})
    return pd.DataFrame(records, columns=list(col_names.values()))


def read_xlsx(path):
    """Прочитать первый лист .xlsx в DataFrame (header в первой строке)."""
    try:
        import openpyxl  # noqa: F401
        return pd.read_excel(path)
    except Exception:
        return _read_xlsx_fallback(path)


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #
def _to_float(value):
    """Привести ячейку к float, нормализуя десятичную запятую. Пусто -> None."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    if s == '' or s.lower() == 'nan':
        return None
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        return None


def _is_year(name):
    try:
        y = int(float(str(name)))
    except (ValueError, TypeError):
        return None
    return y if 1900 <= y <= 2100 else None


# --------------------------------------------------------------------------- #
# 1. Временные ряды
# --------------------------------------------------------------------------- #
def migrate_time_series():
    """JSON-ряды -> time_series.csv + time_series_meta.csv."""
    series = {}   # (tid, parm, year, day) -> value   (last-write-wins, дедуп дублей)
    meta = {}     # (tid, parm, year) -> (first_date, last_date)
    name_re = re.compile(r'^(?P<parm>.+)_(?P<reg>[^_]+)_(?P<dist>[^_]+)_(?P<year>\d{4})$')

    files = glob.glob(os.path.join(TIMEROWS_DIR, '*.json'))
    used, skipped = 0, 0
    for path in files:
        base = os.path.basename(path)
        if base.startswith('~$'):
            continue
        stem = base[:-5] if base.endswith('.json') else base
        m = name_re.match(stem)
        if not m:                       # «копия», не-годовые и т.п.
            skipped += 1
            continue
        tid = db.territory_id(m.group('reg'), m.group('dist'))
        parm = m.group('parm')
        year = int(m.group('year'))

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (ValueError, OSError):
            skipped += 1
            continue

        block = data.get('data', {})
        if not block:
            skipped += 1
            continue
        first = block.get('1') or next(iter(block.values()))
        xy = first.get('xy') or []
        if not xy:
            skipped += 1
            continue

        for day_str, value in xy:
            if value is None:
                continue
            series[(tid, parm, year, int(float(day_str)))] = float(value)

        labels = first.get('labels') or {}
        if labels:
            dates = sorted(str(v) for v in labels.values())
            meta[(tid, parm, year)] = (dates[0], dates[-1])
        used += 1

    rows = [{'territory_id': tid, 'parm': parm, 'year': year, 'day': day, 'value': value}
            for (tid, parm, year, day), value in series.items()]
    df = pd.DataFrame(rows, columns=['territory_id', 'parm', 'year', 'day', 'value'])
    df = df.sort_values(['territory_id', 'parm', 'year', 'day']).reset_index(drop=True)
    db.save('time_series', df)

    meta_rows = [{'territory_id': tid, 'parm': parm, 'year': year,
                  'first_date': fd, 'last_date': ld}
                 for (tid, parm, year), (fd, ld) in meta.items()]
    meta_df = pd.DataFrame(meta_rows, columns=['territory_id', 'parm', 'year',
                                               'first_date', 'last_date'])
    db.save('time_series_meta', meta_df)

    print(f'  time_series: файлов учтено {used}, пропущено {skipped}, '
          f'строк {len(df)}, рядов(meta) {len(meta_df)}')
    return df


# --------------------------------------------------------------------------- #
# 2. Территории и урожайности
# --------------------------------------------------------------------------- #
_META_COLS = {'id_country', 'region', 'district', 'culture', 'id_region', 'id_district'}
_IGNORE_COLS = {'Average productivity', 'dispersion'}


def _cell(row, name):
    if name not in row:
        return None
    v = row[name]
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return None if s == '' or s.lower() == 'nan' else s


def migrate_destination_excel():
    """Широкие листы -> territories.csv + yields.csv."""
    territories = {}                    # tid -> record
    yields = {}                         # (tid, culture, year) -> value

    patterns = glob.glob(os.path.join(DESTINATION_DIR, '**', '*.xlsx'), recursive=True)
    files = [p for p in patterns
             if not os.path.basename(p).startswith('~$')
             and '_reserw' not in os.path.basename(p).lower()]

    for path in sorted(files):
        try:
            df = read_xlsx(path)
        except Exception as exc:                       # noqa: BLE001
            print(f'    ! пропуск {os.path.basename(path)}: {exc}')
            continue
        if df.empty or 'id_region' not in df.columns:
            continue

        year_cols = {c: _is_year(c) for c in df.columns}
        year_cols = {c: y for c, y in year_cols.items() if y is not None}

        for _, row in df.iterrows():
            id_region = _cell(row, 'id_region')
            if id_region is None:
                continue
            id_district = _cell(row, 'id_district')
            tid = db.territory_id(id_region, id_district)

            territories[tid] = {
                'territory_id': tid,
                'id_country': _cell(row, 'id_country') or '',
                'id_region': db._norm_id(id_region),
                'id_district': db._norm_id(id_district),
                'region': _cell(row, 'region') or '',
                'district': _cell(row, 'district') or '',
                'level': 'district' if id_district is not None else 'region',
            }

            culture = _cell(row, 'culture')
            if culture is None:
                continue
            for col, year in year_cols.items():
                val = _to_float(row.get(col))
                if val is not None:
                    yields[(tid, culture, year)] = val

    terr_df = pd.DataFrame(list(territories.values()),
                           columns=['territory_id', 'id_country', 'id_region',
                                    'id_district', 'region', 'district', 'level'])
    db.save('territories', terr_df)

    yld_rows = [{'territory_id': tid, 'culture': culture, 'year': year, 'yield': value}
                for (tid, culture, year), value in yields.items()]
    yld_df = pd.DataFrame(yld_rows, columns=['territory_id', 'culture', 'year', 'yield'])
    yld_df = yld_df.sort_values(['territory_id', 'culture', 'year']).reset_index(drop=True)
    db.save('yields', yld_df)

    print(f'  territories: {len(terr_df)} (из {len(files)} файлов), yields: {len(yld_df)}')
    return terr_df, yld_df


# --------------------------------------------------------------------------- #
# 3. Модели
# --------------------------------------------------------------------------- #
def _model_type(name):
    if 'NN' in name:
        return 'NN'
    if 'RF' in name:
        return 'RF'
    if 'EXM' in name:
        return 'EXM'
    if 'LM' in name:
        return 'LM'
    return 'UNKNOWN'


def _model_culture(name):
    parts = re.split(r'_v\d+_', name, maxsplit=1)
    return parts[-1] if len(parts) > 1 else ''


def _weights_file(name, model_dir):
    if 'NN' in name and os.path.exists(os.path.join(model_dir, 'model_scripted.pt')):
        return 'model_scripted.pt'
    if 'RF' in name and os.path.exists(os.path.join(model_dir, 'my_model.pickle')):
        return 'my_model.pickle'
    return ''


def migrate_models():
    """parms.json по папкам моделей -> models.csv + model_params.csv."""
    model_rows = []
    param_rows = []

    for name in sorted(os.listdir(MODELS_DIR)):
        model_dir = os.path.join(MODELS_DIR, name)
        parms_path = os.path.join(model_dir, 'parms.json')
        if not os.path.isdir(model_dir) or not os.path.exists(parms_path):
            continue
        with open(parms_path, 'r', encoding='utf-8') as f:
            parms = json.load(f)

        parm_list = parms.get('parm_list', [])
        normalization = parms.get('normalization', [])
        scalars = set(parms.get('scalars', []) or [])
        borders = parms.get('time_borders', [None, None])

        model_rows.append({
            'model_name': name,
            'model_type': _model_type(name),
            'culture': _model_culture(name),
            'weights_file': _weights_file(name, model_dir),
            'time_border_first': borders[0],
            'time_border_last': borders[1],
        })

        for i, parm in enumerate(parm_list):
            norm = normalization[i] if i < len(normalization) else None
            param_rows.append({
                'model_name': name,
                'param_index': i,
                'parm': parm,
                'normalization': norm,
                'scalar': 1 if parm in scalars else '',
            })
        if len(parm_list) != len(normalization):
            print(f'    ! {name}: parm_list({len(parm_list)}) != normalization({len(normalization)})')

    models_df = pd.DataFrame(model_rows, columns=['model_name', 'model_type', 'culture',
                                                  'weights_file', 'time_border_first',
                                                  'time_border_last'])
    db.save('models', models_df)

    params_df = pd.DataFrame(param_rows, columns=['model_name', 'param_index', 'parm',
                                                  'normalization', 'scalar'])
    db.save('model_params', params_df)

    print(f'  models: {len(models_df)}, model_params: {len(params_df)}')
    return models_df, params_df


# --------------------------------------------------------------------------- #
# 4. Справочники
# --------------------------------------------------------------------------- #
def migrate_reference():
    """Cultures_masks.xlsx -> cultures_masks.csv; Regions_uid_dict.json -> regions.csv."""
    masks_path = os.path.join(COMMON_DIR, 'Cultures_masks.xlsx')
    masks_df = read_xlsx(masks_path)
    masks_df = masks_df.loc[:, [c for c in masks_df.columns if not str(c).startswith('Unnamed')]]
    db.save('cultures_masks', masks_df)

    regions_path = os.path.join(COMMON_DIR, 'Regions_uid_dict.json')
    with open(regions_path, 'r', encoding='utf-8') as f:
        regions = json.load(f)
    regions_df = pd.DataFrame(
        [{'region': k, 'region_uid': str(v)} for k, v in regions.items()],
        columns=['region', 'region_uid'])
    db.save('regions', regions_df)

    print(f'  cultures_masks: {masks_df.shape}, regions: {len(regions_df)}')
    return masks_df, regions_df


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(db.DB_DIR, exist_ok=True)
    print('Миграция данных Nerual Rabbit -> Database/')
    print('1) Временные ряды:')
    migrate_time_series()
    print('2) Территории и урожайности:')
    migrate_destination_excel()
    print('3) Модели:')
    migrate_models()
    print('4) Справочники:')
    migrate_reference()
    db.reload()
    print('Готово.')


if __name__ == '__main__':
    main()
