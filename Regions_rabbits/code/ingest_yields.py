"""Наполнение БД урожайностей из исходных файлов БДПМО (PMODB_init/*.xls).

Порт логики ``Read_bdpmo.py`` с вынесенными во внешние конфиги параметрами:
* культуры для поиска     — ``workspace/config/bdpmo_cultures.csv`` (столбец ``culture``)
* триггеры района/региона — ``workspace/config/bdpmo_triggers.json``
* порог похожести и пр.   — ``workspace/config/ingest.json``

Стадии: парс .xls -> очистка спецзначения -> fuzzy-сопоставление район->id по
``Regions_and_dist_tree.json`` + ``Regions_uid_dict.json`` -> длинная таблица
``yields.csv`` (territory_id, culture, year, yield) + ``territories.csv``.

Требует ``xlrd`` для чтения старых ``.xls``.  Запуск:  ``python ingest_yields.py``
"""

import difflib
import json
import os
import re

import numpy as np
import pandas as pd

import config
import db


def _load_cfg():
    cultures = list(pd.read_csv(config.BDPMO_CULTURES_FILE, dtype=str, encoding='utf-8-sig')['culture'])
    with open(config.BDPMO_TRIGGERS_FILE, 'r', encoding='utf-8-sig') as f:
        triggers = json.load(f)
    with open(config.INGEST_FILE, 'r', encoding='utf-8-sig') as f:
        ingest = json.load(f)
    return cultures, triggers, ingest


def _is_region_heading(heading, dist_triggers, reg_triggers):
    s = str(heading)
    for rgx in reg_triggers:
        if re.compile(str(rgx)).search(s):
            return True
    return any(part in dist_triggers for part in s.split(' '))


def _read_one_xls(path, sheet_names):
    last_err = None
    for sheet in sheet_names + [os.path.splitext(os.path.basename(path))[0]]:
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except Exception as exc:                       # noqa: BLE001
            last_err = exc
    raise last_err


def parse_bdpmo(cultures, triggers, ingest):
    """Разобрать все .xls -> длинный DataFrame [region, district, culture, year, yield]."""
    dist_triggers = triggers.get('district', ['район', 'город'])
    reg_triggers = triggers.get('region', ['ский', 'СКИЙ'])
    sheet_names = ingest.get('sheet_names', ['Первый лист', 'Report'])
    missing_value = ingest.get('missing_value', 999999999.0)
    yield_max = ingest.get('yield_max', 5000.0)   # ц/га; больше — сентинел пропуска/мусор

    records = []
    for fname in sorted(os.listdir(config.PMODB_DIR)):
        if not fname.lower().endswith(('.xls', '.xlsx')):
            continue
        region = os.path.splitext(fname)[0]
        excel = _read_one_xls(os.path.join(config.PMODB_DIR, fname), sheet_names)

        year_row = list(excel.iloc[0])[1:]
        if str(year_row[-1]) == 'nan':
            year_row = list(excel.iloc[1])[1:]

        dist_name = None
        for i in range(len(excel)):
            row = list(excel.iloc[i])
            heading = row[0]
            if _is_region_heading(heading, dist_triggers, reg_triggers):
                dist_name = heading
            elif heading in cultures and dist_name is not None:
                culture = heading
                for year, value in zip(year_row, row[1:]):
                    try:
                        y = int(float(str(year)))
                        v = float(str(value).replace(',', '.'))
                    except (ValueError, TypeError):
                        continue
                    if v == missing_value or v < 0 or v > yield_max:
                        continue            # пропуск/сентинел/мусор — не в БД
                    records.append({'region': region, 'district': dist_name,
                                    'culture': culture, 'year': y, 'yield': v})
    return pd.DataFrame(records)


def _similarity(a, b):
    return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _clean_dist(name):
    s = str(name)
    for token in ('муниципальный', 'Муниципальный', 'район'):
        s = s.replace(token, '')
    s = re.sub(r'\([\s\S]+?\)', '', s)
    return s.strip()


def build_id_resolver(threshold):
    """Возвращает (region->id_region, (region,district)->id_district) по дереву."""
    with open(config.REGIONS_TREE_FILE, 'r', encoding='utf-8-sig') as f:
        tree = json.load(f)
    with open(config.REGIONS_UID_FILE, 'r', encoding='utf-8-sig') as f:
        reg_uid = json.load(f)
    root = tree.get('Российская Федерация', tree)

    def resolve(region, district):
        id_region = reg_uid.get(region)
        id_district = None
        if region in root and isinstance(root[region], dict):
            best, best_name = 0.0, None
            target = _clean_dist(district)
            for cand, payload in root[region].items():
                score = _similarity(target, _clean_dist(cand))
                if score > best:
                    best, best_name = score, cand
            if best_name is not None and best > threshold:
                payload = root[region][best_name]
                if isinstance(payload, dict) and 'Number' in payload:
                    id_district = payload['Number']
        return id_region, id_district

    return resolve


def ingest_yields():
    config.ensure_dirs()
    try:
        import xlrd  # noqa: F401  — нужен для чтения старого формата .xls (BIFF)
    except ImportError:
        print('Для чтения PMODB *.xls нужен пакет xlrd. Установите его:')
        print('    pip install xlrd')
        print('(если есть готовый свод по культуре в .pkl/.xlsx — можно обойтись '
              'ingest_yields_from_df.py)')
        return None
    cultures, triggers, ingest = _load_cfg()
    threshold = ingest.get('fuzzy_threshold', 0.65)

    df = parse_bdpmo(cultures, triggers, ingest)
    if df.empty:
        print('Не найдено ни одной записи урожайности.')
        return df

    resolve = build_id_resolver(threshold)

    territories = {}
    yields_rows = []
    cache = {}
    for _, r in df.iterrows():
        key = (r['region'], r['district'])
        if key not in cache:
            cache[key] = resolve(r['region'], r['district'])
        id_region, id_district = cache[key]
        if id_region is None:
            continue
        tid = db.territory_id(id_region, id_district)
        territories[tid] = {
            'territory_id': tid,
            'id_country': '',
            'id_region': db._norm_id(id_region),
            'id_district': db._norm_id(id_district),
            'region': r['region'],
            'district': r['district'],
            'level': 'district' if id_district is not None else 'region',
        }
        yields_rows.append({'territory_id': tid, 'culture': r['culture'],
                            'year': int(r['year']), 'yield': float(r['yield'])})

    terr_df = pd.DataFrame(list(territories.values()),
                           columns=['territory_id', 'id_country', 'id_region',
                                    'id_district', 'region', 'district', 'level'])
    db.save('territories', terr_df)

    yld_df = pd.DataFrame(yields_rows, columns=['territory_id', 'culture', 'year', 'yield'])
    yld_df = yld_df.drop_duplicates(['territory_id', 'culture', 'year'], keep='last')
    yld_df = yld_df.sort_values(['territory_id', 'culture', 'year']).reset_index(drop=True)
    db.save('yields', yld_df)

    print(f'territories: {len(terr_df)}, yields: {len(yld_df)} '
          f'(культур: {yld_df["culture"].nunique()})')
    return yld_df


if __name__ == '__main__':
    ingest_yields()
