"""Альтернативное наполнение БД урожайностей из готового свода (Final_result_df).

Если БДПМО уже сведён в широкий файл со столбцами
``region, district, culture, <годы...>, id_region, id_district`` (pickle или xlsx),
этот скрипт разворачивает его в длинную ``yields.csv`` + ``territories.csv`` без
повторного парсинга ``.xls`` (не требует ``xlrd``). Для регенерации с нуля из
сырых ``PMODB_init/*.xls`` используйте :mod:`ingest_yields`.

Запуск:  ``python ingest_yields_from_df.py <path_to_df.(pkl|xlsx)>``
"""

import os
import sys

import numpy as np
import pandas as pd

import config
import db


def _is_year(name):
    try:
        y = int(float(str(name)))
    except (ValueError, TypeError):
        return None
    return y if 1900 <= y <= 2100 else None


def ingest(path):
    config.ensure_dirs()
    if path.lower().endswith('.pkl'):
        df = pd.read_pickle(path)
    else:
        from xlsx_read import read_xlsx
        df = read_xlsx(path)

    year_cols = {c: _is_year(c) for c in df.columns}
    year_cols = {c: y for c, y in year_cols.items() if y is not None}

    territories = {}
    yields_rows = []
    for _, row in df.iterrows():
        id_region = db._norm_id(row.get('id_region'))
        id_district = db._norm_id(row.get('id_district'))
        if id_region == 'nan':
            continue
        tid = db.territory_id(id_region, id_district)
        territories[tid] = {
            'territory_id': tid, 'id_country': '',
            'id_region': id_region, 'id_district': id_district,
            'region': str(row.get('region', '') or ''),
            'district': str(row.get('district', '') or ''),
            'level': 'district' if id_district != 'nan' else 'region',
        }
        culture = str(row.get('culture', '') or '')
        for col, year in year_cols.items():
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            try:
                v = float(str(val).replace(',', '.'))
            except (ValueError, TypeError):
                continue
            if v < 0 or v > 5000.0:        # сентинел пропуска БДПМО (999999999) / мусор
                continue
            yields_rows.append({'territory_id': tid, 'culture': culture,
                                'year': year, 'yield': v})

    terr_df = pd.DataFrame(list(territories.values()),
                           columns=['territory_id', 'id_country', 'id_region',
                                    'id_district', 'region', 'district', 'level'])
    db.save('territories', terr_df)

    yld_df = pd.DataFrame(yields_rows, columns=['territory_id', 'culture', 'year', 'yield'])
    yld_df = yld_df.drop_duplicates(['territory_id', 'culture', 'year'], keep='last')
    yld_df = yld_df.sort_values(['territory_id', 'culture', 'year']).reset_index(drop=True)
    db.save('yields', yld_df)

    print(f'territories: {len(terr_df)}, yields: {len(yld_df)} '
          f'(культур: {yld_df["culture"].nunique() if not yld_df.empty else 0})')
    return yld_df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Использование: python ingest_yields_from_df.py <path_to_df.(pkl|xlsx)>')
        sys.exit(1)
    ingest(sys.argv[1])
