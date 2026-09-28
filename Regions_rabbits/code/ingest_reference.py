"""Наполнение справочных таблиц БД из исходных файлов.

* ``Cultures_masks.xlsx`` -> ``workspace/database/cultures_masks.csv`` (маски NDVI по культурам)
* ``Regions_uid_dict.json`` -> ``workspace/database/regions.csv`` (имя региона -> uid)

Запуск:  ``python ingest_reference.py``
"""

import json

import pandas as pd

import config
import db
from xlsx_read import read_xlsx


def ingest_masks():
    df = read_xlsx(config.CULTURES_MASKS_XLSX)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith('Unnamed')]]
    db.save('cultures_masks', df)
    print(f'cultures_masks: {df.shape}, столбцы: {list(df.columns)}')
    return df


def ingest_regions():
    with open(config.REGIONS_UID_FILE, 'r', encoding='utf-8-sig') as f:
        regions = json.load(f)
    df = pd.DataFrame([{'region': k, 'region_uid': str(v)} for k, v in regions.items()],
                      columns=['region', 'region_uid'])
    db.save('regions', df)
    print(f'regions: {len(df)}')
    return df


def main():
    config.ensure_dirs()
    ingest_masks()
    ingest_regions()


if __name__ == '__main__':
    main()
