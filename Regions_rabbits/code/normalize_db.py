"""Нормализация БД временных рядов ради минимума памяти.

Преобразует таблицы СТАРОЙ схемы
    time_series.csv       : territory_id, parm, year, day, value   (повтор текста)
    time_series_meta.csv  : territory_id, parm, year, first_date, last_date
в НОВУЮ нормализованную схему
    parms.csv             : parm_id, parm
    series.csv            : series_id, territory_id, parm_id, year, first_date, last_date
    time_series.csv       : series_id, day, value                  (только числа)

Повторяющийся текст ``parm`` и повторяющиеся ключи ``territory_id``/``year``
выносятся в маленькие справочники; гигантская таблица сводится к числам.
Идемпотентна: если ``time_series.csv`` уже нормализован (есть ``series_id``),
ничего не делает.

Запуск:  ``python normalize_db.py``
"""

import os

import pandas as pd

import config
import db


def normalize():
    ts_path = db.path_of('time_series')
    if not os.path.exists(ts_path):
        print('time_series.csv не найден — нечего нормализовать.')
        return

    header = pd.read_csv(ts_path, nrows=0).columns.tolist()
    if 'series_id' in header:
        print('БД уже нормализована (есть series_id). Пропуск.')
        return
    print(f'Старый размер time_series.csv: {round(os.path.getsize(ts_path)/1e6,1)} МБ')

    # category для текстовых ключей -> компактно в RAM
    old = pd.read_csv(ts_path, dtype={'territory_id': 'category', 'parm': 'category'})

    # parms: справочник параметров
    parm_cats = list(pd.Categorical(old['parm']).categories)
    parms = pd.DataFrame({'parm_id': range(len(parm_cats)), 'parm': parm_cats})
    pmap = {name: pid for pid, name in zip(parms['parm_id'], parms['parm'])}
    old['parm_id'] = old['parm'].map(pmap).astype('int32')

    # series_id = номер группы (territory_id, parm_id, year)
    old['series_id'] = old.groupby(['territory_id', 'parm_id', 'year'],
                                   sort=False, observed=True).ngroup().astype('int32')

    # таблица series (первое вхождение каждой группы) + даты из meta
    series = (old.drop_duplicates('series_id')[['series_id', 'territory_id', 'parm_id', 'year']]
              .reset_index(drop=True))
    series['territory_id'] = series['territory_id'].astype(str)

    meta_path = db.path_of('time_series_meta')
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path, dtype={'territory_id': str, 'parm': str})
        meta['parm_id'] = meta['parm'].map(pmap)
        meta = meta.dropna(subset=['parm_id'])
        meta['parm_id'] = meta['parm_id'].astype('int32')
        series = series.merge(meta[['territory_id', 'parm_id', 'year', 'first_date', 'last_date']],
                              on=['territory_id', 'parm_id', 'year'], how='left')
    if 'first_date' not in series.columns:
        series['first_date'] = ''
        series['last_date'] = ''
    series[['first_date', 'last_date']] = series[['first_date', 'last_date']].fillna('')

    points = old[['series_id', 'day', 'value']].sort_values(['series_id', 'day']).reset_index(drop=True)

    db.save('parms', parms)
    db.save('series', series.sort_values('series_id').reset_index(drop=True))
    db.save('time_series', points)
    if os.path.exists(meta_path):
        os.remove(meta_path)        # объединена в series.csv

    print(f'parms: {len(parms)}, series: {len(series)}, points: {len(points)}')
    print(f'Новый размер time_series.csv: {round(os.path.getsize(ts_path)/1e6,1)} МБ '
          f'(+ series.csv {round(os.path.getsize(db.path_of("series"))/1e6,2)} МБ)')


if __name__ == '__main__':
    config.ensure_dirs()
    normalize()
