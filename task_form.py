"""Тонкая форма расчётного задания.

Вместо широкого Excel, где смешаны территории, история урожайностей и колонки
результатов, задание теперь описывается узкой таблицей (CSV или XLSX) только из
того, что нужно ЗАДАТЬ:

    id_region | id_district | culture | model | predict_year

Одна строка — один запрос прогноза (территория × культура × модель × год).
``id_district`` можно оставить пустым для регионального уровня. История
урожайностей и временные ряды берутся из БД (см. :mod:`db`), результаты пишутся
в таблицу ``predictions``, отчёт собирается :mod:`report`.
"""

import os

import numpy as np
import pandas as pd

import db


FORM_COLUMNS = ['id_region', 'id_district', 'culture', 'model', 'predict_year']


def _clean(value):
    if value is None:
        return ''
    if isinstance(value, float) and np.isnan(value):
        return ''
    return str(value).strip()


def load_task_form(path):
    """Прочитать форму задания и собрать структуру задания.

    Возвращает dict::

        {
          'jobs': [ {territory_id, id_region, id_district, culture,
                     model, predict_year}, ... ],   # по строке формы
          'territory_ids': [...],                   # уникальные, в порядке появления
          'model_year_pairs': [(model, year), ...], # уникальные, в порядке появления
          'culture_by_territory': {territory_id: culture},
          'years': None,                            # годы истории для отчёта (по умолч. все)
        }
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path, dtype=str, encoding='utf-8')
    else:
        import migrate  # переиспользуем устойчивый читатель .xlsx
        df = migrate.read_xlsx(path)

    missing = [c for c in ('id_region', 'culture', 'model', 'predict_year')
               if c not in df.columns]
    if missing:
        raise ValueError(f'В форме задания нет обязательных столбцов: {missing}')

    jobs = []
    territory_ids = []
    model_year_pairs = []
    culture_by_territory = {}

    for _, row in df.iterrows():
        id_region = _clean(row.get('id_region'))
        if id_region == '':
            continue
        id_district = _clean(row.get('id_district'))
        culture = _clean(row.get('culture'))
        model = _clean(row.get('model'))
        year_raw = _clean(row.get('predict_year'))
        if model == '' or year_raw == '':
            continue
        predict_year = int(float(year_raw))

        tid = db.territory_id(id_region, id_district)
        jobs.append({
            'territory_id': tid,
            'id_region': id_region,
            'id_district': id_district if id_district else 'nan',
            'culture': culture,
            'model': model,
            'predict_year': predict_year,
        })
        if tid not in territory_ids:
            territory_ids.append(tid)
        if (model, predict_year) not in model_year_pairs:
            model_year_pairs.append((model, predict_year))
        culture_by_territory.setdefault(tid, culture)

    return {
        'jobs': jobs,
        'territory_ids': territory_ids,
        'model_year_pairs': model_year_pairs,
        'culture_by_territory': culture_by_territory,
        'years': None,
    }


def write_sample_form(path, n=3):
    """Сохранить пример формы задания из реальных территорий БД (для шаблона)."""
    terr = db.load('territories')
    yld = db.load('yields')
    rows = []
    if not terr.empty:
        sample = terr[terr['level'] == 'district'].head(n)
        for _, t in sample.iterrows():
            tid = t['territory_id']
            sub = yld[yld['territory_id'] == tid] if not yld.empty else pd.DataFrame()
            culture = sub['culture'].iloc[0] if not sub.empty else 'Пшеница озимая'
            rows.append({
                'id_region': db._norm_id(t['id_region']),
                'id_district': db._norm_id(t['id_district']),
                'culture': culture,
                'model': 'NN_CN_v7_winter_wheat',
                'predict_year': 2025,
            })
    df = pd.DataFrame(rows, columns=FORM_COLUMNS)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f'Пример формы задания: {path} ({len(df)} строк)')
    return df
