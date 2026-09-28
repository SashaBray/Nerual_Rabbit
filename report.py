"""Генерация отчёта (широкий лист) из CSV-таблиц БД — в Excel или CSV.

Хранение и представление разделены: данные лежат в ``Database/`` (см. :mod:`db`),
а отчёт собирается по запросу. Макет повторяет исторический широкий лист:

    id_country | region | district | culture | id_region | id_district |
    <год> ... <год> | Average productivity |
    <model>_predict_<year> ... | dispersion

Историю урожайностей берём из ``yields``, прогнозы — из свежайших записей
``predictions`` (по ``made_at``). Конвенция имён столбцов ``_predict_`` живёт
только здесь, на стороне вывода.
"""

import os

import numpy as np
import pandas as pd

import db


_META_COLUMNS = ['id_country', 'region', 'district', 'culture', 'id_region', 'id_district']


def _territory_meta(tid, culture):
    """Метастолбцы территории для строки отчёта."""
    rec = db.get_territory(tid) or {}
    return {
        'id_country': rec.get('id_country', ''),
        'region': rec.get('region', ''),
        'district': rec.get('district', ''),
        'culture': culture,
        'id_region': db._norm_id(rec.get('id_region')),
        'id_district': db._norm_id(rec.get('id_district')),
    }


def build_frame(territory_ids, years=None, model_year_pairs=None, culture_by_territory=None):
    """Собрать DataFrame отчёта.

    :param territory_ids: список ``territory_id``.
    :param years: годы для столбцов истории; ``None`` — объединение всех годов
        из ``yields`` по выбранным территориям.
    :param model_year_pairs: список пар ``(model_name, predict_year)`` —
        какие столбцы прогнозов добавить.
    :param culture_by_territory: ``{territory_id: culture}``; если не задано,
        культура берётся из таблицы ``yields`` (первая по территории).
    """
    model_year_pairs = model_year_pairs or []
    culture_by_territory = culture_by_territory or {}

    yld = db.load('yields')

    def culture_of(tid):
        if tid in culture_by_territory:
            return culture_by_territory[tid]
        sub = yld[yld['territory_id'] == tid] if not yld.empty else pd.DataFrame()
        return sub['culture'].iloc[0] if not sub.empty else ''

    # Набор годов истории
    if years is None:
        if yld.empty:
            years = []
        else:
            years = sorted(int(y) for y in
                           yld[yld['territory_id'].isin(territory_ids)]['year'].unique())
    years = list(years)

    predict_columns = [f'{model}_predict_{year}' for model, year in model_year_pairs]

    rows = []
    for tid in territory_ids:
        culture = culture_of(tid)
        row = _territory_meta(tid, culture)

        year_yields = db.get_yields(tid, culture)
        for y in years:
            row[y] = year_yields.get(int(y))

        present = [v for v in year_yields.values() if v is not None]
        row['Average productivity'] = float(np.mean(present)) if present else None

        dispersion = None
        for (model, predict_year), col in zip(model_year_pairs, predict_columns):
            pred = db.get_prediction(tid, culture, model, predict_year)
            row[col] = pred['value'] if pred else None
            if pred and pred.get('dispersion') is not None and not (
                    isinstance(pred['dispersion'], float) and np.isnan(pred['dispersion'])):
                dispersion = pred['dispersion']
        row['dispersion'] = dispersion

        rows.append(row)

    columns = _META_COLUMNS + years + ['Average productivity'] + predict_columns + ['dispersion']
    return pd.DataFrame(rows, columns=columns)


def build_report(out_path, territory_ids, years=None, model_year_pairs=None,
                 culture_by_territory=None):
    """Собрать отчёт и сохранить.

    Формат выбирается по расширению ``out_path``: ``.csv`` -> CSV (без зависимостей),
    ``.xlsx``/``.xls`` -> Excel (требует openpyxl в окружении). В отчёте нет формул,
    поэтому CSV полностью равноценен.
    """
    frame = build_frame(territory_ids, years, model_year_pairs, culture_by_territory)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    ext = os.path.splitext(out_path)[1].lower()
    if ext == '.csv':
        frame.to_csv(out_path, index=False, encoding='utf-8')
    else:
        frame.to_excel(out_path)

    print(f'Отчёт сохранён: {out_path}  ({frame.shape[0]} строк, {frame.shape[1]} столбцов)')
    return frame


def build_report_from_task(task, out_path):
    """Собрать отчёт по расчётному заданию (см. :mod:`task_form`).

    ``task`` — dict с ключами ``territory_ids``, ``model_year_pairs``,
    ``culture_by_territory`` (как возвращает ``task_form.load_task_form``).
    """
    return build_report(
        out_path,
        territory_ids=task['territory_ids'],
        years=task.get('years'),
        model_year_pairs=task['model_year_pairs'],
        culture_by_territory=task.get('culture_by_territory'),
    )
