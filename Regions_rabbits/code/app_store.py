"""Хранилище приложения: постоянные списки интересующих районов/регионов и выборка прогнозов.

Таблица ``user_lists``: list_id, account_id, kind('region'|'district'), id_region, id_district,
territory_id, label, added_at. Прогнозы лежат в таблице ``predictions`` (см. db.upsert_prediction).
"""
import json
from datetime import datetime

import pandas as pd

import db

_LIST_COLUMNS = ['list_id', 'account_id', 'kind', 'id_region', 'id_district',
                 'territory_id', 'label', 'added_at']


# --------------------------------------------------------------------------- #
# Списки интересов пользователя (районы / регионы)
# --------------------------------------------------------------------------- #
def add_list_item(account_id, kind, id_region, id_district=None, label=None):
    """Добавить регион/район в постоянный список пользователя (без дублей по территории)."""
    if kind not in ('region', 'district'):
        raise ValueError("kind должен быть 'region' или 'district'")
    tid = db.territory_id(id_region, id_district if kind == 'district' else None)
    df = db.load('user_lists')
    if not df.empty:
        dup = df[(df['account_id'] == int(account_id)) & (df['territory_id'] == tid)]
        if not dup.empty:
            return int(dup.iloc[0]['list_id'])         # уже в списке
    next_id = 1 if df.empty else int(df['list_id'].max()) + 1
    rec = {'list_id': next_id, 'account_id': int(account_id), 'kind': kind,
           'id_region': int(id_region),
           'id_district': None if kind == 'region' or id_district is None else int(id_district),
           'territory_id': tid, 'label': None if label is None else str(label),
           'added_at': datetime.now().isoformat(timespec='seconds')}
    df = pd.concat([df, pd.DataFrame([rec], columns=_LIST_COLUMNS)], ignore_index=True)
    db.save('user_lists', df)
    return next_id


def list_items(account_id, kind=None):
    """Сохранённые элементы списка пользователя (по желанию — только region/district)."""
    df = db.load('user_lists')
    if df.empty:
        return []
    sub = df[df['account_id'] == int(account_id)]
    if kind is not None:
        sub = sub[sub['kind'] == kind]
    return sub.sort_values('added_at').to_dict('records')


def remove_item(list_id):
    df = db.load('user_lists')
    if df.empty:
        return
    df = df[df['list_id'] != int(list_id)]
    db.save('user_lists', df)


# --------------------------------------------------------------------------- #
# Выборка прогнозов (для вкладки «Обзор прогнозов»)
# --------------------------------------------------------------------------- #
def list_predictions(account_id=None, culture=None, territory_id=None, predict_year=None,
                     model_name=None):
    """Прогнозы с фильтрами, отсортированы по времени (свежие сверху), с именем автора."""
    df = db.load('predictions')
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if account_id is not None and 'account_id' in df:
        df = df[df['account_id'] == int(account_id)]
    if culture is not None:
        df = df[df['culture'] == str(culture)]
    if territory_id is not None:
        df = df[df['territory_id'] == str(territory_id)]
    if predict_year is not None:
        df = df[df['predict_year'] == int(predict_year)]
    if model_name is not None:
        df = df[df['model_name'] == str(model_name)]
    # имя автора
    acc = db.load('accounts')
    if not acc.empty and 'account_id' in df:
        names = dict(zip(acc['account_id'], acc['name']))
        df['автор'] = df['account_id'].map(lambda a: names.get(a) if a == a else None)
    return df.sort_values('made_at', ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Шаблоны территорий (сохранённые наборы для прогноза)
# --------------------------------------------------------------------------- #
def list_templates():
    """Названия сохранённых шаблонов (по алфавиту). Хранятся в таблице ``templates`` БД."""
    df = db.load('templates')
    if df.empty:
        return []
    return sorted(df['name'].astype(str).unique())


def template_exists(name):
    df = db.load('templates')
    return (not df.empty) and (df['name'].astype(str) == str(name)).any()


def save_template(name, items):
    """Сохранить шаблон территорий в БД: items = [(id_region, id_district|None, label), ...].

    items хранится JSON-строкой в ячейке таблицы — отдельных JSON-файлов не создаётся.
    """
    df = db.load('templates')
    if not df.empty:                                       # перезапись по имени
        df = df[df['name'].astype(str) != str(name)]
    next_id = 1 if df.empty else int(df['template_id'].max()) + 1
    items_json = json.dumps([[int(r) if r is not None else None,
                              int(d) if d is not None else None, str(lab)] for r, d, lab in items],
                            ensure_ascii=False)
    rec = {'template_id': next_id, 'name': str(name), 'items': items_json,
           'saved_at': datetime.now().isoformat(timespec='seconds')}
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    db.save('templates', df)


def load_template(name):
    """Вернуть список (id_region, id_district|None, label) из шаблона (из БД)."""
    df = db.load('templates')
    if df.empty:
        return []
    sub = df[df['name'].astype(str) == str(name)]
    if sub.empty:
        return []
    items = json.loads(sub.iloc[-1]['items'])
    return [(it[0], it[1], it[2]) for it in items]


def prediction_history(territory_id, culture, predict_year, model_name=None):
    """Вся история прогнозов для (территория, культура, год) — для графика истории прогноза."""
    df = list_predictions(territory_id=territory_id, culture=culture, predict_year=predict_year,
                          model_name=model_name)
    return df.sort_values('made_at') if not df.empty else df


# --------------------------------------------------------------------------- #
# Партии прогнозов (один запуск по списку территорий = одна партия)
# --------------------------------------------------------------------------- #
def list_forecast_batches(account_id=None):
    """Список партий прогноза: id, когда, автор, культура, число территорий, комментарий."""
    df = db.load('predictions')
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if 'batch_id' not in df:
        df['batch_id'] = None
    df['batch_id'] = df['batch_id'].fillna('—')
    if account_id is not None and 'account_id' in df:
        df = df[df['account_id'] == int(account_id)]
    acc = db.load('accounts')
    names = dict(zip(acc['account_id'], acc['name'])) if not acc.empty else {}
    rows = []
    for bid, g in df.groupby('batch_id'):
        aid = g['account_id'].dropna().iloc[0] if ('account_id' in g and g['account_id'].notna().any()) else None
        rows.append({'batch_id': bid, 'когда': g['made_at'].min(),
                     'автор': names.get(aid) if aid is not None else None,
                     'культура': ', '.join(sorted(map(str, g['culture'].dropna().unique()))),
                     'территорий': int(g['territory_id'].nunique()), 'прогнозов': int(len(g)),
                     'комментарий': (g['comment'].dropna().iloc[0] if g['comment'].notna().any() else '')})
    return pd.DataFrame(rows).sort_values('когда', ascending=False).reset_index(drop=True)


def batch_predictions(batch_id):
    """Все прогнозы партии."""
    df = db.load('predictions')
    if df.empty:
        return df
    df = df.copy()
    if 'batch_id' not in df:
        df['batch_id'] = None
    return df[df['batch_id'].fillna('—') == str(batch_id)].copy()
