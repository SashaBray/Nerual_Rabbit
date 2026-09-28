"""Учётные записи приложения и сессия (вход/выход).

Таблица ``accounts`` в файловой БД: account_id, name, email, ukey, created_at.
При первом запуске пользователь вводит имя/почту/ukey -> создаётся запись и выполняется вход.
Вход сохраняется в таблице ``app_state`` файловой БД (account_id) и держится до явного выхода.
ukey активной учётки используется для запросов к Vega.
"""
import os
from datetime import datetime

import pandas as pd

import config
import db

AVATARS_DIR = os.path.join(config.CONFIG_DIR, 'avatars')
_COLUMNS = ['account_id', 'name', 'email', 'ukey', 'description', 'gender', 'birthdate',
            'avatar', 'created_at']


# --------------------------------------------------------------------------- #
# CRUD учётных записей
# --------------------------------------------------------------------------- #
def list_accounts():
    """Все учётные записи как список dict (без раскрытия — ukey включён, нужен приложению)."""
    df = db.load('accounts')
    return [] if df.empty else df.to_dict('records')


def get_account(account_id):
    df = db.load('accounts')
    if df.empty or account_id is None:
        return None
    hit = df[df['account_id'] == int(account_id)]
    return hit.iloc[0].to_dict() if not hit.empty else None


def find_by_email(email):
    df = db.load('accounts')
    if df.empty or not email:
        return None
    hit = df[df['email'].str.lower() == str(email).strip().lower()]
    return hit.iloc[0].to_dict() if not hit.empty else None


def create_account(name, email, ukey):
    """Создать учётную запись и вернуть её account_id. Почта должна быть уникальной."""
    name, email, ukey = str(name).strip(), str(email).strip(), str(ukey or '').strip()
    if not name or not email:                          # ukey не обязателен: без него прогноз считается по
        raise ValueError('Имя и электронная почта обязательны.')   # данным базы, а план покажет, чего не хватит
    if find_by_email(email):
        raise ValueError(f'Учётная запись с почтой {email} уже существует.')
    df = db.load('accounts')
    next_id = 1 if df.empty else int(df['account_id'].max()) + 1
    rec = {'account_id': next_id, 'name': name, 'email': email, 'ukey': ukey,
           'description': '', 'gender': '', 'birthdate': '', 'avatar': '',
           'created_at': datetime.now().isoformat(timespec='seconds')}
    df = pd.concat([df, pd.DataFrame([rec], columns=_COLUMNS)], ignore_index=True)
    db.save('accounts', df)
    return next_id


def update_account(account_id, **fields):
    """Обновить поля учётки: name,email,ukey,description,gender,birthdate,avatar."""
    df = db.load('accounts').copy()
    if df.empty:
        return
    for col in ('description', 'gender', 'birthdate', 'avatar'):  # старые записи без новых столбцов
        if col not in df.columns:
            df[col] = None
    i = df.index[df['account_id'] == int(account_id)]
    if len(i) == 0:
        return
    for k, v in fields.items():
        if v is None or k not in _COLUMNS:
            continue
        df[k] = df[k].astype('object')                 # иначе запись строки в столбец float64 (all-NaN) падает
        df.loc[i, k] = (str(v).strip() if isinstance(v, str) else v)
    db.save('accounts', df)


def set_avatar(account_id, src_path):
    """Скопировать файл-аватар в config/avatars и сохранить путь в учётке. Вернуть путь."""
    import shutil
    if not src_path or not os.path.exists(src_path):
        return None
    os.makedirs(AVATARS_DIR, exist_ok=True)
    ext = os.path.splitext(src_path)[1].lower() or '.png'
    dest = os.path.join(AVATARS_DIR, f'avatar_{int(account_id)}{ext}')
    shutil.copyfile(src_path, dest)
    update_account(account_id, avatar=dest)
    return dest


# --------------------------------------------------------------------------- #
# Сессия (вход/выход; сохраняется между запусками)
# --------------------------------------------------------------------------- #
_SESSION_KEY = 'current_account_id'


def login(account_id):
    """Запомнить активную учётную запись в БД (таблица ``app_state``), до явного выхода."""
    if get_account(account_id) is None:
        raise ValueError(f'Нет учётной записи с id={account_id}.')
    df = db.load('app_state')
    if not df.empty:
        df = df[df['key'].astype(str) != _SESSION_KEY]
    rec = {'key': _SESSION_KEY, 'value': str(int(account_id))}
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    db.save('app_state', df)


def logout():
    df = db.load('app_state')
    if df.empty:
        return
    db.save('app_state', df[df['key'].astype(str) != _SESSION_KEY])


def current_account_id():
    df = db.load('app_state')
    if df.empty:
        return None
    sub = df[df['key'].astype(str) == _SESSION_KEY]
    if sub.empty:
        return None
    try:
        return int(sub.iloc[-1]['value'])
    except (ValueError, TypeError):
        return None


def current_account():
    """Активная учётка как dict либо None (если не входили или запись удалена)."""
    return get_account(current_account_id())


def is_logged_in():
    return current_account() is not None


def current_ukey():
    """ukey активной учётки; запасной — из ukey.txt."""
    acc = current_account()
    if acc and str(acc.get('ukey') or '').strip():
        return str(acc['ukey']).strip()
    return config.read_ukey()


if __name__ == '__main__':       # быстрый самотест
    n = len(list_accounts())
    print(f'учётных записей в БД: {n}; активная: {current_account()}')
