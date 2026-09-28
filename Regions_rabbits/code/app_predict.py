"""Движок прогнозов: единый инференс для одной территории/культуры/года всеми моделями.

Модели: нейросеть (стандартная *_all_nn), случайный лес, линейная (Ridge), классическая
регрессия по NDVI, ансамбль (среднее доступных). Все строят прогноз из БД (временные ряды),
поэтому работают и для текущего года (после докачки свежих рядов из Vega — см. app_freshness).

Главная функция: predict_for_territory(id_region, id_district, culture_title, year, models).
"""
import datetime
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

import build_dataset as B
import config
import db
import explore_models as E
import tasks as tasks_mod
import train_nn as T
from ndvi_regression import fit_exp, pred_exp

MODEL_KEYS = ['nn', 'rf', 'linear', 'ndvi', 'ensemble']
MODEL_NAMES = {'nn': 'Нейросеть', 'rf': 'Случайный лес', 'linear': 'Линейная (Ridge)',
               'ndvi': 'Классическая регрессия по NDVI', 'ensemble': 'Ансамбль'}
_KIND_ORDER = {'nn': 0, 'rf': 1, 'linear': 2, 'ndvi': 3, 'ensemble': 4}
_FILE_KINDS = ('nn', 'rf', 'linear')          # модели-файлы (могут иметь несколько версий)
L_SKLEARN = 50
_cache = {}                                   # кэш: задания, модели, sklearn-фиты


# --------------------------------------------------------------------------- #
# Токены моделей (плоская структура models/):
#   модель-файл (nn/rf/linear) — токен РАВЕН имени её папки в models/, напр.
#   'winter_wheat_all_nn_v4'; 'ndvi'/'ensemble' — особые токены (без папки).
#   Поддерживаются и старые токены ('nn@v4', 'nn') для совместимости со старыми прогнозами.
# --------------------------------------------------------------------------- #
_FILE_RE = re.compile(r'^(.+)_(nn|rf|linear)_v(\d+)$')      # <task>_<kind>_v<N> (имя папки)


def parse_token(token):
    """Вернуть (kind, version|None). Понимает имя-папки, старое 'nn@v4' и голый 'nn'."""
    s = str(token)
    if s in ('ndvi', 'ensemble'):
        return s, None
    m = _FILE_RE.match(s)                                  # имя папки <task>_<kind>_v<N>
    if m:
        return m.group(2), int(m.group(3))
    if '@v' in s:                                          # старый формат 'nn@v4'
        k, v = s.split('@v', 1)
        return k, (int(v) if v.isdigit() else None)
    return s, None                                         # голый 'nn'/'rf'/'linear'


def _model_file(kind):
    return 'model.pt' if kind == 'nn' else 'model.pkl'


def _token_dir(token):
    """Путь к папке модели-файла по токену (== имя папки) или None для ndvi/ensemble/голых."""
    if token in ('ndvi', 'ensemble', 'nn', 'rf', 'linear'):
        return None
    d = os.path.join(config.MODELS_DIR, str(token))
    return d if os.path.isdir(d) else None


def token_label(token):
    """Имя модели для UI/отчётов: для nn/rf/linear — имя папки; для ndvi/ensemble — читаемое."""
    if token in ('ndvi', 'ensemble'):
        return MODEL_NAMES[token]
    if _token_dir(token):                                  # токен == имя папки
        return str(token)
    kind, v = parse_token(token)                           # старые/голые токены — читаемо
    base = MODEL_NAMES.get(kind, str(token))
    return base if v is None else f'{base} v{v}'


def token_sort_key(token):
    kind, v = parse_token(token)
    return (_KIND_ORDER.get(kind, 9), v or 0)


def _model_dirs_for(task_id, kind):
    """Папки моделей-файлов культуры для (task, kind): [(имя_папки, версия)], версии по убыванию."""
    out = []
    pref = f'{task_id}_{kind}_v'
    if os.path.isdir(config.MODELS_DIR):
        for d in os.listdir(config.MODELS_DIR):
            if d.startswith(pref) and d[len(pref):].isdigit():
                if os.path.exists(os.path.join(config.MODELS_DIR, d, _model_file(kind))):
                    out.append((d, int(d[len(pref):])))
    return sorted(out, key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- #
# Резолвер культур
# --------------------------------------------------------------------------- #
def _all_tasks_by_culture():
    if 'tasks_by_culture' not in _cache:
        df = pd.read_csv(config.DATASET_TASKS_FILE)
        m = {}
        for _, r in df.iterrows():
            tid = str(r['task_id'])
            if tid.endswith('_all'):           # рабочее задание культуры
                m[int(r['culture_id'])] = tid
        _cache['tasks_by_culture'] = m
    return _cache['tasks_by_culture']


def _cultures_df():
    return pd.read_csv(config.CULTURES_FILE)


def list_cultures():
    """Список названий культур (для выпадающего списка), у которых есть рабочее задание."""
    cdf = _cultures_df(); tb = _all_tasks_by_culture()
    return [r['title'] for _, r in cdf.iterrows() if int(r['culture_id']) in tb]


def _harvest_doy(task_id):
    df = pd.read_csv(os.path.join(config.CONFIG_DIR, 'harvest_dates.csv'))
    hit = df[df['task'] == task_id]
    return int(hit.iloc[0]['harvest_doy']) if not hit.empty else 210


def _latest_nn_dir(task_id):
    """Путь к папке последней версии нейросети культуры (плоская структура) или None."""
    dirs = _model_dirs_for(task_id, 'nn')
    return os.path.join(config.MODELS_DIR, dirs[0][0]) if dirs else None


def resolve_culture(title):
    """Вернуть конфигурацию культуры: задание, маска, NN-модель, окно уборки."""
    cdf = _cultures_df()
    hit = cdf[cdf['title'] == title]
    if hit.empty:
        raise ValueError(f'Неизвестная культура: {title}')
    row = hit.iloc[0]
    cid = int(row['culture_id'])
    task_id = _all_tasks_by_culture().get(cid)
    if task_id is None:
        raise ValueError(f'Нет рабочего задания для культуры {title}')
    return {'culture_id': cid, 'title': title, 'bdpmo_culture': row['bdpmo_culture'],
            'mask_culture': row['mask_culture'], 'task_id': task_id,
            'nn_dir': _latest_nn_dir(task_id), 'harvest_doy': _harvest_doy(task_id)}


def _task(task_id):
    if ('task', task_id) not in _cache:
        _cache[('task', task_id)] = tasks_mod.load_task(task_id)
    return _cache[('task', task_id)]


def available_models(culture_title):
    """Только реально доступные модели культуры. Токен модели-файла == имя её папки в models/.

    Нейросеть / случайный лес / линейная — по токену на каждую сохранённую папку культуры;
    классическая регрессия по NDVI — всегда (на лету); ансамбль — всегда.
    Обучить отсутствующие RF/линейную можно во вкладке «Модели» (``train_sklearn``).
    """
    cfg = resolve_culture(culture_title)
    tid = cfg['task_id']; out = []
    for kind in ('nn', 'rf', 'linear'):
        out += [name for name, _v in _model_dirs_for(tid, kind)]   # токен = имя папки
    out += ['ndvi', 'ensemble']
    return out


def train_sklearn(culture_title, kinds=('rf', 'linear')):
    """Обучить и сохранить (папка <task>_<kind>_v1) случайный лес / линейную. -> [(kind, статус), ...]."""
    cfg = resolve_culture(culture_title); res = []
    for kind in kinds:
        if _model_dirs_for(cfg['task_id'], kind):
            res.append((kind, 'уже обучена')); continue
        _cache.pop(('sk', _sk_default_dir(cfg['task_id'], kind)), None)
        _fit_one_sklearn(cfg, kind, None)                 # обучает на датасете культуры и сохраняет папку
        res.append((kind, 'обучена (v1)'))
    return res


# --------------------------------------------------------------------------- #
# Заметка пользователя о модели
#
# Хранится РЯДОМ С МОДЕЛЬЮ — простым файлом ``comment.txt`` в её папке: так заметка
# уезжает вместе с папкой модели (копирование, архив, передача коллеге) и правится
# блокнотом. Метаданные обучения (``details.json`` у нейросети, ``meta.json`` у
# sklearn-моделей) при этом не трогаются.
# --------------------------------------------------------------------------- #
MODEL_COMMENT_FILE = 'comment.txt'


def model_dir(token):
    """Папка модели-файла по токену (== имя папки) либо ``None`` для ndvi/ensemble."""
    return _token_dir(token)


def get_model_comment(token):
    """Заметка о модели (текст из ``<папка модели>/comment.txt``); нет файла -> ``''``."""
    d = _token_dir(token)
    if not d:
        return ''
    path = os.path.join(d, MODEL_COMMENT_FILE)
    if not os.path.exists(path):
        return ''
    try:
        with open(path, encoding='utf-8-sig') as f:
            return f.read().strip()
    except OSError:
        return ''


def set_model_comment(token, text):
    """Записать заметку в папку модели; пустой текст удаляет файл. -> путь к файлу или ``None``."""
    d = _token_dir(token)
    if not d:
        raise ValueError('У этой модели нет папки (ndvi/ансамбль считаются на лету) — '
                         'заметку хранить негде.')
    path = os.path.join(d, MODEL_COMMENT_FILE)
    text = (text or '').strip()
    if not text:
        if os.path.exists(path):
            os.remove(path)
        return None
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def model_autonote(token):
    """Краткая справка о модели из её метаданных — подсказка при написании заметки.

    Нейросеть (``details.json``): архитектура, длительность блока, окно урожайности, эпохи,
    лучшая val-MSE, сид, сплит. sklearn (``meta.json``): тип, число обучающих примеров, сжатие.
    Пусто, если метаданных нет.
    """
    d = _token_dir(token)
    if not d:
        return ''
    parts = []
    det = os.path.join(d, 'details.json')
    meta = os.path.join(d, 'meta.json')
    try:
        if os.path.exists(det):
            j = json.load(open(det, encoding='utf-8'))
            parts.append(f"задание {j.get('task')}")
            if j.get('arch'):
                parts.append(f"архитектура {j['arch']}")
            if j.get('duration_years'):
                parts.append(f"{j['duration_years']} г. рядов")
            if j.get('prod_hist_last'):
                parts.append(f"окно урожайности {j['prod_hist_last']} л.")
            if j.get('best_val_mse') is not None:
                parts.append(f"лучшая val MSE {float(j['best_val_mse']):.2f}")
            if j.get('epochs_run'):
                parts.append(f"эпох {j['epochs_run']}")
            if j.get('split'):
                parts.append(f"сплит {j['split']}")
        elif os.path.exists(meta):
            j = json.load(open(meta, encoding='utf-8'))
            parts.append(f"задание {j.get('task')}")
            if j.get('type'):
                parts.append(MODEL_NAMES.get(j['type'], j['type']))
            if j.get('n_train'):
                parts.append(f"обучена на {j['n_train']} примерах")
            if j.get('L'):
                parts.append(f"сжатие до {j['L']} точек")
    except (OSError, ValueError):
        return ''
    created = model_created_at(token)
    if created:
        parts.append(f"сохранена {created}")
    return ', '.join(str(p) for p in parts if p)


def model_created_at(token):
    """Когда модель сохранена: дата файла весов (``ГГГГ-ММ-ДД``) либо ``''``."""
    d = _token_dir(token)
    if not d:
        return ''
    kind, _v = parse_token(token)
    for fname in (_model_file(kind), 'model.pt', 'model.pkl'):
        path = os.path.join(d, fname)
        if os.path.exists(path):
            return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d')
    return ''


def model_inventory():
    """Мониторинг арсенала: по строке на каждую папку-модель и культуру.

    Поля: культура, mkey (токен = имя папки), модель (имя), сохранена, расположение,
    создана (дата файла весов), комментарий (заметка из ``comment.txt`` в папке модели).
    """
    rows = []
    for title in list_cultures():
        cfg = resolve_culture(title); tid = cfg['task_id']
        for kind in _FILE_KINDS:
            dirs = _model_dirs_for(tid, kind)
            if dirs:
                for name, _v in dirs:                     # токен и имя == имя папки
                    rows.append({'культура': title, 'mkey': name, 'модель': name,
                                 'сохранена': True, 'расположение': name,
                                 'создана': model_created_at(name),
                                 'комментарий': get_model_comment(name)})
            else:
                note = ('не обучена — кнопка «Обучить RF/линейную»' if kind in ('rf', 'linear')
                        else 'нет обученной нейросети')
                rows.append({'культура': title, 'mkey': kind, 'модель': MODEL_NAMES[kind],
                             'сохранена': False, 'расположение': note,
                             'создана': '', 'комментарий': ''})
        rows.append({'культура': title, 'mkey': 'ndvi', 'модель': MODEL_NAMES['ndvi'],
                     'сохранена': False, 'расположение': 'на лету (классическая регрессия по району)',
                     'создана': '', 'комментарий': ''})
        rows.append({'культура': title, 'mkey': 'ensemble', 'модель': MODEL_NAMES['ensemble'],
                     'сохранена': False, 'расположение': 'среднее выбранных моделей',
                     'создана': '', 'комментарий': ''})
    return rows


def districts_with_data(id_region, culture_title):
    """Множество id районов региона, у которых есть история урожайности по культуре.

    Для авто-выбора в прогнозе: по районам без истории прогноз построить нельзя
    (NDVI — нет точек регрессии; признаки — нет рядов), поэтому их не отмечаем.
    """
    import app_geo as _G
    cfg = resolve_culture(culture_title)
    have = set()
    for did, _name in _G.list_districts(id_region):
        yld = db.get_yields(db.territory_id(id_region, did), cfg['bdpmo_culture'])
        if any(v is not None and v == v for v in yld.values()):
            have.add(did)
    return have


# --------------------------------------------------------------------------- #
# Докачка недостающих рядов из Vega (#7)
# --------------------------------------------------------------------------- #
def ensure_inputs(cfg, id_region, id_district, year, ukey=None):
    """Скачать из Vega недостающие ряды, нужные для прогноза (все признаки × блок-годы + ndvi × годы урожая)."""
    import download
    from vega import VegaClient
    task = _task(cfg['task_id']); masks_df = db.get_masks_df()
    n_years = int(task['duration_years'])
    block_years = B._block_years(int(year), n_years, task['concat_newest_first'])
    tid = db.territory_id(id_region, id_district)
    client = VegaClient(ukey=ukey)
    res = []
    need = []                                              # (parm, year)
    for feat in task['features']:
        parm, _h = B._resolve_parm(feat, cfg['mask_culture'], id_region, masks_df)
        if parm:
            need += [(parm, y) for y in block_years]
    ndvi_feat = next((f for f in task['features'] if f.get('is_ndvi')), None)
    if ndvi_feat is not None:
        parm, _h = B._resolve_parm(ndvi_feat, cfg['mask_culture'], id_region, masks_df)
        if parm:
            need += [(parm, y) for y in db.get_yields(tid, cfg['bdpmo_culture'])]
    for parm, y in set(need):
        if not db.has_time_series(tid, parm, int(y)):
            ok, st = download.ensure_series(client, id_region, id_district, parm, int(y), offline=False)
            res.append((parm, int(y), st))
    return res, client


# --------------------------------------------------------------------------- #
# Сборка одиночного примера из БД (по заданному списку имён признаков)
# --------------------------------------------------------------------------- #
def _resolve_named(name, cfg, id_region, masks_df):
    """Имя признака -> (parm, historical). ndvi разрешается по маске культуры."""
    hist = name.endswith('_historical') or name.endswith('_hist')
    base = name.replace('_historical', '').replace('_hist', '')
    if base == 'ndvi':
        parm, _ = B._resolve_parm({'name': 'ndvi', 'is_ndvi': True, 'historical': hist},
                                  cfg['mask_culture'], id_region, masks_df)
    else:
        parm = base
    return parm, hist


def _build_X(cfg, id_region, id_district, year, names):
    """X (len(names), n_years*size) из БД для заданных признаков; (X, ok, note)."""
    task = _task(cfg['task_id'])
    size = int(task['time_rows_size']); n_years = int(task['duration_years'])
    N = int(task.get('feature_hist_last', 4))
    tid = db.territory_id(id_region, id_district); masks_df = db.get_masks_df()
    block_years = B._block_years(int(year), n_years, task['concat_newest_first'])
    X = np.zeros((len(names), n_years * size), np.float32)
    for pi, name in enumerate(names):
        parm, hist = _resolve_named(name, cfg, id_region, masks_df)
        if parm is None:
            return None, False, f'нет параметра {name}'
        row = []
        for y in block_years:
            arr = (db.mean_over_years(tid, parm, y, N, size, task['padding']) if hist
                   else db.get_time_series_array(tid, parm, y, size, task['padding']))
            if arr.shape[0] == 0:
                return None, False, f'нет ряда {name} за {y} год'
            if task['edge_fixup']:
                arr = B._edge_fixup(arr)
            row.append(arr)
        X[pi] = np.concatenate(row)
    return X, True, 'ok'


def _prod_hist_val(cfg, tid, year):
    task = _task(cfg['task_id'])
    ph = B._prod_hist(db.get_yields(tid, cfg['bdpmo_culture']), int(year), int(task['prod_hist_last']))
    return ph


# --------------------------------------------------------------------------- #
# Нейросеть (строится по СПИСКУ признаков самой модели)
# --------------------------------------------------------------------------- #
def _load_nn(nn_dir):
    if ('nn', nn_dir) not in _cache:
        arch = json.load(open(os.path.join(nn_dir, 'architecture.json'), encoding='utf-8'))
        norm = json.load(open(os.path.join(nn_dir, 'normalization.json'), encoding='utf-8'))
        import torch
        model = T.make_model(arch['arch_name'], arch['in_channels'], arch['in_length'])[0]
        model.load_state_dict(torch.load(os.path.join(nn_dir, 'model.pt'), map_location='cpu'))
        model.eval()
        _cache[('nn', nn_dir)] = (model, np.array(norm['mean'], np.float32),
                                  np.array(norm['std'], np.float32), list(norm['features']))
    return _cache[('nn', nn_dir)]


def predict_nn(cfg, id_region, id_district, year, nn_dir=None):
    nn_dir = nn_dir or cfg.get('nn_dir')                   # явная версия или последняя
    if not nn_dir or not os.path.exists(os.path.join(nn_dir, 'model.pt')):
        return None, 'нет обученной нейросети'
    model, mean, std, feats = _load_nn(nn_dir)
    names = [f for f in feats if f != 'prod_hist_ma']      # сырьё; prod_hist добавим строкой
    X, ok, note = _build_X(cfg, id_region, id_district, year, names)
    if not ok:
        return None, note
    tid = db.territory_id(id_region, id_district)
    ph = _prod_hist_val(cfg, tid, year)
    ph_val = ph if ph == ph else float(mean[-1])
    Xfull = np.concatenate([X, np.full((1, X.shape[1]), ph_val, np.float32)], axis=0)[None]
    Xn = ((Xfull - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    return float(T.predict(model, Xn, 'cpu')[0]), 'ok'


# --------------------------------------------------------------------------- #
# Случайный лес и линейная (обучаются на датасете культуры, кэш)
# --------------------------------------------------------------------------- #
def _sk_default_dir(task_id, kind):
    """Папка модели по умолчанию при обучении на лету (плоская): models/<task>_<kind>_v1."""
    return os.path.join(config.MODELS_DIR, f'{task_id}_{kind}_v1')


def _save_sklearn(kind, model, cfg, names, med, n_train, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    meta = {'type': kind, 'culture': cfg['title'], 'task': cfg['task_id'], 'L': L_SKLEARN,
            'features': names, 'ph_median': med, 'n_train': int(n_train),
            'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'description': f'{"Случайный лес" if kind == "rf" else "Линейная (Ridge)"} '
                           f'для культуры «{cfg["title"]}», сжатие времени до {L_SKLEARN} точек/канал.'}
    with open(os.path.join(outdir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _sk_dataset(cfg):
    """Подготовленная матрица датасета культуры для sklearn (кэш): (Xf, y, keep, med, names)."""
    key = ('skdata', cfg['task_id'])
    if key in _cache:
        return _cache[key]
    base = os.path.join(config.DATASETS_DIR, cfg['task_id'])
    meta = json.load(open(os.path.join(base, 'meta.json'), encoding='utf-8'))
    names = list(meta['features']); P = len(names)
    size = int(meta['time_rows_size']); ny = int(meta['duration_years'])
    mat = pd.read_csv(os.path.join(base, 'matrix.csv'), sep=' ', header=None, dtype='float32').values
    sc = pd.read_csv(os.path.join(base, 'scalar.csv'), sep=' ', header=None).values
    y = sc[:, 3].astype(float); ph = sc[:, 4].astype(float)
    Xc = mat.reshape(mat.shape[0], P, ny * size)
    Xv = E.to_vectors(E.downsample(Xc, L_SKLEARN))
    med = float(np.nanmedian(ph[~np.isnan(y)]))
    Xf = np.concatenate([Xv, np.where(np.isnan(ph), med, ph)[:, None]], axis=1)
    keep = ~np.isnan(y) & (y >= 0) & (y <= E.YIELD_MAX)
    _cache[key] = (Xf, y, keep, med, names)
    return _cache[key]


def _fit_one_sklearn(cfg, kind, model_dir=None):
    """Загрузить sklearn-модель (kind=rf|linear) из папки model_dir или обучить+сохранить (v1).

    Возвращает (model, ph_median, feature_names).
    """
    d = model_dir or _sk_default_dir(cfg['task_id'], kind)
    key = ('sk', d)
    if key in _cache:
        return _cache[key]
    pkl = os.path.join(d, 'model.pkl'); meta_p = os.path.join(d, 'meta.json')
    if os.path.exists(pkl) and os.path.exists(meta_p):
        m = json.load(open(meta_p, encoding='utf-8'))
        with open(pkl, 'rb') as f:
            model = pickle.load(f)
        _cache[key] = (model, float(m['ph_median']), list(m['features']))
        return _cache[key]
    if model_dir is not None:                              # запрошенной папки модели нет на диске
        raise FileNotFoundError(f'нет модели {kind} в {os.path.basename(d)}')
    Xf, y, keep, med, names = _sk_dataset(cfg)              # обучить модель по умолчанию и сохранить (v1)
    if kind == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42).fit(Xf[keep], y[keep])
    else:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=10.0).fit(Xf[keep], y[keep])
    _save_sklearn(kind, model, cfg, names, med, int(keep.sum()), d)
    _cache[key] = (model, med, names)
    return _cache[key]


def _sklearn_vector(cfg, id_region, id_district, year, names, med):
    X, ok, note = _build_X(cfg, id_region, id_district, year, names)
    if not ok:
        return None, note
    xv = E.to_vectors(E.downsample(X[None], L_SKLEARN))[0]
    tid = db.territory_id(id_region, id_district)
    ph = _prod_hist_val(cfg, tid, year)
    ph_val = ph if ph == ph else med
    return np.concatenate([xv, [ph_val]])[None], 'ok'


def predict_rf(cfg, id_region, id_district, year, model_dir=None):
    model, med, names = _fit_one_sklearn(cfg, 'rf', model_dir)
    vec, note = _sklearn_vector(cfg, id_region, id_district, year, names, med)
    if vec is None:
        return None, note
    return float(model.predict(vec)[0]), 'ok'


def predict_linear(cfg, id_region, id_district, year, model_dir=None):
    model, med, names = _fit_one_sklearn(cfg, 'linear', model_dir)
    vec, note = _sklearn_vector(cfg, id_region, id_district, year, names, med)
    if vec is None:
        return None, note
    return float(model.predict(vec)[0]), 'ok'


# --------------------------------------------------------------------------- #
# Классическая регрессия по NDVI (на лету по району)
# --------------------------------------------------------------------------- #
def _ndvi_parm(cfg, id_region):
    masks_df = db.get_masks_df()
    task = _task(cfg['task_id'])
    ndvi_feat = next((f for f in task['features'] if f.get('is_ndvi')), None)
    if ndvi_feat is None:
        return None
    parm, _h = B._resolve_parm(ndvi_feat, cfg['mask_culture'], id_region, masks_df)
    return parm


def _max_ndvi(cfg, parm, tid, year, size=365):
    # Мемоизация: один и тот же (район, parm, год) при бэктесте/прогнозе читается многократно;
    # кэш в _cache переживает db.save() (которое сбрасывает кэши рядов БД при каждом upsert).
    key = ('ndvimax', tid, parm, int(year), int(cfg['harvest_doy']), size)
    if key not in _cache:
        arr = db.get_time_series_array(tid, parm, int(year), size, 'zeros')
        _cache[key] = None if arr.shape[0] == 0 else float(arr[:cfg['harvest_doy']].max())
    return _cache[key]


def predict_ndvi(cfg, id_region, id_district, year):
    tid = db.territory_id(id_region, id_district)
    parm = _ndvi_parm(cfg, id_region)
    if parm is None:
        return None, 'нет NDVI-маски'
    yields = db.get_yields(tid, cfg['bdpmo_culture'])
    pts = []                                               # (maxNDVI, yield) по годам района, кроме целевого
    for yy, val in yields.items():
        if yy == int(year) or val is None or val != val:
            continue
        mx = _max_ndvi(cfg, parm, tid, yy)
        if mx is not None and mx > 0:
            pts.append((mx, float(val)))
    if len(pts) < 3:
        return None, f'мало точек для регрессии ({len(pts)})'
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    fit = fit_exp(xs, ys)
    mx_t = _max_ndvi(cfg, parm, tid, year)
    if mx_t is None or mx_t <= 0:
        return None, 'нет NDVI расчётного года'
    if fit is None:
        return float(np.mean(ys)), 'откат к среднему (экспонента не подгоналась)'
    xc = min(max(mx_t, float(xs.min())), float(xs.max()))  # без экстраполяции
    cap = min(E.YIELD_MAX, 1.5 * float(ys.max()))
    return min(pred_exp(fit, xc), cap), 'ok'


# --------------------------------------------------------------------------- #
# Единая точка: прогноз всеми выбранными моделями
# --------------------------------------------------------------------------- #
def predict_for_territory(id_region, id_district, culture_title, year, models=None,
                          download=False, ukey=None, ensemble_of=None):
    """Прогноз выбранными моделями. download=True — докачать недостающие ряды из Vega.

    ensemble_of — список моделей, входящих в ансамбль (по умолчанию все точечные).
    Возвращает {'values': {model: float|None}, 'notes': {...}, 'prod_hist': float|None, 'cfg':...}.
    """
    models = models or ['nn', 'rf', 'linear', 'ndvi', 'ensemble']
    cfg = resolve_culture(culture_title)
    tid = db.territory_id(id_region, id_district)
    dl = None                                             # сводка докачки рядов из Vega
    if download:
        try:
            res, client = ensure_inputs(cfg, id_region, id_district, year, ukey=ukey)
            sts = [st for _p, _y, st in res]
            dl = {'requested': len(sts),
                  'downloaded': sts.count('downloaded'),
                  'empty': sts.count('empty'),
                  'no_uid': sts.count('no_uid'),
                  'net_fail': int(getattr(client, 'net_fail_count', 0)),
                  'bad_response': int(getattr(client, 'bad_response_count', 0))}
        except Exception:                                 # noqa: BLE001
            dl = None
    comps = ensemble_of or ['nn', 'rf', 'linear', 'ndvi']
    need = set(m for m in models if m != 'ensemble')
    if 'ensemble' in models:
        need |= set(comps)                                # для ансамбля нужны его составляющие
    values, notes = {}, {}
    for token in need:                                    # token: имя папки / 'rf' / 'ndvi' / ...
        kind, _ver = parse_token(token)
        mdir = _token_dir(token)                           # папка конкретной модели (или None — последняя/по умолч.)
        try:
            if kind == 'nn':
                v, note = predict_nn(cfg, id_region, id_district, year, nn_dir=mdir or cfg.get('nn_dir'))
            elif kind == 'rf':
                v, note = predict_rf(cfg, id_region, id_district, year, model_dir=mdir)
            elif kind == 'linear':
                v, note = predict_linear(cfg, id_region, id_district, year, model_dir=mdir)
            elif kind == 'ndvi':
                v, note = predict_ndvi(cfg, id_region, id_district, year)
            else:
                v, note = None, f'неизвестная модель: {token}'
        except Exception as exc:                          # noqa: BLE001
            v, note = None, f'ошибка: {exc}'
        values[token], notes[token] = v, note
    if 'ensemble' in models:
        pts = [values[m] for m in comps if values.get(m) is not None]
        values['ensemble'] = float(np.mean(pts)) if pts else None
        notes['ensemble'] = (f'среднее моделей: {", ".join(token_label(m) for m in comps if values.get(m) is not None)}'
                             if pts else 'нет доступных моделей для ансамбля')
    task = _task(cfg['task_id'])
    ph = B._prod_hist(db.get_yields(tid, cfg['bdpmo_culture']), int(year), int(task['prod_hist_last']))
    return {'values': values, 'notes': notes, 'prod_hist': (ph if ph == ph else None),
            'cfg': cfg, 'download': dl}
