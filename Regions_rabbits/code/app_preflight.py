"""Обследование базы перед прогнозом и учёт итогов: что есть, чего нет, сколько прогнозов получится.

Программой можно пользоваться и без ukey. Поэтому перед запуском прогноза партия
«территории × годы × модели» проверяется по базе, и пользователь заранее видит:

* какие входные ряды уже в базе, а каких нет (по параметрам и годам);
* сколько прогнозов посчитается сразу, сколько — только после докачки с Vega (нужен ukey),
  а сколько не посчитается ни при каких условиях и почему;
* оценку: «без докачки — R из N», «с докачкой — до R + D из N».

После расчёта по каждому запланированному прогнозу пишется итог с причиной в таблицу БД
``forecast_outcomes`` — её показывает вкладка «Прогнозы и отчёты» и кладёт в выгрузку партии.

Статусы плана:
    ready     — все входные данные уже в базе;
    download  — не хватает рядов, которые можно докачать с Vega (посчитается, только если есть
                ukey и Vega эти ряды отдаст — поэтому оценка «до»);
    blocked   — посчитать нельзя: нет модели, маски NDVI, истории урожайности, район не
                сопоставлен с Vega, год ещё не наступил и т.п.
"""
import datetime
import io
import json
import os

import pandas as pd

import app_predict as P
import build_dataset as B
import db

READY, DOWNLOAD, BLOCKED = 'ready', 'download', 'blocked'
STATUS_TITLES = {READY: 'готово к расчёту', DOWNLOAD: 'нужна докачка с Vega',
                 BLOCKED: 'не посчитать'}
OUTCOMES_TABLE = 'forecast_outcomes'
OUTCOME_COLUMNS = ['outcome_id', 'batch_id', 'territory_id', 'label', 'culture', 'year', 'model',
                   'planned', 'status', 'reason', 'has_ukey', 'made_at']


# --------------------------------------------------------------------------- #
# Что нужно модели
# --------------------------------------------------------------------------- #
_feat_cache = {}


def _model_features(cfg, token):
    """Список признаков модели-файла и путь к её папке: (names | None, reason | None)."""
    kind, _v = P.parse_token(token)
    d = P.model_dir(token)
    if kind == 'nn':
        d = d or cfg.get('nn_dir')
        if not d or not os.path.exists(os.path.join(d, 'model.pt')):
            return None, 'нет обученной нейросети'
        key = ('nn', d)
        if key not in _feat_cache:
            norm = json.load(io.open(os.path.join(d, 'normalization.json'), encoding='utf-8'))
            _feat_cache[key] = [f for f in norm['features'] if f != 'prod_hist_ma']
        return _feat_cache[key], None
    if kind in ('rf', 'linear'):
        d = d or P._sk_default_dir(cfg['task_id'], kind)
        meta = os.path.join(d, 'meta.json')
        if not os.path.exists(os.path.join(d, 'model.pkl')) or not os.path.exists(meta):
            return None, 'модель не обучена (нет файла модели)'
        key = (kind, d)
        if key not in _feat_cache:
            _feat_cache[key] = list(json.load(io.open(meta, encoding='utf-8'))['features'])
        return _feat_cache[key], None
    return None, f'неизвестная модель: {token}'


def _downloadable(id_region, id_district, year):
    """Можно ли докачать недостающие ряды территории с Vega: (True, None) | (False, причина)."""
    if int(year) > datetime.date.today().year:
        return False, 'год ещё не наступил — спутниковых и метеорядов за него нет'
    if id_district is not None:
        if db.current_vega_uid(db.territory_id(id_region, id_district)) is None:
            return False, 'район не сопоставлен с Vega — недостающие ряды не докачать'
    return True, None


def _check_features(cfg, id_region, id_district, year, names, masks_df, task):
    """Входы модели-файла: (blocker, present_set, missing_set). Наборы — пары (parm, год)."""
    tid = db.territory_id(id_region, id_district)
    n_years = int(task['duration_years'])
    N = int(task.get('feature_hist_last', 4))
    block_years = B._block_years(int(year), n_years, task['concat_newest_first'])
    present, missing = set(), set()
    for name in names:
        parm, hist = P._resolve_named(name, cfg, id_region, masks_df)
        if parm is None:
            return f'нет маски NDVI культуры для региона ({name})', present, missing
        for y in block_years:
            if not hist:
                (present if db.has_time_series(tid, parm, y) else missing).add((parm, int(y)))
                continue
            prev = [int(y) - i for i in range(1, N + 1)]    # климатология: хватит любого из N прошлых лет
            have = [(parm, p) for p in prev if db.has_time_series(tid, parm, p)]
            if have:
                present.update(have)
            else:
                missing.update((parm, p) for p in prev)
    return None, present, missing


def _check_ndvi(cfg, id_region, id_district, year):
    """Входы регрессии NDVI: (blocker, present_set, missing_set, note)."""
    tid = db.territory_id(id_region, id_district)
    parm = P._ndvi_parm(cfg, id_region)
    if parm is None:
        return 'нет маски NDVI культуры для региона', set(), set()
    yl = [yy for yy, v in db.get_yields(tid, cfg['bdpmo_culture']).items()
          if int(yy) != int(year) and v is not None and v == v]
    if len(yl) < 3:
        return (f'мало лет урожайности для регрессии NDVI ({len(yl)}, нужно ≥ 3)'
                if yl else 'нет истории урожайности района'), set(), set()
    have = {(parm, int(yy)) for yy in yl if db.has_time_series(tid, parm, int(yy))}
    lack = {(parm, int(yy)) for yy in yl} - have
    present, missing = set(have), set()
    if len(have) < 3:                                      # до трёх точек регрессии не хватает —
        missing |= lack                                    # нужны недостающие годы; иначе они не обязательны
    target = (parm, int(year))
    (present if db.has_time_series(tid, parm, int(year)) else missing).add(target)
    return None, present, missing


# --------------------------------------------------------------------------- #
# Обследование партии
# --------------------------------------------------------------------------- #
def survey(targets, culture_title, years, models, ensemble_of=None, has_ukey=False, progress=None):
    """Обследовать базу под партию прогноза.

    ``targets`` — [(id_region, id_district | None, подпись)], ``models`` — токены моделей.
    Возвращает dict: ``items`` (по строке на прогноз), ``data`` (есть/нет по параметрам и годам),
    ``reasons`` (сводка причин), ``summary`` (числа для оценки).
    """
    cfg = P.resolve_culture(culture_title)
    task = P._task(cfg['task_id'])
    masks_df = db.get_masks_df()
    comps = list(ensemble_of or [m for m in models if m != 'ensemble'])
    need_models = [m for m in models if m != 'ensemble']
    if 'ensemble' in models:
        need_models += [c for c in comps if c not in need_models]

    items, data_need = [], {}                              # (parm, год) -> [нужно, есть]
    total = max(len(targets) * len(years), 1)
    step = 0
    for reg, did, label in targets:
        for yr in years:
            step += 1
            if progress:
                progress(int(100 * step / total))
            tid = db.territory_id(reg, did)
            can_dl, dl_reason = _downloadable(reg, did, yr)
            per_model = {}
            for token in need_models:
                kind, _v = P.parse_token(token)
                if kind == 'ndvi':
                    blocker, present, missing = _check_ndvi(cfg, reg, did, yr)
                else:
                    names, blocker = _model_features(cfg, token)
                    present, missing = set(), set()
                    if blocker is None:
                        blocker, present, missing = _check_features(cfg, reg, did, yr, names,
                                                                    masks_df, task)
                for key in present:
                    rec = data_need.setdefault(key, [0, 0]); rec[0] += 1; rec[1] += 1
                for key in missing:
                    rec = data_need.setdefault(key, [0, 0]); rec[0] += 1
                if blocker:
                    status, reason = BLOCKED, blocker
                elif not missing:
                    status, reason = READY, ''
                elif can_dl:
                    sample = ', '.join(f'{p} {y}' for p, y in sorted(missing)[:3])
                    more = f' и ещё {len(missing) - 3}' if len(missing) > 3 else ''
                    status, reason = DOWNLOAD, f'нет в базе {len(missing)} рядов: {sample}{more}'
                else:
                    status, reason = BLOCKED, f'{dl_reason} (нет в базе {len(missing)} рядов)'
                per_model[token] = (status, reason, len(missing))

            for token in models:
                if token == 'ensemble':
                    st = [per_model[c][0] for c in comps if c in per_model]
                    if READY in st:
                        status, reason = READY, ''
                    elif DOWNLOAD in st:
                        status, reason = DOWNLOAD, 'составляющие ансамбля считаются только после докачки'
                    else:
                        status, reason = BLOCKED, 'ни одна модель ансамбля не считается'
                    miss = 0
                else:
                    status, reason, miss = per_model[token]
                items.append({'territory_id': tid, 'label': label, 'year': int(yr), 'model': token,
                              'модель': P.token_label(token), 'status': status, 'reason': reason,
                              'missing': miss})

    items_df = pd.DataFrame(items, columns=['territory_id', 'label', 'year', 'model', 'модель',
                                            'status', 'reason', 'missing'])
    parm_title = {str(k): v for k, v in _PARM_TITLES.items()}
    data_rows = [{'параметр': parm_title.get(p, p), 'код': p, 'год': y, 'нужно рядов': n,
                  'есть в базе': have, 'нет': n - have}
                 for (p, y), (n, have) in data_need.items()]
    data_df = pd.DataFrame(data_rows, columns=['параметр', 'код', 'год', 'нужно рядов',
                                               'есть в базе', 'нет'])
    if not data_df.empty:
        data_df = data_df.sort_values(['нет', 'параметр', 'год'],
                                      ascending=[False, True, True]).reset_index(drop=True)

    reasons_df = _group_reasons(items_df[items_df['status'] != READY], 'reason', 'status')
    planned = int(len(items_df))
    ready = int((items_df['status'] == READY).sum())
    download = int((items_df['status'] == DOWNLOAD).sum())
    blocked = int((items_df['status'] == BLOCKED).sum())
    summary = {'planned': planned, 'ready': ready, 'download': download, 'blocked': blocked,
               'has_ukey': bool(has_ukey), 'territories': len(targets), 'years': len(years),
               'models': len(models),
               'expected_without_download': ready,
               'expected_with_download': ready + download,
               'expected': ready + (download if has_ukey else 0)}
    return {'items': items_df, 'data': data_df, 'reasons': reasons_df, 'summary': summary}


def _group_reasons(df, reason_col, status_col=None):
    """Сводка причин: короткая причина (без перечня рядов) -> сколько прогнозов."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['причина', 'прогнозов'])
    short = df[reason_col].astype(str).str.replace(r':.*$', '', regex=True).str.strip()
    short = short.str.replace(r'\(нет в базе \d+ рядов\)', '(нет рядов в базе)', regex=True)
    short = short.str.replace(r'^нет в базе \d+ рядов$', 'нет части входных рядов в базе', regex=True)
    short = short.str.replace(r'\s*\([^)]*\d[^)]*\)$', '', regex=True).str.strip()   # «(2, нужно ≥ 3)» и т.п.
    if status_col is not None:
        short = df[status_col].map(STATUS_TITLES).fillna('') + ' — ' + short
    out = short.value_counts().rename_axis('причина').reset_index(name='прогнозов')
    return out


def estimate_text(summary):
    """Человекочитаемая оценка плана (для окна перед запуском)."""
    s = summary
    n = max(s['planned'], 1)
    pct = lambda k: f'{100 * k / n:.0f}%'
    lines = [f'Запланировано прогнозов: {s["planned"]} '
             f'({s["territories"]} террит. × {s["years"]} лет × {s["models"]} моделей).',
             f'• готово к расчёту сразу: {s["ready"]} ({pct(s["ready"])})',
             f'• нужна докачка рядов с Vega: {s["download"]} ({pct(s["download"])})',
             f'• не посчитать при любых условиях: {s["blocked"]} ({pct(s["blocked"])})', '']
    if s['has_ukey']:
        lines.append(f'ukey указан — недостающие ряды будут докачаны. Ожидается до '
                     f'{s["expected_with_download"]} из {s["planned"]} '
                     f'({pct(s["expected_with_download"])}); точное число зависит от того, '
                     'отдаст ли Vega ряды.')
    else:
        lines.append(f'ukey не указан — докачки не будет. Ожидается {s["ready"]} из {s["planned"]} '
                     f'({pct(s["ready"])}). С ukey было бы до {s["expected_with_download"]} '
                     f'({pct(s["expected_with_download"])}).')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Итоги расчёта
# --------------------------------------------------------------------------- #
def run_reason(plan_status, plan_reason, note, has_ukey, download_stats=None):
    """Почему прогноз не посчитан — понятной фразой по плану, примечанию модели и докачке."""
    note = str(note or '').strip()
    if plan_status == BLOCKED and plan_reason:
        return plan_reason
    if plan_status == DOWNLOAD and not has_ukey:
        return 'нет входных рядов в базе; ukey не указан — докачка невозможна'
    if plan_status == DOWNLOAD and has_ukey:
        d = download_stats or {}
        if d.get('net_fail'):
            return f'сбой сети при докачке с Vega ({note})' if note else 'сбой сети при докачке с Vega'
        return f'Vega не отдала недостающие ряды ({note})' if note else 'Vega не отдала недостающие ряды'
    return note or 'причина не установлена'


def save_outcomes(batch_id, culture, rows, has_ukey):
    """Записать итоги партии: по строке на запланированный прогноз."""
    if not rows:
        return 0
    now = datetime.datetime.now().isoformat(timespec='seconds')
    df = db.load(OUTCOMES_TABLE)
    start = 1 if df.empty else int(df['outcome_id'].max()) + 1
    new = pd.DataFrame([{
        'outcome_id': start + i, 'batch_id': str(batch_id), 'territory_id': r['territory_id'],
        'label': r.get('label'), 'culture': str(culture), 'year': int(r['year']),
        'model': r['model'], 'planned': r.get('planned'), 'status': r['status'],
        'reason': r.get('reason') or '', 'has_ukey': bool(has_ukey), 'made_at': now,
    } for i, r in enumerate(rows)], columns=OUTCOME_COLUMNS)
    if not df.empty:
        df = df[df['batch_id'].astype(str) != str(batch_id)]
    db.save(OUTCOMES_TABLE, pd.concat([df, new], ignore_index=True))
    return len(new)


def batch_outcomes(batch_id):
    """Итоги партии из БД: (summary | None, таблица причин, таблица нерассчитанных)."""
    df = db.load(OUTCOMES_TABLE)
    if df.empty or 'batch_id' not in df:
        return None, pd.DataFrame(), pd.DataFrame()
    sub = df[df['batch_id'].astype(str) == str(batch_id)]
    if sub.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    ok = sub['status'] == 'ok'
    planned_ready = int((sub['planned'] == READY).sum())
    summary = {'planned': int(len(sub)), 'done': int(ok.sum()), 'failed': int((~ok).sum()),
               'planned_ready': planned_ready,
               'planned_download': int((sub['planned'] == DOWNLOAD).sum()),
               'planned_blocked': int((sub['planned'] == BLOCKED).sum()),
               'has_ukey': bool(str(sub['has_ukey'].iloc[0]).lower() in ('true', '1')),
               'made_at': str(sub['made_at'].iloc[0])}
    failed = sub[~ok].copy()
    reasons = _group_reasons(failed, 'reason')
    failed_view = failed.assign(модель=failed['model'].map(P.token_label))[
        ['label', 'year', 'модель', 'reason']].rename(
        columns={'label': 'территория', 'year': 'год', 'reason': 'почему не рассчитан'})
    return summary, reasons, failed_view.reset_index(drop=True)


def outcome_text(summary):
    """Итог партии одной-двумя фразами."""
    if not summary:
        return ('Для этой партии нет сведений о плане и итогах — она сделана до появления учёта '
                'причин нерасчёта.')
    s = summary
    n = max(s['planned'], 1)
    ukey = 'с ukey (докачка включена)' if s['has_ukey'] else 'без ukey (без докачки)'
    return (f'Рассчитано {s["done"]} из {s["planned"]} запланированных ({100 * s["done"] / n:.0f}%), '
            f'не рассчитано {s["failed"]}. Прогноз шёл {ukey}. По плану было: готово сразу '
            f'{s["planned_ready"]}, нужна докачка {s["planned_download"]}, '
            f'не посчитать {s["planned_blocked"]}.')


# Понятные имена параметров для таблицы «есть / нет в базе»
_PARM_TITLES = {
    'mean_temp': 'температура', 'mean_temp_acc': 'накопленное тепло', 'mean_prec': 'осадки',
    'mean_prec_acc': 'накопленные осадки', 'mean_rh': 'влажность воздуха', 'mean_p': 'атм. давление',
    'mean_snod': 'высота снега', 'mean_snowc': 'снежный покров', 'mean_soilw10': 'влага почвы 10 см',
    'mean_tmpgr10': 'температура почвы 10 см',
    'mean_ndvi_7dc_modis_int_ozim': 'NDVI (маска озимых)',
    'mean_ndvi_7dc_modis_int_spring': 'NDVI (маска яровых)',
    'mean_ndvi_7dc_modis_int_agro': 'NDVI (маска пашни)',
}
