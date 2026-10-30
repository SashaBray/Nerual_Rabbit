"""Метрики мониторинга: умеет ли модель упорядочить районы ВНУТРИ незнакомого года.

Зачем отдельно от ``forest_eval.py``. Прогнозная задача спрашивает «какая будет урожайность»,
и там мы уперлись: уровень незнакомого года по погоде не называется. Но продукт спрашивает
другое — «где хуже, чем вокруг». Этот вопрос внутрисезонный и относительный, поэтому
постоянный сдвиг по году в нём сокращается и мерить его надо не RMSE.

Отсечка по году при этом ОБЯЗАТЕЛЬНА и здесь сохранена: в июле года Y урожайность года Y не
известна нигде, урожай не собран. Законный вход текущего сезона — только погода и NDVI до
сегодня, а метки года Y недоступны. Поэтому схемы те же, что в прогнозной ветке
(``ww_loo``, ``ww_ladder``), меняются только метрики:

* ``внутр_rmse`` — RMSE после центрирования факта и прогноза в пределах тестового года:
  сколько ошибки остаётся, если уровень года не наказывать;
* ``спирмен`` — ранговая корреляция прогноза с фактом по районам одного года;
* ``трети`` — доля районов, отнесённых в верную треть своего года.

Каждая метрика считается и для модели, и для эталона (скользящей средней), потому что эталон
тоже умеет упорядочивать районы — вопрос в том, добавляет ли погода к этому хоть что-то.

    python code/monitor_eval.py --name ww_mon --scheme ww_loo --models ridge rf et \\
           --features core_d --cutoff 0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import ensemble_single as ES
import explore_models as E
import folds as FD
import forest_eval as FE
import run_experiment as R

COLUMNS = ['task', 'fold', 'model', 'features', 'cutoff', 'seed', 'n_feat', 'n_train', 'n_test',
           'shrink', 'rmse', 'base_rmse', 'внутр_rmse', 'внутр_base_rmse',
           'спирмен', 'спирмен_base', 'трети', 'трети_base',
           'спирмен_откл', 'спирмен_ndvi', 'ndvi_признак', 'лет_в_тесте', 'minutes']


def _dev_rank(y, pred, ph, years):
    """Ранговая корреляция ОТКЛОНЕНИЙ внутри года, усреднённая по тестовым годам.

    Это и есть продуктовая метрика. Предыдущая (порядок самих урожайностей) почти целиком
    определяется тем, что сильные районы всегда сильнее слабых: у скользящей средней там
    ρ = 0.80 без всякой погоды. Продукт же спрашивает «хуже ли, чем ДОЛЖНО быть здесь в этом
    году», то есть порядок отклонений от собственной нормы района. У эталона это постоянная
    величина, его ранговая корреляция здесь тождественно не определена, поэтому любое
    устойчивое положительное значение — добавленная информация, а не повторение истории.
    """
    d = pd.DataFrame({'dy': np.asarray(y, float) - np.asarray(ph, float),
                      'dp': np.asarray(pred, float) - np.asarray(ph, float), 'g': years})
    out = []
    for _g, s in d.groupby('g'):
        if len(s) < 20 or s['dp'].nunique() < 3:
            continue
        r = spearmanr(s['dy'], s['dp']).statistic
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else np.nan


def _feature_rank(y, ph, years, col):
    """То же для одного признака: «а если просто смотреть на NDVI?»"""
    d = pd.DataFrame({'dy': np.asarray(y, float) - np.asarray(ph, float),
                      'x': np.asarray(col, float), 'g': years}).dropna()
    out = []
    for _g, s in d.groupby('g'):
        if len(s) < 20 or s['x'].nunique() < 3:
            continue
        r = spearmanr(s['dy'], s['x']).statistic
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else np.nan


def _centred_rmse(y, pred, years):
    """RMSE после вычитания средних по тестовому году из факта и из прогноза."""
    d = pd.DataFrame({'y': y, 'p': pred, 'g': years})
    d['y'] -= d.groupby('g')['y'].transform('mean')
    d['p'] -= d.groupby('g')['p'].transform('mean')
    return float(np.sqrt(np.mean((d['y'] - d['p']) ** 2)))


def _rank_skill(y, pred, years):
    """Ранговая корреляция и доля верных третей, усреднённые по тестовым годам."""
    rho, hit, w = [], [], []
    for _g, s in pd.DataFrame({'y': y, 'p': pred, 'g': years}).groupby('g'):
        if len(s) < 20 or s['p'].nunique() < 3:
            continue
        r = spearmanr(s['y'], s['p']).statistic
        if np.isfinite(r):
            rho.append(r)
        qy = pd.qcut(s['y'].rank(method='first'), 3, labels=False)
        qp = pd.qcut(s['p'].rank(method='first'), 3, labels=False)
        hit.append(float((qy == qp).mean()))
        w.append(len(s))
    if not rho:
        return np.nan, np.nan
    return float(np.mean(rho)), float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser(description='Метрики мониторинга на честных фолдах')
    ap.add_argument('--name', required=True)
    ap.add_argument('--scheme', default='ww_loo')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--models', nargs='+', default=['ridge'], choices=list(FE.MODELS))
    ap.add_argument('--features', nargs='+', default=['core_d'],
                    choices=list(FE.FEATURE_MODULE))
    ap.add_argument('--cutoff', type=int, default=0, help='0 — весь сезон, иначе день от 1 янв')
    ap.add_argument('--folds', nargs='*')
    ap.add_argument('--seeds', type=int, default=1)
    a = ap.parse_args()

    cutoff = a.cutoff or None
    scheme = FD.read_definition(a.scheme)
    names = a.folds or sorted(str(x) for x in scheme['fold'].unique())
    specs = [FD.fold_spec(scheme, f) for f in names]
    out = os.path.join(config.REPORTS_DIR, 'experiments', a.name)
    os.makedirs(out, exist_ok=True)
    runs = os.path.join(out, 'runs.csv')

    done = set()
    if os.path.exists(runs):
        prev = pd.read_csv(runs, sep=';', decimal=',', encoding='utf-8-sig')
        done = set(zip(prev['fold'].astype(str), prev['model'].astype(str),
                       prev['features'].astype(str), prev['cutoff'].astype(int),
                       prev['seed'].astype(int)))

    plan = [(str(s['fold']), m, fs, a.cutoff, sd) for s in specs for m in a.models
            for fs in a.features for sd in range(1, a.seeds + 1)]
    todo = [p for p in plan if p not in done]
    print(f'Опыт «{a.name}»: схема {a.scheme}, отсечка {cutoff or "весь сезон"} -> '
          f'прогонов {len(plan)}, осталось {len(todo)}')
    if not todo:
        print('нечего считать')
        return

    tables = {m: m.table_for(a.task, cutoff)
              for m in {FE.FEATURE_MODULE[fs] for fs in a.features}}
    ds = R.Dataset(a.task, 'raw', 0)
    years = ds.ids['year'].to_numpy(dtype=int)
    cols = {fs: FE.FEATURE_MODULE[fs].select_columns(tables[FE.FEATURE_MODULE[fs]], fs)
            for fs in a.features}

    for i, (fold, model, fs, cut, seed) in enumerate(todo, 1):
        t0 = time.time()
        spec = next(s for s in specs if str(s['fold']) == fold)
        _Xn, y, ph, tr, va, te = ds.fold_arrays(spec)
        X, _found = FE.align(tables[FE.FEATURE_MODULE[fs]], ds.tid, years, cols[fs])
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med)
        keep = X[tr].std(axis=0) > 0
        X, kept = X[:, keep], [c for c, k in zip(cols[fs], keep) if k]
        fit_y, clip = ES.ratio_target(y, ph, tr)

        best = None                                        # отбор тот же, что в прогнозной ветке
        for params in FE.grid(model):
            m = FE.build(model, params, seed)
            m.fit(X[tr], fit_y[tr])
            sv = FE.shrink_rmse(m.predict(X[va]), y[va], ph[va], clip)
            oof = FE.oof_by_group(model, params, seed, X[tr], fit_y[tr], ds.tid[tr])
            if oof is None:
                continue
            ok = np.isfinite(oof)
            sg = FE.shrink_rmse(oof[ok], y[tr][ok], ph[tr][ok], clip)
            for lam in FE.SHRINK:
                sel = (sv[lam] + sg[lam]) / 2
                if best is None or sel < best[0]:
                    best = (sel, params, lam)
        _sel, params, lam = best

        final = FE.build(model, params, seed)
        final.fit(X[tr], fit_y[tr])
        pred = ES.to_absolute(final.predict(X[te]) * lam, ph[te], clip)
        yt, pt, bt, gt = y[te], pred, ph[te], years[te]
        rho, hit = _rank_skill(yt, pt, gt)
        rho_b, hit_b = _rank_skill(yt, bt, gt)
        # одиночный NDVI-признак как простейший соперник: «просто посмотреть на NDVI»
        nd = next((c for c in kept if 'ndvi' in c.lower() and 'труб' in c),
                  next((c for c in kept if 'ndvi' in c.lower()), None))
        rho_nd = (_feature_rank(yt, bt, gt, X[te][:, kept.index(nd)])
                  if nd is not None else np.nan)
        row = {'task': a.task, 'fold': fold, 'model': model, 'features': fs, 'cutoff': cut,
               'seed': seed, 'n_feat': len(kept), 'n_train': len(tr), 'n_test': len(te),
               'shrink': lam,
               'rmse': round(float(E.metrics(yt, pt)['RMSE']), 4),
               'base_rmse': round(float(E.metrics(yt, bt)['RMSE']), 4),
               'внутр_rmse': round(_centred_rmse(yt, pt, gt), 4),
               'внутр_base_rmse': round(_centred_rmse(yt, bt, gt), 4),
               'спирмен': round(rho, 4), 'спирмен_base': round(rho_b, 4),
               'трети': round(hit, 4), 'трети_base': round(hit_b, 4),
               'спирмен_откл': round(_dev_rank(yt, pt, bt, gt), 4),
               'спирмен_ndvi': (round(rho_nd, 4) if np.isfinite(rho_nd) else ''),
               'ndvi_признак': nd or '',
               'лет_в_тесте': int(len(np.unique(gt))),
               'minutes': round((time.time() - t0) / 60, 1)}
        FE._append(runs, row, COLUMNS)
        print(f'[{i}/{len(todo)}] {fold}, {model}, {fs}, отсечка {cut}: λ={lam} -> '
              f'внутр. RMSE {row["внутр_rmse"]:.2f} (эталон {row["внутр_base_rmse"]:.2f}), '
              f'спирмен {rho:+.3f} (эталон {rho_b:+.3f})', flush=True)

    with open(os.path.join(out, 'experiment.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': a.name, 'scheme': a.scheme, 'task': a.task, 'models': a.models,
                   'features': a.features, 'cutoff': a.cutoff, 'folds': names,
                   'metric': 'monitoring'}, f, ensure_ascii=False, indent=2)

    d = pd.read_csv(runs, sep=';', decimal=',', encoding='utf-8-sig')
    print('\n=== среднее по фолдам')
    print(d.pivot_table(index=['features', 'model'],
                        values=['внутр_rmse', 'внутр_base_rmse', 'спирмен', 'спирмен_base',
                                'трети', 'трети_base', 'спирмен_откл', 'спирмен_ndvi'],
                        aggfunc='mean').round(3).to_string())


if __name__ == '__main__':
    main()
