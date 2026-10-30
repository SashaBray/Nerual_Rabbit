"""Продуктовая метрика против дня сезона: когда мониторинг начинает что-то знать.

Мониторинг спрашивает в любой момент года, поэтому результат — не одно число, а кривая.
Метрика — ранговая корреляция ОТКЛОНЕНИЙ от нормы района внутри тестового года: продукт
спрашивает «хуже ли здесь, чем должно быть здесь в этом году», а не «какой район сильнее».
У скользящей средней эта величина постоянна, то есть планка равна нулю.

Год изымается (схема ``ww_loo``): урожайности тестового года не видит ни обучение, ни отбор.
Признак выбирается по обучающим годам фолда, значение докладывается по тестовому — поэтому
«лучший признак» не подсмотрен.

Сравниваются три всё более сложных предсказателя, чтобы видеть, за что мы платим сложностью:

1. один признак, выбранный по обучению;
2. среднее z-оценок трёх лучших по обучению;
3. заранее названный NDVI в трубкование — тот, что доступен раньше всех прочих.

    python code/rank_curve.py --name ww_curve --cutoffs 90 120 151 181 0
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import agro_features as AF
import config
import ensemble_single as ES
import folds as FD
import forest_eval as FE
import run_experiment as R
import series_features as SF

COLUMNS = ['task', 'scheme', 'cutoff', 'fold', 'n_train', 'n_test', 'лет_в_тесте',
           'лучший_признак', 'обуч', 'тест_один', 'тест_топ3', 'тест_ndvi', 'minutes']
NDVI_EARLY = 'Δтруб_ndvi'
SETS = {'agro': (AF, 'agro_all'), 'series': (SF, 'full')}


def rank_by_year(dev, x, yr, min_n=20):
    """Средняя по годам ранговая корреляция отклонения с величиной. -> float или NaN."""
    d = (pd.DataFrame({'d': np.asarray(dev, float), 'x': np.asarray(x, float),
                       'g': np.asarray(yr)})
         .replace([np.inf, -np.inf], np.nan).dropna())
    out = []
    for _g, s in d.groupby('g'):
        if len(s) < min_n or s['x'].nunique() < 3:
            continue
        r = spearmanr(s['d'], s['x']).statistic
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else np.nan


def _columns(task, cutoff):
    """Матрицы признаков обоих модулей при данной отсечке. -> [(имя, вектор)]."""
    out = []
    for mod, kind in SETS.values():
        tbl = mod.table_for(task, cutoff or None)
        cols = mod.select_columns(tbl, kind)
        X, _found = FE.align(tbl, _DS.tid, _YEARS, cols)
        for j, name in enumerate(cols):
            out.append((name, X[:, j].astype(float)))
    return out


def main():
    ap = argparse.ArgumentParser(description='Кривая продуктовой метрики по дням сезона')
    ap.add_argument('--name', required=True)
    ap.add_argument('--scheme', default='ww_loo')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--cutoffs', nargs='+', type=int, default=[90, 120, 151, 181, 0],
                    help='дни от 1 января года урожая; 0 — весь сезон')
    ap.add_argument('--top', type=int, default=3)
    a = ap.parse_args()

    global _DS, _YEARS
    _DS = R.Dataset(a.task, 'raw', 0)
    _YEARS = _DS.ids['year'].to_numpy(dtype=int)
    scheme = FD.read_definition(a.scheme)
    names = sorted(str(x) for x in scheme['fold'].unique())
    out = os.path.join(config.REPORTS_DIR, 'experiments', a.name)
    os.makedirs(out, exist_ok=True)
    runs = os.path.join(out, 'runs.csv')

    done = set()
    if os.path.exists(runs):
        prev = pd.read_csv(runs, sep=';', decimal=',', encoding='utf-8-sig')
        done = set(zip(prev['cutoff'].astype(int), prev['fold'].astype(str)))

    for cutoff in a.cutoffs:
        todo = [f for f in names if (cutoff, f) not in done]
        if not todo:
            print(f'отсечка {cutoff or "весь сезон"}: уже есть')
            continue
        print(f'\n=== отсечка {cutoff or "весь сезон"}: строю признаки', flush=True)
        feats = _columns(a.task, cutoff)
        print(f'    признаков {len(feats)}', flush=True)
        for fold in todo:
            t0 = time.time()
            spec = FD.fold_spec(scheme, fold)
            _Xn, y, ph, tr, va, te = _DS.fold_arrays(spec)
            fit_y, _clip = ES.ratio_target(y, ph, tr)
            trv = np.concatenate([tr, va])             # «прошлые годы» целиком

            sc = {}
            for i, (name, col) in enumerate(feats):
                s = rank_by_year(fit_y[trv], col[trv], _YEARS[trv])
                if np.isfinite(s):
                    sc[i] = s
            if not sc:
                print(f'   {fold}: ни один признак не оценился'); continue
            top = sorted(sc, key=lambda i: -abs(sc[i]))[:a.top]
            best = top[0]

            def signed(i):
                return (np.sign(sc[i]) or 1.0) * feats[i][1]

            acc = np.zeros(len(y))
            for i in top:                              # z-оценки по обучению, знаки по обучению
                col = feats[i][1]
                mu, sd = np.nanmean(col[trv]), (np.nanstd(col[trv]) or 1.0)
                acc += (np.sign(sc[i]) or 1.0) * np.nan_to_num((col - mu) / sd)
            nd = next((i for i, (n, _c) in enumerate(feats) if n == NDVI_EARLY), None)

            row = {'task': a.task, 'scheme': a.scheme, 'cutoff': cutoff, 'fold': fold,
                   'n_train': len(trv), 'n_test': len(te),
                   'лет_в_тесте': int(len(np.unique(_YEARS[te]))),
                   'лучший_признак': feats[best][0], 'обуч': round(sc[best], 4),
                   'тест_один': round(rank_by_year(fit_y[te], signed(best)[te], _YEARS[te]), 4),
                   'тест_топ3': round(rank_by_year(fit_y[te], acc[te], _YEARS[te]), 4),
                   'тест_ndvi': (round(rank_by_year(fit_y[te], signed(nd)[te], _YEARS[te]), 4)
                                 if nd is not None else ''),
                   'minutes': round((time.time() - t0) / 60, 2)}
            FE._append(runs, row, COLUMNS)
            print(f'   {fold}: {feats[best][0]} обуч {sc[best]:+.3f} -> тест '
                  f'{row["тест_один"]:+.3f}, топ{a.top} {row["тест_топ3"]:+.3f}, '
                  f'{NDVI_EARLY} {row["тест_ndvi"]}', flush=True)

    d = pd.read_csv(runs, sep=';', decimal=',', encoding='utf-8-sig')
    print('\n=== кривая: ранговая корреляция отклонений внутри незнакомого года')
    g = d.groupby('cutoff')
    t = pd.DataFrame({
        'фолдов': g.size(),
        'один_признак': g['тест_один'].mean(),
        'ст.ошибка': g['тест_один'].std() / np.sqrt(g.size()),
        'выше_0': d.assign(b=d['тест_один'] > 0).groupby('cutoff')['b'].sum(),
        'топ3': g['тест_топ3'].mean(),
        NDVI_EARLY: g['тест_ndvi'].mean(),
    })
    print(t.round(3).to_string())
    print('\nчаще всего выбираемый признак по отсечкам:')
    print(d.groupby('cutoff')['лучший_признак'].agg(lambda s: s.value_counts().idxmax())
          .to_string())


if __name__ == '__main__':
    main()
