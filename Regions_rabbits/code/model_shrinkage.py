"""Регрессия по NDVI с частичным объединением районов (shrinkage).

Задача. Связь «NDVI к норме района → урожайность к скользящей средней» внутри района
заметно теснее (медиана r = 0.74), чем в общем облаке (r = 0.50), поэтому регрессию строят
по району. Но на район приходится всего 4–9 лет, и наклон по ним определяется неуверенно:
районные наклоны лежат в пределах 2.3…5.2 вокруг глобального 3.6, то есть расходятся в
основном из-за шума оценки, а не потому, что районы устроены по-разному.

Решение. Брать районный наклон, но подтягивать его к глобальному тем сильнее, чем меньше у
района наблюдений::

    k = w * k_район + (1 - w) * k_глобальный,   w = n / (n + λ)

Сила подтягивания λ подбирается по ВАЛИДАЦИОННОМУ году фолда, тест не используется.
λ = 0 — чистая районная регрессия, λ → ∞ — чистая глобальная; обе считаются для сравнения.

    python code/model_shrinkage.py --name ww2x2

Нужны точки из ``plot_ndvi_relation.py`` (``ndvi_points.csv``) и схема фолдов.
"""
import argparse
import os

import numpy as np
import pandas as pd

import build_dataset as B
import config
import db
import folds as FD
import app_predict as P

LAMBDAS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 1e9]


def build_points(name, culture='Озимая пшеница'):
    """Точки за ВСЕ доступные годы, а не только за годы датасета.

    Районная регрессия имеет право опираться на всю историю района до целевого года —
    это прошлое, а не утечка. Урожайности в базе есть с 2007 года, максимумы NDVI — с 2009
    (кэш ``maxndvi.csv``), поэтому на район набирается 10–13 точек вместо 3–7, как было бы
    по годам одного фолда.
    """
    out = os.path.join(config.REPORTS_DIR, 'experiments', name)
    full = os.path.join(out, 'ndvi_points_full.csv')
    if os.path.exists(full):
        df = pd.read_csv(full, sep=';', decimal=',', encoding='utf-8-sig')
        df['territory_id'] = df['territory_id'].astype(str)
        return df, out
    nd = pd.read_csv(os.path.join(out, 'maxndvi.csv'), encoding='utf-8-sig')
    ndvi = {(str(t), int(y)): v for t, y, v in zip(nd['territory_id'], nd['year'], nd['max_ndvi'])}
    cfg = P.resolve_culture(culture)
    rows = []
    for tid in sorted({t for t, _y in ndvi}):
        yields = db.get_yields(tid, cfg['bdpmo_culture'])
        for year, val in yields.items():
            if val is None or val != val or val <= 0:
                continue
            n = ndvi.get((tid, int(year)), np.nan)
            hist = [ndvi.get((tid, y), np.nan) for y in range(int(year) - 4, int(year))]
            hist = [h for h in hist if h == h and h > 0]
            ph = B._prod_hist(yields, int(year), 4)
            if not (n == n and n > 0) or not hist or not (ph == ph and ph > 0):
                continue
            rows.append({'territory_id': tid, 'year': int(year), 'max_ndvi': n,
                         'ndvi_norm': float(np.mean(hist)), 'yield': float(val),
                         'prod_hist': float(ph)})
    df = pd.DataFrame(rows)
    df['ndvi_rel'] = df['max_ndvi'] / df['ndvi_norm']
    df['yield_rel'] = df['yield'] / df['prod_hist']
    df.to_csv(full, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    return df, out


def _fit_line(x, y):
    """Наклон и свободный член; при вырожденных данных — нулевой наклон."""
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0, float(np.mean(y)) if len(y) else 1.0
    k, b = np.polyfit(x, y, 1)
    return float(k), float(b)


def predict(points, train_idx, target_idx, lam):
    """Прогноз относительной урожайности для ``target_idx`` при силе подтягивания ``lam``."""
    tr = points.loc[train_idx]
    k_g, b_g = _fit_line(tr['ndvi_rel'].to_numpy(), tr['yield_rel'].to_numpy())
    gx, gy = tr['ndvi_rel'].mean(), tr['yield_rel'].mean()

    by_terr = {}
    for tid, g in tr.groupby('territory_id'):
        n = len(g)
        k_d, _b = _fit_line(g['ndvi_rel'].to_numpy(), g['yield_rel'].to_numpy())
        w = 0.0 if lam >= 1e8 else (1.0 if lam == 0 else n / (n + lam))
        k = w * k_d + (1 - w) * k_g
        mx = w * g['ndvi_rel'].mean() + (1 - w) * gx          # центр тоже подтягиваем
        my = w * g['yield_rel'].mean() + (1 - w) * gy
        by_terr[tid] = (k, mx, my, n)

    out = []
    for _i, row in points.loc[target_idx].iterrows():
        p = by_terr.get(row['territory_id'])
        if p is None:                                         # района не было в обучении
            rel = b_g + k_g * row['ndvi_rel']
        else:
            k, mx, my, _n = p
            rel = my + k * (row['ndvi_rel'] - mx)
        out.append(np.clip(rel, 0.3, 2.5) * row['prod_hist'])
    return np.array(out)


def rmse_r2(y, pred):
    e = np.asarray(pred, float) - np.asarray(y, float)
    return (float(np.sqrt(np.mean(e ** 2))),
            1 - float(np.sum(e ** 2)) / float(np.sum((y - np.mean(y)) ** 2)))


def run(name, scheme_name='winter_wheat'):
    pts, out = build_points(name)
    scheme = FD.read_definition(scheme_name)
    print(f'точек всего: {len(pts)}, районов {pts["territory_id"].nunique()}, '
          f'годы {pts["year"].min()}–{pts["year"].max()}; '
          f'лет на район в среднем {len(pts) / pts["territory_id"].nunique():.1f}')

    rows = []
    for fold_year in sorted(scheme['fold'].unique()):
        spec = FD.fold_spec(scheme, int(fold_year))
        test_pairs = {(t, y) for (t, y) in spec['pairs'] if y == spec['test_year']}
        val_pairs = {(t, y) for (t, y) in spec['pairs'] if y == spec['val_year']}
        pair_of = list(zip(pts['territory_id'], pts['year']))
        te_idx = pts.index[[p in test_pairs for p in pair_of]]
        va_idx = pts.index[[p in val_pairs for p in pair_of]]
        # обучение — ВСЁ прошлое до целевого года (для подбора λ — всё до валидационного)
        past_test = pts.index[pts['year'].to_numpy() < spec['test_year']]
        past_val = pts.index[pts['year'].to_numpy() < spec['val_year']]

        scores = {}
        for lam in LAMBDAS:
            pv = predict(pts, past_val, va_idx, lam)
            scores[lam] = rmse_r2(pts.loc[va_idx, 'yield'].to_numpy(), pv)[0]
        lam_best = min(scores, key=scores.get)

        y_te = pts.loc[te_idx, 'yield'].to_numpy()
        res = {'фолд': int(fold_year), 'тест': len(te_idx),
               'λ (по валидации)': ('∞' if lam_best >= 1e8 else f'{lam_best:g}')}
        for tag, lam in (('районная (λ=0)', 0.0), ('глобальная (λ=∞)', 1e9),
                         ('подтянутая', lam_best)):
            r, q = rmse_r2(y_te, predict(pts, past_test, te_idx, lam))
            res[f'{tag} RMSE'] = round(r, 2)
            res[f'{tag} R²'] = round(q, 3)
        rows.append(res)
        print(f'  фолд {fold_year}: λ={res["λ (по валидации)"]}, '
              f'районная {res["районная (λ=0) RMSE"]}, глобальная {res["глобальная (λ=∞) RMSE"]}, '
              f'подтянутая {res["подтянутая RMSE"]}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'shrinkage.csv'), index=False, sep=';', decimal=',',
              encoding='utf-8-sig')
    print()
    print(df.to_string(index=False))
    print()
    for tag in ('районная (λ=0)', 'глобальная (λ=∞)', 'подтянутая'):
        print(f'  {tag:18s} среднее RMSE {df[f"{tag} RMSE"].mean():5.2f}   '
              f'R² {df[f"{tag} R²"].mean():.3f}')
    print('\nсохранено:', os.path.join(out, 'shrinkage.csv'))
    return df


def main():
    ap = argparse.ArgumentParser(description='NDVI-регрессия с частичным объединением районов')
    ap.add_argument('--name', default='ww2x2')
    ap.add_argument('--scheme', default='winter_wheat')
    a = ap.parse_args()
    run(a.name, a.scheme)


if __name__ == '__main__':
    main()
