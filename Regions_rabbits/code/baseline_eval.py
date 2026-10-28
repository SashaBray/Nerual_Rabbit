"""Эталоны без нейросети на той же схеме фолдов: с чем честно сравнивать сеть.

Считаются три предсказателя на тех же тестовых парах «район, год», что и сетка опытов:

* **скользящая средняя** — среднее урожайностей района за N последних лет С ДАННЫМИ до
  целевого (тот же признак ``prod_hist_ma``, который сеть получает отдельным каналом);
* **среднее обучения** — одно число на всё, нужно лишь как нижняя граница шкалы;
* **регрессия по NDVI** — классическая экспонента «урожайность ↔ максимум NDVI», подогнанная
  по годам района. Здесь она считается ЧЕСТНО: только по годам СТРОГО ДО целевого, тогда как
  в приложении (``app_predict.predict_ndvi``) подгонка идёт по всем годам района кроме
  целевого, то есть использует и будущее — на бэктесте это завышает её точность.

    python code/baseline_eval.py --scheme winter_wheat --task winter_wheat_raw_v2

Результат — ``workspace/reports/experiments/<имя>/baselines.csv``.
"""
import argparse
import os

import numpy as np
import pandas as pd

import app_predict as P
import build_dataset as B
import config
import db
import folds as FD
from ndvi_regression import fit_exp, pred_exp


def metrics(y, pred):
    e = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    r2 = 1 - float(np.sum(e ** 2)) / float(np.sum((y - np.mean(y)) ** 2))
    return rmse, r2


def ndvi_past_only(cfg, tid, year, parm, min_points=3):
    """Экспонента «урожайность ↔ maxNDVI» по годам СТРОГО ДО целевого. -> прогноз или None."""
    yields = db.get_yields(tid, cfg['bdpmo_culture'])
    pts = []
    for yy, val in yields.items():
        if yy >= int(year) or val is None or val != val:
            continue                                        # только прошлое, без целевого года
        mx = P._max_ndvi(cfg, parm, tid, yy)
        if mx is not None and mx > 0:
            pts.append((mx, float(val)))
    if len(pts) < min_points:
        return None
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    mx_t = P._max_ndvi(cfg, parm, tid, int(year))
    if mx_t is None or mx_t <= 0:
        return None
    fit = fit_exp(xs, ys)
    if fit is None:
        return float(np.mean(ys))
    xc = min(max(mx_t, float(xs.min())), float(xs.max()))   # без экстраполяции, как в приложении
    return float(min(pred_exp(fit, xc), min(5000.0, 1.5 * float(ys.max()))))


def run(scheme_name, task, culture, prod_hist_last=4, out_name=None):
    scheme = FD.read_definition(scheme_name)
    plan = FD.load_plan(task)
    plan = plan[FD.valid_mask(plan)]
    y_of = {(t, int(y)): float(v) for t, y, v in
            zip(plan['territory_id'], plan['year'], plan['productive'])}
    cfg = P.resolve_culture(culture)

    rows, per_pair = [], []
    for fold_year in sorted(scheme['fold'].unique()):
        spec = FD.fold_spec(scheme, int(fold_year))
        test = [(t, y) for (t, y) in sorted(spec['pairs'])
                if y == spec['test_year'] and (t, y) in y_of]
        train = [(t, y) for (t, y) in spec['pairs']
                 if y in spec['train_years'] and (t, y) in y_of]
        y_true = np.array([y_of[k] for k in test])
        train_mean = float(np.mean([y_of[k] for k in train]))

        ph, ndvi, n_ndvi = [], [], 0
        for tid, year in test:
            v = B._prod_hist(db.get_yields(tid, cfg['bdpmo_culture']), int(year), prod_hist_last)
            ph.append(v)
            parm = P._ndvi_parm(cfg, int(str(tid).split('_')[0]))
            pv = ndvi_past_only(cfg, tid, int(year), parm) if parm else None
            if pv is not None:
                n_ndvi += 1
            ndvi.append(pv)
        ph = np.array(ph, dtype=float)
        med = float(np.nanmedian([B._prod_hist(db.get_yields(t, cfg['bdpmo_culture']), int(y),
                                               prod_hist_last) for t, y in train[:400]]))
        ph_filled = np.where(np.isnan(ph), med, ph)

        r_ph, q_ph = metrics(y_true, ph_filled)
        r_mean, q_mean = metrics(y_true, np.full_like(y_true, train_mean))
        ok = np.array([v is not None for v in ndvi])
        nd = np.array([v if v is not None else np.nan for v in ndvi], dtype=float)
        r_nd, q_nd = metrics(y_true[ok], nd[ok]) if ok.sum() > 10 else (np.nan, np.nan)
        # NDVI там, где он есть, иначе скользящая средняя — чтобы сравнивать на всей выборке
        mix = np.where(ok, np.nan_to_num(nd, nan=0.0), ph_filled)
        r_mix, q_mix = metrics(y_true, mix)

        rows.append({'фолд': int(fold_year), 'тест': len(test),
                     'скольз. средняя RMSE': round(r_ph, 2), 'R²': round(q_ph, 3),
                     'NDVI-регрессия RMSE': round(r_nd, 2), 'R² ': round(q_nd, 3),
                     'NDVI посчитан': f'{ok.sum()}/{len(test)}',
                     'NDVI+средняя RMSE': round(r_mix, 2), 'R²  ': round(q_mix, 3),
                     'среднее обучения RMSE': round(r_mean, 2), 'R²   ': round(q_mean, 3)})
        for (t, y), yt, p1, p2 in zip(test, y_true, ph_filled, nd):
            per_pair.append({'fold': int(fold_year), 'territory_id': t, 'year': y, 'yield': yt,
                             'prod_hist': p1, 'ndvi_reg': p2})
        print(f'  фолд {fold_year}: готово ({len(test)} пар, NDVI посчитан у {ok.sum()})', flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(config.REPORTS_DIR, 'experiments', out_name or scheme_name)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'baselines.csv'), index=False, sep=';', decimal=',',
              encoding='utf-8-sig')
    pd.DataFrame(per_pair).to_csv(os.path.join(out, 'baselines_pairs.csv'), index=False,
                                  sep=';', decimal=',', encoding='utf-8-sig')
    print()
    print(df.to_string(index=False))
    print(f'\nсохранено: {os.path.join(out, "baselines.csv")}')
    return df


def main():
    ap = argparse.ArgumentParser(description='Эталоны без нейросети на схеме фолдов')
    ap.add_argument('--scheme', default='winter_wheat')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--culture', default='Озимая пшеница')
    ap.add_argument('--out', default='ww2x2', help='куда класть отчёт (папка опыта)')
    a = ap.parse_args()
    run(a.scheme, a.task, a.culture, out_name=a.out)


if __name__ == '__main__':
    main()
