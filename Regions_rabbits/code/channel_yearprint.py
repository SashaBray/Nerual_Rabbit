"""Насколько каждый канал ряда опознаёт ГОД, а не район: измерение вместо рассуждения.

Зачем. Ансамбль одноканальных моделей показал, что каналы ведут себя по-разному: `mean_temp`
в режиме с утечкой даёт 8.07 ц/га, а в честном — 11.44, тогда как `ndvi` держится ровно
(7.93 и 9.34), а `prod_hist_ma` вообще не меняется (9.71 и 9.67). Объяснение предлагалось
такое: метеополя почти одинаковы у всех районов внутри одного года, поэтому служат ярлыком
года — подсказкой, которой в настоящем прогнозе не будет.

Здесь это проверяется прямо. Для каждого канала берётся его среднее за целевой сезон (одно
число на пример) и дисперсия раскладывается по двум факторам::

    доля года   — какую часть разброса объясняет принадлежность к году
    доля района — какую часть объясняет принадлежность к району

У «ярлыка года» доля года велика, а доля района мала: зная значение канала, легко назвать
год и невозможно назвать район. У полезного признака должно быть наоборот — либо хотя бы
заметная доля района, потому что урожайность в первую очередь районная.

    python code/channel_yearprint.py --task winter_wheat_raw_v2

Отчёт — ``workspace/reports/experiments/<name>/channel_yearprint.csv``.
"""
import argparse
import os

import numpy as np
import pandas as pd

import config
import run_experiment as R


def factor_share(values, labels):
    """Доля дисперсии ``values``, объясняемая принадлежностью к группе ``labels`` (R² ANOVA)."""
    v = np.asarray(values, float)
    total = float(np.sum((v - v.mean()) ** 2))
    if total <= 0:
        return float('nan')
    within = 0.0
    for lab in np.unique(labels):
        g = v[labels == lab]
        if len(g):
            within += float(np.sum((g - g.mean()) ** 2))
    return 1.0 - within / total


def year_share_anomaly(curves, terr, years):
    """Доля года в АНОМАЛИИ хода канала: сколько общего у разных районов в один год.

    Зачем не просто «доля года»: постоянная подпись района (давление зависит от высоты,
    температура — от широты) занимает почти всю дисперсию уровня и к году отношения не имеет.
    Чтобы увидеть ярлык года, подпись надо снять: из хода каждого примера вычитается средний
    ход ЭТОГО района по всем его годам. Остаётся аномалия — «чем этот сезон отличался от
    обычного здесь». Если у разных районов аномалии одного года похожи, год опознаётся.

    Считается по дням: для каждого дня сезона — доля дисперсии аномалии, объясняемая годом;
    затем среднее по дням (взвешенное по дисперсии дня, чтобы пустые дни не тянули на себя).
    """
    A = np.asarray(curves, float).copy()
    for t in np.unique(terr):
        m = terr == t
        A[m] -= np.nanmean(A[m], axis=0)
    num = den = 0.0
    for d in range(A.shape[1]):
        v = A[:, d]
        ok = np.isfinite(v)
        if ok.sum() < 10:
            continue
        var = float(np.var(v[ok]))
        if var <= 0:
            continue
        sh = factor_share(v[ok], years[ok])
        if sh == sh:
            num += sh * var
            den += var
    return num / den if den > 0 else float('nan')


def run(task, name, channels='raw'):
    ds = R.Dataset(task, channels, 0)
    size = int(ds.meta['time_rows_size'])
    newest_first = bool(ds.meta.get('concat_newest_first', True))
    n_years = ds.Xb.shape[2] // size
    lo = 0 if newest_first else (n_years - 1) * size
    target = ds.Xb[:, :, lo:lo + size]                 # только целевой сезон
    years = ds.ids['year'].to_numpy(dtype=int)
    terr = ds.tid
    y = ds.y
    ph = ds.prod_hist
    rel = y / np.where(np.isnan(ph) | (ph <= 0), np.nan, ph)

    rows = []
    names = list(ds.feature_names) + ['prod_hist_ma']
    extra = np.repeat(np.where(np.isnan(ph), np.nanmedian(ph), ph).reshape(-1, 1, 1),
                      size, axis=2).astype(np.float32)
    target = np.concatenate([target, extra], axis=1)
    for i, nm in enumerate(names):
        m = np.nanmean(target[:, i, :], axis=1)
        ok = np.isfinite(m)
        row = {'канал': nm,
               'доля года': round(factor_share(m[ok], years[ok]), 3),
               'доля района': round(factor_share(m[ok], terr[ok]), 3),
               'доля года (аномалия)': round(year_share_anomaly(target[:, i, :], terr, years), 3)}
        for tag, q in (('r с урожайностью', y), ('r с относительной', rel)):
            k = ok & np.isfinite(q)
            row[tag] = round(float(np.corrcoef(m[k], np.asarray(q, float)[k])[0, 1]), 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['год/район'] = (df['доля года'] / df['доля района'].replace(0, np.nan)).round(2)
    df = df.sort_values('год/район', ascending=False)
    out = os.path.join(config.REPORTS_DIR, 'experiments', name)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'channel_yearprint.csv')
    df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    print()
    print(df.to_string(index=False))
    print('\nсохранено:', path)
    return df


def main():
    ap = argparse.ArgumentParser(description='Доля дисперсии канала по годам и по районам')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--name', default='ww_single')
    ap.add_argument('--channels', default='raw', choices=['raw', 'relative'])
    a = ap.parse_args()
    run(a.task, a.name, a.channels)


if __name__ == '__main__':
    main()
