"""Диаграммы связи «максимум NDVI ↔ урожайность» в трёх постановках.

Три панели по одним и тем же парам «район, год»:

1. **как есть** — maxNDVI против абсолютной урожайности: видно, что общей связи почти нет,
   облако расслоено по районам (у каждого свой уровень);
2. **урожайность относительная** — maxNDVI против урожайности, делённой на скользящую среднюю
   района: нормирован отклик, но не признак;
3. **обе величины относительные** — NDVI к своей норме (среднее за 4 предыдущих года) против
   той же относительной урожайности. Это постановка, в которой одна прямая на все районы
   обыгрывает нейросеть.

    python code/plot_ndvi_relation.py --name ww2x2

Нужен кэш максимумов NDVI ``workspace/reports/experiments/<name>/maxndvi.csv``
(колонки territory_id, year, max_ndvi); если его нет, он собирается из базы (долго).
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

CULTURE = 'Озимая пшеница'


def load_points(name, task, culture):
    """Таблица точек: район, год, maxNDVI, норма NDVI, урожайность, скользящая средняя."""
    out = os.path.join(config.REPORTS_DIR, 'experiments', name)
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, 'maxndvi.csv')
    cfg = P.resolve_culture(culture)
    plan = FD.load_plan(task)
    plan = plan[FD.valid_mask(plan)].copy()

    if os.path.exists(cache):
        nd = pd.read_csv(cache, encoding='utf-8-sig')
        ndvi = {(str(t), int(y)): v for t, y, v in zip(nd['territory_id'], nd['year'], nd['max_ndvi'])}
    else:
        ndvi = {}
        years = range(int(plan['year'].min()) - 5, int(plan['year'].max()) + 1)
        for i, tid in enumerate(sorted(plan['territory_id'].unique()), 1):
            parm = P._ndvi_parm(cfg, int(str(tid).split('_')[0]))
            for y in years:
                v = P._max_ndvi(cfg, parm, tid, y) if parm else None
                ndvi[(str(tid), int(y))] = float(v) if v else np.nan
            if i % 100 == 0:
                print(f'   NDVI: район {i}', flush=True)
        pd.DataFrame([{'territory_id': t, 'year': y, 'max_ndvi': v} for (t, y), v in ndvi.items()]) \
            .to_csv(cache, index=False, encoding='utf-8-sig')

    rows = []
    for tid, year, yld in zip(plan['territory_id'], plan['year'], plan['productive']):
        tid, year = str(tid), int(year)
        n = ndvi.get((tid, year), np.nan)
        hist = [ndvi.get((tid, y), np.nan) for y in range(year - 4, year)]
        hist = [h for h in hist if h == h and h > 0]
        ph = B._prod_hist(db.get_yields(tid, cfg['bdpmo_culture']), year, 4)
        if not (n == n and n > 0) or not hist or not (ph == ph and ph > 0):
            continue
        rows.append({'territory_id': tid, 'year': year, 'max_ndvi': n,
                     'ndvi_norm': float(np.mean(hist)), 'yield': float(yld), 'prod_hist': float(ph)})
    df = pd.DataFrame(rows)
    df['ndvi_rel'] = df['max_ndvi'] / df['ndvi_norm']
    df['yield_rel'] = df['yield'] / df['prod_hist']
    return df, out


def _panel(ax, x, y, xlabel, ylabel, title, ylim=None):
    import matplotlib.pyplot as plt                        # noqa: F401 — стиль задаётся снаружи
    ax.scatter(x, y, s=5, alpha=0.18, color='#4C6EF5', linewidths=0)
    r = float(np.corrcoef(x, y)[0, 1])
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 50)
    ax.plot(xs, k * xs + b, color='#E8590C', lw=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nr = {r:.2f},  R² = {r * r:.2f}', fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(np.percentile(x, 0.5), np.percentile(x, 99.5))
    if ylim:
        ax.set_ylim(*ylim)
    return r


def main():
    ap = argparse.ArgumentParser(description='Диаграммы NDVI ↔ урожайность')
    ap.add_argument('--name', default='ww2x2')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--culture', default=CULTURE)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df, out = load_points(a.name, a.task, a.culture)
    print(f'точек: {len(df)}, районов: {df["territory_id"].nunique()}, '
          f'годы {df["year"].min()}–{df["year"].max()}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    r1 = _panel(axes[0], df['max_ndvi'].values, df['yield'].values,
                'максимум NDVI за сезон', 'урожайность, ц/га',
                '1. Как есть: NDVI → урожайность')
    r2 = _panel(axes[1], df['max_ndvi'].values, df['yield_rel'].values * 100,
                'максимум NDVI за сезон', 'урожайность, % от скользящей средней',
                '2. Отклик относительный', ylim=(20, 220))
    r3 = _panel(axes[2], df['ndvi_rel'].values * 100, df['yield_rel'].values * 100,
                'NDVI, % от нормы района', 'урожайность, % от скользящей средней',
                '3. Обе величины относительные', ylim=(20, 220))
    fig.suptitle(f'{a.culture}: {len(df)} пар «район, год», {df["territory_id"].nunique()} районов, '
                 f'{df["year"].min()}–{df["year"].max()}', fontsize=12)
    fig.tight_layout()
    path = os.path.join(out, 'ndvi_vs_yield.png')
    fig.savefig(path, dpi=150)
    print(f'r: как есть {r1:.2f}, отклик относительный {r2:.2f}, обе относительные {r3:.2f}')
    print('сохранено:', path)
    df.to_csv(os.path.join(out, 'ndvi_points.csv'), index=False, sep=';', decimal=',',
              encoding='utf-8-sig')


if __name__ == '__main__':
    main()
