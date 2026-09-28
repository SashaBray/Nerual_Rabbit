"""Сравнение honest-моделей С аугментацией и БЕЗ — компромисс «пик в 0 vs устойчивость на горизонте».

Обе обучены на честном датасете winter_wheat_honest, одинаковый случайный сплит (seed 42).
  aug   — winter_wheat_honest_e2e_nn/<--ver-aug>   (по умолч. v5)
  noaug — winter_wheat_honest_e2e_nn/<--ver-noaug> (по умолч. v6, обучена только на полных данных)

Запуск:  python code/compare_augment_honest.py [--ver-aug v5 --ver-noaug v6]
Выход:   workspace/reports/honest_multi_winter_wheat/compare_aug_*
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import mean_squared_error as _mse, r2_score as _r2

import config
import explore_models as E
import plots
import report_honest_multi as R
import train_honest as TH


def prodhist_baseline(dataset, seed=42):
    """«Пол» — прогноз только по средней урожайности (prod_hist целевого года, известной к уборке).

    identity = прогноз = средняя урожайность напрямую; те же примеры/сплит, что у honest-моделей.
    """
    z, meta = TH.load_honest(dataset)
    ph = z['ph']; y = z['y'].astype(float); ny = int(meta['duration_years'])
    keep = ~np.isnan(y) & (y >= 0) & (y <= E.YIELD_MAX)
    ph, y = ph[keep], y[keep]
    pht = ph[:, ny - 1]
    tr, te, va = TH.split_random(len(y), seed)
    med = np.nanmedian(pht[tr]); pht = np.where(np.isnan(pht), med, pht)
    out = {}
    for nm, idx in (('test', te), ('val', va)):
        out[nm] = {'mse': float(_mse(y[idx], pht[idx])), 'r2': float(_r2(y[idx], pht[idx]))}
    return out


def main(ver_aug='v12', ver_noaug='v11', dataset='winter_wheat_honest'):
    outdir = os.path.join(config.REPORTS_DIR, 'honest_multi_winter_wheat')
    os.makedirs(outdir, exist_ok=True)
    aug, _, _ = R.curves(dataset, ver_aug)
    noaug, _, _ = R.curves(dataset, ver_noaug)
    base = prodhist_baseline(dataset)
    models = [('с аугментацией', 'with augmentation', aug, '-'),
              ('без аугментации', 'without augmentation', noaug, '--')]

    def fig(split, metric):
        def draw(lang):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for ru, en, res, ls in models:
                c = res[split]
                ax.plot(c.days_before_harvest, c[metric], marker='.', ls=ls,
                        label=plots.tr(ru, en, lang))
            ax.invert_xaxis()
            if metric == 'R2':
                ax.axhline(0, color='gray', lw=0.8, ls=':')
            # «пол»: модель только на средней урожайности (prod_hist)
            bval = base[split]['mse' if metric == 'MSE' else 'r2']
            ax.axhline(bval, color='red', lw=1.3, ls='-.',
                       label=plots.tr(f'только средняя урожайность ({metric}={bval:.0f})'
                                      if metric == 'MSE' else f'только средняя урожайность ({metric}={bval:.2f})',
                                      f'average-yield only ({metric}={bval:.0f})'
                                      if metric == 'MSE' else f'average-yield only ({metric}={bval:.2f})', lang))
            ax.set_xlabel(plots.tr('Дней до уборки (время →, ближе к уборке)',
                                   'Days before harvest (time →, toward harvest)', lang))
            ax.set_ylabel(f'{metric} ({split})')
            ax.set_title(plots.tr(
                f'Аугментация: компромисс ({metric}, {split}) — озимая пшеница',
                f'Augmentation trade-off ({metric}, {split}) — winter wheat', lang))
            ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
            fig.tight_layout()
            return fig
        return draw

    imgs = []
    for split in ('test', 'val'):
        for metric in ('R2', 'MSE'):
            name = f'compare_aug_{split}_{metric.lower()}'
            imgs.append((f'{split} — {metric}', plots.bilingual(outdir, name, fig(split, metric))))

    # краткая сводка в консоль: R² в точке 0 и в дальнем горизонте
    print('\n=== MSE/R² при заблаговременности 0 / макс (и «пол» — средняя урожайность) ===')
    for split in ('test', 'val'):
        print(f'  [{split}] пол (средняя урожайность): MSE={base[split]["mse"]:.1f}  R²={base[split]["r2"]:.3f}')
    for ru, en, res, _ in models:
        for split in ('test', 'val'):
            c = res[split].sort_values('days_before_harvest')
            print(f'  {ru:16s} {split}: MSE(0)={c.iloc[0].MSE:.1f}/R²={c.iloc[0].R2:.3f}  '
                  f'MSE({int(c.iloc[-1].days_before_harvest)}д)={c.iloc[-1].MSE:.1f}/R²={c.iloc[-1].R2:.3f}')
    print(f'\nГотово: {outdir} (compare_aug_*)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ver-aug', default='v12')
    ap.add_argument('--ver-noaug', default='v11')
    ap.add_argument('--dataset', default='winter_wheat_honest')
    a = ap.parse_args()
    main(a.ver_aug, a.ver_noaug, a.dataset)
