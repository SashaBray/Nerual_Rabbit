"""Сравнение способов представления «будущего» на честном датасете — заблаговременность.

Версии (winter_wheat_honest_e2e_nn), все: свёрточная модель + аугментация, случайный сплit:
  v12 — полный: климат-каналы + заполнение будущего климатологией;
  v13 — опыт E: БЕЗ климат-каналов, заполнение будущего климатологией района;
  v14 — опыт F: БЕЗ климат-каналов, заполнение будущего НУЛЯМИ (проще готовить прогноз).
Линия «пол» — прогноз только по средней урожайности (prod_hist).

Запуск:  python code/compare_fill_honest.py
Выход:   workspace/reports/honest_multi_winter_wheat/compare_fill_*
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import compare_augment_honest as CA
import config
import plots
import report_honest_multi as R

DATASET = 'winter_wheat_honest'
VERSIONS = [('v12', 'полный: клим-каналы + клим', 'with clim-channels + clim fill', '-'),
            ('v13', 'E: без клим-каналов, клим', 'E: no clim-channels, clim fill', '--'),
            ('v14', 'F: без клим-каналов, нули', 'F: no clim-channels, zeros fill', ':')]


def main(dataset=DATASET):
    outdir = os.path.join(config.REPORTS_DIR, 'honest_multi_winter_wheat')
    os.makedirs(outdir, exist_ok=True)
    res = [(ru, en, ls, R.curves(dataset, v)[0]) for v, ru, en, ls in VERSIONS]
    base = CA.prodhist_baseline(dataset)

    def fig(split, metric):
        def draw(lang):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for ru, en, ls, cur in res:
                c = cur[split]
                ax.plot(c.days_before_harvest, c[metric], marker='.', ls=ls,
                        label=plots.tr(ru, en, lang))
            ax.invert_xaxis()
            if metric == 'R2':
                ax.axhline(0, color='gray', lw=0.8, ls=':')
            bval = base[split]['mse' if metric == 'MSE' else 'r2']
            ax.axhline(bval, color='red', lw=1.3, ls='-.',
                       label=plots.tr(f'только средняя урожайность ({bval:.0f})' if metric == 'MSE'
                                      else f'только средняя урожайность ({bval:.2f})',
                                      f'average-yield only ({bval:.0f})' if metric == 'MSE'
                                      else f'average-yield only ({bval:.2f})', lang))
            ax.set_xlabel(plots.tr('Дней до уборки (время →, ближе к уборке)',
                                   'Days before harvest (time →, toward harvest)', lang))
            ax.set_ylabel(f'{metric} ({split})')
            ax.set_title(plots.tr(
                f'Представление «будущего»: {metric}, {split} — озимая пшеница',
                f'Future representation: {metric}, {split} — winter wheat', lang))
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            fig.tight_layout()
            return fig
        return draw

    for split in ('test', 'val'):
        for metric in ('R2', 'MSE'):
            plots.bilingual(outdir, f'compare_fill_{split}_{metric.lower()}', fig(split, metric))

    print('\n=== MSE на уборке (0) / на 940 дн (и «пол») ===')
    for split in ('test', 'val'):
        print(f'  [{split}] пол: MSE={base[split]["mse"]:.1f}')
        for ru, en, ls, cur in res:
            c = cur[split].sort_values('days_before_harvest')
            print(f'    {ru:32s} MSE(0)={c.iloc[0].MSE:.1f}/R²={c.iloc[0].R2:.3f}  '
                  f'MSE(940)={c.iloc[-1].MSE:.1f}/R²={c.iloc[-1].R2:.3f}')
    print(f'\nГотово: {outdir} (compare_fill_*)')


if __name__ == '__main__':
    main()
