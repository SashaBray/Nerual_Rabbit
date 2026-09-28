"""MSE по корзинам отклонения |урожай − prod_hist| на равномерном тесте: равномерная vs оверсэмплинг.
Наглядно показывает, что выигрыш оверсэмплинга сосредоточен в верхних корзинах (аномалии).

Нужны preds_none.npz и preds_dev.npz. Запуск: python code/error_vs_dev.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import plots

OUT = os.path.join(config.REPORTS_DIR, 'emphasis')
NB = 5


def main():
    none = np.load(os.path.join(OUT, 'preds_none.npz'))
    dev = np.load(os.path.join(OUT, 'preds_dev.npz'))
    d = none['dev']
    edges = np.quantile(d, np.linspace(0, 1, NB + 1))
    edges[-1] += 1e-6
    idx = np.clip(np.digitize(d, edges) - 1, 0, NB - 1)
    labels, mse_none, mse_dev, ns = [], [], [], []
    for b in range(NB):
        m = idx == b
        labels.append(f'{edges[b]:.0f}–{edges[b+1]:.0f}')
        mse_none.append(float(np.mean((none['y'][m] - none['pred'][m]) ** 2)))
        mse_dev.append(float(np.mean((dev['y'][m] - dev['pred'][m]) ** 2)))
        ns.append(int(m.sum()))
    tab = pd.DataFrame({'|dev|_корзина': labels, 'n': ns,
                        'MSE_равномерная': np.round(mse_none, 1), 'MSE_оверсэмплинг': np.round(mse_dev, 1)})
    tab.to_csv(os.path.join(OUT, 'error_vs_dev.csv'), index=False, encoding='utf-8-sig')
    print(tab.to_string(index=False))

    def draw(lang):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(NB); w = 0.38
        ax.bar(x - w / 2, mse_none, w, color='#9aa0a6',
               label=plots.tr('равномерная модель', 'uniform model', lang))
        ax.bar(x + w / 2, mse_dev, w, color='#d62728',
               label=plots.tr('оверсэмплинг аномалий', 'anomaly oversampling', lang))
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel(plots.tr('Корзина |урожай − prod_hist|, ц/га', '|yield − prod_hist| bin, c/ha', lang))
        ax.set_ylabel(plots.tr('MSE на тесте', 'test MSE', lang))
        ax.set_title(plots.tr('MSE по величине отклонения от средней продуктивности — озимая пшеница',
                              'Test MSE by deviation from average productivity — winter wheat', lang))
        for i in range(NB):
            ax.text(i + w / 2, mse_dev[i], f'{mse_dev[i]:.0f}', ha='center', va='bottom', fontsize=8)
            ax.text(i - w / 2, mse_none[i], f'{mse_none[i]:.0f}', ha='center', va='bottom', fontsize=8)
        ax.legend(); ax.grid(True, axis='y', alpha=0.3); fig.tight_layout()
        return fig

    plots.bilingual(OUT, 'error_vs_dev', draw)
    print(f'\nГрафик: {OUT}\\error_vs_dev_{{ru,en}}.{{png,pdf}}')


if __name__ == '__main__':
    main()
