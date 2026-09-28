"""Диаграмма реальная vs предсказанная урожайность на РАВНОМЕРНОЙ тестовой выборке:
равномерная модель vs модель с оверсэмплингом аномалий. Аномальные точки (верх.25% по |урожай−prod_hist|) выделены.

Нужны preds_none.npz и preds_dev.npz (из train_emphasis.py). Запуск: python code/scatter_emphasis.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
import plots

OUT = os.path.join(config.REPORTS_DIR, 'emphasis')


def main():
    none = np.load(os.path.join(OUT, 'preds_none.npz'))
    dev = np.load(os.path.join(OUT, 'preds_dev.npz'))
    d = none['dev']                                   # |y - prod_hist| на тесте (один и тот же тест)
    anom = d >= np.quantile(d, 0.75)
    panels = [('равномерная модель', 'uniform model', none),
              ('оверсэмплинг аномалий', 'anomaly oversampling', dev)]

    def draw(lang):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        for ax, (ru, en, P) in zip(axes, panels):
            y, pred = P['y'], P['pred']
            lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
            ax.scatter(y[~anom], pred[~anom], s=10, alpha=0.35, color='#9aa0a6', edgecolors='none',
                       label=plots.tr('обычные', 'typical', lang))
            ax.scatter(y[anom], pred[anom], s=18, alpha=0.65, color='#d62728', edgecolors='none',
                       label=plots.tr('аномальные (верх.25% |dev|)', 'anomalous (top 25% |dev|)', lang))
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x')
            o = float(np.mean((y - pred) ** 2)); a = float(np.mean((y[anom] - pred[anom]) ** 2))
            ax.set_title(f'{plots.tr(ru, en, lang)}\n' +
                         plots.tr(f'MSE: общий {o:.1f}, аномальные {a:.1f}',
                                  f'MSE: overall {o:.1f}, anomalous {a:.1f}', lang))
            ax.set_xlabel(plots.tr('Реальная урожайность, ц/га', 'Actual yield, c/ha', lang))
            ax.set_ylabel(plots.tr('Предсказанная урожайность, ц/га', 'Predicted yield, c/ha', lang))
            ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'box'); ax.legend(fontsize=8)
        fig.suptitle(plots.tr('Реальная и предсказанная урожайность (равномерный тест) — озимая пшеница',
                              'Actual vs predicted yield (uniform test) — winter wheat', lang))
        fig.tight_layout()
        return fig

    p = plots.bilingual(OUT, 'scatter_emphasis_test', draw)
    print(f'Диаграмма: {OUT}\\scatter_emphasis_test_*.{{png,pdf}} (RU/EN)')
    return p


if __name__ == '__main__':
    main()
