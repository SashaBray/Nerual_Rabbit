"""Оверсэмплинг «аномальных» примеров (большое |урожай − prod_hist|) для глубокой mse24+BN.

Гипотеза: сеть плохо предсказывает случаи с большим отклонением урожая от средней продуктивности
(регрессия к среднему). Решение — чаще показывать такие примеры через WeightedRandomSampler
(он повторяет редкие примеры с большим |dev| чаще, не меняя сам датасет).

Сравнение: --emphasis none (обычное обучение) vs --emphasis dev (вес ∝ |dev|^gamma).
Оценка — общая и ОТДЕЛЬНО на «аномальной» подвыборке (верхний квартиль по |dev|), на ТЕСТЕ и ВАЛИДАЦИИ.
Честный протокол: ранняя остановка на ТЕСТОВОЙ; ВАЛИДАЦИОННАЯ — финал.

Запуск:  python code/train_emphasis.py --emphasis dev --gamma 2.0 [--batch 512]
Выход:   workspace/reports/emphasis/
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

import archsearch as A
import config
import explore_models as E
import train_nn as T

EPOCHS, PATIENCE, LR, SEED = 150, 18, 1e-3, 42


def dev_weights(dev, gamma):
    """Вес примера ∝ (|dev|, обрезанное по 95-му перцентилю, нормированное на медиану)^gamma + пол."""
    d = np.minimum(dev, np.quantile(dev, 0.95))
    med = np.median(d) + 1e-6
    return ((d / med) + 0.2) ** gamma


def subset_mse(y, pred, dev, frac=0.25):
    """MSE на «аномальной» подвыборке — верхний frac по |dev|."""
    thr = np.quantile(dev, 1 - frac)
    m = dev >= thr
    return float(np.mean((y[m] - pred[m]) ** 2)), int(m.sum())


def main(emphasis='dev', gamma=2.0, batch=512):
    outdir = os.path.join(config.REPORTS_DIR, 'emphasis')
    os.makedirs(outdir, exist_ok=True)
    D = A.prep()
    inch = D['in_ch']
    dev_tr = np.abs(D['ytr'] - D['phtr'])
    dev_te = np.abs(D['ytest'] - D['phtest'])
    dev_va = np.abs(D['yval'] - D['phval'])

    sampler = None
    if emphasis == 'dev':
        w = dev_weights(dev_tr, gamma)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
        top = dev_tr >= np.quantile(dev_tr, 0.75)
        share = w[top].sum() / w.sum()
        print(f'Оверсэмплинг dev: gamma={gamma}; верхний квартиль по |dev| получит ~{share*100:.0f}% '
              f'выборок (без оверсэмплинга было 25%).')
    else:
        print('Без оверсэмплинга (обычное равномерное обучение).')

    cfg = dict(channels=[inch * 2, inch * 4], kernels=[4, 4], strides=[3, 3], bn=True, adaptive=None,
               fc=[1292] * 4 + [500] * 6 + [200] * 9, fc_bn=True, dropout=0.1, act='elu')
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = A.ConvNet(inch, D['in_len'], cfg)
    print(f'mse24_deep10_bn, батч={batch}, эмфазис={emphasis}. Обучение…', flush=True)
    hist, bep, _ = T.train(model, None, D['ytr'], D['Xtest'], D['ytest'], 'cpu',
                           EPOCHS, batch, LR, PATIENCE, train_ds=D['train_ds'],
                           lr_sched=True, sampler=sampler)

    p_te = T.predict(model, D['Xtest'], 'cpu')
    p_va = T.predict(model, D['Xval'], 'cpu')
    mte, mva = E.metrics(D['ytest'], p_te), E.metrics(D['yval'], p_va)
    a_te, n_te = subset_mse(D['ytest'], p_te, dev_te)
    a_va, n_va = subset_mse(D['yval'], p_va, dev_va)
    print(f'\n=== эмфазис={emphasis} (gamma={gamma}, батч={batch}) ===')
    print(f'ТЕСТ     : общий MSE={mte["MSE"]:.2f} R²={mte["R2"]:.3f} | аномальные (верх.25%, n={n_te}) MSE={a_te:.1f}')
    print(f'ВАЛИДАЦИЯ: общий MSE={mva["MSE"]:.2f} R²={mva["R2"]:.3f} | аномальные (верх.25%, n={n_va}) MSE={a_va:.1f}')

    pd.DataFrame([{'split': 'test', 'overall_MSE': round(mte['MSE'], 2), 'overall_R2': round(mte['R2'], 3),
                   'anom_MSE': round(a_te, 1)},
                  {'split': 'val', 'overall_MSE': round(mva['MSE'], 2), 'overall_R2': round(mva['R2'], 3),
                   'anom_MSE': round(a_va, 1)}]).to_csv(
        os.path.join(outdir, f'emphasis_{emphasis}_g{gamma}_b{batch}.csv'),
        index=False, encoding='utf-8-sig')
    # сохраняем предсказания на РАВНОМЕРНОЙ тестовой выборке (для диаграммы реальная vs предсказанная)
    np.savez(os.path.join(outdir, f'preds_{emphasis}.npz'),
             y=D['ytest'], pred=p_te, dev=dev_te)
    print(f'Готово: {outdir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--emphasis', default='dev', choices=['none', 'dev'])
    ap.add_argument('--gamma', type=float, default=2.0)
    ap.add_argument('--batch', type=int, default=512)
    a = ap.parse_args()
    main(a.emphasis, a.gamma, a.batch)
