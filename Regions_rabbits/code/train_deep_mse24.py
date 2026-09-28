"""Углублённая свёрточная mse24: +10 слоёв в FC-воронке и BatchNorm (в conv и FC).

База — наша «MSE 24» (2 strided-conv 2×/4× + широкая FC-воронка, ELU). Здесь:
  - FC-воронка углублена с 9 до 19 скрытых слоёв (исходные [1292×3,500×3,200×3] -> [1292×4,500×6,200×9]);
  - добавлен BatchNorm после каждого conv и каждого FC-слоя;
  - dropout снижен до 0.1 (он применяется после каждого из 19 слоёв; BatchNorm регуляризует).
Датасет winter_wheat_honest (климат-каналы + заполнение климатологией, аугментация), случайный сплит.
Честный протокол: ранняя остановка на ТЕСТОВОЙ; ВАЛИДАЦИОННАЯ — только финал.

Запуск:  python code/train_deep_mse24.py
Выход:   workspace/reports/deep_mse24/
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch

import archsearch as A
import config
import explore_models as E
import train_nn as T

EPOCHS, BATCH, PATIENCE, LR, SEED = 150, 256, 18, 1e-3, 42


def main(batch=BATCH):
    outdir = os.path.join(config.REPORTS_DIR, 'deep_mse24')
    os.makedirs(outdir, exist_ok=True)
    D = A.prep()
    inch = D['in_ch']
    cfg = dict(name='mse24_deep10_bn',
               channels=[inch * 2, inch * 4], kernels=[4, 4], strides=[3, 3],
               bn=True, adaptive=None,
               fc=[1292] * 4 + [500] * 6 + [200] * 9,          # 19 скрытых = исходные 9 + 10
               fc_bn=True, dropout=0.1, act='elu')
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = A.ConvNet(inch, D['in_len'], cfg)
    npar = sum(p.numel() for p in model.parameters())
    print(f'mse24_deep10_bn: conv {cfg["channels"]} (k4 s3, +BN), FC 19 слоёв {cfg["fc"]} (+BN, do0.1); '
          f'flat={model.flat}, параметров={npar:,}')

    print(f'Обучение (батч={batch}, ранняя остановка на ТЕСТОВОЙ)…', flush=True)
    hist, bep, bmse = T.train(model, None, D['ytr'], D['Xtest'], D['ytest'], 'cpu',
                              EPOCHS, batch, LR, PATIENCE, train_ds=D['train_ds'], lr_sched=True)
    p_te = T.predict(model, D['Xtest'], 'cpu')
    p_va = T.predict(model, D['Xval'], 'cpu')
    mte, mva = E.metrics(D['ytest'], p_te), E.metrics(D['yval'], p_va)
    tr_last = hist[-1]['train_mse']; gap = mva['MSE'] - tr_last
    print(f'\n=== батч {batch} ===')
    print(f'TRAIN MSE (посл.): {tr_last:.2f}')
    print(f'ТЕСТ (отбор):      MSE={mte["MSE"]:.2f}  R²={mte["R2"]:.3f}')
    print(f'ВАЛИДАЦИЯ (финал): MSE={mva["MSE"]:.2f}  R²={mva["R2"]:.3f}  (разрыв вал−train={gap:.1f})')
    print(f'(эталон v12: тест 26.7 / вал 27.4; батч 256 дал: тест 25.74 / вал 29.54 / train 9.3; эпох {len(hist)})')

    tag = f'_b{batch}'
    pd.DataFrame([{'split': 'test', 'batch': batch, **mte}, {'split': 'val', 'batch': batch, **mva}]).to_csv(
        os.path.join(outdir, f'deep_mse24_metrics{tag}.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(hist).to_csv(os.path.join(outdir, f'training_history{tag}.csv'),
                              index=False, encoding='utf-8-sig')
    print(f'Готово: {outdir}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args()
    main(batch=a.batch)
