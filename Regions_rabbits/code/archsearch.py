"""Самостоятельный поиск архитектуры свёрточной сети на честном датасете.

Датасет winter_wheat_honest (климат-каналы + заполнение будущего средней климатологией, аугментация),
случайный сплит 70/15/15. Перебираем конфигурации свёрточной сети: BatchNorm, тип пулинга
(max / avg / adaptive), число и ширину conv-слоёв, голову FC. Отбор по val MSE, отчёт по test.
Все обучения — одинаковые честные настройки (мягкий планировщик LR, 150 эпох, ранняя остановка).

Запуск:  python code/archsearch.py
Выход:   workspace/reports/archsearch/
"""

import gc
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import explore_models as E
import plots
import train_honest as TH
import train_nn as T

DATASET = 'winter_wheat_honest'
SEED = 42
EPOCHS, BATCH, PATIENCE, LR, AUG_STEP = 120, 256, 14, 1e-3, 7


def prep():
    z, meta = TH.load_honest(DATASET)
    raw, clim, ph, valid = z['raw'], z['clim'], z['ph'], z['valid']
    ids, y = z['ids'], z['y'].astype(np.float32)
    keep = ~np.isnan(y) & (y >= 0) & (y <= E.YIELD_MAX)
    raw, clim, ph, valid, ids, y = (a[keep] for a in (raw, clim, ph, valid, ids, y))
    n_years = int(meta['duration_years']); size = int(meta['time_rows_size']); P = int(meta['n_params'])
    Tlen = n_years * size
    tr, te, va = TH.split_random(len(y), SEED)
    # Конвенция: ТЕСТОВАЯ (te) — для ранней остановки и отбора архитектуры; ВАЛИДАЦИОННАЯ (va) —
    # только финальная проверка, при доработке НЕ используется.
    rawn, climn, phn, _ = TH.normalize(raw, clim, ph, tr)
    rawf = rawn.reshape(len(y), P, Tlen)
    h = (n_years - 1) * size + TH.HARVEST_DOY
    train_ds = TH.HonestAugDataset(rawf[tr], climn[tr], phn[tr], y[tr], valid[tr],
                                   AUG_STEP, n_years, size, augment=True, fill='clim', clim_channels=True)
    bi = lambda idx: TH.build_input(rawf[idx], climn[idx], phn[idx], h, n_years - 1, n_years, size)
    pht = ph[:, n_years - 1].astype(np.float64)               # prod_hist целевого года (raw, как у y)
    med = float(np.nanmedian(pht[tr])); pht = np.where(np.isnan(pht), med, pht)
    return dict(train_ds=train_ds, ytr=y[tr], Xtest=bi(te), ytest=y[te],
                Xval=bi(va), yval=y[va], in_ch=2 * P + 1, in_len=Tlen,
                n_tr=len(tr), n_test=len(te), n_val=len(va),
                phtr=pht[tr], phtest=pht[te], phval=pht[va])    # средняя продуктивность по выборкам


class ConvNet(nn.Module):
    def __init__(self, in_ch, in_len, cfg):
        super().__init__()
        Act = {'elu': nn.ELU, 'relu': nn.ReLU}[cfg.get('act', 'elu')]
        layers = []; c = in_ch
        for out, k, s in zip(cfg['channels'], cfg['kernels'], cfg['strides']):
            layers.append(nn.Conv1d(c, out, k, stride=s, padding=k // 2))
            if cfg.get('bn'):
                layers.append(nn.BatchNorm1d(out))
            layers.append(Act())
            bp = cfg.get('between_pool')
            if bp == 'max':
                layers.append(nn.MaxPool1d(2))
            elif bp == 'avg':
                layers.append(nn.AvgPool1d(2))
            c = out
        ad = cfg.get('adaptive')
        if ad:
            typ, n = ad
            layers.append(nn.AdaptiveMaxPool1d(n) if typ == 'max' else nn.AdaptiveAvgPool1d(n))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            self.flat = self.conv(torch.zeros(1, in_ch, in_len)).flatten(1).shape[1]
        fc = [nn.Flatten()]; prev = self.flat
        for hsz in cfg['fc']:
            fc.append(nn.Linear(prev, hsz))
            if cfg.get('fc_bn'):
                fc.append(nn.BatchNorm1d(hsz))
            fc.append(Act())
            if cfg.get('dropout'):
                fc.append(nn.Dropout(cfg['dropout']))
            prev = hsz
        fc.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*fc)

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


RND = '_honest'
# Только STRIDED-конфигурации (быстрые на CPU; в раунде 1 именно они побеждали).
CONFIGS = [
    dict(name='nobn_strided_adaptmax8', channels=[32, 64, 128], kernels=[7, 5, 3], strides=[2, 2, 2],
         bn=False, adaptive=('max', 8), fc=[256, 128], dropout=0.3),
    dict(name='bn_strided_adaptmax8', channels=[32, 64, 128], kernels=[7, 5, 3], strides=[2, 2, 2],
         bn=True, adaptive=('max', 8), fc=[256, 128], dropout=0.3),
    dict(name='wide_strided_adaptmax8', channels=[64, 128, 256], kernels=[9, 5, 3], strides=[2, 2, 2],
         bn=True, adaptive=('max', 8), fc=[512, 256], dropout=0.3),
    dict(name='wide_strided_adaptmax16', channels=[64, 128, 256], kernels=[9, 5, 3], strides=[2, 2, 2],
         bn=True, adaptive=('max', 16), fc=[512, 256], dropout=0.3),
    # «большие» модели — влезающие (макс 256 каналов) идут первыми; широкая-384 медленная -> меньше эпох
    dict(name='wide_strided_fcbig', channels=[64, 128, 256], kernels=[9, 5, 3], strides=[2, 2, 2],
         bn=True, adaptive=('max', 8), fc=[1024, 512], fc_bn=True, dropout=0.3),
    dict(name='wide_strided_4layer', channels=[64, 128, 256, 256], kernels=[9, 5, 3, 3],
         strides=[2, 2, 2, 1], bn=True, adaptive=('max', 8), fc=[512, 256], dropout=0.3),
    dict(name='wider_strided_adaptmax8', channels=[96, 192, 384], kernels=[9, 5, 3], strides=[2, 2, 2],
         bn=True, adaptive=('max', 8), fc=[512, 256], dropout=0.3, epochs=70),
]


def main():
    outdir = os.path.join(config.REPORTS_DIR, 'archsearch')
    os.makedirs(outdir, exist_ok=True)
    D = prep()
    print(f'Вход {D["in_ch"]}×{D["in_len"]}; train {D["n_tr"]}, ТЕСТ (отбор) {D["n_test"]}, '
          f'ВАЛИДАЦИЯ (финал) {D["n_val"]}. Конфигураций: {len(CONFIGS)}')
    cfg_by_name = {c['name']: c for c in CONFIGS}
    tmpdir = os.path.join(outdir, '_models'); os.makedirs(tmpdir, exist_ok=True)
    results_csv = os.path.join(outdir, f'archsearch_results{RND}.csv')
    done = {}                                                  # возобновление: пропускаем готовые
    if os.path.exists(results_csv):
        for _, r in pd.read_csv(results_csv).iterrows():
            if os.path.exists(os.path.join(tmpdir, f'{r["config"]}.pt')):
                done[r['config']] = r.to_dict()
    rows = []
    for i, cfg in enumerate(CONFIGS, 1):
        if cfg['name'] in done:
            rows.append(done[cfg['name']])
            print(f'\n[{i}/{len(CONFIGS)}] {cfg["name"]}: уже обучена — пропуск (resume)', flush=True)
            continue
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = ConvNet(D['in_ch'], D['in_len'], cfg)
        npar = sum(p.numel() for p in model.parameters())
        print(f'\n[{i}/{len(CONFIGS)}] {cfg["name"]}: flat={model.flat}, параметров={npar:,}', flush=True)
        # ранняя остановка — на ТЕСТОВОЙ (доработка); валидационную не трогаем
        hist, bep, bmse = T.train(model, None, D['ytr'], D['Xtest'], D['ytest'], 'cpu',
                                  cfg.get('epochs', EPOCHS), BATCH, LR, PATIENCE,
                                  train_ds=D['train_ds'], lr_sched=True)
        mte = E.metrics(D['ytest'], T.predict(model, D['Xtest'], 'cpu'))
        rows.append({'config': cfg['name'], 'params': npar, 'flat': model.flat, 'best_epoch': bep,
                     'test_MSE': round(mte['MSE'], 2), 'test_R2': round(mte['R2'], 3)})
        torch.save(model.state_dict(), os.path.join(tmpdir, f'{cfg["name"]}.pt'))  # на диск, не в ОЗУ
        print(f'    -> ТЕСТ MSE={mte["MSE"]:.2f} R²={mte["R2"]:.3f}', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(outdir, f'archsearch_results{RND}.csv'),
                                  index=False, encoding='utf-8-sig')
        del model, hist; gc.collect()                          # освободить память между конфигурациями

    df = pd.DataFrame(rows).sort_values('test_MSE').reset_index(drop=True)
    print('\n=== Отбор по ТЕСТОВОЙ выборке (валидационную не смотрели) ===')
    print(df.to_string(index=False))

    # ФИНАЛ: валидационную выборку раскрываем только теперь — для топ-3 по тесту (перезагружаем с диска)
    print('\n=== ФИНАЛ на ВАЛИДАЦИОННОЙ выборке (top-3 по тесту) ===')
    fin = []
    for name in df['config'].head(3):
        m = ConvNet(D['in_ch'], D['in_len'], cfg_by_name[name])
        m.load_state_dict(torch.load(os.path.join(tmpdir, f'{name}.pt'), map_location='cpu'))
        m.eval()
        mval = E.metrics(D['yval'], T.predict(m, D['Xval'], 'cpu'))
        del m; gc.collect()
        t = df[df['config'] == name].iloc[0]
        fin.append({'config': name, 'test_MSE': t['test_MSE'], 'test_R2': t['test_R2'],
                    'val_MSE': round(mval['MSE'], 2), 'val_R2': round(mval['R2'], 3)})
    fdf = pd.DataFrame(fin)
    print(fdf.to_string(index=False))
    df.to_csv(os.path.join(outdir, f'archsearch_results{RND}.csv'), index=False, encoding='utf-8-sig')
    fdf.to_csv(os.path.join(outdir, f'archsearch_final{RND}.csv'), index=False, encoding='utf-8-sig')
    _bar(df, fdf, outdir)
    json.dump({c['name']: c for c in CONFIGS}, open(os.path.join(outdir, 'configs.json'), 'w',
              encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    w = fdf.iloc[0]
    print(f'\nЛучшая (отбор по тесту): {w["config"]} — ТЕСТ MSE {w["test_MSE"]}, '
          f'ВАЛИДАЦИЯ MSE {w["val_MSE"]}\nГотово: {outdir}')


def _bar(df, fdf, outdir):
    valmap = dict(zip(fdf['config'], fdf['val_MSE']))

    def draw(lang):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(df))
        ax.bar(x, df['test_MSE'], 0.55,
               label=plots.tr('ТЕСТ MSE (отбор)', 'test MSE (selection)', lang), color='#ff7f0e')
        vx = [i for i, c in enumerate(df['config']) if c in valmap]
        ax.scatter(vx, [valmap[df['config'][i]] for i in vx], color='#1f77b4', zorder=5, s=55,
                   label=plots.tr('ВАЛИДАЦИЯ MSE (топ-3, финал)', 'validation MSE (top-3, final)', lang))
        ax.axhline(26.7, color='red', ls='-.', lw=1.2,
                   label=plots.tr('свёрточная v12 (тест 26.7)', 'conv v12 (test 26.7)', lang))
        ax.set_xticks(x); ax.set_xticklabels(df['config'], rotation=25, ha='right', fontsize=7)
        ax.set_ylabel('MSE'); ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)
        ax.set_title(plots.tr('Поиск архитектуры свёрточной сети — озимая пшеница',
                              'Conv-net architecture search — winter wheat', lang))
        fig.tight_layout()
        return fig
    return plots.bilingual(outdir, 'archsearch_mse'+RND, draw)


if __name__ == '__main__':
    main()
