"""Свёрточный автоэнкодер рядов: сжать сезон в K чисел, заморозить, предсказывать по ним.

Замысел (предложен в работе над статьёй). Все прежние попытки упирались в одно: на входе
12 каналов × 1095 дней = 13 140 чисел, а обучающих примеров 809–887. Архитектуры перебраны от
12.2 млн весов до 32 тысяч — ни одна на честном сплите не обошла скользящую среднюю района.
Размерность входа при этом не менялась ни разу.

Здесь она сокращается до обучения с метками, и сокращается БЕЗ МЕТОК:

1. Автоэнкодер сжимает ход одного канала за сезон в K чисел и восстанавливает его обратно.
   Учится на одних рядах, урожайности не видит.
2. Энкодер один и тот же на все каналы. Тогда ему достаётся 12 кривых с каждого примера:
   850 примеров × 12 ≈ 10 200 кривых на 25 тысяч весов. Впервые за всю работу примеров
   больше, чем весов.
3. Веса замораживаются, сверху ставится персептрон 12×K -> 64 -> 1, и он учится предсказывать
   отклонение урожайности от скользящей средней. Межканальные связи сохраняются (персептрон
   видит все каналы разом), но вход у него десятки чисел, а не тринадцать тысяч.

Зачем это нужно, даже если персептрон проиграет. Сейчас отрицательный результат слабого вида:
«ничего из пробованного не сработало» — всегда можно возразить, что плохо искали. Если же
персептрон не обойдёт эталон, А ОШИБКА ВОССТАНОВЛЕНИЯ ПРИ ЭТОМ МАЛА, вывод становится
сильным: вот сжатое представление, из которого ряд восстанавливается с точностью таких-то
процентов, и урожайности в нём нет. Поэтому ошибка восстановления по каналам — такая же часть
отчёта, как RMSE.

Две особенности, без которых затея не работает.

**Кодируется АНОМАЛИЯ, а не сырая кривая.** Автоэнкодер оптимизирует восстановление, а не
полезность: он потратит горло на то, что даёт больше всего разброса. А разброс сырой кривой —
это на 96–99% постоянная подпись района (высота, широта) плюс сезонный цикл, одинаковый каждый
год (измерено в ``channel_yearprint.py``). Межгодовая аномалия — малая добавка, и её он
выбросил бы первой. Поэтому из целевого сезона вычитается среднее двух предыдущих сезонов
ТОГО ЖЕ примера: они уже лежат во входе, поэтому аномалия честна по построению — только
прошлое, ничего из будущего.

**Энкодер получает указатель канала.** Каналы устроены по-разному: NDVI — гладкая горка,
накопленные осадки — монотонная лестница, высота снега — ноль полгода и зимние пики. Один
базис на всё это восстанавливал бы их неравномерно, поэтому и энкодеру, и декодеру подаётся
короткий обучаемый признак канала. Декодер знает канал, значит коду не нужно тратиться на
хранение того, какой это канал.

Энкодер учится ТОЛЬКО на кривых обучающего блока фолда: иначе базис подсмотрел бы тестовые
годы и честность сплита развалилась бы.

    python code/autoencoder.py --name ww_ae --scheme ww_blocks --codes 4 8 16 32 --seeds 3

Отчёты: ``runs.csv`` (строка на прогон персептрона) и ``recon.csv`` (ошибка восстановления
по каналам).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
import ensemble_single as ES
import explore_models as E
import folds as FD
import run_experiment as R
import train_nn as T

RUN_COLUMNS = ['task', 'fold', 'code', 'seed', 'ae_params', 'head_params', 'n_train', 'n_val',
               'n_test', 'ae_epochs', 'ae_val_mse', 'recon_r2_test', 'head_epochs',
               'head_best_epoch', 'head_val_mse', 'val_rmse', 'base_val_rmse', 'test_rmse',
               'test_r2', 'base_rmse', 'base_r2', 'minutes',
               'finished_at']
RECON_COLUMNS = ['fold', 'code', 'seed', 'ch', 'ch_name', 'recon_r2', 'доля разброса']


def _append(path, row, columns):
    pd.DataFrame([row], columns=columns).to_csv(
        path, mode='a', header=not os.path.exists(path), index=False, sep=';', decimal=',',
        encoding='utf-8-sig')


class SeriesAutoencoder(nn.Module):
    """Общий на все каналы автоэнкодер одной сезонной кривой: (1, L) -> K чисел -> (1, L)."""

    def __init__(self, n_channels, length, code=8, emb=4, width=(16, 24, 32)):
        super().__init__()
        c1, c2, c3 = width
        self.length, self.code_size = length, code
        self.emb = nn.Embedding(n_channels, emb)
        self.enc = nn.Sequential(
            nn.Conv1d(1 + emb, c1, kernel_size=9, padding=4), nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(c1, c2, kernel_size=7, padding=3), nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2), nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )
        self.enc_len = length // 4 // 4
        self.to_code = nn.Linear(c3 * self.enc_len, code)
        self.from_code = nn.Linear(code + emb, c3 * self.enc_len)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=4), nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=4), nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, 1, kernel_size=9, padding=4),
        )
        self.c3 = c3

    def encode(self, x, ch):
        e = self.emb(ch).unsqueeze(2).expand(-1, -1, x.shape[2])
        h = self.enc(torch.cat([x, e], dim=1))
        return self.to_code(h.flatten(1))

    def decode(self, z, ch):
        h = self.from_code(torch.cat([z, self.emb(ch)], dim=1))
        h = self.dec(h.view(-1, self.c3, self.enc_len))
        return F.interpolate(h, size=self.length, mode='linear', align_corners=False)

    def forward(self, x, ch):
        return self.decode(self.encode(x, ch), ch)


def anomaly_curves(Xn, meta):
    """Аномалия целевого сезона: он сам минус среднее двух предыдущих. -> (n, каналов, L).

    Прошлые сезоны уже лежат во входе примера, поэтому вычитание не трогает ни одного дня,
    которого не было бы у прогнозиста в момент прогноза.
    """
    size = int(meta['time_rows_size'])
    n_years = Xn.shape[2] // size
    blocks = Xn.reshape(Xn.shape[0], Xn.shape[1], n_years, size)
    newest_first = bool(meta.get('concat_newest_first', True))
    tgt = 0 if newest_first else n_years - 1
    past = [k for k in range(n_years) if k != tgt]
    return (blocks[:, :, tgt] - blocks[:, :, past].mean(axis=2)).astype(np.float32)


def flatten_curves(A, idx):
    """Из (n, C, L) сделать набор одиночных кривых: (n*C, 1, L) и номера каналов (n*C,)."""
    sub = A[idx]
    n, C, L = sub.shape
    x = sub.reshape(n * C, 1, L)
    ch = np.tile(np.arange(C), n).astype(np.int64)
    return np.ascontiguousarray(x), ch


def train_ae(model, Xtr, chtr, Xva, chva, epochs, batch, lr, patience, quiet=False):
    """Обучить автоэнкодер на восстановление. -> (эпох, лучшая val MSE)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(chtr)),
                    batch_size=batch, shuffle=True)
    Xva_t, chva_t = torch.from_numpy(Xva), torch.from_numpy(chva)
    best, best_state, bad, ep = float('inf'), None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, cb in dl:
            opt.zero_grad()
            loss = lossf(model(xb, cb), xb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = float(lossf(model(Xva_t, chva_t), Xva_t).item())
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if not quiet and (ep % 10 == 0 or ep == 1):
            print(f'      АЭ эпоха {ep:3d}: val MSE восстановления {va:.4f} (лучшая {best:.4f})',
                  flush=True)
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return ep, best


@torch.no_grad()
def codes_of(model, A, batch=256):
    """Коды всех примеров: (n, каналов*K)."""
    model.eval()
    n, C, L = A.shape
    out = np.empty((n, C * model.code_size), dtype=np.float32)
    ch_all = torch.arange(C)
    for i in range(0, n, batch):
        sub = A[i:i + batch]
        b = sub.shape[0]
        x = torch.from_numpy(np.ascontiguousarray(sub.reshape(b * C, 1, L)))
        ch = ch_all.repeat(b)
        z = model.encode(x, ch).cpu().numpy()
        out[i:i + b] = z.reshape(b, C * model.code_size)
    return out


@torch.no_grad()
def recon_quality(model, A, batch=256):
    """Доля удержанного разброса кривой по каналам. -> (recon_r2 по каналам, доля разброса)."""
    model.eval()
    n, C, L = A.shape
    sse = np.zeros(C)
    ch_all = torch.arange(C)
    for i in range(0, n, batch):
        sub = A[i:i + batch]
        b = sub.shape[0]
        x = torch.from_numpy(np.ascontiguousarray(sub.reshape(b * C, 1, L)))
        ch = ch_all.repeat(b)
        rec = model(x, ch).cpu().numpy().reshape(b, C, L)
        sse += ((rec - sub) ** 2).sum(axis=(0, 2))
    mean = A.mean(axis=(0, 2), keepdims=True)
    sst = ((A - mean) ** 2).sum(axis=(0, 2))
    r2 = 1.0 - sse / np.where(sst > 0, sst, np.nan)
    return r2, sst / sst.sum()


def build_head(n_in, hidden=64, dropout=0.2):
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                        nn.Linear(hidden, 1))
    return T.SqueezeHead(net)


def main():
    ap = argparse.ArgumentParser(description='Автоэнкодер рядов + персептрон на замороженных кодах')
    ap.add_argument('--name', default='ww_ae')
    ap.add_argument('--scheme', default='ww_blocks')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--codes', nargs='+', type=int, default=[4, 8, 16, 32])
    ap.add_argument('--folds', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--ae-epochs', type=int, default=200)
    ap.add_argument('--ae-patience', type=int, default=25)
    ap.add_argument('--ae-batch', type=int, default=256)
    ap.add_argument('--ae-lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--patience', type=int, default=80)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    scheme = FD.read_definition(a.scheme)
    fold_names = a.folds or sorted(str(x) for x in scheme['fold'].unique())
    specs = [FD.fold_spec(scheme, f) for f in fold_names]
    out = os.path.join(config.REPORTS_DIR, 'experiments', a.name)
    os.makedirs(out, exist_ok=True)
    runs_path, recon_path = os.path.join(out, 'runs.csv'), os.path.join(out, 'recon.csv')
    done = set()
    if os.path.exists(runs_path):
        prev = pd.read_csv(runs_path, sep=';', decimal=',', encoding='utf-8-sig')
        done = set(zip(prev['fold'].astype(str), prev['code'].astype(int), prev['seed'].astype(int)))

    plan = [(str(s['fold']), k, sd) for s in specs for k in a.codes
            for sd in range(1, a.seeds + 1)]
    todo = [p for p in plan if p not in done]
    print(f'Опыт «{a.name}»: фолдов {len(specs)}, размеров кода {len(a.codes)}, seed {a.seeds} '
          f'-> прогонов {len(plan)}, осталось {len(todo)}')
    print(f'   коды: {a.codes}; схема {a.scheme}; цель — отклонение от скользящей средней')
    if a.dry_run:
        return
    with open(os.path.join(out, 'experiment.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': a.name, 'scheme': a.scheme, 'task': a.task, 'codes': a.codes,
                   'folds': fold_names, 'seeds': a.seeds, 'target': 'ratio',
                   'anomaly': 'целевой сезон минус среднее двух предыдущих',
                   'ae_epochs': a.ae_epochs, 'epochs': a.epochs}, f, ensure_ascii=False, indent=2)

    ds = R.Dataset(a.task, 'raw', 0)
    names = list(ds.feature_names)        # без prod_hist_ma: он не кодируется (см. ниже)
    # Автоэнкодер не зависит от seed персептрона: он один и тот же для пары (фолд, код),
    # поэтому обучается ОДИН РАЗ, а на его кодах учатся несколько персептронов. Так тренировок
    # автоэнкодера 16 вместо 48, и разброс между seed меряется там, где он и есть — в голове.
    groups = {}
    for (fold, K, seed) in todo:
        groups.setdefault(fold, {}).setdefault(K, []).append(seed)
    n_pairs = sum(len(v) for v in groups.values())
    t_start, i = time.time(), 0

    for fold in sorted(groups):
        spec = next(s for s in specs if str(s['fold']) == fold)
        Xn, y, ph, tr, va, te = ds.fold_arrays(spec)
        # последний канал (prod_hist_ma) постоянен по времени, его аномалия — тождественный
        # ноль: кодировать нечего. Уровень района отдаётся персептрону отдельным числом.
        A = anomaly_curves(Xn, ds.meta)[:, :-1]
        fit_y, clip = ES.ratio_target(y, ph, tr)
        Xc_tr, ch_tr = flatten_curves(A, tr)
        Xc_va, ch_va = flatten_curves(A, va)

        for K in sorted(groups[fold]):
            seeds = sorted(groups[fold][K])
            i += 1
            t0 = time.time()
            torch.manual_seed(10_000 + K)                  # у автоэнкодера свой постоянный seed
            np.random.seed(10_000 + K)
            ae = SeriesAutoencoder(A.shape[1], A.shape[2], code=K)
            ae_params = sum(p.numel() for p in ae.parameters())
            print(f'[{i}/{n_pairs}] фолд {fold}, код {K}: кривых на автоэнкодер '
                  f'{len(Xc_tr)}, весов {ae_params:,}; персептронов {len(seeds)}…', flush=True)
            ae_ep, ae_val = train_ae(ae, Xc_tr, ch_tr, Xc_va, ch_va, a.ae_epochs, a.ae_batch,
                                     a.ae_lr, a.ae_patience)
            for prm in ae.parameters():
                prm.requires_grad_(False)

            r2_ch, share = recon_quality(ae, A[te])
            for j, nm in enumerate(names):
                _append(recon_path, {'fold': fold, 'code': K, 'seed': 0, 'ch': j, 'ch_name': nm,
                                     'recon_r2': round(float(r2_ch[j]), 4),
                                     'доля разброса': round(float(share[j]), 4)}, RECON_COLUMNS)
            r2_all = float(np.nansum(r2_ch * share))
            ae_min = (time.time() - t0) / 60
            print(f'     восстановление: удержано {r2_all * 100:.1f}% разброса '
                  f'(худший канал {names[int(np.nanargmin(r2_ch))]} '
                  f'{np.nanmin(r2_ch) * 100:.0f}%), эпох {ae_ep}, {ae_min:.1f} мин', flush=True)

            Z = np.concatenate([codes_of(ae, A), ph.reshape(-1, 1)], axis=1)  # + уровень района
            zm, zs = Z[tr].mean(axis=0), Z[tr].std(axis=0) + 1e-6
            Zn = ((Z - zm) / zs).astype(np.float32)

            for seed in seeds:
                t1 = time.time()
                torch.manual_seed(seed)
                np.random.seed(seed)
                head = build_head(Zn.shape[1])
                head_params = sum(p.numel() for p in head.parameters())
                hist, best_epoch, _best_val = T.train(
                    head, Zn[tr], fit_y[tr], Zn[va], fit_y[va], 'cpu', a.epochs, a.batch, a.lr,
                    a.patience)
                pred = ES.to_absolute(T.predict(head, Zn[te], 'cpu'), ph[te], clip)
                m, base = E.metrics(y[te], pred), E.metrics(y[te], ph[te])
                pred_va = ES.to_absolute(T.predict(head, Zn[va], 'cpu'), ph[va], clip)
                mv, basev = E.metrics(y[va], pred_va), E.metrics(y[va], ph[va])
                _append(runs_path, {
                    'task': a.task, 'fold': fold, 'code': K, 'seed': seed,
                    'ae_params': ae_params, 'head_params': head_params, 'n_train': len(tr),
                    'n_val': len(va), 'n_test': len(te), 'ae_epochs': ae_ep,
                    'ae_val_mse': round(float(ae_val), 5), 'recon_r2_test': round(r2_all, 4),
                    'head_epochs': len(hist), 'head_best_epoch': best_epoch,
                    'head_val_mse': round(float(_best_val), 4),
                    'val_rmse': round(float(mv['RMSE']), 4),
                    'base_val_rmse': round(float(basev['RMSE']), 4),
                    'test_rmse': round(float(m['RMSE']), 4),
                    'test_r2': round(float(m['R2']), 4),
                    'base_rmse': round(float(base['RMSE']), 4),
                    'base_r2': round(float(base['R2']), 4),
                    'minutes': round((time.time() - t1) / 60, 1),
                    'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, RUN_COLUMNS)
                print(f'     персептрон seed {seed} ({head_params:,} весов): тест RMSE '
                      f'{m["RMSE"]:.2f} / R² {m["R2"]:.3f} против эталона {base["RMSE"]:.2f}',
                      flush=True)
            left = (n_pairs - i) * (time.time() - t_start) / i / 60
            print(f'     осталось примерно {left:.0f} мин', flush=True)

    if os.path.exists(runs_path):
        d = pd.read_csv(runs_path, sep=';', decimal=',', encoding='utf-8-sig')
        print('\n=== RMSE и восстановление по размеру кода:')
        print(d.pivot_table(index='code', values=['test_rmse', 'base_rmse', 'recon_r2_test'],
                            aggfunc='mean').round(3).to_string())
        print('\n=== по фолдам:')
        print(d.pivot_table(index='fold', columns='code', values='test_rmse',
                            aggfunc='mean').round(2).to_string())
    if os.path.exists(recon_path):
        c = pd.read_csv(recon_path, sep=';', decimal=',', encoding='utf-8-sig')
        print('\n=== удержано разброса кривой по каналам (recon R²):')
        print(c.pivot_table(index='ch_name', columns='code', values='recon_r2',
                            aggfunc='mean').round(3).to_string())


if __name__ == '__main__':
    main()
