"""Обучение ЧЕСТНОЙ нейросети заблаговременности на датасете winter_wheat_honest.

Вход модели на отсечке a (день): [сырьё(маск.) | климатология | prod_hist] = (2P+1) каналов × T.
  - сырьё: реальные ряды до дня a; дни > a заполнены климатологией года-прогноза b=a//size;
  - климатология: «типичный год», известный при прогнозе в году b (mean за предыдущие N лет);
  - prod_hist: средняя урожайность, известная при прогнозе в году b (урожаи <= year_b-1).
Аугментация: каждый пример обучается со случайной отсечкой a (валидные блоки), будущее -> климатология.
Сплит — ПО ГОДАМ (leave-one-year-out): train<=2020, val=2021, test=2022.

Запуск:  python code/train_honest.py [--dataset winter_wheat_honest] [--epochs 80 --batch 256]
Выход:   workspace/models/winter_wheat_honest_e2e_nn/vN/  (+ reports/nn/.../vN/)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import config
import explore_models as E
import plots
import train_nn as T

DATASET = 'winter_wheat_honest'
HARVEST_DOY = 210            # озимая пшеница (как в harvest_dates.csv)


def load_honest(dataset):
    d = os.path.join(config.DATASETS_DIR, dataset)
    z = np.load(os.path.join(d, 'honest.npz'))
    meta = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
    return z, meta


def split_random(n, seed, val_frac=0.15, test_frac=0.15):
    """Полностью случайный сплит train/test/val (70/15/15) по примерам.

    Честность вспомогательных признаков (prod_hist/климатология по дате прогноза) обеспечивается
    самим датасетом, а не сплитом.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac)); n_test = int(round(n * test_frac))
    va = idx[:n_val]; te = idx[n_val:n_val + n_test]; tr = idx[n_val + n_test:]
    return tr, te, va


def normalize(raw, clim, ph, tr):
    """raw,clim: (n,P,Y,size); ph:(n,Y). Стат по train; raw и clim — общая (одни единицы)."""
    mu = raw[tr].mean(axis=(0, 2, 3)).astype(np.float32)          # (P,)
    sd = (raw[tr].std(axis=(0, 2, 3)) + 1e-6).astype(np.float32)
    rawn = ((raw - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float32)
    climn = ((clim - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float32)
    ph_med = float(np.nanmedian(ph[tr]))
    phf = np.where(np.isnan(ph), ph_med, ph).astype(np.float32)
    ph_mu = float(phf[tr].mean()); ph_sd = float(phf[tr].std() + 1e-6)
    phn = ((phf - ph_mu) / ph_sd).astype(np.float32)
    norm = {'mu': mu.tolist(), 'sd': sd.tolist(), 'ph_median': ph_med,
            'ph_mu': ph_mu, 'ph_sd': ph_sd}
    return rawn, climn, phn, norm


def build_input(rawn, climn, phn, a, b, n_years, size, fill='clim', clim_channels=True):
    """Собрать вход для отсечки a (блок b): сырьё(маск.) [| клим-каналы по-блочно] | prod_hist.

    rawn: (n,P,Y*size); climn:(n,P,Y,size); phn:(n,Y). a — день отсечки, b — индекс блока-прогноза.
    fill: 'clim' — будущее (дни>a) заполняется климатологией года-прогноза clim(b); 'zeros' — нулями.
    clim_channels: добавлять ли 11 клим-каналов (по-блочно clim(min(j,b))). Без них вход = P+1 каналов.
    """
    n, P, Tlen = rawn.shape
    x_raw = rawn.copy()
    if a < Tlen:
        if fill == 'zeros':
            x_raw[:, :, a:] = 0.0                                  # «нет данных» -> 0 (нормированное)
        else:
            clim_fill = np.tile(climn[:, :, b, :], (1, 1, n_years))
            x_raw[:, :, a:] = clim_fill[:, :, a:]
    parts = [x_raw]
    if clim_channels:
        parts.append(np.concatenate([climn[:, :, min(j, b), :] for j in range(n_years)], axis=2))
    parts.append(np.broadcast_to(phn[:, b][:, None, None], (n, 1, Tlen)))
    return np.concatenate(parts, axis=1).astype(np.float32)


class HonestAugDataset(torch.utils.data.Dataset):
    """Случайная честная отсечка на пример: будущее -> климатология года-прогноза."""

    def __init__(self, rawn, climn, phn, y, valid, aug_step, n_years, size, augment=True,
                 fill='clim', clim_channels=True):
        self.raw = rawn; self.clim = climn; self.ph = phn
        self.y = y.astype(np.float32)
        self.n_years = n_years; self.size = size; self.T = rawn.shape[2]
        self.fill = fill; self.clim_channels = clim_channels
        if not augment:
            # без аугментации: только полные данные (прогноз в день уборки целевого года)
            self.cuts = [[self.T] for _ in range(rawn.shape[0])]
            return
        # список валидных отсечек на пример (блок отсечки должен иметь климатологию)
        all_a = list(range(0, self.T + 1, aug_step))
        self.cuts = []
        for i in range(rawn.shape[0]):
            ai = [a for a in all_a if valid[i, min(a // size, n_years - 1)]]
            self.cuts.append(ai if ai else [self.T])

    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, i):
        a = self.cuts[i][int(torch.randint(len(self.cuts[i]), (1,)).item())]
        b = min(a // self.size, self.n_years - 1)
        x_raw = self.raw[i].copy()
        if a < self.T:
            if self.fill == 'zeros':
                x_raw[:, a:] = 0.0
            else:
                x_raw[:, a:] = np.tile(self.clim[i, :, b, :], (1, self.n_years))[:, a:]
        parts = [x_raw]
        if self.clim_channels:
            parts.append(np.concatenate([self.clim[i, :, min(j, b), :]
                                         for j in range(self.n_years)], axis=1))
        parts.append(np.full((1, self.T), self.ph[i, b], np.float32))
        x = np.concatenate(parts, axis=0).astype(np.float32)
        return torch.from_numpy(x), self.y[i]


def run(dataset=DATASET, arch='cnn', epochs=80, batch=256, lr=1e-3, patience=12, seed=42,
        aug_step=7, name=None, augment=True, fill='clim', clim_channels=True):
    torch.manual_seed(seed); np.random.seed(seed)
    group = name or f'{dataset}_e2e_nn'
    ver = T._next_version(group); tag = f'v{ver}'; disp = f'{group}_{tag}'
    outdir = os.path.join(config.MODELS_DIR, f'{group}_{tag}')       # плоская папка модели
    report_dir = os.path.join(config.REPORTS_DIR, 'nn', group, tag)
    err_dir = os.path.join(report_dir, 'error_tables')
    os.makedirs(outdir, exist_ok=True); os.makedirs(err_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[1/5] Загрузка честного датасета {dataset}…')
    z, meta = load_honest(dataset)
    raw, clim, ph, valid = z['raw'], z['clim'], z['ph'], z['valid']
    ids, y = z['ids'], z['y'].astype(np.float32)
    n_years = int(meta['duration_years']); size = int(meta['time_rows_size'])
    P = int(meta['n_params']); Tlen = n_years * size
    keep = ~np.isnan(y) & (y >= 0) & (y <= E.YIELD_MAX)            # без метки/сентинелы урожайности
    n_drop = int((~keep).sum())
    raw, clim, ph, valid = raw[keep], clim[keep], ph[keep], valid[keep]
    ids, y = ids[keep], y[keep]
    print(f'      отброшено без валидной урожайности: {n_drop}; осталось {len(y)}')
    years = ids[:, 2]
    tr, te, va = split_random(len(y), seed)
    in_channels = (2 * P + 1) if clim_channels else (P + 1)
    print(f'      примеров {len(y)}; P={P}, n_years={n_years}, size={size}; вход C={in_channels} '
          f'(клим-каналы={clim_channels}, заполнение будущего={fill}), L={Tlen}; '
          f'сплит СЛУЧАЙНЫЙ 70/15/15 (train {len(tr)}, test {len(te)}, val {len(va)}); {device}')

    rawn, climn, phn, norm = normalize(raw, clim, ph, tr)
    rawf = rawn.reshape(len(y), P, Tlen)                           # (n,P,T) хронологически
    bi = lambda R, C, Ph, a, b: build_input(R, C, Ph, a, b, n_years, size, fill, clim_channels)

    print(f'[2/5] Сборка нейросети (arch={arch}, in_channels={in_channels})…')
    model, summary_text, n_params, arch_cfg = T.make_model(arch, in_channels, Tlen)
    print(summary_text)

    # вход для валидации/раннего стопа = ПОЛНЫЕ данные (прогноз в день уборки целевого года)
    harvest_abs = (n_years - 1) * size + HARVEST_DOY
    Xva_full = bi(rawf[va], climn[va], phn[va], harvest_abs, n_years - 1)

    train_ds = HonestAugDataset(rawf[tr], climn[tr], phn[tr], y[tr], valid[tr],
                                aug_step, n_years, size, augment=augment,
                                fill=fill, clim_channels=clim_channels)
    print(f'[3/5] Обучение (epochs={epochs}, batch={batch}, lr={lr}, '
          f'{"аугментация честная, шаг " + str(aug_step) if augment else "БЕЗ аугментации (только полные данные)"})…')
    history, best_epoch, best_mse = T.train(
        model, None, y[tr], Xva_full, y[va], device, epochs, batch, lr, patience,
        train_ds=train_ds, lr_sched=True)

    print('[4/5] Оценка (прогноз в день уборки целевого года) и таблицы…')
    metric_rows, imgs = [], []
    for split_name, idx in (('test', te), ('val', va)):
        Xf = bi(rawf[idx], climn[idx], phn[idx], harvest_abs, n_years - 1)
        pred = T.predict(model, Xf, device)
        metric_rows.append({'split': split_name, **E.metrics(y[idx], pred)})
        ids_df = pd.DataFrame({'id_region': ids[idx, 0], 'id_district': ids[idx, 1], 'year': ids[idx, 2]})
        E.error_table(ids_df, y[idx], pred).to_csv(
            os.path.join(err_dir, f'errors_{split_name}.csv'), index=False, encoding='utf-8-sig')
        if split_name == 'test':
            imgs.append(('Предсказание vs реальность (test, уборка)', E.scatter_plot(y[idx], pred, 'test', report_dir)))
    metrics_df = pd.DataFrame(metric_rows)
    print(metrics_df.to_string(index=False))
    imgs.append(('Кривая обучения', T.training_curve(history, report_dir)))

    print('[5/5] Сохранение модели и отчёта…')
    torch.save(model.state_dict(), os.path.join(outdir, 'model.pt'))
    with open(os.path.join(outdir, 'architecture.json'), 'w', encoding='utf-8') as f:
        json.dump({'arch_name': arch, 'in_channels': in_channels, 'in_length': Tlen,
                   'config': arch_cfg, 'n_params': n_params, 'summary': summary_text}, f,
                  ensure_ascii=False, indent=2)
    details = {
        'dataset': dataset, 'group': group, 'version': ver, 'arch': arch, 'honest_e2e': True,
        'features': meta['features'], 'n_params': P, 'duration_years': n_years,
        'time_rows_size': size, 'feature_hist_last': meta['feature_hist_last'],
        'prod_hist_last': meta['prod_hist_last'], 'harvest_doy': HARVEST_DOY,
        'in_channels': in_channels, 'in_length': Tlen, 'augment': bool(augment),
        'aug_step': aug_step if augment else None,
        'fill': fill, 'clim_channels': bool(clim_channels),
        'split_mode': 'random', 'split': [len(tr), len(te), len(va)],
        'seed': seed, 'epochs_run': len(history), 'best_epoch': best_epoch, 'best_val_mse': best_mse,
        'optimizer': 'Adam', 'lr': lr, 'batch': batch, 'patience': patience, 'loss': 'MSE',
        'channels_layout': ('raw(P) masked[%s]' % fill
                            + (' | climatology(P) по-блочно' if clim_channels else '')
                            + ' | prod_hist(1)'),
    }
    with open(os.path.join(outdir, 'details.json'), 'w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, 'normalization.json'), 'w', encoding='utf-8') as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    metrics_df.to_csv(os.path.join(report_dir, 'metrics.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(history).to_csv(os.path.join(report_dir, 'training_history.csv'),
                                 index=False, encoding='utf-8-sig')
    _docx(disp, details, summary_text, metrics_df, imgs, report_dir)
    print(f'\nГотово. Модель {group}/{tag}: {outdir}')
    return outdir


def _docx(name, details, summary_text, metrics_df, imgs, outdir):
    from docx import Document
    from docx.shared import Inches, Pt
    doc = Document()
    doc.add_heading(f'Честная нейросеть заблаговременности: {name}', 0)
    doc.add_paragraph(
        'Вспомогательные признаки (средняя урожайность и климатология) рассчитаны по данным, '
        'доступным на момент прогноза: при прогнозе в году Y известны урожаи годов <= Y-1 '
        '(статистика за прошлый год известна к Новому году) и климатология за годы до Y. '
        f'Каналы: {details["channels_layout"]} = {details["in_channels"]}; длина ряда {details["in_length"]}. '
        f'Сплит СЛУЧАЙНЫЙ 70/15/15 по примерам (train/test/val={details["split"]}). '
        + (f'Аугментация честная (шаг {details["aug_step"]} дн.).' if details.get('augment')
           else 'БЕЗ аугментации (обучение только на полных данных, прогноз в день уборки).'))
    doc.add_heading('Конфигурация нейросети', level=1)
    r = doc.add_paragraph().add_run(summary_text); r.font.name = 'Consolas'; r.font.size = Pt(9)
    doc.add_heading('Метрики (прогноз в день уборки целевого года)', level=1)
    E._table_from_df(doc, metrics_df)
    for title, img in imgs:
        doc.add_heading(title, level=1)
        if img:
            doc.add_picture(img, width=Inches(5.8))
    p = os.path.join(outdir, f'report_{name}.docx')
    try:
        doc.save(p)
    except PermissionError:
        p = os.path.join(outdir, f'report_{name}_new.docx'); doc.save(p)
    print(f'Отчёт: {p}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Обучение честной нейросети заблаговременности.')
    ap.add_argument('--dataset', default=DATASET)
    ap.add_argument('--arch', default='cnn', choices=['cnn', 'mse24', 'conv'])
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--aug-step', type=int, default=7)
    ap.add_argument('--name', default=None)
    ap.add_argument('--no-augment', dest='augment', action='store_false',
                    help='обучать только на полных данных (без аугментации заблаговременности)')
    ap.add_argument('--no-clim-channels', dest='clim_channels', action='store_false',
                    help='без 11 климат-каналов (вход = сырьё + prod_hist, P+1 каналов)')
    ap.add_argument('--fill', default='clim', choices=['clim', 'zeros'],
                    help='чем заполнять будущее: clim (климатология района) или zeros (нулями)')
    a = ap.parse_args()
    run(a.dataset, a.arch, a.epochs, a.batch, a.lr, a.patience, a.seed, a.aug_step, a.name,
        augment=a.augment, fill=a.fill, clim_channels=a.clim_channels)
