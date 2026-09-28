"""Обучение свёрточной нейросети для прогноза урожайности (PyTorch).

Вход — длинная матрица примера (признаки × время) СО строкой скользящего среднего урожайности.
Так как ряд длинный, сначала свёрточные слои (Conv1d) сжимают время, затем глубокая полносвязная
голова (~15 слоёв) с Dropout, на выходе одно число — урожайность в АБСОЛЮТНОМ значении (выход НЕ
нормируется). Активация ReLU. Подготовка данных, разбиение 70/15/15 и табличные выводы — те же,
что в разведке (переиспользуется ``explore_models``).

Папка модели называется ``<датасет>_nn`` (без атрибутов в имени), внутри — ВЕРСИИ ``v1, v2, …``:
каждый новый прогон пишется НОВОЙ версией (старое не перезаписывается). Все детали запуска
(arch, lr, batch, epochs, нормировка, признаки, глубина лет, n скольз.среднего) — в
сопроводительном ``details.json``. В ``workspace/models/<датасет>_nn/vN/`` лежат ТОЛЬКО файлы для
ИСПОЛЬЗОВАНИЯ: ``model.pt``, ``architecture.json``, ``details.json``, ``normalization.json``.
Отчёт об обучении — ОТДЕЛЬНО, в ``workspace/reports/nn/<датасет>_nn/vN/`` (рядом с разведкой
``reports/experiments/<task>/``): ``report_*.docx``, ``metrics.csv``, ``training_history.csv``,
``error_tables/``, ``training_curve.png``, ``scatter_test.png``, ``hist_test.png``.

Запуск:
    python code/train_nn.py winter_wheat_all
    python code/train_nn.py winter_wheat_all --name ww_cnn_v1 --epochs 100 --batch 64
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
import explore_models as E
import plots

# --------------------------------------------------------------------------- #
# Конфигурация архитектуры (попадает в отчёт и architecture.json)
# --------------------------------------------------------------------------- #
ARCH = {
    'conv_channels': [32, 64, 128, 128],   # выходные каналы Conv1d-блоков
    'conv_kernels': [7, 7, 5, 3],
    'pool': 2,                              # MaxPool1d после каждого conv-блока
    'adaptive_out': 8,                      # AdaptiveMaxPool1d перед головой
    'conv_dropout': 0.10,
    'hidden': 256,                          # ширина полносвязных слоёв
    'n_hidden_layers': 13,                  # +вход +выход => 15 Linear-слоёв в голове
    'dense_dropout': 0.30,
    'dropout_every': 3,                     # Dropout после каждых k скрытых слоёв
    'batchnorm': True,                      # BN для устойчивого обучения глубокой головы
    'activation': 'relu',
}


class CNNRegressor(nn.Module):
    """Conv1d-экстрактор + глубокая FC-голова -> одно число (урожайность)."""

    def __init__(self, in_channels, in_length, arch):
        super().__init__()
        self.arch = arch
        # --- свёрточный экстрактор (сжимает длинный ряд) ---
        conv = []
        c = in_channels
        for out_c, k in zip(arch['conv_channels'], arch['conv_kernels']):
            conv += [nn.Conv1d(c, out_c, kernel_size=k, padding=k // 2),
                     nn.ReLU(inplace=True),
                     nn.MaxPool1d(arch['pool']),
                     nn.Dropout(arch['conv_dropout'])]
            c = out_c
        conv += [nn.AdaptiveMaxPool1d(arch['adaptive_out'])]
        self.conv = nn.Sequential(*conv)
        flat = arch['conv_channels'][-1] * arch['adaptive_out']

        # --- глубокая полносвязная голова (~15 Linear), ReLU + Dropout ---
        h = arch['hidden']
        head = [nn.Linear(flat, h)]
        if arch['batchnorm']:
            head += [nn.BatchNorm1d(h)]
        head += [nn.ReLU(inplace=True), nn.Dropout(arch['dense_dropout'])]
        for i in range(arch['n_hidden_layers']):
            head += [nn.Linear(h, h)]
            if arch['batchnorm']:
                head += [nn.BatchNorm1d(h)]
            head += [nn.ReLU(inplace=True)]
            if (i + 1) % arch['dropout_every'] == 0:
                head += [nn.Dropout(arch['dense_dropout'])]
        head += [nn.Dropout(arch['dense_dropout']), nn.Linear(h, 1)]   # выход — без активации
        self.head = nn.Sequential(*head)

    def forward(self, x):                        # x: (B, C, L)
        z = self.conv(x)
        z = z.flatten(1)
        return self.head(z).squeeze(1)           # (B,) — абсолютная урожайность


def arch_summary(model, in_channels, in_length, arch):
    """Текстовое описание архитектуры (для отчёта/architecture.json)."""
    n_params = sum(p.numel() for p in model.parameters())
    lines = [f'CNNRegressor: вход (C={in_channels}, L={in_length}), выход 1 (урожайность, без нормировки)',
             'Свёрточный экстрактор:']
    c = in_channels
    for out_c, k in zip(arch['conv_channels'], arch['conv_kernels']):
        lines.append(f'  Conv1d({c}->{out_c}, k={k}, pad={k//2}) -> ReLU -> '
                     f'MaxPool1d({arch["pool"]}) -> Dropout({arch["conv_dropout"]})')
        c = out_c
    lines.append(f'  AdaptiveMaxPool1d({arch["adaptive_out"]}) -> Flatten '
                 f'({arch["conv_channels"][-1]}*{arch["adaptive_out"]}='
                 f'{arch["conv_channels"][-1]*arch["adaptive_out"]})')
    bn = ' -> BatchNorm' if arch['batchnorm'] else ''
    lines.append(f'Полносвязная голова ({arch["n_hidden_layers"]+2} Linear-слоёв):')
    lines.append(f'  Linear(flat->{arch["hidden"]}){bn} -> ReLU -> Dropout({arch["dense_dropout"]})')
    lines.append(f'  {arch["n_hidden_layers"]}x [ Linear({arch["hidden"]}->{arch["hidden"]}){bn} '
                 f'-> ReLU (Dropout каждые {arch["dropout_every"]}) ]')
    lines.append(f'  Dropout({arch["dense_dropout"]}) -> Linear({arch["hidden"]}->1)')
    lines.append(f'Активация: ReLU. Параметров: {n_params:,}')
    return '\n'.join(lines), n_params


class SqueezeHead(nn.Module):
    """Обёртка: применяет nn.Sequential и схлопывает (B,1)->(B,)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_mse24(in_channels, in_length):
    """Архитектура по образцу пользователя («MSE 24»): 2 strided-conv + широкая FC-воронка, ELU.

    Каналы 2×/4× от входа (как у автора 19*2/19*4); flatten считается под нашу длину ряда
    динамически (у автора было зашито 1292 под ~157 точек, у нас ряд 1095). BatchNorm нет.
    """
    c1, c2 = in_channels * 2, in_channels * 4
    conv = nn.Sequential(
        nn.Conv1d(in_channels, c1, kernel_size=4, stride=3), nn.ELU(),
        nn.Conv1d(c1, c2, kernel_size=4, stride=3), nn.ELU(),
        nn.MaxPool1d(1),                       # как в оригинале (фактически no-op)
    )
    with torch.no_grad():
        flat = conv(torch.zeros(1, in_channels, in_length)).flatten(1).shape[1]
    fc = [1292, 1292, 1292, 500, 500, 500, 200, 200, 200, 1]
    net = nn.Sequential(
        conv, nn.Flatten(), nn.Dropout(0.3),
        nn.Linear(flat, 1292), nn.ELU(),
        nn.Linear(1292, 1292), nn.ELU(),
        nn.Linear(1292, 1292), nn.ELU(),
        nn.Linear(1292, 500), nn.ELU(),
        nn.Linear(500, 500), nn.ELU(),
        nn.Dropout(0.3),
        nn.Linear(500, 500), nn.ELU(),
        nn.Linear(500, 200), nn.ELU(),
        nn.Linear(200, 200), nn.ELU(),
        nn.Linear(200, 200), nn.ELU(),
        nn.Linear(200, 1),
    )
    model = SqueezeHead(net)
    n_params = sum(p.numel() for p in model.parameters())
    summary = (
        'CNN-MSE24 (по образцу пользователя; ELU, без BatchNorm):\n'
        f'  Conv1d({in_channels}->{c1}, k=4, s=3) -> ELU\n'
        f'  Conv1d({c1}->{c2}, k=4, s=3) -> ELU -> MaxPool1d(1) [no-op]\n'
        f'  Flatten({flat}) -> Dropout(0.3)\n'
        f'  Linear({flat}->1292)->ELU -> 1292->1292->ELU -> 1292->1292->ELU\n'
        f'  -> 1292->500->ELU -> 500->500->ELU -> Dropout(0.3)\n'
        f'  -> 500->500->ELU -> 500->200->ELU -> 200->200->ELU -> 200->200->ELU -> 200->1\n'
        f'  Активация: ELU. Выход без нормировки. Параметров: {n_params:,}')
    cfg = {'style': 'mse24', 'conv_channels': [c1, c2], 'kernel': 4, 'stride': 3,
           'flatten': int(flat), 'fc': fc, 'dropout': 0.3, 'activation': 'ELU', 'batchnorm': False}
    return model, summary, n_params, cfg


def make_model(arch, in_channels, in_length):
    """Вернуть (model, summary_text, n_params, cfg) по имени архитектуры."""
    if arch in ('mse24', 'conv'):       # 'conv' — псевдоним: «свёрточная модель» (свёрточные слои + ELU)
        return build_mse24(in_channels, in_length)
    model = CNNRegressor(in_channels, in_length, ARCH)
    summary, n_params = arch_summary(model, in_channels, in_length, ARCH)
    return model, summary, n_params, {'style': 'cnn', **ARCH}


# --------------------------------------------------------------------------- #
# Обучение
# --------------------------------------------------------------------------- #
class LeadAugmentDataset(torch.utils.data.Dataset):
    """Аугментация заблаговременности: на каждый доступ случайная отсечка a (кратная aug_step),
    всё ПОСЛЕ a -> климатология. Один пример порождает десятки вариантов с разной известной частью."""

    def __init__(self, X, y, clim, cutoffs):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.clim = torch.from_numpy(clim)
        self.cutoffs = list(cutoffs)
        self.T = X.shape[2]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i].clone()
        a = self.cutoffs[int(torch.randint(len(self.cutoffs), (1,)).item())]
        if a < self.T:
            x[:, a:] = self.clim[i][:, a:]
        return x, self.y[i]


def train(model, Xtr, ytr, Xva, yva, device, epochs, batch, lr, patience, train_ds=None,
          lr_sched=False, lr_patience=10, sampler=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # планировщик: снижаем LR вдвое только после ДЛИТЕЛЬНОГО плато val (мягко, чтобы не задушить
    # обучение на ранних шумных эпохах). min_lr не даём упасть слишком низко.
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                        patience=lr_patience, min_lr=1e-4)
             if lr_sched else None)
    lossf = nn.MSELoss()
    ds = train_ds if train_ds is not None else TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    # sampler (напр. WeightedRandomSampler) задаёт частоту примеров; иначе обычное перемешивание
    dl = (DataLoader(ds, batch_size=batch, sampler=sampler) if sampler is not None
          else DataLoader(ds, batch_size=batch, shuffle=True))
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    history = []
    best_mse, best_state, best_epoch, bad = float('inf'), None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(ytr)

        model.eval()
        with torch.no_grad():
            va_pred = model(Xva_t)
            va_mse = float(lossf(va_pred, yva_t).item())          # MSE — функция потерь
        if sched is not None:
            sched.step(va_mse)
        cur_lr = opt.param_groups[0]['lr']
        history.append({'epoch': ep, 'train_mse': tr_loss, 'val_mse': va_mse, 'lr': cur_lr})

        if va_mse < best_mse - 1e-3:
            best_mse, best_epoch, bad = va_mse, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f'      эпоха {ep:3d}: train MSE={tr_loss:.3f}, val MSE={va_mse:.3f} '
                  f'(лучшая {best_mse:.3f} @ {best_epoch}; lr={cur_lr:.1e})')
        if bad >= patience:
            print(f'      ранняя остановка на эпохе {ep} (нет улучшения {patience})')
            break

    model.load_state_dict(best_state)
    return history, best_epoch, best_mse


@torch.no_grad()
def predict(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Графики/отчёт
# --------------------------------------------------------------------------- #
def training_curve(history, outdir):
    h = pd.DataFrame(history)

    def draw(lang):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(h.epoch, h.train_mse, label='train MSE')
        ax.plot(h.epoch, h.val_mse, label='val MSE')
        ax.set_xlabel(plots.tr('эпоха', 'epoch', lang)); ax.set_ylabel('MSE')
        ax.set_title(plots.tr('Кривая обучения (MSE — функция потерь)',
                              'Training curve (MSE — loss function)', lang))
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    return plots.bilingual(outdir, 'training_curve', draw)


def build_docx(name, meta, details, summary_text, metrics_df, imgs, outdir):
    from docx import Document
    from docx.shared import Inches, Pt
    doc = Document()
    doc.add_heading(f'Нейросеть: {name}', level=0)
    doc.add_paragraph(f'Датасет: {meta.get("output_name", details["task"])} — культура '
                      f'{meta.get("culture_title", "")}.')
    doc.add_paragraph(f'Группа: {details["group"]}, версия: v{details["version"]}, '
                      f'архитектура: {details["arch"]}; lr={details["lr"]}, '
                      f'batch={details["batch"]}, optimizer={details["optimizer"]}.')
    doc.add_paragraph(f'Глубина лет: {details["duration_years"]}; '
                      f'n скользящего среднего урожайности: {details["prod_hist_last"]}; '
                      f'признаков (с prod_hist): {details["in_channels"]}, '
                      f'длина ряда: {details["in_length"]}.')
    doc.add_paragraph(f'Разбиение train/test/val = {details["split"]} (70/15/15). '
                      f'Выход НЕ нормируется. Обучено эпох: {details["epochs_run"]} '
                      f'(лучшая {details["best_epoch"]}).')

    doc.add_heading('Конфигурация нейросети', level=1)
    pre = doc.add_paragraph()
    run = pre.add_run(summary_text)
    run.font.name = 'Consolas'; run.font.size = Pt(9)

    doc.add_heading('Метрики', level=1)
    E._table_from_df(doc, metrics_df)

    for title, img in imgs:
        doc.add_heading(title, level=1)
        doc.add_picture(img, width=Inches(5.8))

    p = os.path.join(outdir, f'report_{name}.docx')
    doc.save(p)
    return p


def _next_version(group):
    """Следующая версия vN: по плоским папкам models/<group>_vN и (старые отчёты) reports/nn/<group>/v*."""
    vers = []
    md = config.MODELS_DIR
    if os.path.isdir(md):
        pref = f'{group}_v'                              # плоская структура models/<group>_vN
        vers += [int(d[len(pref):]) for d in os.listdir(md)
                 if d.startswith(pref) and d[len(pref):].isdigit()]
    rbase = os.path.join(config.REPORTS_DIR, 'nn', group)
    if os.path.isdir(rbase):
        vers += [int(d[1:]) for d in os.listdir(rbase) if d.startswith('v') and d[1:].isdigit()]
    return max(vers, default=0) + 1


# --------------------------------------------------------------------------- #
# Главный сценарий
# --------------------------------------------------------------------------- #
def run(task, name, epochs, batch, lr, patience, seed, arch='cnn', augment=False, aug_step=7,
        split_mode='random', aug_scope='full'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    E.SPLIT_MODE = split_mode               # 'random' или 'year' (leave-one-year-out)
    group = name or f'{task}_nn'           # имя = датасет + _nn, без атрибутов
    ver = _next_version(group)             # каждый прогон — НОВАЯ версия vN (не перезаписываем)
    tag = f'v{ver}'
    disp = f'{group}_{tag}'                # для заголовка/имени файла отчёта
    outdir = os.path.join(config.MODELS_DIR, f'{group}_{tag}')       # плоская папка модели (ТОЛЬКО файлы)
    report_dir = os.path.join(config.REPORTS_DIR, 'nn', group, tag)  # отчёт — в общей reports/nn/
    err_dir = os.path.join(report_dir, 'error_tables')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[1/5] Подготовка данных {task}…')
    P = E.prepare_data(task, seed)
    Xn, y, ids = P['Xn'].astype(np.float32), P['y'].astype(np.float32), P['ids']
    tr, te, va = P['tr'], P['te'], P['va']
    meta, norm, feats, c = P['meta'], P['norm'], P['feature_names'], P['counts']
    in_channels, in_length = Xn.shape[1], Xn.shape[2]
    split = (len(tr), len(te), len(va))
    sy = P.get('split_years')
    smode = f'year (train<{sy[1]}, val={sy[1]}, test={sy[2]})' if sy else 'random 70/15/15'
    print(f'      примеров {c["n"]} (без метки {c["dropped"]}, выбросов {c["n_outlier"]}, '
          f'битых {c["n_corrupt"]}); вход (C={in_channels}, L={in_length}); сплит {split} [{smode}]; {device}')

    print(f'[2/5] Сборка нейросети (arch={arch})…')
    model, summary_text, n_params, arch_cfg = make_model(arch, in_channels, in_length)
    print(summary_text)

    # аугментация заблаговременности: каждый пример -> варианты с отсечкой (будущее = климатология)
    train_ds = None
    if augment:
        size = int(meta['time_rows_size']); n_years = in_length // size
        if n_years < 2:
            raise SystemExit('Аугментация требует >=2 лет в ряду (для климатологии).')
        nf = bool(meta.get('concat_newest_first', True))
        tgt_idx = 0 if nf else n_years - 1
        blocks = Xn.reshape(Xn.shape[0], in_channels, n_years, size)
        clim_year = np.delete(blocks, tgt_idx, axis=2).mean(axis=2)        # (n,F,size)
        clim_full = np.tile(clim_year, (1, 1, n_years)).astype(np.float32)  # (n,F,T)
        # scope='year' — отсечки ТОЛЬКО внутри целевого года (честно: маскируем лишь будущее
        # прогнозируемого сезона, прошлые годы t-1,t-2 остаются реальными -> нет утечки t-1);
        # scope='full' — отсечки по всему ряду (может «протекать» t-1 через климатологию).
        tgt0 = tgt_idx * size
        lo = tgt0 if aug_scope == 'year' else 0
        cutoffs = list(range(lo, in_length + 1, aug_step))
        train_ds = LeadAugmentDataset(Xn[tr], y[tr], clim_full[tr], cutoffs)
        del blocks, clim_year, clim_full                                  # освободить память
        print(f'      АУГМЕНТАЦИЯ заблаговременности: scope={aug_scope}, шаг {aug_step} дн., '
              f'вариантов отсечки {len(cutoffs)} на пример (будущее -> климатология)')

    print(f'[3/5] Обучение (epochs={epochs}, batch={batch}, lr={lr}'
          f'{", augment" if augment else ""})…')
    history, best_epoch, best_mse = train(
        model, Xn[tr], y[tr], Xn[va], y[va], device, epochs, batch, lr, patience, train_ds=train_ds)

    print('[4/5] Оценка и таблицы…')
    metric_rows, imgs = [], []
    for split_name, idx in (('test', te), ('val', va)):
        pred = predict(model, Xn[idx], device)
        metric_rows.append({'split': split_name, **E.metrics(y[idx], pred)})
        E.error_table(ids.iloc[idx], y[idx], pred).to_csv(
            os.path.join(err_dir, f'errors_{split_name}.csv'), index=False, encoding='utf-8-sig')
        if split_name == 'test':
            imgs.append(('Предсказание vs реальность (test)',
                         E.scatter_plot(y[idx], pred, 'test', report_dir)))
            imgs.append(('Распределение урожайности (test)',
                         E.hist_plot(y[idx], pred, 'test', report_dir)))
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(report_dir, 'metrics.csv'), index=False, encoding='utf-8-sig')
    print(metrics_df.to_string(index=False))
    imgs.insert(0, ('Кривая обучения', training_curve(history, report_dir)))

    print('[5/5] Сохранение модели и отчёта…')
    torch.save(model.state_dict(), os.path.join(outdir, 'model.pt'))
    with open(os.path.join(outdir, 'architecture.json'), 'w', encoding='utf-8') as f:
        json.dump({'arch_name': arch, 'in_channels': in_channels, 'in_length': in_length,
                   'config': arch_cfg, 'n_params': n_params, 'summary': summary_text}, f,
                  ensure_ascii=False, indent=2)
    details = {
        'task': task, 'group': group, 'version': ver, 'arch': arch,
        'features': feats,                                  # список параметров (вкл. prod_hist_ma)
        'duration_years': meta['duration_years'],           # глубина лет
        'prod_hist_last': meta['prod_hist_last'],           # n скользящего среднего урожайности
        'time_rows_size': meta['time_rows_size'],
        'in_channels': in_channels, 'in_length': in_length,
        'split': split, 'seed': seed, 'yield_max': E.YIELD_MAX,
        'split_mode': split_mode, 'split_years': P.get('split_years'),
        'aug_scope': aug_scope if augment else None,
        'epochs_run': len(history), 'best_epoch': best_epoch, 'best_val_mse': best_mse,
        'optimizer': 'Adam', 'lr': lr, 'batch': batch, 'patience': patience, 'loss': 'MSE',
        'output_normalized': False, 'augment_leadtime': bool(augment), 'aug_step': aug_step if augment else None,
    }
    with open(os.path.join(outdir, 'details.json'), 'w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, 'normalization.json'), 'w', encoding='utf-8') as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(report_dir, 'training_history.csv'),
                                 index=False, encoding='utf-8-sig')
    rep = build_docx(disp, meta, details, summary_text, metrics_df, imgs, report_dir)

    print(f'\nГотово. Модель {group} / {tag} (для использования): {outdir}')
    print(f'  model.pt, architecture.json, details.json, normalization.json')
    print(f'  отчёт об обучении (отдельно): {report_dir}')
    print(f'    {os.path.basename(rep)}, metrics.csv, training_history.csv, error_tables/, *.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Обучение CNN-регрессора урожайности (PyTorch).')
    ap.add_argument('task', help='имя датасета, напр. winter_wheat_all')
    ap.add_argument('--arch', default='cnn', choices=['cnn', 'mse24', 'conv'],
                    help='архитектура: cnn (BatchNorm+ReLU) или conv/mse24 (свёрточная модель, ELU)')
    ap.add_argument('--name', default=None,
                    help='базовое имя группы (по умолч. <task>_nn); версия vN добавляется автоматически')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--augment', action='store_true',
                    help='аугментация заблаговременности: примеры с отсечкой (будущее->климатология)')
    ap.add_argument('--aug-step', type=int, default=7, help='шаг отсечки при аугментации, дней')
    ap.add_argument('--split', default='random', choices=['random', 'year'],
                    help='random=70/15/15 по примерам; year=leave-one-year-out '
                         '(test=последний год, val=предпоследний, train=ранние)')
    ap.add_argument('--aug-scope', default='full', choices=['full', 'year'],
                    help='full=отсечки по всему ряду; year=только в целевом году (честно, без утечки t-1)')
    a = ap.parse_args()
    run(a.task, a.name, a.epochs, a.batch, a.lr, a.patience, a.seed, a.arch, a.augment, a.aug_step,
        split_mode=a.split, aug_scope=a.aug_scope)
