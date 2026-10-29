"""Аддитивная модель: ветвь на канал, выходы складываются — обучение и честные вклады каналов.

Замысел (предложен в работе над статьёй). Прежние опыты показали две вещи. Совместная сеть,
получающая все 12 каналов сразу, на честном сплите хуже тривиального эталона (10.78–11.05
против 8.98): она подгоняет случайные сочетания каналов, которые на тест не переносятся.
Ансамбль одноканальных моделей эталон не обходит (8.86–9.10), но там каждая модель обучалась
предсказывать ВСЁ отклонение целиком, в одиночку, и прогнозы потом усреднялись.

Здесь проверяется третье: пусть каждый канал предсказывает свой ВКЛАД в ответ, а не весь
ответ, и пусть ветви обучаются вместе, деля между собой остаток. Межканальных сочетаний в
такой модели нет по построению (``train_nn.AdditiveNet``), поэтому опознать год или район по
сочетанию каналов она не может, а согласовать вклады — может.

Чего такая модель НЕ умеет: взаимодействий. «Жара вредна, только когда сухо» в сумме вкладов
не выражается. Поэтому её провал не был бы доказательством отсутствия сигнала — он означал бы,
что сигнала нет В АДДИТИВНОМ ВИДЕ.

Побочная выгода, едва ли не главная. Выход ветви — это вклад канала в тех же единицах, что
ответ, поэтому вклад признаков считается прямо и на честном тесте. Для каждого канала
выводится:

* ``сила``        — ст. отклонение его вклада по тесту (в процентных пунктах): насколько он
                    вообще шевелит ответ;
* ``r с откл.``   — корреляция вклада с настоящим отклонением урожайности от средней: в ту ли
                    сторону он шевелит;
* ``Δ RMSE без``  — насколько ухудшится прогноз, если этот вклад убрать (ветвь занулить).
                    Положительное — канал помогает, отрицательное — мешает. Переобучать для
                    этого ничего не нужно: сумма разбирается по слагаемым.

    python code/additive_eval.py --name ww_add --scheme ww_blocks --seeds 3

Результаты: ``runs.csv`` (строка на прогон) и ``contributions.csv`` (строка на канал).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import config
import ensemble_single as ES
import explore_models as E
import folds as FD
import run_experiment as R
import train_nn as T

RUN_COLUMNS = ['task', 'arch', 'target', 'years_kept', 'fold', 'seed', 'n_params', 'n_train',
               'n_val', 'n_test', 'epochs_run', 'best_epoch', 'val_mse', 'test_rmse', 'test_r2',
               'base_rmse', 'base_r2', 'minutes', 'finished_at']
CONTRIB_COLUMNS = ['fold', 'seed', 'ch', 'ch_name', 'сила', 'r с откл.', 'Δ RMSE без',
                   'средний вклад']


def _append(path, row, columns):
    pd.DataFrame([row], columns=columns).to_csv(
        path, mode='a', header=not os.path.exists(path), index=False, sep=';', decimal=',',
        encoding='utf-8-sig')


@torch.no_grad()
def contributions(model, X, batch=256):
    """Вклады ветвей по всем примерам. -> (n, каналов)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model.contributions(torch.from_numpy(X[i:i + batch])).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description='Аддитивная модель и честные вклады каналов')
    ap.add_argument('--name', default='ww_add')
    ap.add_argument('--scheme', default='ww_blocks')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--arch', default='additive', choices=['additive', 'additive_wide'])
    ap.add_argument('--target', default='ratio', choices=['ratio', 'absolute'])
    ap.add_argument('--channels', default='raw', choices=['raw', 'relative'])
    ap.add_argument('--years-kept', type=int, default=0)
    ap.add_argument('--folds', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    scheme = FD.read_definition(a.scheme)
    fold_names = a.folds or sorted(str(x) for x in scheme['fold'].unique())
    specs = [FD.fold_spec(scheme, f) for f in fold_names]
    out = os.path.join(config.REPORTS_DIR, 'experiments', a.name)
    os.makedirs(out, exist_ok=True)
    runs_path = os.path.join(out, 'runs.csv')
    contrib_path = os.path.join(out, 'contributions.csv')
    done = set()
    if os.path.exists(runs_path):
        prev = pd.read_csv(runs_path, sep=';', decimal=',', encoding='utf-8-sig')
        done = set(zip(prev['fold'].astype(str), prev['seed'].astype(int)))

    plan = [(str(s['fold']), seed) for s in specs for seed in range(1, a.seeds + 1)]
    todo = [p for p in plan if p not in done]
    print(f'Опыт «{a.name}»: фолдов {len(specs)}, seed {a.seeds} -> прогонов {len(plan)}, '
          f'уже посчитано {len(plan) - len(todo)}, осталось {len(todo)}')
    print(f'   архитектура {a.arch}, цель {a.target}, схема {a.scheme}')
    print(f'   результаты: {runs_path}, {contrib_path}')
    if a.dry_run:
        return
    with open(os.path.join(out, 'experiment.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': a.name, 'scheme': a.scheme, 'task': a.task, 'arch': a.arch,
                   'target': a.target, 'channels': a.channels, 'years_kept': a.years_kept,
                   'folds': fold_names, 'seeds': a.seeds, 'epochs': a.epochs, 'batch': a.batch,
                   'lr': a.lr, 'patience': a.patience}, f, ensure_ascii=False, indent=2)

    ds = R.Dataset(a.task, a.channels, a.years_kept)
    names = list(ds.feature_names) + ['prod_hist_ma']
    t_start = time.time()
    for i, (fold, seed) in enumerate(todo, 1):
        spec = next(s for s in specs if str(s['fold']) == fold)
        Xn, y, ph, tr, va, te = ds.fold_arrays(spec)
        if a.target == 'ratio':
            fit_y, clip = ES.ratio_target(y, ph, tr)
            to_abs = lambda q, idx: ES.to_absolute(q, ph[idx], clip)
        else:
            fit_y, clip = y, None
            to_abs = lambda q, _idx: np.asarray(q, float)

        torch.manual_seed(seed)
        np.random.seed(seed)
        model, summary, n_params, _cfg = T.make_model(a.arch, Xn.shape[1], Xn.shape[2])
        if i == 1:
            print(summary, flush=True)
            with open(os.path.join(out, 'architecture.txt'), 'w', encoding='utf-8') as f:
                f.write(summary + '\n')
        print(f'[{i}/{len(todo)}] фолд {fold}, seed {seed}: обучение {len(tr)}, '
              f'валидация {len(va)}, тест {len(te)}…', flush=True)
        t0 = time.time()
        hist, best_epoch, best_val = T.train(model, Xn[tr], fit_y[tr], Xn[va], fit_y[va], 'cpu',
                                             a.epochs, a.batch, a.lr, a.patience)
        pred = to_abs(T.predict(model, Xn[te], 'cpu'), te)
        m = E.metrics(y[te], pred)
        base = E.metrics(y[te], ph[te])
        _append(runs_path, {
            'task': a.task, 'arch': a.arch, 'target': a.target,
            'years_kept': a.years_kept or None, 'fold': fold, 'seed': seed,
            'n_params': n_params, 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
            'epochs_run': len(hist), 'best_epoch': best_epoch,
            'val_mse': round(float(best_val), 4), 'test_rmse': round(float(m['RMSE']), 4),
            'test_r2': round(float(m['R2']), 4), 'base_rmse': round(float(base['RMSE']), 4),
            'base_r2': round(float(base['R2']), 4), 'minutes': round((time.time() - t0) / 60, 1),
            'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, RUN_COLUMNS)
        print(f'     эпох {len(hist)} (лучшая {best_epoch}), тест RMSE {m["RMSE"]:.2f} / '
              f'R² {m["R2"]:.3f} (эталон {base["RMSE"]:.2f}), {(time.time() - t0) / 60:.1f} мин',
              flush=True)

        # --- вклады каналов: центруем по ОБУЧЕНИЮ, уровень уходит в общее смещение
        C_tr = contributions(model, Xn[tr])
        C_te = contributions(model, Xn[te])
        mu = C_tr.mean(axis=0)
        Cc = C_te - mu
        bias = float(model.bias.detach().cpu().numpy().ravel()[0])
        full = Cc.sum(axis=1) + bias + mu.sum()
        rmse_full = E.metrics(y[te], to_abs(full, te))['RMSE']
        dev_true = (y[te] / np.maximum(ph[te], 1.0) - 1.0) * 100      # настоящее отклонение, п.п.
        rows = []
        for j, nm in enumerate(names):
            wo = full - Cc[:, j]
            r_wo = E.metrics(y[te], to_abs(wo, te))['RMSE']
            sd = float(np.std(Cc[:, j]))
            rr = (float(np.corrcoef(Cc[:, j], dev_true)[0, 1]) if sd > 1e-9 else float('nan'))
            row = {'fold': fold, 'seed': seed, 'ch': j, 'ch_name': nm,
                   'сила': round(sd, 3), 'r с откл.': round(rr, 3),
                   'Δ RMSE без': round(float(r_wo - rmse_full), 3),
                   'средний вклад': round(float(mu[j]), 3)}
            _append(contrib_path, row, CONTRIB_COLUMNS)
            rows.append(row)
        top = sorted(rows, key=lambda r: -(r['Δ RMSE без'] if r['Δ RMSE без'] == r['Δ RMSE без'] else 0))[:3]
        print('     вклады (сила, п.п. / Δ RMSE если убрать): '
              + ', '.join(f'{r["ch_name"]} {r["сила"]:.1f}/{r["Δ RMSE без"]:+.2f}' for r in top),
              flush=True)
        left = (len(todo) - i) * (time.time() - t_start) / i / 60
        print(f'     осталось примерно {left:.0f} мин', flush=True)

    if os.path.exists(runs_path):
        d = pd.read_csv(runs_path, sep=';', decimal=',', encoding='utf-8-sig')
        print('\n=== по фолдам:')
        print(d.pivot_table(index='fold', values=['test_rmse', 'test_r2', 'base_rmse'],
                            aggfunc='mean').round(3).to_string())
        print(f'\nв среднем RMSE {d["test_rmse"].mean():.2f} против эталона '
              f'{d["base_rmse"].mean():.2f}; разброс между seed '
              f'{d.groupby("fold")["test_rmse"].std().mean():.2f}')
    if os.path.exists(contrib_path):
        c = pd.read_csv(contrib_path, sep=';', decimal=',', encoding='utf-8-sig')
        print('\n=== вклады каналов (среднее по фолдам и seed):')
        print(c.pivot_table(index='ch_name', values=['сила', 'r с откл.', 'Δ RMSE без'],
                            aggfunc='mean').round(3).sort_values('Δ RMSE без',
                                                                 ascending=False).to_string())


if __name__ == '__main__':
    main()
