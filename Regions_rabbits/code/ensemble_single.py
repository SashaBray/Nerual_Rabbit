"""Ансамбль одноканальных моделей: по модели на каждый параметр ряда, затем усреднение.

Зачем. Лесенка утечек показала, что на случайном сплите сеть достигает RMSE 5.8 ц/га, а на
честном (незнакомые годы) — 9.4, то есть хуже тривиального эталона 9.3. Разрыв не закрылся
ни сокращением входа до одного года, ни уменьшением сети с 12.2 млн весов до 38 тысяч
(``ww_ladder``, ``ww_1year``, ``ww_small``, ``ww_compact``, ``ww_compact_pc``). Значит дело не
в ёмкости: модель, получая 12 каналов сразу, опирается на их СОЧЕТАНИЕ — а оно опознаёт
конкретный год, общий для обучения и теста при случайном сплите.

Идея опыта. Лишить модель этой возможности структурно: обучить по отдельной сети на КАЖДЫЙ
канал (вход 1×T) и усреднить их прогнозы. Одного параметра мало, чтобы опознать год, зато
достаточно, чтобы сказать «сезон по этому параметру вышел лучше или хуже обычного».

Целевая величина — процент от скользящей средней района (100 — как обычно). Иначе
одноканальная модель не смогла бы предсказывать абсолютную урожайность: уровень района ей
неизвестен. Уровень возвращается умножением на ``prod_hist``, метрики — на абсолютной шкале,
поэтому цифры сопоставимы с лесенкой.

Способы сложить прогнозы (все веса подбираются ТОЛЬКО по валидации фолда):

* ``среднее``          — простое среднее по всем каналам;
* ``среднее без ур.``  — то же без канала ``prod_hist_ma`` (он не несёт сведений о годе);
* ``по 1/val``         — веса обратно пропорциональны ошибке канала на валидации;
* ``лучшие N``         — среднее по N лучшим каналам валидации;
* ``лучший одиночный`` — канал с наименьшей ошибкой на валидации.

    python code/ensemble_single.py --name ww_single --scheme ww_ladder --seeds 1

Прогнозы каждой модели кладутся в ``preds/``, поэтому запуск можно прервать и продолжить, а
способы сложения пересчитать (``--combine-only``) без переобучения.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import config
import explore_models as E
import folds as FD
import train_nn as T
import run_experiment as R

SINGLE_COLUMNS = ['task', 'arch', 'channels', 'years_kept', 'fold', 'seed', 'ch', 'ch_name',
                  'n_params', 'n_train', 'n_val', 'n_test', 'epochs_run', 'best_epoch',
                  'val_mse_ratio', 'val_rmse', 'test_rmse', 'test_r2', 'minutes', 'finished_at']
ENSEMBLE_COLUMNS = ['fold', 'seed', 'method', 'n_used', 'used', 'val_rmse', 'test_rmse',
                    'test_r2', 'base_rmse', 'base_r2']
TOP_N = (3, 5)


def _dirs(name):
    out = os.path.join(config.REPORTS_DIR, 'experiments', name)
    preds = os.path.join(out, 'preds')
    os.makedirs(preds, exist_ok=True)
    return out, preds


def _append(path, row, columns):
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False, sep=';',
              decimal=',', encoding='utf-8-sig')


def ratio_target(y, ph, tr):
    """Цель — ОТКЛОНЕНИЕ от скользящей средней в процентных пунктах; границы обрезки по
    обучающей части.

    Ноль значит «год как обычно», +20 — на пятую часть выше обычного. Отсчёт от нуля, а не от
    100, выбран ради обучения: свежая сеть выдаёт величины около нуля, и ей не приходится
    сначала несколько десятков эпох подниматься к уровню 100.
    """
    ratio = y / np.maximum(ph, 1.0)
    clip = (float(np.percentile(ratio[tr], 1)), float(np.percentile(ratio[tr], 99)))
    return ((np.clip(ratio, *clip) - 1.0) * 100).astype(np.float32), clip


def to_absolute(pred_dev, ph, clip):
    return np.clip(np.asarray(pred_dev, float) / 100.0 + 1.0, *clip) * ph


def train_single(Xn, fit_y, ch, tr, va, te, seed, arch, epochs, batch, lr, patience):
    """Обучить сеть на ОДНОМ канале. -> прогнозы в процентах на валидации и тесте."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xc = np.ascontiguousarray(Xn[:, ch:ch + 1, :])
    model, _summary, n_params, _cfg = T.make_model(arch, 1, Xc.shape[2])
    hist, best_epoch, best_val = T.train(model, Xc[tr], fit_y[tr], Xc[va], fit_y[va], 'cpu',
                                         epochs, batch, lr, patience)
    pv = np.asarray(T.predict(model, Xc[va], 'cpu'), float).ravel()
    pt = np.asarray(T.predict(model, Xc[te], 'cpu'), float).ravel()
    return pv, pt, {'n_params': n_params, 'epochs_run': len(hist), 'best_epoch': best_epoch,
                    'val_mse_ratio': round(float(best_val), 4)}


def combine(preds_v, preds_t, names, y, ph, clip, va, te):
    """Способы сложения прогнозов. Веса и отбор каналов — только по валидации."""
    chs = sorted(preds_v)
    val_rmse = {}
    for c in chs:
        val_rmse[c] = E.metrics(y[va], to_absolute(preds_v[c], ph[va], clip))['RMSE']
    order = sorted(chs, key=lambda c: val_rmse[c])

    def scored(used, weights=None):
        w = np.ones(len(used)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        rv = sum(wi * preds_v[c] for wi, c in zip(w, used))
        rt = sum(wi * preds_t[c] for wi, c in zip(w, used))
        mv = E.metrics(y[va], to_absolute(rv, ph[va], clip))
        mt = E.metrics(y[te], to_absolute(rt, ph[te], clip))
        return {'n_used': len(used), 'used': '+'.join(names[c] for c in used),
                'val_rmse': round(float(mv['RMSE']), 4),
                'test_rmse': round(float(mt['RMSE']), 4), 'test_r2': round(float(mt['R2']), 4)}

    level = [c for c in chs if names[c] == 'prod_hist_ma']
    no_level = [c for c in chs if c not in level]
    out = {'среднее': scored(chs)}
    if no_level and level:
        out['среднее без ур.'] = scored(no_level)
    out['по 1/val'] = scored(chs, [1.0 / max(val_rmse[c], 1e-6) ** 2 for c in chs])
    for n in TOP_N:
        if n < len(order):
            out[f'лучшие {n}'] = scored(order[:n])
    out['лучший одиночный'] = scored(order[:1])
    return out, val_rmse


def main():
    ap = argparse.ArgumentParser(description='Ансамбль одноканальных моделей')
    ap.add_argument('--name', default='ww_single')
    ap.add_argument('--scheme', default='ww_ladder')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--arch', default='compact',
                    choices=['cnn', 'mse24', 'conv', 'compact', 'compact_pc'])
    ap.add_argument('--channels', default='raw', choices=['raw', 'relative'])
    ap.add_argument('--years-kept', type=int, default=0)
    ap.add_argument('--folds', nargs='*', default=None)
    ap.add_argument('--only-channels', nargs='*', type=int, default=None)
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--combine-only', action='store_true',
                    help='не обучать, только пересчитать способы сложения по готовым прогнозам')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    scheme = FD.read_definition(a.scheme)
    fold_names = a.folds or sorted(str(x) for x in scheme['fold'].unique())
    specs = [FD.fold_spec(scheme, f) for f in fold_names]
    out, preds_dir = _dirs(a.name)
    singles_path = os.path.join(out, 'singles.csv')
    ens_path = os.path.join(out, 'ensemble.csv')

    print(f'Опыт «{a.name}»: фолдов {len(specs)}, seed {a.seeds}, архитектура {a.arch}, '
          f'цель — процент от скользящей средней')
    print(f'   фолды: {", ".join(str(s["fold"]) for s in specs)}')
    print(f'   результаты: {singles_path}, {ens_path}')
    if a.dry_run:
        return
    with open(os.path.join(out, 'experiment.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': a.name, 'scheme': a.scheme, 'task': a.task, 'arch': a.arch,
                   'channels': a.channels, 'years_kept': a.years_kept, 'target': 'ratio',
                   'folds': fold_names, 'seeds': a.seeds, 'epochs': a.epochs, 'batch': a.batch,
                   'lr': a.lr, 'patience': a.patience, 'per_channel_models': True},
                  f, ensure_ascii=False, indent=2)

    ds = R.Dataset(a.task, a.channels, a.years_kept)
    names = list(ds.feature_names) + ['prod_hist_ma']
    if os.path.exists(ens_path):
        os.remove(ens_path)                       # сводка пересчитывается заново каждый запуск

    t_start, done_runs = time.time(), 0
    for spec in specs:
        fold = str(spec['fold'])
        tag = fold.replace('|', '-')          # в имени файла «|» недопустим
        Xn, y, ph, tr, va, te = ds.fold_arrays(spec)
        fit_y, clip = ratio_target(y, ph, tr)
        chs = a.only_channels if a.only_channels is not None else list(range(Xn.shape[1]))
        base = E.metrics(y[te], ph[te])
        print(f'\n=== фолд {fold}: обучение {len(tr)}, валидация {len(va)}, тест {len(te)}; '
              f'каналов {len(chs)}; эталон RMSE {base["RMSE"]:.2f}', flush=True)

        for seed in range(1, a.seeds + 1):
            pv_all, pt_all = {}, {}
            for ch in chs:
                npz = os.path.join(preds_dir, f'{tag}_s{seed}_c{ch}.npz')
                if os.path.exists(npz):
                    z = np.load(npz)
                    pv_all[ch], pt_all[ch] = z['pv'], z['pt']
                    continue
                if a.combine_only:
                    continue
                print(f'  фолд {fold}, seed {seed}, канал {ch} ({names[ch]})…', flush=True)
                t0 = time.time()
                pv, pt, info = train_single(Xn, fit_y, ch, tr, va, te, seed, a.arch, a.epochs,
                                            a.batch, a.lr, a.patience)
                mv = E.metrics(y[va], to_absolute(pv, ph[va], clip))
                mt = E.metrics(y[te], to_absolute(pt, ph[te], clip))
                np.savez(npz, pv=pv, pt=pt)
                pv_all[ch], pt_all[ch] = pv, pt
                _append(singles_path, {
                    'task': a.task, 'arch': a.arch, 'channels': a.channels,
                    'years_kept': a.years_kept or None, 'fold': fold, 'seed': seed, 'ch': ch,
                    'ch_name': names[ch], 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
                    'val_rmse': round(float(mv['RMSE']), 4),
                    'test_rmse': round(float(mt['RMSE']), 4), 'test_r2': round(float(mt['R2']), 4),
                    'minutes': round((time.time() - t0) / 60, 1),
                    'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), **info}, SINGLE_COLUMNS)
                done_runs += 1
                print(f'     вал. RMSE {mv["RMSE"]:.2f}, тест RMSE {mt["RMSE"]:.2f} / '
                      f'R² {mt["R2"]:.3f}, {(time.time() - t0) / 60:.1f} мин '
                      f'(всего {(time.time() - t_start) / 60:.0f} мин)', flush=True)

            if len(pv_all) < 2:
                print(f'  фолд {fold}, seed {seed}: прогнозов меньше двух, сложение пропущено')
                continue
            res, val_rmse = combine(pv_all, pt_all, names, y, ph, clip, va, te)
            print(f'  -- фолд {fold}, seed {seed}: лучший канал по валидации — '
                  f'{names[min(val_rmse, key=val_rmse.get)]}')
            for method, r in res.items():
                _append(ens_path, {'fold': fold, 'seed': seed, 'method': method,
                                   'base_rmse': round(float(base['RMSE']), 4),
                                   'base_r2': round(float(base['R2']), 4), **r}, ENSEMBLE_COLUMNS)
                print(f'     {method:18s} тест RMSE {r["test_rmse"]:6.2f} / R² {r["test_r2"]:6.3f} '
                      f'(вал. {r["val_rmse"]:.2f}, каналов {r["n_used"]})', flush=True)

    if os.path.exists(ens_path):
        df = pd.read_csv(ens_path, sep=';', decimal=',', encoding='utf-8-sig')
        df['режим'] = df['fold'].astype(str).str.split('|').str[-1]
        print('\n=== Сводка: средний тестовый RMSE по режимам')
        print(df.pivot_table(index='method', columns='режим', values='test_rmse',
                             aggfunc='mean').round(2).to_string())
        print(f'\nэталон (скользящая средняя): {df["base_rmse"].mean():.2f}')
    print(f'\nГотово. Обучено моделей за этот запуск: {done_runs}. '
          f'Прогнозы: {preds_dir}')


if __name__ == '__main__':
    main()
