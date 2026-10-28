"""Сетка опытов статьи: конфигурации × фолды по годам × seed, с общей тестовой выборкой.

Отвечает на вопрос, ради которого всё затевалось: что даёт климатология, что даёт
аугментация отсечкой, и различимы ли эти вклады на фоне разброса между запусками.

Четыре конфигурации (опыт 2×2)::

    raw-noaug   12 каналов, без аугментации       (рецепт v1)
    raw-aug     12 каналов, с аугментацией        (рецепт v4)
    clim-noaug  23 канала,  без аугментации       (рецепт v2)
    clim-aug    23 канала,  с аугментацией        (рецепт v3)

Каждая обучается на каждом фолде (обучение <= Y-2, валидация Y-1, тест Y) по нескольку раз
с разными seed. Разбиение задаётся годом и одинаково для всех конфигураций, тестовая выборка —
общая (пересечение заданий), поэтому цифры сопоставимы между собой.

    python code/run_experiment.py --name ww2x2 --seeds 3
    python code/run_experiment.py --name ww2x2 --seeds 3 --configs raw-aug clim-aug
    python code/run_experiment.py --name ww2x2 --dry-run        # только расписание

Результаты дописываются в ``workspace/reports/experiments/<name>/runs.csv`` после каждого
прогона: запуск можно прервать и продолжить — уже посчитанные строки пропускаются.
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

CONFIGS = {
    'raw-noaug':  {'task': 'winter_wheat_raw_v2', 'augment': False, 'прежний': 'nn_v1'},
    'raw-aug':    {'task': 'winter_wheat_raw_v2', 'augment': True,  'прежний': 'nn_v4'},
    'clim-noaug': {'task': 'winter_wheat_all_v2', 'augment': False, 'прежний': 'nn_v2'},
    'clim-aug':   {'task': 'winter_wheat_all_v2', 'augment': True,  'прежний': 'nn_v3'},
}
ARCH = 'mse24'          # по умолчанию; переопределяется ключом --arch
COLUMNS = ['config', 'task', 'arch', 'channels', 'years_kept', 'target', 'fold', 'seed', 'augment', 'in_channels', 'n_train',
           'n_val', 'n_test', 'epochs_run', 'best_epoch', 'val_mse', 'test_mse', 'test_rmse',
           'test_r2', 'base_rmse', 'base_r2', 'minutes', 'finished_at']


def to_relative_channels(Xb, names):
    """Из сырых и климатологических каналов сделать ОТНОСИТЕЛЬНЫЕ: ряд минус норма района.

    В 23-канальном задании у каждого параметра есть пара: сам ряд и его климатология —
    межгодовое среднее за 4 предыдущих года по той же территории. Разность между ними и есть
    «насколько этот год отличается от обычного для этого района» — та самая система координат,
    в которой двухпараметрическая модель обыгрывает сеть на сырых рядах.

    Почему разность, а не отношение: часть каналов (высота снега, накопленные осадки) рядом с
    нулём, и деление на них даёт выбросы. Для NDVI, где норма около 0.8 и устойчива, разность
    пропорциональна отношению, так что смысл сохраняется. Масштаб каналов всё равно выравнивает
    нормировка по обучающей части.
    """
    suffix = '_historical'
    hist = {n[:-len(suffix)]: i for i, n in enumerate(names) if n.endswith(suffix)}
    raw = [(i, n) for i, n in enumerate(names) if not n.endswith(suffix)]
    missing = [n for _i, n in raw if n not in hist]
    if missing:
        raise SystemExit('Нет климатологической пары для каналов: ' + ', '.join(missing)
                         + ' — нужен датасет со списком параметров 4')
    out = np.empty((Xb.shape[0], len(raw), Xb.shape[2]), dtype=np.float32)
    for j, (i, n) in enumerate(raw):
        out[:, j] = Xb[:, i] - Xb[:, hist[n]]
    return out, [n + '_anom' for _i, n in raw]


def _out_dir(name):
    d = os.path.join(config.REPORTS_DIR, 'experiments', name)
    os.makedirs(d, exist_ok=True)
    return d


def _load_done(path):
    """Уже посчитанные прогоны: множество (config, fold, seed)."""
    if not os.path.exists(path):
        return set(), None
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
    return set(zip(df['config'], df['fold'].astype(int), df['seed'].astype(int))), df


def _append(path, row):
    df = pd.DataFrame([row], columns=COLUMNS)
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False, sep=';', decimal=',',
              encoding='utf-8-sig')


class Dataset:
    """Датасет, загруженный один раз: матрица, метки, prod_hist, территории.

    Чтение ``matrix.csv`` — самая долгая часть (гигабайты текста), поэтому держим его в памяти
    и переиспользуем на всех фолдах и seed. Нормировка считается заново на каждом фолде, по
    его обучающей части: иначе статистика теста протекла бы в обучение.
    """

    def __init__(self, task, channels='raw', years_kept=0):
        t0 = time.time()
        Xb, y, prod_hist, ids, meta, feature_names = E.load_dataset(task)
        keep = ~np.isnan(y)
        Xb, y, prod_hist, ids = Xb[keep], y[keep], prod_hist[keep], ids[keep].reset_index(drop=True)
        ok = (y >= 0) & (y <= E.YIELD_MAX) & (np.isnan(prod_hist) | (prod_hist <= E.YIELD_MAX))
        Xb, y, prod_hist, ids = Xb[ok], y[ok], prod_hist[ok], ids[ok].reset_index(drop=True)
        Xb, y, prod_hist, ids, _n_corrupt = E.drop_corrupt(Xb, y, prod_hist, ids)
        if channels == 'relative':
            Xb, feature_names = to_relative_channels(Xb, list(feature_names))
        self.channels = channels
        self.years_kept = int(years_kept or 0)             # 0 — брать ряд целиком
        self.task, self.Xb, self.y, self.prod_hist = task, Xb, y, prod_hist
        self.ids, self.meta, self.feature_names = ids, meta, feature_names
        self.tid = (ids['id_region'].astype(int).astype(str) + '_'
                    + ids['id_district'].astype(int).astype(str)).to_numpy()
        print(f'   {task}: {Xb.shape[0]} примеров, вход {Xb.shape[1]}×{Xb.shape[2]} '
              f'({channels}), чтение {time.time() - t0:.0f} с')

    def fold_arrays(self, spec):
        """(Xn, y, ph, tr, va, te) для фолда: медиана и нормировка — только по обучению."""
        years = self.ids['year'].to_numpy(dtype=int)
        if spec.get('test_pairs') is not None:             # роли заданы парами (лесенка утечек)
            sel = lambda pp: np.where(np.array([(t, int(yy)) in pp
                                                for t, yy in zip(self.tid, years)]))[0]
            tr, va, te = sel(spec['train_pairs']), sel(spec['val_pairs']), sel(spec['test_pairs'])
        else:
            allowed = np.array([(t, int(yy)) in spec['pairs'] for t, yy in zip(self.tid, years)])
            sel_m = lambda m: np.where(allowed & m)[0]
            tr = sel_m(np.isin(years, spec['train_years']))
            va = sel_m(years == spec['val_year'])
            te = sel_m(years == spec['test_year'])
        med = float(np.nanmedian(self.prod_hist[tr]))      # медиана обучения, не всего датасета
        ph = np.where(np.isnan(self.prod_hist), med, self.prod_hist)
        X, names = E.add_prod_hist_row(self.Xb, ph, list(self.feature_names))
        norm = E.fit_normalization(X[tr], names)
        Xn = E.apply_normalization(X, norm).astype(np.float32)
        if self.years_kept:
            # оставить только последние годы ряда, остальные занулить. Ноль ПОСЛЕ нормировки —
            # это среднее значение канала, то есть «ничего не сказано», а не выброс.
            size = int(self.meta['time_rows_size'])
            n_years = Xn.shape[2] // size
            keep = min(self.years_kept, n_years)
            tgt_first = bool(self.meta.get('concat_newest_first', True))
            if tgt_first:
                Xn[:, :, keep * size:] = 0.0               # целевой год в начале ряда
            else:
                Xn[:, :, :(n_years - keep) * size] = 0.0   # целевой год в конце
        return (Xn,
                self.y.astype(np.float32), ph.astype(np.float32), tr, va, te)


def run_one(ds, spec, seed, augment, epochs, batch, lr, patience, aug_step, aug_scope,
            target='absolute', arch=ARCH):
    """Одно обучение. -> строка результатов.

    Целевая величина (``target``):

    * ``absolute`` — урожайность как есть;
    * ``anomaly``  — ОТКЛОНЕНИЕ от скользящей средней района (ц/га), уровень прибавляется
      обратно;
    * ``ratio``    — ПРОЦЕНТ от скользящей средней: 100 — как обычно, 120 — на пятую часть
      выше обычного, 70 — недород. Уровень возвращается умножением.

    Смысл двух последних: не заставлять сеть заново выучивать уровень каждого района (на
    сплите по годам она этого не умеет) и оставить ей ровно ту работу, ради которой её ставят —
    понять, каким вышел конкретный год. Процент вдобавок уравнивает районы между собой:
    отклонение +5 ц/га значит разное там, где обычно собирают 15 и 45 ц/га.

    Метрики во всех случаях считаются на АБСОЛЮТНОЙ шкале — цифры сопоставимы между собой.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xn, y, ph, tr, va, te = ds.fold_arrays(spec)
    clip = None
    if target == 'anomaly':
        fit_y = y - ph
    elif target == 'ratio':
        ratio = y / np.maximum(ph, 1.0)                    # знаменатель заведомо >= 4 ц/га
        # хвост до 595% (десяток примеров) перетянул бы на себя MSE: обрезаем обучающую цель
        # по краям ОБУЧАЮЩЕЙ части (1% и 99%), тест не трогаем
        clip = (float(np.percentile(ratio[tr], 1)), float(np.percentile(ratio[tr], 99)))
        fit_y = (np.clip(ratio, *clip) * 100).astype(np.float32)
    else:
        fit_y = y
    in_channels, in_length = Xn.shape[1], Xn.shape[2]
    model, _summary, n_params, _cfg = T.make_model(arch, in_channels, in_length)

    train_ds = None
    if augment:
        size = int(ds.meta['time_rows_size'])
        n_years = in_length // size
        tgt_idx = 0 if bool(ds.meta.get('concat_newest_first', True)) else n_years - 1
        blocks = Xn.reshape(Xn.shape[0], in_channels, n_years, size)
        clim = np.tile(np.delete(blocks, tgt_idx, axis=2).mean(axis=2), (1, 1, n_years)).astype(np.float32)
        # scope='year': отсечки только внутри целевого года — прошлые годы остаются реальными,
        # поэтому t-1 не протекает в прогноз через климатологию
        lo = tgt_idx * size if aug_scope == 'year' else 0
        train_ds = T.LeadAugmentDataset(Xn[tr], fit_y[tr], clim[tr], list(range(lo, in_length + 1, aug_step)))
        del blocks, clim

    t0 = time.time()
    # train сама возвращает модель с лучшими весами (ранняя остановка по валидации)
    hist, best_epoch, best_val = T.train(
        model, Xn[tr], fit_y[tr], Xn[va], fit_y[va], 'cpu', epochs, batch, lr, patience,
        train_ds=train_ds)
    out = T.predict(model, Xn[te], 'cpu')
    if target == 'anomaly':
        pred = out + ph[te]
    elif target == 'ratio':
        pred = np.clip(out / 100.0, *clip) * ph[te]        # те же границы, что при обучении
    else:
        pred = out
    m = E.metrics(y[te], pred)
    base = E.metrics(y[te], ph[te])             # эталон: сама скользящая средняя
    return {
        'config': None, 'task': ds.task, 'arch': arch, 'channels': ds.channels,
        'years_kept': ds.years_kept or None, 'target': target, 'fold': spec['fold'], 'seed': seed,
        'augment': int(bool(augment)), 'in_channels': in_channels,
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'epochs_run': len(hist), 'best_epoch': best_epoch, 'val_mse': round(float(best_val), 4),
        'test_mse': round(float(m['MSE']), 4), 'test_rmse': round(float(m['RMSE']), 4),
        'test_r2': round(float(m['R2']), 4),
        'base_rmse': round(float(base['RMSE']), 4), 'base_r2': round(float(base['R2']), 4),
        'minutes': round((time.time() - t0) / 60, 1),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def main():
    ap = argparse.ArgumentParser(description='Сетка опытов: конфигурации × фолды × seed')
    ap.add_argument('--name', default='ww2x2', help='имя опыта (папка отчёта)')
    ap.add_argument('--scheme', default='winter_wheat', help='схема фолдов из workspace/tasks/folds')
    ap.add_argument('--configs', nargs='*', default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument('--folds', nargs='*', default=None, help='имена фолдов (по умолчанию все)')
    ap.add_argument('--seeds', type=int, default=3, help='сколько seed на конфигурацию и фолд')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--aug-step', type=int, default=7)
    ap.add_argument('--aug-scope', default='year', choices=['year', 'full'],
                    help='year — отсечки только в целевом году (честно), full — по всему ряду')
    ap.add_argument('--channels', default='raw', choices=['raw', 'relative'],
                    help='raw — ряды как есть; relative — ряд минус норма района (нужен список 4)')
    ap.add_argument('--arch', default=ARCH, choices=['cnn', 'mse24', 'conv', 'compact', 'compact_pc'],
                    help='архитектура: mse24 — 12.2 млн весов, cnn — 1.2 млн, '
                         'compact — ~50 тыс. (свёртки + усреднение по времени), '
                         'compact_pc — то же с раздельным первым слоем')
    ap.add_argument('--years-kept', type=int, default=0,
                    help='сколько последних лет ряда оставить (0 — весь ряд); '
                         'остальные зануляются после нормировки')
    ap.add_argument('--target', default='absolute', choices=['absolute', 'anomaly', 'ratio'],
                    help='что предсказывает сеть: урожайность, отклонение от скользящей средней '
                         'или процент от неё (100 — как обычно)')
    ap.add_argument('--dry-run', action='store_true', help='показать расписание и выйти')
    a = ap.parse_args()

    scheme = FD.read_definition(a.scheme)
    fold_years = a.folds or sorted(str(x) for x in scheme['fold'].unique())
    specs = [FD.fold_spec(scheme, f) for f in fold_years]
    out = _out_dir(a.name)
    runs_path = os.path.join(out, 'runs.csv')
    done, _prev = _load_done(runs_path)
    done = {(c, str(f), int(sd)) for c, f, sd in done}

    plan = [(c, str(s['fold']), seed) for c in a.configs for s in specs
            for seed in range(1, a.seeds + 1)]
    todo = [p for p in plan if p not in done]
    print(f'Опыт «{a.name}»: конфигураций {len(a.configs)}, фолдов {len(specs)}, seed {a.seeds} '
          f'-> прогонов {len(plan)}, уже посчитано {len(plan) - len(todo)}, осталось {len(todo)}')
    print(f'   фолды: {", ".join(str(s["fold"]) for s in specs)}; аугментация scope={a.aug_scope}')
    print(f'   результаты: {runs_path}')
    if a.dry_run:
        return

    with open(os.path.join(out, 'experiment.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': a.name, 'scheme': a.scheme, 'configs': {c: CONFIGS[c] for c in a.configs},
                   'folds': fold_years, 'seeds': a.seeds, 'arch': a.arch, 'epochs': a.epochs,
                   'channels': a.channels, 'years_kept': a.years_kept, 'target': a.target, 'batch': a.batch, 'lr': a.lr, 'patience': a.patience,
                   'aug_step': a.aug_step, 'aug_scope': a.aug_scope}, f, ensure_ascii=False, indent=2)

    datasets = {}
    t_start = time.time()
    for i, (cfg_name, fold_year, seed) in enumerate(todo, 1):
        cfg = CONFIGS[cfg_name]
        if cfg['task'] not in datasets:
            print(f'[{i}/{len(todo)}] Чтение датасета {cfg["task"]}…', flush=True)
            datasets[cfg['task']] = Dataset(cfg['task'], a.channels, a.years_kept)
        spec = next(s for s in specs if str(s['fold']) == str(fold_year))
        print(f'[{i}/{len(todo)}] {cfg_name}, фолд {fold_year}, seed {seed}…', flush=True)
        row = run_one(datasets[cfg['task']], spec, seed, cfg['augment'], a.epochs, a.batch,
                      a.lr, a.patience, a.aug_step, a.aug_scope, a.target, a.arch)
        row['config'] = cfg_name
        _append(runs_path, row)
        print(f'      эпох {row["epochs_run"]} (лучшая {row["best_epoch"]}), val MSE {row["val_mse"]:.2f}, '
              f'тест RMSE {row["test_rmse"]:.2f} / R² {row["test_r2"]:.3f} '
              f'(эталон {row["base_rmse"]:.2f} / {row["base_r2"]:.3f}), {row["minutes"]:.1f} мин',
              flush=True)
        left = (len(todo) - i) * (time.time() - t_start) / i / 60
        print(f'      осталось примерно {left:.0f} мин', flush=True)

    print(f'\nГотово. Все прогоны: {runs_path}')


if __name__ == '__main__':
    main()
