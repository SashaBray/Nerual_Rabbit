"""Выкинуть БИТЫЕ примеры из уже собранных датасетов (без пересборки).

Битые = строки matrix.csv, где значения одного признака затекли в другой (легаси-порча:
давление ~1020 в ndvi/осадках/снеге и т.п.). Детектор тот же, что в explore_models:
пик признака на пример (max по времени) > 5× от 95-перцентиля ненулевых пиков этого признака.

Для каждого датасета синхронно фильтрует matrix.csv / scalar.csv / plan.csv по одним и тем же
строкам (потоково, сохраняя исходный формат чисел) и правит n_samples в meta.json.
Несопоставленные по метке примеры (NaN productive) НЕ трогаются — это не «битые», а без метки.

Запуск:
    python code/clean_datasets.py                 # сухой прогон по всем датасетам
    python code/clean_datasets.py --apply
    python code/clean_datasets.py soy_all --apply  # только один датасет
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import config

YIELD_MAX = 5000.0   # ц/га; productive/historical_productive больше — сентинел/мусор


def detect_yield_outlier(scalar_path):
    """Маска примеров с неправдоподобной урожайностью в scalar (productive[3]/hist[4])."""
    s = np.loadtxt(scalar_path)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    y, ph = s[:, 3], s[:, 4]
    bad_y = ~np.isnan(y) & ((y < 0) | (y > YIELD_MAX))
    bad_ph = ~np.isnan(ph) & ((ph < 0) | (ph > YIELD_MAX))
    return bad_y | bad_ph


def detect_corrupt(matrix_path, n_features, time_len):
    """Вернуть булеву маску битых строк (как в explore_models.drop_corrupt)."""
    mat = pd.read_csv(matrix_path, sep=' ', header=None, dtype=np.float32,
                      engine='c').dropna(axis=1, how='all').values
    if mat.shape[1] != n_features * time_len:
        raise SystemExit(f'{matrix_path}: ширина {mat.shape[1]} != {n_features}*{time_len}')
    X = mat.reshape(mat.shape[0], n_features, time_len)
    rowmax = X.max(axis=2)                       # (n, F): пик признака на пример
    ceil = np.empty(n_features)
    for f in range(n_features):
        nz = rowmax[:, f][rowmax[:, f] > 0]
        ceil[f] = 5.0 * (np.percentile(nz, 95) if nz.size else 0.0) + 1e-9
    corrupt = (rowmax > ceil.reshape(1, n_features)).any(axis=1)
    return corrupt


def _filter_lines(path, keep, has_header, encoding='utf-8'):
    """Переписать текстовый файл, оставив строки данных, где keep[i] == True."""
    tmp = path + '.tmp'
    with open(path, 'r', encoding=encoding, newline='') as fin, \
         open(tmp, 'w', encoding=encoding, newline='') as fout:
        if has_header:
            fout.write(fin.readline())
        for i, line in enumerate(fin):
            if i < len(keep) and keep[i]:
                fout.write(line)
    os.replace(tmp, path)


def clean_one(task, apply):
    ddir = config.dataset_dir(task)
    meta_path = os.path.join(ddir, 'meta.json')
    if not os.path.exists(meta_path):
        print(f'  {task}: нет meta.json — пропуск')
        return
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    n_features = int(meta['n_features'])
    time_len = int(meta['time_rows_size']) * int(meta['duration_years'])

    matrix_path = os.path.join(ddir, 'matrix.csv')
    scalar_path = os.path.join(ddir, 'scalar.csv')
    plan_path = os.path.join(ddir, 'plan.csv')

    corrupt = detect_corrupt(matrix_path, n_features, time_len)
    yld = detect_yield_outlier(scalar_path)
    bad = corrupt | yld
    n = len(bad)
    n_bad = int(bad.sum())
    print(f'  {task}: примеров {n}, к удалению {n_bad} '
          f'(misalignment {int(corrupt.sum())}, выброс урожайности {int(yld.sum())})')
    if n_bad == 0 or not apply:
        return n, n_bad

    keep = (~bad).tolist()
    _filter_lines(matrix_path, keep, has_header=False)
    _filter_lines(scalar_path, keep, has_header=False)
    _filter_lines(plan_path, keep, has_header=True, encoding='utf-8-sig')

    meta['n_samples'] = n - n_bad
    meta['dropped_corrupt'] = int(meta.get('dropped_corrupt', 0)) + n_bad
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'     -> оставлено {n - n_bad}, meta.n_samples обновлён')
    return n, n_bad


def main(tasks, apply):
    if not tasks:
        tasks = sorted(d for d in os.listdir(config.DATASETS_DIR)
                       if os.path.exists(os.path.join(config.DATASETS_DIR, d, 'meta.json')))
    print(('ПРИМЕНЕНИЕ' if apply else 'СУХОЙ ПРОГОН') + f' — датасетов: {len(tasks)}')
    total = 0
    for t in tasks:
        r = clean_one(t, apply)
        if r:
            total += r[1]
    print(f'\nИтого битых {"удалено" if apply else "к удалению"}: {total}')
    if not apply:
        print('Это сухой прогон. Для удаления добавьте --apply.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Удалить битые примеры из собранных датасетов.')
    ap.add_argument('tasks', nargs='*', help='имена датасетов (по умолчанию — все)')
    ap.add_argument('--apply', action='store_true', help='переписать файлы (без флага — сухой прогон)')
    a = ap.parse_args()
    main(a.tasks, a.apply)
