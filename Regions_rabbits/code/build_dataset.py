"""Сборка датасета по id задания.

Запуск:  ``python build_dataset.py <task_id> [--online]``

По id из ``workspace/tasks/dataset_tasks.csv`` находятся все детали (культура,
набор признаков, длительность N, размер ряда и т.д.). Урожайности и временные
ряды читаются из БД (``workspace/database``). Признак-ряд каждого параметра —
конкатенация N лет (целевой год первым), историч. признаки — межгодовое среднее.

Выход в ``workspace/datasets/<task_id>/``:
* ``matrix.csv``  — (n_samples, n_features * N * time_rows_size)
* ``scalar.csv``  — (n_samples, [id_region, id_district, year, productive, historical_productive])
* ``plan.csv``    — перечень собранных примеров
* ``meta.json``   — разрешённое задание (самоописание датасета)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import config
import db
import download
import tasks as tasks_mod
from vega import VegaClient


def _resolve_parm(feature, mask_culture, id_region, masks_df):
    """Развернуть признак (dict из tasks) в реальный продукт Vega.

    ``feature`` = ``{name, is_ndvi, historical, ...}``. Если ``is_ndvi`` — продукт
    берётся из маски культуры по региону; иначе ``name`` — это и есть продукт.
    """
    historical = bool(feature['historical'])
    if feature['is_ndvi']:
        # сравнение id региона через _norm_id (устойчиво к '64.0' vs '64')
        rid = db._norm_id(id_region)
        res = masks_df[masks_df['id_region'].apply(db._norm_id) == rid]
        if res.empty:
            return None, historical
        # столбец культуры ищем регистронезависимо ('Яровая пшеница' -> 'яровая пшеница')
        cols = {str(c).strip().lower(): c for c in masks_df.columns}
        col = cols.get(str(mask_culture).strip().lower())
        if col is None:
            return None, historical
        val = res[col].values[0]
        if not isinstance(val, str) or 'ndvi' not in val:
            return None, historical    # маска не задана для этого региона/культуры
        parm = val.strip()
    else:
        parm = str(feature['name'])
    return parm, historical


def _block_years(target_year, n, newest_first):
    years = [int(target_year) - j for j in range(n)]
    return years if newest_first else list(reversed(years))


def _edge_fixup(arr):
    if arr.shape[0] >= 2:
        arr = arr.copy()
        arr[0] = arr[1]
        arr[-1] = arr[-2]
    return arr


def _prod_hist(all_years_yields, target_year, m):
    """Скользящее среднее урожайности за m последних лет С ДАННЫМИ до целевого.

    Год без значения (пустая клетка в БД -> NaN) «имеющимся» не считается: он
    пропускается, и окно берёт следующий по давности год с данными. Иначе одна дыра
    в истории района давала бы ``mean`` с NaN, то есть NaN на весь признак, и модель
    молча подставляла бы медиану датасета вместо истории самого района (в БД пусты
    ~21% строк урожайности). Сюда же попадают значения, введённые пользователем
    вручную, — ``db.get_yields`` подмешивает их к данным источников.
    """
    prior = sorted((y for y, v in all_years_yields.items()
                    if y < target_year and v is not None and v == v), reverse=True)[:int(m)]
    vals = [all_years_yields[y] for y in prior]
    return float(np.mean(vals)) if vals else np.nan


def _normalize(matrix, mode):
    if mode == 'zscore':
        mu = matrix.mean(axis=0)
        sd = matrix.std(axis=0)
        sd[sd == 0] = 1.0
        return (matrix - mu) / sd
    if mode == 'minmax':
        lo = matrix.min(axis=0)
        hi = matrix.max(axis=0)
        rng = hi - lo
        rng[rng == 0] = 1.0
        return (matrix - lo) / rng
    return matrix


def build_plan(task):
    """Список примеров: (tid, id_region, id_district, target_year, prod, prod_hist)."""
    yld = db.yields_frame()          # источники + введённое пользователем вручную
    sub = yld[yld['culture'] == task['bdpmo_culture']] if not yld.empty else pd.DataFrame()
    plan = []
    for tid in sub['territory_id'].unique() if not sub.empty else []:
        id_region, id_district = (tid.split('_', 1) + ['nan'])[:2]
        # пропускаем территории без числового района (region-level/'[nan]' — для районного датасета непригодны)
        if not str(id_district).isdigit():
            continue
        if task['regions_filter'] is not None and id_region not in task['regions_filter']:
            continue
        year_yields = db.get_yields(tid, task['bdpmo_culture'])
        for target_year in task['years']:
            if target_year not in year_yields:
                continue
            prod = year_yields[target_year]
            prod_hist = _prod_hist(year_yields, target_year, task['prod_hist_last'])
            plan.append((tid, id_region, id_district, target_year, prod, prod_hist))
    return plan


def _line_count(path):
    if not os.path.exists(path):
        return 0
    with open(path, 'rb') as f:
        return sum(1 for _ in f)


def _truncate_lines(path, keep):
    """Оставить в файле первые ``keep`` строк (для выравнивания после сбоя)."""
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    if len(lines) > keep:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.writelines(lines[:keep])


def _load_progress(matrix_path, scalar_path, plan_path):
    """Возобновление: выровнять чекпойнт-файлы и вернуть множество готовых (tid, year).

    plan.csv пишется ПОСЛЕДНИМ, поэтому консервативен. Если matrix/scalar успели
    получить лишние строки из-за сбоя на середине flush — обрезаем по минимуму.
    """
    if not os.path.exists(plan_path):
        return set()
    m = _line_count(matrix_path)
    s = _line_count(scalar_path)
    p = max(_line_count(plan_path) - 1, 0)        # минус заголовок
    target = min(m, s, p)
    if m > target:
        _truncate_lines(matrix_path, target)
    if s > target:
        _truncate_lines(scalar_path, target)
    if p > target:
        _truncate_lines(plan_path, target + 1)    # +заголовок
    if target == 0:
        return set()
    pdf = pd.read_csv(plan_path, encoding='utf-8-sig')
    return {(str(r['territory_id']), int(r['year'])) for _, r in pdf.iterrows()}


def _make_sample(item, task, masks_df, client, offline, fetch_stats):
    """Собрать один пример. Возвращает (плоская строка матрицы | None, scalar_row, plan_row)."""
    tid, id_region, id_district, target_year, prod, prod_hist = item
    size, n = task['time_rows_size'], task['duration_years']
    sample = []

    for feature in task['features']:
        parm, historical = _resolve_parm(feature, task['mask_culture'], id_region, masks_df)
        if parm is None:
            return None, None, None
        feature_row = np.array([])
        for y in _block_years(target_year, n, task['concat_newest_first']):
            _ok, status = download.ensure_series(client, id_region, id_district, parm, y, offline)
            fetch_stats[status] = fetch_stats.get(status, 0) + 1
            if historical:
                block = db.mean_over_years(tid, parm, y, task['feature_hist_last'], size, task['padding'])
            else:
                block = db.get_time_series_array(tid, parm, y, size, task['padding'])
            if block is None or block.shape[0] == 0:
                return None, None, None
            if task['edge_fixup']:
                block = _edge_fixup(block)
            feature_row = np.concatenate((feature_row, block), axis=0)
        sample.append(feature_row)

    flat = np.concatenate(sample)                 # n_features*N*size (порядок = vstack().reshape(-1))
    scalar_row = [int(id_region), int(id_district), int(target_year),
                  float(prod), float(prod_hist) if prod_hist == prod_hist else np.nan]
    plan_row = {'territory_id': tid, 'id_region': id_region, 'id_district': id_district,
                'year': target_year, 'productive': prod, 'historical_productive': prod_hist}
    return flat, scalar_row, plan_row


def _meta_dict(task, offline, status, n_done, skipped, fetch_stats, normalized):
    size, n = task['time_rows_size'], task['duration_years']
    meta = {k: v for k, v in task.items() if k not in ('ukey', 'features')}
    meta['features'] = [f['name'] + ('_historical' if f['historical'] else '') for f in task['features']]
    meta['regions_filter'] = sorted(task['regions_filter']) if task['regions_filter'] else 'all'
    meta.update({
        'status': status,                 # in_progress | complete
        'n_samples': n_done,
        'n_features': len(task['features']),
        'matrix_width': len(task['features']) * n * size,
        'skipped': skipped,
        'fetch_stats': fetch_stats,
        'offline': offline,
        'normalized': normalized,
    })
    return meta


def _write_meta(meta, out_dir):
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_dataset(task_id, offline=True, checkpoint_every=None):
    task = tasks_mod.load_task(task_id)
    masks_df = db.get_masks_df()
    client = None if offline else VegaClient(ukey=task['ukey'])   # ukey развёрнут из ukey_id
    if client is not None and (not client.ukey or client.ukey == 'your_ukey_here'):
        print('ВНИМАНИЕ: ukey не задан (your_ukey_here) — докачка с Vega работать не будет.')
        print('  Впишите настоящий ukey в workspace/config/ukeys.csv. Продолжаю только из кэша БД.')
        client, offline = None, True
    if checkpoint_every is None:
        checkpoint_every = task.get('checkpoint_every', 200)

    features = task['features']
    out_dir = config.dataset_dir(task_id)
    os.makedirs(out_dir, exist_ok=True)
    matrix_path = os.path.join(out_dir, 'matrix.csv')
    scalar_path = os.path.join(out_dir, 'scalar.csv')
    plan_path = os.path.join(out_dir, 'plan.csv')
    meta_path = os.path.join(out_dir, 'meta.json')

    # прежний флаг normalized + множество уже готовых примеров (возобновление)
    prior_normalized = False
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as f:
                prior_normalized = bool(json.load(f).get('normalized', False))
        except (ValueError, OSError):
            pass
    done = _load_progress(matrix_path, scalar_path, plan_path)

    plan = build_plan(task)
    todo = [item for item in plan if (item[0], item[3]) not in done]
    print(f'Задание {task_id}: всего {len(plan)}, уже готово {len(done)}, к сборке {len(todo)}, '
          f'признаков {len(features)}, N={task["duration_years"]}')

    fetch_stats = {'cached': 0, 'downloaded': 0, 'missing': 0, 'empty': 0}
    built = len(done)
    skipped = 0
    buf = []                                       # [(flat, scalar_row, plan_row), ...]

    def flush():
        if not buf:
            return
        with open(matrix_path, 'a', encoding='utf-8') as f:
            np.savetxt(f, np.array([b[0] for b in buf]))
        with open(scalar_path, 'a', encoding='utf-8') as f:
            np.savetxt(f, np.array([b[1] for b in buf], dtype=float))
        header = _line_count(plan_path) == 0       # заголовок только для нового файла
        pd.DataFrame([b[2] for b in buf]).to_csv(
            plan_path, mode='a', header=header, index=False, encoding='utf-8-sig')
        buf.clear()
        _write_meta(_meta_dict(task, offline, 'in_progress', built, skipped,
                               fetch_stats, prior_normalized), out_dir)

    for item in todo:
        # быстрый останов ТОЛЬКО при системном сетевом сбое (а не «нет данных/битый район»)
        if client is not None and client.net_fail_count >= 15 and client.ok_count == 0:
            print('\nОстановка докачки: 15+ сетевых сбоёв подряд (0 успешных запросов).')
            print('  Проверьте сеть/прокси/ukey: python code/vega_test.py <ukey>')
            print('  Уже собранное сохранено; перезапуск продолжит с этого места.')
            break

        flat, scalar_row, plan_row = _make_sample(item, task, masks_df, client, offline, fetch_stats)
        if flat is None:
            skipped += 1
            continue
        buf.append((flat, scalar_row, plan_row))
        built += 1
        print(f'  [{built}] загружен пример: территория {item[0]}, год {item[3]}')
        if len(buf) >= checkpoint_every:
            flush()
            print(f'  -- чекпойнт сохранён: {built} примеров --')
    flush()

    # финальная нормировка (над всей матрицей; только если ещё не нормировано)
    normalized = prior_normalized
    if task['normalize'] != 'none' and not prior_normalized and built > 0:
        matrix = np.loadtxt(matrix_path)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        np.savetxt(matrix_path, _normalize(matrix, task['normalize']))
        normalized = True

    meta = _meta_dict(task, offline, 'complete', built, skipped, fetch_stats, normalized)
    _write_meta(meta, out_dir)

    if built == 0:
        print('Внимание: ни одного полного примера не собрано.')
    print(f'Готово: примеров {built}, ширина {meta["matrix_width"]}, '
          f'пропущено {skipped}, загрузки {fetch_stats}')
    print('Каталог:', out_dir)
    return meta


def main():
    parser = argparse.ArgumentParser(description='Сборка датасета по id задания.')
    parser.add_argument('task_id', help='id задания из dataset_tasks.csv')
    parser.add_argument('--online', action='store_true',
                        help='докачивать недостающие ряды с Vega (по умолчанию offline)')
    parser.add_argument('--checkpoint-every', type=int, default=None,
                        help='сохранять датасет каждые N примеров (по умолчанию из build_defaults.json)')
    parser.add_argument('--restart', action='store_true',
                        help='начать сборку заново (удалить прежний частичный датасет)')
    args = parser.parse_args()

    if args.restart:
        out_dir = config.dataset_dir(args.task_id)
        for name in ('matrix.csv', 'scalar.csv', 'plan.csv', 'meta.json'):
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                os.remove(path)

    build_dataset(args.task_id, offline=not args.online, checkpoint_every=args.checkpoint_every)


if __name__ == '__main__':
    main()
