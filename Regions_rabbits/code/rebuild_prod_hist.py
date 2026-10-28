"""Новая версия датасета с пересчитанным признаком ``prod_hist_ma``.

Зачем. Скользящая средняя урожайности района (5-й столбец ``scalar.csv``) считалась при
сборке датасета по прежнему правилу: год без значения урожайности попадал в окно, среднее
становилось NaN, и при обучении вместо истории района подставлялась медиана всего датасета.
Правило исправлено в :func:`build_dataset._prod_hist` — год без данных пропускается, и окно
берёт следующий по давности год С ДАННЫМИ.

Матрица наблюдений (``matrix.csv``, гигабайты) при этом НЕ меняется: признак в неё не входит,
он добавляется отдельной строкой при подготовке обучения. Поэтому скрипт делает новую версию
датасета за минуты: пересчитывает ``scalar.csv`` и ``plan.csv`` из базы, а матрицу подключает
жёсткой ссылкой (на NTFS — та же самая, без удвоения места; если не вышло, копирует).

    python code/rebuild_prod_hist.py winter_wheat_raw winter_wheat_raw_v2
    python code/rebuild_prod_hist.py winter_wheat_all winter_wheat_all_v2 --copy-matrix

Прежний датасет остаётся нетронутым: обученные на нём модели продолжают воспроизводиться.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

import build_dataset as B
import config
import db

SMALL_FILES = ('plan.csv', 'meta.json', 'normalization.json')


def _link_or_copy(src, dst, force_copy=False):
    """Подключить матрицу: жёсткая ссылка (мгновенно) либо копия. -> способ."""
    if os.path.exists(dst):
        return 'уже на месте'
    if not force_copy:
        try:
            os.link(src, dst)
            return 'жёсткая ссылка'
        except OSError:
            pass
    shutil.copyfile(src, dst)
    return 'копия'


def rebuild(src_task, dst_task, force_copy=False, verbose=True):
    """Собрать новую версию датасета с пересчитанным prod_hist. -> сводка (dict)."""
    src = config.dataset_dir(src_task)
    dst = config.dataset_dir(dst_task)
    if not os.path.isdir(src):
        raise SystemExit(f'Нет исходного датасета: {src}')
    if os.path.isdir(dst) and os.listdir(dst):
        raise SystemExit(f'Каталог {dst} уже существует и не пуст — выберите другое имя версии')
    os.makedirs(dst, exist_ok=True)

    meta = json.load(open(os.path.join(src, 'meta.json'), encoding='utf-8'))
    # урожайности в БД хранятся под именем БДПМО («Пшеница озимая»), а не под названием
    # культуры в приложении («Озимая пшеница») — сборщик датасета берёт именно его
    culture = meta.get('bdpmo_culture') or meta.get('culture_title')
    m = int(meta.get('prod_hist_last') or 4)
    plan = pd.read_csv(os.path.join(src, 'plan.csv'), encoding='utf-8-sig')

    # пересчёт по тому же правилу, что теперь в build_dataset
    old = plan['historical_productive'].to_numpy(dtype=float)
    new = np.empty(len(plan), dtype=float)
    for i, (tid, year) in enumerate(zip(plan['territory_id'], plan['year'])):
        new[i] = B._prod_hist(db.get_yields(str(tid), culture), int(year), m)

    both = ~np.isnan(old) & ~np.isnan(new)
    changed = int((np.abs(old[both] - new[both]) > 1e-9).sum())
    filled = int((np.isnan(old) & ~np.isnan(new)).sum())
    lost = int((~np.isnan(old) & np.isnan(new)).sum())
    still_nan = int(np.isnan(new).sum())
    diff = np.abs(old[both] - new[both])
    diff = diff[diff > 1e-9]

    plan['historical_productive'] = new
    plan.to_csv(os.path.join(dst, 'plan.csv'), index=False, encoding='utf-8-sig')

    scalar = np.column_stack([plan['id_region'].to_numpy(dtype=float),
                              plan['id_district'].to_numpy(dtype=float),
                              plan['year'].to_numpy(dtype=float),
                              plan['productive'].to_numpy(dtype=float), new])
    np.savetxt(os.path.join(dst, 'scalar.csv'), scalar)       # тот же формат, что у сборщика

    how = _link_or_copy(os.path.join(src, 'matrix.csv'), os.path.join(dst, 'matrix.csv'), force_copy)
    for name in SMALL_FILES:
        s = os.path.join(src, name)
        if name != 'plan.csv' and os.path.exists(s):
            shutil.copyfile(s, os.path.join(dst, name))

    meta.update({
        'task_id': dst_task, 'output_name': dst_task,
        'derived_from': src_task,
        'derived_change': 'prod_hist_ma пересчитан: год без урожайности пропускается, '
                          'окно берёт следующий год с данными (прежде давал NaN -> медиана датасета)',
        'derived_stats': {'n': int(len(plan)), 'changed': changed, 'filled': filled,
                          'lost': lost, 'still_nan': still_nan},
        'matrix_link': how,
    })
    with open(os.path.join(dst, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f'{src_task} -> {dst_task}')
        print(f'   примеров: {len(plan)}')
        print(f'   признак изменился: {changed} ({changed / len(plan):.1%})'
              + (f', медиана сдвига {np.median(diff):.2f} ц/га, максимум {diff.max():.2f}'
                 if len(diff) else ''))
        print(f'   было пусто, стало значение: {filled}; осталось пустым: {still_nan}'
              + (f'; потеряно: {lost}' if lost else ''))
        print(f'   матрица: {how}')
    return {'n': len(plan), 'changed': changed, 'filled': filled, 'lost': lost,
            'still_nan': still_nan, 'matrix': how}


def main():
    ap = argparse.ArgumentParser(description='Новая версия датасета с пересчитанным prod_hist_ma')
    ap.add_argument('source', help='исходный датасет (task_id), например winter_wheat_raw')
    ap.add_argument('dest', help='имя новой версии, например winter_wheat_raw_v2')
    ap.add_argument('--copy-matrix', action='store_true',
                    help='копировать матрицу вместо жёсткой ссылки (нужно, если версии поедут на разные диски)')
    a = ap.parse_args()
    rebuild(a.source, a.dest, a.copy_matrix)


if __name__ == '__main__':
    main()
