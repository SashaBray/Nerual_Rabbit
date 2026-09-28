"""Сборка ЧЕСТНОГО датасета заблаговременности (озимая пшеница).

Ключевая идея: вспомогательные признаки (средняя урожайность prod_hist и климатология)
рассчитываются по тому, что было ИЗВЕСТНО на момент прогноза, а не относительно целевого года.

Для каждого примера (территория, целевой год t) и каждого «года-прогноза» Y из блоков
[t-2, t-1, t] (хронологически) сохраняем:
  raw[p, b, :]   — реальный суточный ряд параметра p для года block_years[b];
  clim[p, b, :]  — климатология (среднее за ПРЕДЫДУЩИЕ N лет) для года Y=block_years[b]
                   = db.mean_over_years(.., Y, N). Это «типичный год», известный при прогнозе в году Y;
  ph[b]          — средняя урожайность, известная при прогнозе в году Y: mean(yields Y-1..Y-m).
                   (правило: статистика министерства за прошлый год становится известна к Новому году,
                    т.е. при прогнозе в году Y известны урожаи годов <= Y-1);
  valid[b]       — есть ли климатология для года Y (>=1 предыдущий год в БД).

При обучении/прогнозе на отсечке a (день, абсолютный): блок-прогноза b = a // size,
будущее (дни > a) заполняется clim[:, b, :], строки-климатологии = clim[:, b, :], prod_hist = ph[b].
Так на ЛЮБОЙ заблаговременности используются только доступные на тот момент данные — без утечки t-1.

Запуск:  python code/build_honest_dataset.py [--task winter_wheat_raw] [--out winter_wheat_honest]
Выход:   workspace/datasets/<out>/honest.npz (+ meta.json)
"""

import argparse
import json
import os

import numpy as np

import build_dataset as B
import config
import db
import tasks as tasks_mod


def build(task_id='winter_wheat_raw', out_name='winter_wheat_honest'):
    task = tasks_mod.load_task(task_id)
    size = int(task['time_rows_size'])
    n_years = int(task['duration_years'])
    N = int(task['feature_hist_last'])       # окно климатологии (лет)
    m = int(task['prod_hist_last'])          # окно средней урожайности (лет)
    pad = task['padding']
    edge = bool(task['edge_fixup'])
    nf = bool(task['concat_newest_first'])
    feats = [f for f in task['features']]
    P = len(feats)
    masks_df = db.get_masks_df()
    plan = B.build_plan(task)
    print(f'Задание {task_id}: {P} сырых параметров, n_years={n_years}, size={size}, '
          f'N(клим)={N}, m(урож)={m}; примеров в плане: {len(plan)}')

    raws, clims, phs, valids, ids, ys = [], [], [], [], [], []
    kept = skipped = 0
    for k, item in enumerate(plan):
        tid, id_region, id_district, t, prod, _ = item
        block_years = B._block_years(t, n_years, nf)          # [t-2, t-1, t] при newest_first=False
        year_yields = db.get_yields(tid, task['bdpmo_culture'])

        raw = np.zeros((P, n_years, size), np.float32)
        clim = np.zeros((P, n_years, size), np.float32)
        valid = np.ones(n_years, bool)
        ok = True
        for pi, feature in enumerate(feats):
            parm, _h = B._resolve_parm(feature, task['mask_culture'], id_region, masks_df)
            if parm is None:
                ok = False
                break
            for b, Y in enumerate(block_years):
                arr = db.get_time_series_array(tid, parm, Y, size, pad)
                if arr.shape[0] == 0:                         # нет реального ряда года Y -> пример непригоден
                    ok = False
                    break
                cl = db.mean_over_years(tid, parm, Y, N, size, pad)
                if edge:
                    arr = B._edge_fixup(arr)
                    if cl.shape[0]:
                        cl = B._edge_fixup(cl)
                raw[pi, b, :] = arr
                if cl.shape[0] == 0:
                    valid[b] = False                          # нет климатологии для года Y
                else:
                    clim[pi, b, :] = cl
            if not ok:
                break
        if not ok:
            skipped += 1
            continue

        ph = np.array([B._prod_hist(year_yields, Y, m) for Y in block_years], np.float32)
        raws.append(raw); clims.append(clim); phs.append(ph); valids.append(valid)
        ids.append((int(id_region), int(id_district), int(t))); ys.append(float(prod))
        kept += 1
        if (k + 1) % 500 == 0:
            print(f'  обработано {k + 1}/{len(plan)} (годных {kept}, пропущено {skipped})')

    raw = np.stack(raws).astype(np.float32)       # (n, P, n_years, size)
    clim = np.stack(clims).astype(np.float32)
    ph = np.stack(phs).astype(np.float32)         # (n, n_years)
    valid = np.stack(valids)                       # (n, n_years) bool
    ids = np.array(ids, np.int64)                  # (n, 3)
    y = np.array(ys, np.float32)

    out_dir = os.path.join(config.DATASETS_DIR, out_name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, 'honest.npz'),
                        raw=raw, clim=clim, ph=ph, valid=valid, ids=ids, y=y)
    meta = {
        'source_task': task_id, 'output_name': out_name,
        'features': [f['name'] for f in feats], 'n_params': P,
        'time_rows_size': size, 'duration_years': n_years,
        'concat_newest_first': nf, 'feature_hist_last': N, 'prod_hist_last': m,
        'n_samples': int(kept), 'skipped': int(skipped),
        'block_year_offsets': [int(B._block_years(0, n_years, nf)[b]) for b in range(n_years)],
        'note': 'raw/clim shape (n,P,n_years,size); clim[:,:,b,:]=climatology(block_year b); '
                'ph[:,b]=prod_hist known at forecast year b; valid[:,b]=climatology available',
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # сводка по годам и валидности блоков
    print(f'\nГотово: {kept} примеров (пропущено {skipped}). -> {out_dir}\\honest.npz')
    yrs = ids[:, 2]
    for yy in sorted(set(yrs.tolist())):
        msk = yrs == yy
        vb = valid[msk].mean(axis=0).round(2)
        print(f'  t={yy}: примеров {int(msk.sum())}, доля валидных блоков [t-2,t-1,t] = {vb.tolist()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Сборка честного датасета заблаговременности.')
    ap.add_argument('--task', default='winter_wheat_raw', help='исходное задание (сырые параметры)')
    ap.add_argument('--out', default='winter_wheat_honest', help='имя выходного датасета')
    a = ap.parse_args()
    build(a.task, a.out)
