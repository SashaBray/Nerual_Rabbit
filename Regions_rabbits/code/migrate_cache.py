"""Миграция существующего кэша временных рядов (~80k JSON) в БД (time_series.csv).

Кэш лежит в двух раскладках внутри корня подпроекта:

* плоская   ``Dist_info_actual/<parm>_<dist>_<year>.json``  — регион в имени/пути отсутствует;
* вложенная ``<task>/Dist_info_actual/id_reg_<r>/id_dist_<d>/<parm>/<parm>_<dist>_<year>.json``.

Регион для плоских файлов восстанавливается по справочнику район->регион
(``Regions_and_dist_tree.json`` + ``Regions_uid_dict.json``). Файлы ``*_mean`` —
производные (межгодовые средние) — НЕ мигрируются, они пересчитываются
``db.mean_over_years`` при сборке. Дедуп last-write-wins, вложенные приоритетнее
(у них регион достоверен из пути). Нерешённые плоские -> ``workspace/legacy/unresolved_cache.csv``.

Запуск:  ``python migrate_cache.py``
"""

import glob
import json
import os
import re

import pandas as pd

import config
import db


NAME_RE = re.compile(r'^(?P<parm>.+)_(?P<dist>\d+)_(?P<year>\d{4})(?P<mean>_mean)?$')
IDREG_RE = re.compile(r'id_reg_(\d+)')


def build_district_to_region():
    """Словарь ``district_number(str) -> id_region(str)`` из дерева регионов."""
    with open(config.REGIONS_TREE_FILE, 'r', encoding='utf-8-sig') as f:
        tree = json.load(f)
    with open(config.REGIONS_UID_FILE, 'r', encoding='utf-8-sig') as f:
        reg_uid = json.load(f)

    root = tree.get('Российская Федерация', tree)
    dist_to_region = {}
    for region_name, districts in root.items():
        id_region = reg_uid.get(region_name)
        if id_region is None:
            continue
        if not isinstance(districts, dict):
            continue
        for _district_name, payload in districts.items():
            if isinstance(payload, dict) and 'Number' in payload:
                dist_to_region[str(payload['Number'])] = str(id_region)
    return dist_to_region


def _cache_roots():
    """Каталоги кэша: вложенные (с достоверным регионом) ПЕРВЫМИ, плоский — последним.

    Порядок важен: вложенные обрабатываются раньше и фиксируются в ``nested_keys``,
    после чего одноимённые плоские ряды пропускаются (дешёвая проверка членства).
    """
    roots = []
    for nested in glob.glob(os.path.join(config.ROOT, '*', 'Dist_info_actual')):
        roots.append(nested)
    flat = os.path.join(config.ROOT, 'Dist_info_actual')
    if os.path.isdir(flat):
        roots.append(flat)
    return roots


def migrate_cache():
    dist_to_region = build_district_to_region()

    series = {}        # (tid, parm, year, day) -> value
    meta = {}          # (tid, parm, year) -> (first_date, last_date)
    nested_keys = set()  # (tid, parm, year), пришедшие из вложенной раскладки (приоритет)
    unresolved = []
    skipped_mean = 0
    used = 0

    for root in _cache_roots():
        for path in glob.iglob(os.path.join(root, '**', '*.json'), recursive=True):
            base = os.path.basename(path)
            if base.startswith('~$'):
                continue
            stem = base[:-5] if base.endswith('.json') else base
            m = NAME_RE.match(stem)
            if not m:
                continue
            if m.group('mean'):
                skipped_mean += 1
                continue

            parm = m.group('parm')
            dist = m.group('dist')
            year = int(m.group('year'))

            # id_region: из пути (вложенная) либо из резолвера (плоская)
            path_reg = IDREG_RE.search(path)
            from_nested = path_reg is not None
            if from_nested:
                id_region = path_reg.group(1)
            else:
                id_region = dist_to_region.get(dist)
                if id_region is None:
                    unresolved.append({'file': base, 'dist': dist, 'parm': parm, 'year': year})
                    continue

            tid = db.territory_id(id_region, dist)
            key3 = (tid, parm, year)

            # вложенные идут первыми; одноимённый плоский ряд пропускаем
            if not from_nested and key3 in nested_keys:
                continue

            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                block = data.get('data', {})
                first = block.get('1') or next(iter(block.values()))
                xy = first.get('xy') or []
                labels = first.get('labels') or {}
            except (ValueError, OSError, StopIteration, AttributeError):
                continue
            if not xy:
                continue

            if from_nested:
                nested_keys.add(key3)

            for day_str, value in xy:
                if value is None:
                    continue
                series[(tid, parm, year, int(float(day_str)))] = float(value)

            if labels:
                dates = sorted(str(v) for v in labels.values())
                meta[key3] = (dates[0], dates[-1])
            used += 1

    # --- запись в нормализованную схему (parms / series / time_series) ---
    parm_names = sorted({parm for (_tid, parm, _year, _day) in series})
    parm_id = {name: i for i, name in enumerate(parm_names)}
    parms_df = pd.DataFrame({'parm_id': list(parm_id.values()), 'parm': list(parm_id.keys())})

    series_id = {}             # (tid, parm_id, year) -> series_id
    point_rows = []
    for (tid, parm, year, day), value in series.items():
        key = (tid, parm_id[parm], year)
        sid = series_id.get(key)
        if sid is None:
            sid = len(series_id)
            series_id[key] = sid
        point_rows.append((sid, day, value))

    series_rows = []
    for (tid, pid, year), sid in series_id.items():
        fd, ld = meta.get((tid, parm_names[pid], year), ('', ''))
        series_rows.append({'series_id': sid, 'territory_id': tid, 'parm_id': pid,
                            'year': year, 'first_date': fd, 'last_date': ld})
    series_df = pd.DataFrame(series_rows, columns=['series_id', 'territory_id', 'parm_id',
                                                   'year', 'first_date', 'last_date'])

    points_df = pd.DataFrame(point_rows, columns=['series_id', 'day', 'value'])
    points_df = points_df.sort_values(['series_id', 'day']).reset_index(drop=True)

    db.save('parms', parms_df)
    db.save('series', series_df.sort_values('series_id').reset_index(drop=True))
    db.save('time_series', points_df)

    if unresolved:
        os.makedirs(config.LEGACY_DIR, exist_ok=True)
        pd.DataFrame(unresolved).to_csv(
            os.path.join(config.LEGACY_DIR, 'unresolved_cache.csv'), index=False, encoding='utf-8-sig')

    print(f'Миграция кэша: рядов учтено {used}, parms {len(parms_df)}, series {len(series_df)}, '
          f'точек {len(points_df)}, пропущено _mean {skipped_mean}, нерешённых {len(unresolved)}')
    return points_df


if __name__ == '__main__':
    config.ensure_dirs()
    migrate_cache()
