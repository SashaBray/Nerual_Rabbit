"""Сборка релизной папки: архивы моделей, снимки БД и манифест ``catalog.json``.

Это сторона АВТОРА. Скрипт собирает всё, что раздаётся пользователям, в одну папку
(по умолчанию ``workspace/release``); её целиком выкладывают в публичный доступ
(Яндекс.Диск), а ссылку вставляют в программе (вкладка «Каталог») или зашивают
в сборку через ``assets.DEFAULT_SOURCE``.

    python code/make_release.py                      # модели + db-lite + db-full
    python code/make_release.py --db lite            # только лёгкий снимок базы
    python code/make_release.py --models winter_wheat_all_nn_v4 soy_all_nn_v1
    python code/make_release.py --no-db --out D:/release
    python code/make_release.py --only-datasets            # только обучающие выборки

Что получается::

    release/
        catalog.json
        models/<имя папки модели>.zip
        db/db-lite-<версия>.zip
        db/db-full-<версия>.zip
        datasets/<задание>.zip

Снимок базы НЕ содержит пользовательских таблиц (``db.USER_TABLES``): ручные урожайности,
аккаунты, прогнозы и шаблоны принадлежат владельцу программы и не раздаются. Тяжёлые
таблицы кладутся в Parquet (вдвое-втрое меньше при скачивании, ``db.load`` читает его сам).
"""
import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

import app_predict as P
import config
import db

# Что входит в снимки базы. lite — программа сразу живая (данные, проверка урожайности,
# классическая регрессия по загруженным рядам); full — плюс сами временные ряды, прогноз
# работает офлайн.
LITE_TABLES = ['territories', 'regions', 'cultures_masks', 'territory_uids', 'yields']
FULL_TABLES = LITE_TABLES + ['parms', 'series', 'time_series']

# Таблицы, которые в снимке кладём колоночным форматом (крупные и однотипные).
PARQUET_TABLES = {'time_series', 'series', 'yields'}

MODEL_FILES = ('model.pt', 'model.pkl')

# Датасеты: матрица занимает 99% объёма и хранится текстом по 24 символа на число
# (``np.savetxt`` по умолчанию). В раздачу кладём её как float32 — ровно в той точности,
# в которой её читают все скрипты (`pd.read_csv(..., dtype='float32')`), и это в 9 раз
# меньше исходного CSV. Остальные файлы задания едут как есть.
DATASET_SIDE_FILES = ('scalar.csv', 'plan.csv', 'meta.json', 'normalization.json')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _zip_dir(src_dir, out_zip, arc_root=None, skip=()):
    """Упаковать папку в zip (внутри — одна корневая папка ``arc_root``).

    ``skip`` — имена файлов, которые в архив не кладём. Так из архивов моделей исключается
    ``comment.txt``: заметка едет в манифесте и пишется при установке, поэтому правка
    комментария не меняет sha256 архива и не требует перезаливать модель заново.
    """
    arc_root = arc_root or os.path.basename(src_dir.rstrip(os.sep))
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    tmp = out_zip + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, _dirs, files in os.walk(src_dir):
            for name in files:
                if name in skip:
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, src_dir)
                z.write(full, os.path.join(arc_root, rel))
    os.replace(tmp, out_zip)
    return out_zip


def _zip_files(paths, out_zip):
    """Упаковать набор файлов в zip плоско (без общей папки) — так ставится снимок БД."""
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    tmp = out_zip + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, os.path.basename(p))
    os.replace(tmp, out_zip)
    return out_zip


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #
def _model_rows():
    """Инвентарь сохранённых моделей: {токен: строка инвентаря} (культура, заметка, дата)."""
    return {str(r['mkey']): r for r in P.model_inventory() if r.get('сохранена')}


def build_models(outdir, only=None, verbose=True, reuse=False):
    """Собрать по архиву на каждую сохранённую модель. -> список записей каталога.

    ``reuse=True`` — архивы не пересобирать, взять уже лежащие в ``outdir`` (режим
    ``--manifest-only``): так можно обновить комментарии и метрики в каталоге, не меняя
    sha256 файлов и не перезаливая гигабайты.
    """
    rows = _model_rows()
    tokens = [t for t in sorted(rows) if not only or t in set(only)]
    if only:
        missing = set(only) - set(rows)
        if missing:
            raise SystemExit(f'Нет таких сохранённых моделей: {", ".join(sorted(missing))}')
    out = []
    for token in tokens:
        src = P.model_dir(token)
        if not src or not any(os.path.exists(os.path.join(src, f)) for f in MODEL_FILES):
            continue
        rel = f'models/{token}.zip'
        path = os.path.join(outdir, rel.replace('/', os.sep))
        if reuse:
            if not os.path.exists(path):
                if verbose:
                    print(f'  пропуск {token}: архива нет в {outdir}')
                continue
        else:
            path = _zip_dir(src, path, arc_root=token, skip=(P.MODEL_COMMENT_FILE,))
        row = rows[token]
        culture = str(row.get('культура') or '')
        kind, version = P.parse_token(token)
        met = db.get_model_metrics(culture).get((culture, token), {})
        rnd = lambda v: None if v is None or v != v else round(float(v), 3)
        out.append({
            'id': f'model:{token}', 'kind': 'model',
            'title': f'{culture} — {P.MODEL_NAMES.get(kind, kind)}'
                     + (f' v{version}' if version else ''),
            'culture': culture, 'version': version or 1,
            'file': rel, 'bytes': os.path.getsize(path), 'sha256': _sha256(path),
            'installs_to': f'models/{token}',
            'comment': row.get('комментарий') or '',
            'details': P.model_autonote(token),
            'created': row.get('создана') or '',
            'metrics': {'mse': rnd(met.get('mse')), 'rmse': rnd(met.get('rmse')),
                        'r2': rnd(met.get('r2'))},
        })
        if verbose:
            note = ' (архив прежний)' if reuse else ''
            print(f'  модель {token}: {os.path.getsize(path) / 1048576:.1f} МБ{note}')
    return out


# --------------------------------------------------------------------------- #
# Датасеты обучения
# --------------------------------------------------------------------------- #
def _dataset_dirs():
    """Каталоги датасетов с матрицей и описанием: {задание: путь}."""
    out = {}
    root = config.DATASETS_DIR
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'meta.json')):
            out[name] = d                                  # матрица CSV/NPY либо .npz (честный датасет)
    return out


def _dataset_comment(name, meta, rows, cols):
    """Описание датасета для каталога — по тому, что реально известно из его meta.json."""
    parts = [f'Обучающая выборка задания «{name}»']
    if rows and cols:
        parts.append(f'{rows} примеров × {cols} чисел')
    elif meta.get('n_samples'):
        parts.append(f'{int(meta["n_samples"])} примеров')
    if meta.get('duration_years') and meta.get('time_rows_size'):
        parts.append(f'{meta["duration_years"]} г. рядов по {meta["time_rows_size"]} точек')
    years = [int(y) for y in (meta.get('years') or [])]
    if years:
        parts.append(f'годы {min(years)}–{max(years)}')
    if meta.get('source_task'):                            # честный датасет заблаговременности
        parts.append(f'собран из «{meta["source_task"]}» с честным сплитом по годам (.npz)')
    return ('; '.join(parts) + '. Нужен только для переобучения моделей — '
            'для прогноза не требуется.')


def _npy_shape_in_zip(zip_path, member='matrix.npy'):
    """Размер матрицы из заголовка ``matrix.npy`` внутри архива (данные не распаковываются)."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            if member not in z.namelist():
                return None, None
            with z.open(member) as f:
                major, minor = np.lib.format.read_magic(f)
                reader = getattr(np.lib.format, f'read_array_header_{major}_{minor}')
                shape, _fortran, _dtype = reader(f)        # только заголовок, данные не читаем
        return (int(shape[0]), int(shape[1])) if len(shape) >= 2 else (int(shape[0]), None)
    except Exception:                                      # noqa: BLE001 — не смогли, и ладно
        return None, None


def build_datasets(outdir, only=None, verbose=True, reuse=False):
    """Собрать архив на каждый датасет обучения. -> список записей каталога.

    Матрица перекладывается в ``matrix.npy`` (float32); программа при установке
    разворачивает её обратно в ``matrix.csv``, так что обучающие скрипты не меняются.
    """
    dirs = _dataset_dirs()
    names = [n for n in dirs if not only or n in set(only)]
    if only:
        missing = set(only) - set(dirs)
        if missing:
            raise SystemExit(f'Нет таких датасетов: {", ".join(sorted(missing))}')
    out = []
    for name in sorted(names):
        src = dirs[name]
        rel = f'datasets/{name}.zip'
        path = os.path.join(outdir, rel.replace('/', os.sep))
        meta = json.load(open(os.path.join(src, 'meta.json'), encoding='utf-8'))
        rows = cols = None
        if reuse:
            if not os.path.exists(path):
                if verbose:
                    print(f'  пропуск {name}: архива нет в {outdir}')
                continue
            prev = _previous_entry(outdir, rel) or {}
            rows, cols = prev.get('rows'), prev.get('cols')
            if rows is None or cols is None:               # прежней записи нет — читаем ЗАГОЛОВОК npy
                rows, cols = _npy_shape_in_zip(path)       # из архива, без распаковки данных
        else:
            staging = os.path.join(config.WORKSPACE, '.release_staging', 'ds_' + name)
            if os.path.exists(staging):
                shutil.rmtree(staging)
            os.makedirs(staging)
            csv_path = os.path.join(src, 'matrix.csv')
            npy_path = os.path.join(src, 'matrix.npy')
            if os.path.exists(csv_path) or os.path.exists(npy_path):
                if os.path.exists(csv_path):
                    if verbose:
                        print(f'  {name}: читаю матрицу '
                              f'({os.path.getsize(csv_path) / 1048576:.0f} МБ текста)…')
                    mat = pd.read_csv(csv_path, sep=' ', header=None, dtype='float32').values
                else:
                    mat = np.load(npy_path)
                rows, cols = int(mat.shape[0]), int(mat.shape[1])
                np.save(os.path.join(staging, 'matrix.npy'), mat)
                del mat
                for f in DATASET_SIDE_FILES:
                    p_src = os.path.join(src, f)
                    if os.path.exists(p_src):
                        shutil.copyfile(p_src, os.path.join(staging, f))
            else:
                # датасет не матричный (например, honest.npz — уже упакованный массив):
                # кладём папку как есть, разворачивать при установке нечего
                for f in sorted(os.listdir(src)):
                    p_src = os.path.join(src, f)
                    if os.path.isfile(p_src):
                        shutil.copyfile(p_src, os.path.join(staging, f))
                rows = int(meta.get('n_samples') or 0) or None
            path = _zip_files([os.path.join(staging, f) for f in sorted(os.listdir(staging))], path)
            shutil.rmtree(staging, ignore_errors=True)

        culture = str(meta.get('culture_title') or '')
        out.append({
            'id': f'dataset:{name}', 'kind': 'dataset', 'optional': True,
            'title': f'Датасет обучения: {culture or name}' + (f' ({name})' if culture else ''),
            'culture': culture, 'version': name,
            'file': rel, 'bytes': os.path.getsize(path), 'sha256': _sha256(path),
            'installs_to': f'datasets/{name}',
            'rows': rows, 'cols': cols,
            'comment': _dataset_comment(name, meta, rows, cols),
        })
        if verbose:
            note = ' (архив прежний)' if reuse else ''
            print(f'  датасет {name}: {os.path.getsize(path) / 1048576:.0f} МБ{note}')
    return out


# --------------------------------------------------------------------------- #
# Снимки базы
# --------------------------------------------------------------------------- #
TIMESERIES_TABLES = ('parms', 'series', 'time_series')


def merged_timeseries(verbose=True):
    """``parms``, ``series``, ``time_series`` для полного снимка = база ∪ оверлей докачки.

    Ряды живут в двух местах: базовая таблица и оверлей — всё, что докачивалось с Vega при
    сборке датасетов (у автора в оверлее больше рядов и территорий, чем в базе). Снимок из одной
    базы был бы молча беднее авторской. Объединяем по тому же правилу, что ``db.get_time_series``:
    ряд из оверлея ПЕРЕКРЫВАЕТ одноимённый ряд базы. Рабочая база автора не меняется —
    объединение строится только в памяти, для снимка.

    Возвращает ``(parms, series, time_series, stats)``.
    """
    parms = db.load('parms').copy()
    series = db.load('series').copy()
    base_ts = db.load('time_series')
    overlay = db._downloads()
    keys = sorted(overlay.keys())

    to_id = {str(n): int(i) for i, n in zip(parms['parm_id'], parms['parm'])}
    next_pid = (max(to_id.values()) + 1) if to_id else 0
    sidx = {(str(t), int(pid), int(y)): int(sid) for sid, t, pid, y in
            zip(series['series_id'], series['territory_id'], series['parm_id'], series['year'])}
    next_sid = (int(series['series_id'].max()) + 1) if len(series) else 0

    new_parms, new_series, replaced = [], [], set()
    sids, days, vals = [], [], []
    for n, key in enumerate(keys, 1):
        tid, parm, year = str(key[0]), str(key[1]), int(key[2])
        pts = overlay.points(key)
        if not pts:
            continue
        pid = to_id.get(parm)
        if pid is None:                                    # параметр, которого нет в справочнике базы
            pid = next_pid
            next_pid += 1
            to_id[parm] = pid
            new_parms.append({'parm_id': pid, 'parm': parm})
        sid = sidx.get((tid, pid, year))
        if sid is None:                                    # ряда в базе нет — новая строка series
            sid = next_sid
            next_sid += 1
            sidx[(tid, pid, year)] = sid
            new_series.append({'series_id': sid, 'territory_id': tid, 'parm_id': pid, 'year': year,
                               'first_date': f'{year}-01-01', 'last_date': f'{year}-12-31'})
        else:                                              # есть — точки оверлея заменяют точки базы
            replaced.add(sid)
        arr = np.asarray(pts, dtype=np.float64)
        sids.append(np.full(len(arr), sid, dtype=np.int64))
        days.append(arr[:, 0].astype(np.int64))
        vals.append(arr[:, 1])
        if verbose and n % 10000 == 0:
            print(f'    оверлей: разобрано {n} из {len(keys)} рядов…')

    kept = base_ts[~base_ts['series_id'].isin(replaced)] if replaced else base_ts
    over_ts = pd.DataFrame({'series_id': np.concatenate(sids) if sids else np.array([], np.int64),
                            'day': np.concatenate(days) if days else np.array([], np.int64),
                            'value': np.concatenate(vals) if vals else np.array([], np.float64)})
    time_series = pd.concat([kept, over_ts], ignore_index=True)
    if new_parms:
        parms = pd.concat([parms, pd.DataFrame(new_parms)], ignore_index=True)
    if new_series:
        series = pd.concat([series, pd.DataFrame(new_series)], ignore_index=True)

    stats = {'base_series': int(len(db.load('series'))), 'overlay_series': len(keys),
             'replaced': len(replaced), 'added': len(new_series),
             'series': int(len(series)), 'points': int(len(time_series)),
             'territories': int(series['territory_id'].nunique())}
    return parms, series, time_series, stats


def _export_table(table, staging, frame=None):
    """Выгрузить таблицу БД в staging (Parquet для тяжёлых, иначе CSV). -> путь или None."""
    frame = db.load(table) if frame is None else frame
    if frame.empty:
        return None
    if table in PARQUET_TABLES and db._has_parquet():
        path = os.path.join(staging, table + '.parquet')
        frame.to_parquet(path, index=False)
    else:
        path = os.path.join(staging, table + '.csv')
        frame.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def build_db(outdir, flavour='full', version=None, verbose=True):
    """Собрать снимок базы (``lite`` или ``full``). -> запись каталога либо None."""
    tables = LITE_TABLES if flavour == 'lite' else FULL_TABLES
    tables = [t for t in tables if t not in db.USER_TABLES]     # страховка от опечатки в списке
    version = version or datetime.now().strftime('%Y-%m')
    staging = os.path.join(config.WORKSPACE, '.release_staging', flavour)
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    merged, ts_stats = {}, None
    if any(t in TIMESERIES_TABLES for t in tables):        # полный снимок: база ∪ оверлей
        if verbose:
            print('    объединяю ряды базы и оверлея докачки…')
        m_parms, m_series, m_ts, ts_stats = merged_timeseries(verbose=verbose)
        merged = {'parms': m_parms, 'series': m_series, 'time_series': m_ts}
        if verbose:
            print(f'    рядов: база {ts_stats["base_series"]}, оверлей {ts_stats["overlay_series"]} '
                  f'(из них заменили ряд базы {ts_stats["replaced"]}, новых {ts_stats["added"]}) '
                  f'-> итог {ts_stats["series"]} рядов, {ts_stats["points"]:,} точек, '
                  f'{ts_stats["territories"]} территорий')

    paths, made = [], []
    for t in tables:
        p = _export_table(t, staging, frame=merged.get(t))
        if p:
            paths.append(p); made.append(t)
            if verbose:
                print(f'    {t}: {os.path.getsize(p) / 1048576:.1f} МБ')
        elif verbose:
            print(f'    {t}: пусто — пропуск')
    if not paths:
        shutil.rmtree(staging, ignore_errors=True)
        return None

    rel = f'db/db-{flavour}-{version}.zip'
    path = _zip_files(paths, os.path.join(outdir, rel.replace('/', os.sep)))
    shutil.rmtree(staging, ignore_errors=True)
    title = ('База: справочники и урожайности' if flavour == 'lite'
             else 'База: справочники, урожайности и временные ряды')
    if verbose:
        print(f'  снимок {flavour}: {os.path.getsize(path) / 1048576:.1f} МБ')
    return {
        'id': f'db:{flavour}-{version}', 'kind': 'db', 'title': f'{title} ({version})',
        'version': version, 'file': rel, 'bytes': os.path.getsize(path), 'sha256': _sha256(path),
        'db_schema_version': db.SCHEMA_VERSION, 'tables': made,
        'comment': ('Лёгкий снимок: работают разделы данных и проверка урожайности, '
                    'для прогноза ряды докачиваются с Vega.' if flavour == 'lite'
                    else 'Полный снимок: прогноз работает без обращения к Vega. Временных рядов '
                         f'{ts_stats["series"]} по {ts_stats["territories"]} территориям '
                         '(база и всё докачанное с Vega).' if ts_stats
                    else 'Полный снимок: прогноз работает без обращения к Vega.'),
        'timeseries': ts_stats,
    }


def _previous_assets(outdir):
    """Записи прежнего манифеста (пустой список, если его нет или он битый)."""
    path = os.path.join(outdir, 'catalog.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as f:
            return list(json.load(f).get('assets', []))
    except (OSError, ValueError):
        return []


def _previous_entry(outdir, rel_file):
    """Запись прежнего манифеста по имени файла — чтобы не пересчитывать sha уже выложенного."""
    path = os.path.join(outdir, 'catalog.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            old = json.load(f)
    except (OSError, ValueError):
        return None
    for a in old.get('assets', []):
        if a.get('file') == rel_file:
            return dict(a)
    return None


def write_catalog(outdir, entries, note=''):
    """Записать ``catalog.json`` в корень релизной папки."""
    catalog = {
        'catalog_version': 1,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'app': 'Nerual Rabbit',
        'db_schema_version': db.SCHEMA_VERSION,
        'note': note,
        'assets': entries,
    }
    path = os.path.join(outdir, 'catalog.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description='Сборка релизной папки (модели + снимки БД + каталог)')
    ap.add_argument('--out', default=os.path.join(config.WORKSPACE, 'release'),
                    help='куда собирать (по умолчанию workspace/release)')
    ap.add_argument('--models', nargs='*', help='только эти модели (по именам папок)')
    ap.add_argument('--no-models', action='store_true', help='не собирать модели')
    ap.add_argument('--db', choices=['lite', 'full', 'both', 'no'], default='both',
                    help='какие снимки базы собирать (по умолчанию оба)')
    ap.add_argument('--datasets', nargs='*',
                    help='собрать датасеты обучения: без значений — все, иначе перечислить задания')
    ap.add_argument('--only-datasets', action='store_true',
                    help='собрать только датасеты (модели и базу не трогать)')
    ap.add_argument('--db-version', help='метка версии снимка (по умолчанию ГГГГ-ММ)')
    ap.add_argument('--note', default='', help='примечание в манифесте (видно в программе)')
    ap.add_argument('--manifest-only', action='store_true',
                    help='не пересобирать архивы: обновить только catalog.json (комментарии, '
                         'метрики) поверх уже выложенных файлов — перезалить нужно один файл')
    args = ap.parse_args()

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    print(f'Релизная папка: {outdir}')

    entries = []
    if args.only_datasets:
        args.no_models = True
        args.db = 'no'
        if args.datasets is None:
            args.datasets = []                             # пустой список = «все датасеты»
    if not args.no_models:
        print('Модели:')
        entries += build_models(outdir, only=args.models, reuse=args.manifest_only)
    if args.manifest_only:                                 # снимки базы тоже берём как есть
        for rel in sorted(os.listdir(os.path.join(outdir, 'db'))) if os.path.isdir(
                os.path.join(outdir, 'db')) else []:
            path = os.path.join(outdir, 'db', rel)
            prev = _previous_entry(outdir, f'db/{rel}')
            if prev:
                prev['bytes'] = os.path.getsize(path)
                entries.append(prev)
                print(f'  снимок {rel}: {os.path.getsize(path) / 1048576:.1f} МБ (архив прежний)')
    elif args.db != 'no':
        for flavour in (['lite', 'full'] if args.db == 'both' else [args.db]):
            print(f'Снимок базы ({flavour}):')
            rec = build_db(outdir, flavour, version=args.db_version)
            if rec:
                entries.append(rec)

    if args.datasets is not None:
        print('Датасеты обучения:')
        entries += build_datasets(outdir, only=args.datasets or None, reuse=args.manifest_only)

    # Переносим из прежнего манифеста всё, что в этом запуске не собирали: иначе «собрать
    # только датасеты» выкинуло бы из каталога модели и базу, а пересборка ОДНОГО датасета —
    # все остальные. Раздел считается пересобранным целиком, только если по нему не было
    # фильтра: тогда отсутствие записи означает, что автор её убрал намеренно.
    built_kinds = set()
    if not args.no_models and not args.models:
        built_kinds.add('model')
    if args.db == 'both':
        built_kinds.add('db')
    if args.datasets is not None and not args.datasets:
        built_kinds.add('dataset')
    carried = 0
    for prev in _previous_assets(outdir):
        if prev.get('kind') in built_kinds or any(e['id'] == prev.get('id') for e in entries):
            continue
        if not os.path.exists(os.path.join(outdir, str(prev.get('file', '')).replace('/', os.sep))):
            continue                                       # файла больше нет — запись не переносим
        entries.append(prev)
        carried += 1
    if carried:
        print(f'Перенесено из прежнего каталога: {carried} записей (файлы на месте)')

    path = write_catalog(outdir, entries, note=args.note)
    total = sum(int(e['bytes']) for e in entries)
    print(f'\nМанифест: {path}')
    print(f'Записей: {len(entries)}, суммарно {total / 1048576:.1f} МБ')
    print('Выложите содержимое папки целиком (файлами, не папкой-архивом) и дайте '
          'публичную ссылку — её программа спрашивает во вкладке «Каталог».')


if __name__ == '__main__':
    main()
