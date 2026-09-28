"""Одноразовая свёртка оверлея скачанных рядов к одной точке в сутки.

До 2026-09 ``put_time_series`` писал метеопараметры Веги как есть — 4 значения в сутки под
одним целым номером дня (~1448 точек на ряд-год вместо 366), из-за чего оверлей вырос до
3.7 ГБ, а результат интерполяции зависел от порядка точек внутри суток (см.
``docs/plan_subdaily_overlay.md``). Теперь ``put_time_series`` сводит ряд к суткам сам
(:func:`db.to_daily`), а этот скрипт приводит к тому же виду уже накопленный оверлей:

* обычный параметр — среднее за сутки, накопительный (``*_acc``) — значение на конец суток;
* ряды с одной точкой в сутки (NDVI, ``mean_temp_acc``) переписываются без изменений;
* формат файла, столбцы и провенанс ``vega_uid`` сохраняются (``purge_overlay`` работает);
* повторно скачанные ряды: остаётся последний блок (как при чтении).

Файл переписывается потоково через индекс (память — несколько сотен МБ), исходный
оверлей сохраняется рядом как ``time_series_downloaded.subdaily.bak.csv`` — удалите его
вручную, когда убедитесь, что всё в порядке (или ``--no-backup``).

Запуск:
    python code/compact_overlay_daily.py            # сухой прогон: оценка по выборке рядов
    python code/compact_overlay_daily.py --apply    # переписать оверлей
"""

import argparse
import os
import random
import time

import config
import db
import overlay_index as oi


def _block_bytes(key, uid, pts):
    prefix = f'{key[0]},{key[1]},{key[2]},{"" if uid is None else uid},'
    return ''.join(f'{prefix}{int(d)},{oi._fmt_value(v)}{oi.EOL}' for d, v in pts).encode('utf-8')


def dry_run(idx, sample=2000):
    entries = idx._index()
    keys = list(entries)
    random.seed(0)
    keys = random.sample(keys, min(sample, len(keys)))
    before = after = nbytes_before = nbytes_after = 0
    for key in keys:
        pts = idx.points(key)
        daily = db.to_daily(key[1], pts)
        before += len(pts)
        after += len(daily)
        nbytes_before += entries[key][1]
        nbytes_after += len(_block_bytes(key, entries[key][3], daily))
    share = nbytes_after / max(nbytes_before, 1)
    size = os.path.getsize(idx.csv_path)
    print(f'Выборка {len(keys)} рядов: точек {before:,} -> {after:,} '
          f'({after / max(before, 1):.1%}), байт {share:.1%}')
    print(f'Оценка: оверлей {size / 1e6:.0f} МБ -> ~{size * share / 1e6:.0f} МБ')
    print('Сухой прогон. Для перезаписи запустите с --apply.')


def apply(idx, keep_backup=True):
    path = idx.csv_path
    entries = idx._index()
    order = sorted(entries.items(), key=lambda kv: kv[1][0])      # исходный порядок файла
    tmp = path + '.daily.tmp'
    size0 = os.path.getsize(path)
    t0 = time.time()
    pts_before = pts_after = 0
    next_report = t0 + 15.0
    with open(tmp, 'wb') as out:
        out.write((oi.HEADER + oi.EOL).encode('utf-8'))
        for i, (key, entry) in enumerate(order, 1):
            pts = idx.points(key)
            daily = db.to_daily(key[1], pts)
            pts_before += len(pts)
            pts_after += len(daily)
            out.write(_block_bytes(key, entry[3], daily))
            if time.time() > next_report:
                print(f'  ...{i:,} / {len(order):,} рядов', flush=True)
                next_report = time.time() + 15.0

    idx.invalidate()                      # закрыть дескриптор (Windows не даст переименовать) и сбросить индекс
    bak = os.path.join(os.path.dirname(path), db.OVERLAY_TABLE + '.subdaily.bak.csv')
    if keep_backup:
        if os.path.exists(bak):
            os.remove(bak)
        os.replace(path, bak)
    os.replace(tmp, path)
    db.reload()

    n = len(db._downloads())              # пересборка индекса по новому файлу
    size1 = os.path.getsize(path)
    print(f'Готово за {time.time() - t0:.0f} с: рядов {n:,} (было {len(order):,}), '
          f'точек {pts_before:,} -> {pts_after:,}, '
          f'файл {size0 / 1e6:.0f} МБ -> {size1 / 1e6:.0f} МБ')
    if keep_backup:
        print(f'Исходный оверлей сохранён: {bak} — удалите вручную после проверки.')


def main():
    ap = argparse.ArgumentParser(description='Свёртка оверлея к одной точке в сутки.')
    ap.add_argument('--apply', action='store_true', help='переписать оверлей (без --apply — сухой прогон)')
    ap.add_argument('--no-backup', action='store_true', help='не сохранять исходный оверлей')
    args = ap.parse_args()

    config.ensure_dirs()
    idx = db._downloads()
    if not os.path.exists(idx.csv_path):
        print('Оверлея нет — сворачивать нечего.')
        return
    if args.apply:
        apply(idx, keep_backup=not args.no_backup)
    else:
        dry_run(idx)


if __name__ == '__main__':
    main()
