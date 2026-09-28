"""Состояние и пересборка индекса оверлея скачанных рядов.

Индекс (``time_series_downloaded.index.csv`` + ``.index.json``) строится сам при первом
чтении оверлея, но на файле в гигабайты это один проход по диску (~100 с на 3.7 ГБ) —
удобнее заплатить его один раз из консоли, а не внутри приложения. Скрипт также годится
как диагностика: сколько рядов и точек лежит в оверлее и не пора ли его чистить.

Запуск:
    python code/overlay_reindex.py             # показать состояние (построить, если индекса нет)
    python code/overlay_reindex.py --rebuild    # выбросить индекс и пересобрать с нуля
    python code/overlay_reindex.py --check      # прочитать по индексу 200 случайных рядов
"""

import argparse
import os
import random
import time

import config
import db


def _mb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


def main():
    ap = argparse.ArgumentParser(description='Индекс оверлея скачанных рядов.')
    ap.add_argument('--rebuild', action='store_true', help='пересобрать индекс с нуля')
    ap.add_argument('--check', action='store_true', help='контрольное чтение случайных рядов')
    args = ap.parse_args()

    config.ensure_dirs()
    idx = db._downloads()
    if not os.path.exists(idx.csv_path):
        print('Оверлея нет — индексировать нечего:', idx.csv_path)
        return

    if args.rebuild:
        print('Сбрасываю индекс...')
        idx.invalidate()

    t0 = time.time()
    stats = idx.stats()
    dt = time.time() - t0

    print(f'оверлей : {os.path.basename(idx.csv_path)}  {_mb(idx.csv_path):.0f} МБ')
    print(f'индекс  : {os.path.basename(idx.index_path)}  {_mb(idx.index_path):.2f} МБ '
          f'(загрузка/сборка {dt:.1f} с)')
    print(f'рядов   : {stats["series"]:,}')
    print(f'точек   : {stats["points"]:,}  '
          f'(~{stats["points"] / max(stats["series"], 1):.0f} на ряд)')
    print(f'покрыто : {stats["covered_bytes"]:,} из {os.path.getsize(idx.csv_path):,} байт')

    if args.check:
        keys = list(idx.keys())
        random.seed(0)
        sample = random.sample(keys, min(200, len(keys)))
        t = time.time()
        total = sum(len(db.get_time_series(*k)) for k in sample)
        dt = time.time() - t
        print(f'контроль: {len(sample)} рядов, {total:,} точек, {dt:.2f} с '
              f'({dt / len(sample) * 1000:.2f} мс/ряд)')


if __name__ == '__main__':
    main()
