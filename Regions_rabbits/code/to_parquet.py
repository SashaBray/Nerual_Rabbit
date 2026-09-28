"""Конвертация тяжёлой таблицы ``time_series`` из CSV в Parquet (и обратно).

Parquet (колоночный формат со словарным/RLE/delta-кодированием и сжатием) сильно
уменьшает размер ``time_series`` и ускоряет чтение. Требует ``pyarrow`` (или
``fastparquet``). Если движок не установлен — печатает подсказку и ничего не меняет.
``db.load``/``db.save`` работают с обоими форматами прозрачно (Parquet приоритетнее).

Запуск:  ``python to_parquet.py``            # CSV -> Parquet
         ``python to_parquet.py --to-csv``   # Parquet -> CSV (обратно)
"""

import os
import sys

import config
import db


def _mb(path):
    return round(os.path.getsize(path) / 1e6, 2) if os.path.exists(path) else 0.0


def to_parquet(table='time_series'):
    if not db._has_parquet():
        print('Не найден движок Parquet. Установите: pip install pyarrow')
        return
    csv = db._csv_path(table)
    pq = db._parquet_path(table)
    before = _mb(csv) or _mb(pq)
    db.reload()
    df = db.load(table)
    if df.empty:
        print(f'Таблица {table} пуста — нечего конвертировать.')
        return
    db.save(table, df)        # запишет Parquet и удалит CSV (table в _PARQUET_TABLES)
    print(f'{table}: {before} МБ (CSV) -> {_mb(pq)} МБ (Parquet), строк {len(df)}')


def to_csv(table='time_series'):
    """Принудительно вернуть таблицу в CSV (например, для просмотра)."""
    db.reload()
    df = db.load(table)
    pq = db._parquet_path(table)
    df.to_csv(db._csv_path(table), index=False, encoding='utf-8-sig')
    if os.path.exists(pq):
        os.remove(pq)
    db._cache.clear()
    db._index_cache.clear()
    print(f'{table}: записан CSV {_mb(db._csv_path(table))} МБ, строк {len(df)}')


if __name__ == '__main__':
    config.ensure_dirs()
    if '--to-csv' in sys.argv:
        to_csv()
    else:
        to_parquet()
