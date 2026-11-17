"""Шестичасовые ряды — отдельное хранилище рядом с суточным, своё время внутри суток.

Вега отдаёт метеопараметры с шагом 0.25 суток: x приходит строкой вида ``"1.25"``, где
дробная часть и есть время (1.0 — 00:00, 1.25 — 06:00, 1.5 — полдень, 1.75 — 18:00).
Проверено живыми запросами 2026-11-17: 1463 точки на год на районном уровне.

Почему отдельное хранилище, а не дробные дни в существующем оверлее. ``db.get_time_series``
приводит день к ``int``, а ``db.py`` при экспорте в JSON делает ``str(int(d))``; десяток
скриптов проекта читает ряды через эти функции и ждёт целых дней. Положив дробные дни в тот
же файл, мы бы молча испортили всё, что считается на суточном разрешении. Поэтому суточное
хранилище остаётся неприкосновенным, а подробное живёт рядом со своим индексом.

Устройство — то же, что у суточного оверлея (``overlay_index.OverlayIndex``): append-лог
``time_series_6h.csv`` плюс индекс смещений, чтение одного ряда через ``seek``. Разница одна —
``fractional_days=True``, то есть день пишется и читается как число с дробной частью.

Свёртки к суткам здесь НЕТ сознательно: смысл хранилища в том, чтобы сохранить то, что
суточное теряет. Нужна суточная величина — её считает потребитель (среднее за сутки для
обычных параметров, конец суток для ``*_acc``; правило откалибровано в
``compact_overlay_daily.py``).
"""
import os

import config
import overlay_index

TABLE = 'time_series_6h'
_store = {}


def path():
    """Путь к файлу подробного хранилища."""
    return os.path.join(config.DB_DIR, TABLE + '.csv')


def store():
    """Ленивый индекс подробного хранилища (один на процесс)."""
    idx = _store.get('index')
    if idx is None:
        idx = overlay_index.OverlayIndex(path(), fractional_days=True)
        _store['index'] = idx
    return idx


def get(tid, parm, year):
    """Ряд ``[[день_с_дробью, значение], ...]`` по возрастанию; пустой список, если нет."""
    pts = store().points((str(tid), str(parm), int(year)))
    return [] if pts is None else [[float(d), float(v)] for d, v in pts]


def has(tid, parm, year):
    """Есть ли ряд в подробном хранилище (по индексу, без чтения данных)."""
    return (str(tid), str(parm), int(year)) in store()


def put(tid, parm, year, xy, vega_uid=None):
    """Дописать ряд как есть, с дробными днями и без свёртки. -> число записанных точек."""
    pts = [[float(d), float(v)] for d, v in xy if v is not None]
    if not pts:
        return 0
    pts.sort(key=lambda p: p[0])
    store().append_series((str(tid), str(parm), int(year)), pts, vega_uid=vega_uid)
    return len(pts)


def keys():
    """Ключи ``(territory_id, parm, year)``, уже лежащие в хранилище."""
    return set(store().keys())


def stats():
    """Сводка по хранилищу: рядов, точек, байт — по файлу индекса, без чтения данных."""
    import pandas as pd

    size = os.path.getsize(path()) if os.path.exists(path()) else 0
    ipath = overlay_index.index_paths(path())[0]
    if not os.path.exists(ipath):
        return {'рядов': len(store()), 'точек': None, 'байт': size}
    ix = pd.read_csv(ipath)
    return {'рядов': len(ix), 'точек': int(ix['npoints'].sum()), 'байт': size,
            'параметров': ix['parm'].nunique(), 'районов': ix['territory_id'].nunique(),
            'годы': (int(ix['year'].min()), int(ix['year'].max())) if len(ix) else None}
