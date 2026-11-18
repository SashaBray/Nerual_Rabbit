"""Пересборка суточных метеорядов из подробного хранилища по одному правилу.

Зачем. Суточные метеоряды приходят из двух источников, и читались они по ДВУМ РАЗНЫМ
договорённостям:

* **оверлей** ``time_series_downloaded.csv`` — записан текущим кодом, то есть через
  :func:`db.to_daily`: обычный параметр сведён к СРЕДНЕМУ за сутки, накопительный к
  значению на конец суток. Это правильно;
* **базовая таблица** ``time_series.csv`` — наследство прежнего загрузчика. Он писал
  шестичасовые точки с целым номером дня (``int(float(d))``), и при схлопывании дубликатов
  уцелела одна точка из четырёх — ПОСЛЕДНЯЯ в сутках. По расшифровке дробной части
  (см. ниже) это срез **06 UTC**, то есть мгновенное утреннее измерение, выданное за
  среднесуточное.

Измерено 2026-11-18 на выборке 2500 рядов: из тех, что вообще читаются, **50.2 %**
оказались срезом 06 UTC, а не средним. Расхождение размером со весь суточный ход:
температура до 11 °C, влажность до 41 %, снежный покров до 69 %. Хуже всего то, что
договорённость меняется ВНУТРИ района между годами: у 512 районов часть лет читается
одним способом, часть другим. Межгодовая изменчивость внутри района — именно то, на чём
стоит вся модель, и переключение определения подмешивалось в неё как шум.

Дробная часть номера дня у Веги — фиксированные часы UTC (0.0 = 12 UTC, 0.25 = 18, 0.5 = 00,
0.75 = 06). Проверено по долготе: доля ``(слот0.75 − слот0.5)/(слот0.0 − слот0.5)`` растёт
монотонно с 0.503 (Псков) до 0.917 (Башкортостан), корреляция с долготой 0.978 на 11
регионах. Будь часы местными, зависимости от долготы не было бы.

Что делает скрипт. Для каждого ряда подробного (6-часового) хранилища, которого ещё нет в
оверлее, считает суточный ряд тем же :func:`db.to_daily` и дозаписывает в оверлей. Оверлей
читается раньше базовой таблицы, поэтому старое значение просто перестаёт использоваться;
сама базовая таблица не трогается. Ничего не теряется: обе договорённости выводятся из
подробного хранилища, так что откат — это повторная запись среза вместо среднего.

Запуск:
    python code/rebuild_daily.py --plan
    python code/rebuild_daily.py
"""
import argparse
import time

import db
import series_6h
from download_6h import WEATHER


def tasks(parms, years=None):
    """Ряды подробного хранилища, которых нет в суточном оверлее."""
    over = set(db._downloads().keys())                 # noqa: SLF001  (внутренний индекс)
    out = []
    for tid, parm, year in series_6h.keys():
        if parm not in parms:
            continue
        if years and year not in years:
            continue
        if (tid, parm, year) not in over:
            out.append((tid, parm, year))
    return sorted(out, key=lambda k: (k[2], k[0], k[1]))


def main():
    ap = argparse.ArgumentParser(description='Суточные ряды из 6-часовых по db.to_daily')
    ap.add_argument('--parms', default='all', help='all | список через запятую')
    ap.add_argument('--years', default=None, help='например 2007-2013')
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()

    parms = set(WEATHER if a.parms == 'all' else [p.strip() for p in a.parms.split(',')])
    years = None
    if a.years:
        years = set()
        for part in a.years.split(','):
            if '-' in part:
                lo, hi = part.split('-', 1)
                years.update(range(int(lo), int(hi) + 1))
            elif part.strip():
                years.add(int(part))

    todo = tasks(parms, years)
    if not todo:
        print('нечего пересобирать: всё уже в оверлее')
        return
    yrs = sorted({k[2] for k in todo})
    print('рядов к пересборке: %d, годы %d-%d, районов %d'
          % (len(todo), yrs[0], yrs[-1], len({k[0] for k in todo})))
    print('оценка дозаписи: %.2f ГБ' % (len(todo) * 366 * 41.3 / 2 ** 30))
    if a.plan:
        return

    uids = series_6h.store().uids()
    t0 = time.time()
    done = empty = 0
    for i, (tid, parm, year) in enumerate(todo, 1):
        pts = series_6h.get(tid, parm, year)
        if not pts:
            empty += 1
            continue
        db.put_time_series(tid, parm, year, pts, vega_uid=uids.get((tid, parm, year)))
        done += 1
        if i % 5000 == 0:
            el = time.time() - t0
            print('  %6d / %d  записано %d  прошло %.1f мин, осталось ~%.1f мин'
                  % (i, len(todo), done, el / 60, (len(todo) - i) * el / i / 60), flush=True)
    print('\nготово за %.1f мин: записано %d, пустых %d'
          % ((time.time() - t0) / 60, done, empty))


if __name__ == '__main__':
    main()
