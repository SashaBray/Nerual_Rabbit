"""Докачка подробных рядов: метео с шагом 6 часов, NDVI за новые годы.

Что и зачем. Перебор сводок и автоэнкодер кривых упёрлись в одно и то же: при 11
независимых годах и районных средних добавочной информации в суточных рядах больше не
находится (``peak_residual_screen.py``, ``curve_autoencoder.py``). Две оси, по которым ещё
можно двигаться, обе требуют докачки:

* **годы.** Урожайности есть с 2007 (``yields.csv``), а метео в базе — только с 2013.
  Проверено, что ``yields.csv`` и ``цель_убранная`` из БДПМО совпадают на 7400 общих строк
  (корреляция 1.0000, средняя разница 0.00), поэтому ранние годы — та же цель, а не другая
  система измерения. Это поднимает число независимых лет с 11 до 17;
* **время внутри суток.** Вега отдаёт метео с шагом 0.25 суток, а ``db.put_time_series``
  берёт ``int(float(d))`` и теряет дробную часть. Шестичасовой ход сохраняется в отдельное
  хранилище (:mod:`series_6h`), суточное остаётся неприкосновенным.

Объём считается не по полной сетке, а по делу: качаются только те «район-год», где есть
непустая урожайность хоть по одной культуре и известен ``vega_uid``. Это 13 236 районо-лет
вместо 17 746, то есть 132 тысячи запросов на метео вместо 177 тысяч.

Что куда пишется:

* метео -> ``series_6h`` с дробными днями, без свёртки;
* NDVI (шаг 7 суток, свёртка ему не нужна) -> обычный суточный оверлей через
  ``db.put_time_series``, чтобы не держать две копии одного и того же.

Осторожность с сервисом. Измерено: удачный ответ приходит за 1-4 с, но из шести попыток
подряд удачными были две (ReadTimeout и ConnectionError за 60-82 с). Поэтому ретраи с
нарастающей паузой, умеренная параллельность и возобновляемость: уже скачанное пропускается
по индексу, так что прогон можно останавливать и запускать заново сколько угодно раз.

Запуск:
    python code/download_6h.py --plan                      # только показать объём
    python code/download_6h.py --years 2007-2013 --threads 4
    python code/download_6h.py --threads 4                  # все годы, что есть в yields
    python code/download_6h.py --parms mean_temp --years 2012 --limit 50
"""
import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import db
import series_6h
from vega import VegaClient

WEATHER = ('mean_temp', 'mean_temp_acc', 'mean_prec', 'mean_prec_acc', 'mean_rh',
           'mean_p', 'mean_snod', 'mean_snowc', 'mean_soilw10', 'mean_tmpgr10')
MASKS = ('mean_ndvi_7dc_modis_int_ozim', 'mean_ndvi_7dc_modis_int_spring',
         'mean_ndvi_7dc_modis_int_agro')
TRIES = 4               # попыток на ряд; сервис отвечает примерно через раз
TIMEOUT = 75            # с; удачный ответ приходит за 1-4 с, но сервер бывает вялым
BACKOFF = (2, 8, 20)    # пауза перед повторами
YIELD_MIN, YIELD_MAX = 0.001, 200.0     # отсечь нули и сигнальные 99999999.99
_AGENTS = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:109.0) Gecko/20100101 Firefox/115.0',
)


# --------------------------------------------------------------------------- #
# План
# --------------------------------------------------------------------------- #
def targets_scope(years=None):
    """Районо-годы, где есть непустая урожайность и известен vega_uid.

    Качать полную сетку 934 района на все годы незачем: за 2007 урожайности есть только у
    301 района, и метео остальных прикладывать было бы не к чему.
    """
    y = db.load('yields')
    y['territory_id'] = y['territory_id'].astype(str)
    y['year'] = pd.to_numeric(y['year'], errors='coerce')
    y['yield'] = pd.to_numeric(y['yield'], errors='coerce')
    y = y[(y['yield'] > YIELD_MIN) & (y['yield'] < YIELD_MAX)].dropna(subset=['year'])
    y['year'] = y['year'].astype(int)
    if years:
        y = y[y['year'].isin(years)]
    scope = y[['territory_id', 'year']].drop_duplicates()

    best = {}
    for (tid, _parm, _yr), uid in db.download_uids().items():
        if uid:
            best[str(tid)] = int(uid)
    scope['vega_uid'] = scope['territory_id'].map(best)
    missing = int(scope['vega_uid'].isna().sum())
    scope = scope.dropna(subset=['vega_uid'])
    scope['vega_uid'] = scope['vega_uid'].astype(int)
    return scope.sort_values(['year', 'territory_id']).reset_index(drop=True), missing


def build_tasks(scope, parms, masks):
    """Задания ``(куда, tid, uid, parm, year)`` минус уже скачанное — прогон возобновляем."""
    have6 = series_6h.keys()
    tasks = []
    for tid, yr, uid in scope[['territory_id', 'year', 'vega_uid']].itertuples(index=False):
        for parm in parms:
            if (tid, parm, int(yr)) not in have6:
                tasks.append(('6h', tid, uid, parm, int(yr)))
        for parm in masks:
            if not db.has_time_series(tid, parm, int(yr)):
                tasks.append(('сутки', tid, uid, parm, int(yr)))
    return tasks


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #
class Fetcher:
    """Запросы к Веге с ретраями: свой session на поток, общий счётчик под замком."""

    def __init__(self):
        self.ukey = VegaClient().ukey
        self.local = threading.local()
        self.lock = threading.Lock()
        self.ok = self.fail = self.empty = self.points = 0

    def _session(self):
        se = getattr(self.local, 'session', None)
        if se is None:
            se = requests.Session()
            se.trust_env = False            # прокси из окружения ломают прямой адрес Веги
            self.local.session = se
        return se

    def _url(self, parm, year, uid):
        """URL строит сам клиент: шаблон один, дублировать его тут нельзя."""
        holder = {}
        cl = VegaClient(ukey=self.ukey)
        cl._get = lambda u: (holder.setdefault('u', u), (None, False))[1]
        cl.district(parm, year, uid)
        return holder['u']

    def series(self, parm, year, uid):
        """-> список ``[день, значение]``, пустой список или None, если не ответил."""
        url = self._url(parm, str(year), uid)
        for k in range(TRIES):
            try:
                r = self._session().get(url, headers={'User-Agent': random.choice(_AGENTS)},
                                        timeout=TIMEOUT)
                if r.status_code == 200 and r.text.lstrip().startswith('{'):
                    blocks = list((r.json().get('data') or {}).values())
                    xy = (blocks[0].get('xy') if blocks else []) or []
                    return [[float(p[0]), float(p[1])] for p in xy
                            if p[1] is not None and p[1] != '']
            except Exception:                                      # noqa: BLE001
                pass
            if k < len(BACKOFF):
                time.sleep(BACKOFF[k] * (0.5 + random.random()))
        return None


def run(tasks, threads, fetcher, report_every=200):
    """Выполнить задания; каждый ряд пишется сразу по готовности."""
    t0 = time.time()
    done = [0]
    lock = threading.Lock()

    def one(task):
        where, tid, uid, parm, year = task
        pts = fetcher.series(parm, year, uid)
        with fetcher.lock:
            if pts is None:
                fetcher.fail += 1
            elif not pts:
                fetcher.empty += 1
            else:
                fetcher.ok += 1
                fetcher.points += len(pts)
        if pts:
            if where == '6h':
                series_6h.put(tid, parm, year, pts, vega_uid=uid)
            else:
                db.put_time_series(tid, parm, year, pts, vega_uid=uid)
        with lock:
            done[0] += 1
            n = done[0]
        if n % report_every == 0:
            el = time.time() - t0
            left = (len(tasks) - n) * el / max(n, 1)
            print('  %6d / %d  удачных %d  пусто %d  сбоев %d  прошло %.0f мин, '
                  'осталось ~%.0f мин'
                  % (n, len(tasks), fetcher.ok, fetcher.empty, fetcher.fail,
                     el / 60, left / 60), flush=True)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, tasks))
    return time.time() - t0


# --------------------------------------------------------------------------- #
def parse_years(text):
    """``2007-2013``, ``2012`` или ``2007,2012-2014`` -> множество лет."""
    if not text:
        return None
    out = set()
    for part in str(text).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description='Докачка 6-часовых метеорядов и NDVI')
    ap.add_argument('--years', default=None, help='например 2007-2013 или 2012,2013')
    ap.add_argument('--parms', default='all', help='all | список через запятую')
    ap.add_argument('--masks', default='all', help='all | none | список через запятую')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='взять только первые N заданий')
    ap.add_argument('--plan', action='store_true', help='только показать объём и выйти')
    a = ap.parse_args()

    parms = WEATHER if a.parms == 'all' else tuple(p.strip() for p in a.parms.split(','))
    if a.masks == 'none':
        masks = ()
    elif a.masks == 'all':
        masks = MASKS
    else:
        masks = tuple(m.strip() for m in a.masks.split(','))

    scope, missing = targets_scope(parse_years(a.years))
    if scope.empty:
        raise SystemExit('в этих годах нет ни одной урожайности с известным uid')
    print('районо-лет с урожайностью и uid: %d (без uid пропущено %d)'
          % (len(scope), missing))
    print('годы %d-%d, районов %d' % (scope.year.min(), scope.year.max(),
                                      scope.territory_id.nunique()))
    tasks = build_tasks(scope, parms, masks)
    n6 = sum(1 for t in tasks if t[0] == '6h')
    print('заданий всего %d: метео 6ч %d, NDVI %d' % (len(tasks), n6, len(tasks) - n6))
    print('оценка веса: %.2f ГБ (метео по 1463 точки, NDVI по 32, 41.3 Б на точку)'
          % ((n6 * 1463 + (len(tasks) - n6) * 32) * 41.3 / 2 ** 30))
    for sec in (1.5, 3.0):
        print('  при %.1f с на запрос в %d потоков: %.0f ч'
              % (sec, a.threads, len(tasks) * sec / max(a.threads, 1) / 3600))
    print('уже в подробном хранилище:', series_6h.stats())
    if a.plan:
        return
    if a.limit:
        tasks = tasks[:a.limit]
        print('ограничено до %d заданий' % len(tasks))

    fetcher = Fetcher()
    el = run(tasks, a.threads, fetcher)
    print('\nготово за %.0f мин: удачных %d, пустых %d, сбоев %d, точек %d'
          % (el / 60, fetcher.ok, fetcher.empty, fetcher.fail, fetcher.points))
    print('подробное хранилище:', series_6h.stats())
    print('файл:', series_6h.path())


if __name__ == '__main__':
    main()
