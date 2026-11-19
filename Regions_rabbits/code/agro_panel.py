"""Литературные признаки на панели 2007-2024, с настоящими суточными крайностями.

Зачем ещё один модуль, когда есть ``agro_features``. Тот считает те же механизмы, но по
старому блочному датасету (восемь лет, около 32 независимых наблюдений) и только по суточным
СРЕДНИМ. В его собственном списке невозможного первой строкой стоит: «суточных минимумов и
максимумов нет, только средние за сутки», из-за чего литературные пороги пришлось переводить
на шкалу средних — порог жары 31 °C по максимуму стал 25 °C по средней, а критическая
температура узла кущения оказалась недостижима вовсе. После докачки шестичасовых рядов
(``download_6h.py``) ограничение снято: у каждых суток четыре среза (проверено: 3.99 среза на
сутки), и минимум с максимумом по ним — приемлемое приближение суточных крайностей. Пороги
возвращены к литературным.

Чем приближение отличается от истинных крайностей. Срезы стоят в фиксированные часы UTC:
12, 18, 00 и 06 (см. ``rebuild_daily.py``). На долготе 45° в. д. это 15, 21, 03 и 09 часов по
местному времени, то есть максимум берётся из послеполуденного среза, а минимум — из ночного
или утреннего. Истинный максимум бывает позже 15 часов, поэтому жара слегка недооценена;
измеренная июньская амплитуда 10.8-12.4 °C против реальных 12-14 °C это и показывает. Для
счётчиков «сколько дней выше порога» такое смещение означает консервативную оценку: признак
скорее недосчитает жару, чем придумает её.

Чем этот опыт сильнее перебора. ``peak_residual_screen`` отбирает признаки и считает результат
на одних и тех же годах, поэтому его лучшие числа — верхняя оценка, а не проверка; на 18 годах
объявленная тройка упала с 0.2291 до 0.1258, а ``epoch_transfer`` показал, что отбор между
эпохами не переносится и не отличим от шума. Здесь ОТБОРА НЕТ. Окна, пороги и ожидаемые знаки
выписаны из литературы до того, как посчитана первая корреляция, и не меняются по результату.
Поэтому корреляция равновесной суммы по 18 годам — проверка, а не верхняя оценка, и сравнивать
её надо с шумовым полом (остаток, перемешанный по годам внутри района).

Механизмы и источники — в докстринге ``agro_features``: он не переписан, а продолжен. Здесь
перечислены только отличия в счёте.

Две поправки к счёту, найденные при переносе
============================================

**Осадки.** ``mean_prec`` Веги — это миллиметры за срез, а не за сутки: сумма четырёх срезов
совпадает с приростом ``mean_prec_acc`` (июнь 2012 в районе 13_1336: 67.8 мм прироста против
17.0 суммы суточных средних, отношение ровно 4.0). Суточные осадки здесь считаются суммой
срезов. В ``agro_features`` ГТК складывал суточные средние, то есть был вчетверо меньше
настоящего, и порог «ГТК < 1 — недостаточное увлажнение» к тем числам не применим.

**ГТК Селянинова** = 10·ΣP / Σt, где Σt — сумма СРЕДНИХ СУТОЧНЫХ температур за дни с t > 10 °C
(сумма активных температур). В ``agro_features`` в знаменателе стояла сумма эффективных
температур ``Σ(t − 10)``, что даёт другую величину. Исправлено.

Запуск:
    python code/agro_panel.py --culture "Озимая пшеница"            # собрать признаки
    python code/agro_panel.py --culture "Озимая пшеница" --eval     # и проверить
"""
import argparse
import datetime as dt
import os
import time
import warnings

import numpy as np
import pandas as pd

import config
import db
import peak_regression as P
import series_6h

SEASON_DAYS = 325        # 1 сентября года Y-1 … 21 июля года Y с запасом на високосный
MIN_SLOTS = 3            # суточный минимум и максимум по двум срезам считать нельзя

# Пороги. Каждый — из литературы; в скобках источник указан в докстринге agro_features.
T_DORMANT = 3.0          # начало покоя: трое суток со средней ниже этого
T_HARDEN = 10.0          # закалка идёт, пока грунт между 0 и этим
T_DEHARDEN = 5.0         # оттепель, снимающая закалку
GR_KILL = -10.0          # мороз на 10 см как аналог критического на узле кущения
SNOW_BARE = 0.05         # нет снежного одеяла, м
SNOW_DEEP = 0.30         # мощный снег (предиктор белорусской методики), м
SNOW_THIN = 0.02         # снег на талой почве -> выпревание и ледяная корка, м
T_SPRING = 5.0           # возобновление вегетации: пятисуточная средняя выше этого
T_SPRING_FROST = -2.0    # лёд в точке роста после выхода в трубку
HEAT_ANTH = 31.0         # порог повреждения жарой в цветение, по суточному максимуму
HEAT_FILL = 33.0         # порог повреждения жарой в налив
SUKH_T, SUKH_RH = 25.0, 30.0   # суховей по ГОСТ без ветра: максимум t и минимум влажности
DRY_MM = 1.0             # сухой день, мм

# Канал подробного хранилища -> какие суточные сводки из него нужны.
SUBDAILY = {'t': ('mean_temp', ('сред', 'мин', 'макс')),
            'гр': ('mean_tmpgr10', ('сред', 'мин')),
            'вл': ('mean_rh', ('сред', 'мин')),
            'снег': ('mean_snod', ('сред', 'макс')),
            'почв': ('mean_soilw10', ('сред', 'мин', 'макс')),
            'ос': ('mean_prec', ('сум',))}

# Ожидаемый знак связи с остатком районной линии — из литературы, до счёта корреляций.
# 0 означает, что литература не даёт однозначного направления (например, оптимум).
SIGNS = {
    'осень_длина': 0, 'осень_гдд': 0, 'осень_закалка': +1, 'осень_влага': +1,
    'осень_ос': +1, 'осень_гтк': +1, 'осень_ndvi': +1,
    'зима_грунт_мин': +1, 'зима_дней_м10': -1, 'зима_зимобой': -1, 'зима_снег30': +1,
    'зима_снег_макс': +1, 'зима_разакалка': -1, 'зима_корка': -1,
    'весна_отраст': -1, 'весна_заморозок': +1, 'весна_дней_мороз': -1, 'весна_влага': +1,
    'труб_влага': +1, 'труб_влага_мин': +1, 'труб_ndvi': +1, 'труб_гтк': +1,
    'труб_жара31': -1,
    'колош_гтк': +1, 'колош_жара31': -1, 'колош_влага_мин': +1, 'колош_ndvi_макс': +1,
    'налив_гтк': +1, 'налив_жара33': -1, 'налив_суховей': -1, 'налив_сухостой': -1,
    'налив_ndvi_спад': +1,
    'вегет_гтк': +1, 'вегет_жара31': -1,
}

# Ядро: механизмы, которые в литературе названы определяющими чаще прочих. Равновесная сумма
# этих признаков со знаками из SIGNS — предсказание с нулём подогнанных параметров.
CORE = ('осень_влага', 'зима_зимобой', 'зима_снег30', 'весна_заморозок', 'труб_влага',
        'труб_ndvi', 'колош_гтк', 'колош_жара31', 'налив_суховей', 'налив_жара33',
        'налив_ndvi_спад', 'вегет_гтк')


# --------------------------------------------------------------------------- #
# Суточные сводки из шестичасовых рядов
# --------------------------------------------------------------------------- #
def daily(tid, parm, year, want):
    """Суточные сводки ряда подробного хранилища. -> {сводка: массив по дням года}.

    День берётся как целая часть номера: четыре среза дня D — это 12 и 18 UTC самого дня и
    00 и 06 UTC следующего, то есть окно «день плюс ночь». Крайности по окну с меньше чем
    ``MIN_SLOTS`` срезами не считаются: по одному срезу минимум равен максимуму, и такой
    день притворился бы днём без суточного хода.
    """
    a = np.asarray(series_6h.get(tid, parm, year), dtype=float)
    if a.size == 0:
        return {}
    d = np.floor(a[:, 0] + 1e-9).astype(int)
    v = a[:, 1]
    ok = np.isfinite(v) & (d >= 0) & (d < 400)
    d, v = d[ok], v[ok]
    if d.size == 0:
        return {}
    n = int(d.max()) + 1
    cnt = np.bincount(d, minlength=n).astype(float)
    out = {}
    if 'сред' in want or 'сум' in want:
        s = np.bincount(d, weights=v, minlength=n)
        if 'сум' in want:
            out['сум'] = np.where(cnt > 0, s, np.nan)
        if 'сред' in want:
            out['сред'] = np.where(cnt > 0, s / np.maximum(cnt, 1.0), np.nan)
    if 'мин' in want or 'макс' in want:
        bad = cnt < MIN_SLOTS
        if 'мин' in want:
            mn = np.full(n, np.inf)
            np.minimum.at(mn, d, v)
            mn[bad] = np.nan
            out['мин'] = mn
        if 'макс' in want:
            mx = np.full(n, -np.inf)
            np.maximum.at(mx, d, v)
            mx[bad] = np.nan
            out['макс'] = mx
    return out


def ndvi_daily(tid, parm, year):
    """Ряд маски NDVI по дням года; снимки редкие, между ними NaN."""
    a = np.asarray(db.get_time_series(tid, parm, year), dtype=float)
    out = np.full(400, np.nan)
    if a.size == 0:
        return out
    d = np.floor(a[:, 0] + 1e-9).astype(int)
    v = a[:, 1]
    m = np.isfinite(v) & (v > 0) & (d >= 0) & (d < 400)
    out[d[m]] = v[m]
    return out


def district_daily(tid, years, mask_parm):
    """Суточные сводки района по всем годам. -> {(год, канал, сводка): массив}."""
    cache = {}
    for yr in years:
        for short, (parm, want) in SUBDAILY.items():
            for name, arr in daily(tid, parm, yr, want).items():
                cache[(yr, short, name)] = arr
        cache[(yr, 'ndvi', 'сред')] = ndvi_daily(tid, mask_parm, yr)
    return cache


# --------------------------------------------------------------------------- #
# Ось сезона
# --------------------------------------------------------------------------- #
def season(cache, year, short, name):
    """Канал на оси «сутки от 1 сентября года Y-1». -> массив (SEASON_DAYS,) или None."""
    out = np.full(SEASON_DAYS, np.nan)
    s0 = dt.date(year - 1, 9, 1)
    got = False
    for y in (year - 1, year):
        a = cache.get((y, short, name))
        if a is None:
            continue
        i0 = (dt.date(y, 1, 1) - s0).days
        lo = max(0, -i0)
        hi = min(len(a), SEASON_DAYS - i0)
        if hi <= lo:
            continue
        out[i0 + lo:i0 + hi] = a[lo:hi]
        got = True
    return out if got else None


def win(year, m1, d1, m2, d2):
    """Полуинтервал [начало, конец) на оси сезона; месяц от 9 относится к году Y-1."""
    s0 = dt.date(year - 1, 9, 1)
    a = dt.date(year - 1 if m1 >= 9 else year, m1, d1)
    b = dt.date(year - 1 if m2 >= 9 else year, m2, d2)
    return (a - s0).days, (b - s0).days


# --------------------------------------------------------------------------- #
# Вспомогательные свёртки одного ряда
# --------------------------------------------------------------------------- #
def _nan(f, a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(f(a)) if np.isfinite(a).any() else np.nan


def _count(mask):
    """Число дней, где условие выполнено; NaN считается невыполнением."""
    return float(np.count_nonzero(mask))


def _longest(mask):
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return float(best)


def _runmean(a, k):
    """Скользящее среднее по k суткам, выровненное по левому краю окна."""
    v = np.where(np.isfinite(a), a, 0.0)
    c = np.concatenate([[0.0], np.cumsum(v)])
    n = np.concatenate([[0.0], np.cumsum(np.isfinite(a).astype(float))])
    num, den = c[k:] - c[:-k], n[k:] - n[:-k]
    return np.where(den >= k - 1, num / np.maximum(den, 1.0), np.nan)


def _first(mask, default=np.nan):
    i = np.flatnonzero(mask)
    return float(i[0]) if i.size else default


def _slope(a):
    """Наклон по суткам, ед./сут; None, если точек меньше пяти."""
    ok = np.isfinite(a)
    if ok.sum() < 5:
        return np.nan
    x = np.flatnonzero(ok).astype(float)
    return float(np.polyfit(x, a[ok], 1)[0])


def _gtk(t, p):
    """ГТК Селянинова: 10·ΣP / Σt, обе суммы по дням активной вегетации (t > 10 °C).

    Осадки считаются по тем же дням, что и тепло: иначе в окно с холодным началом попадают
    дожди апреля, а тепла за них нет, и коэффициент раздувается втрое.
    """
    m = np.isfinite(t) & (t > 10.0)
    heat = float(np.nansum(t[m]))
    if heat < 100.0:                      # меньше суммы активных температур делить незачем
        return np.nan
    return 10.0 * float(np.nansum(p[m])) / heat


# --------------------------------------------------------------------------- #
# Признаки
# --------------------------------------------------------------------------- #
def features(cache, year):
    """Признаки одного района-года по механизмам литературы. -> {имя: число}."""
    ch = {}
    for short, (_p, want) in SUBDAILY.items():
        for name in want:
            ch[(short, name)] = season(cache, year, short, name)
    ch[('ndvi', 'сред')] = season(cache, year, 'ndvi', 'сред')

    def g(short, name, m1, d1, m2, d2):
        a = ch.get((short, name))
        if a is None:
            return None
        lo, hi = win(year, m1, d1, m2, d2)
        s = a[lo:hi]
        return s if np.isfinite(s).any() else None

    f = {}

    # --- Осень: длина вегетации, закалка, уход в зиму ---------------------------------
    t = g('t', 'сред', 9, 1, 1, 1)
    if t is not None:
        dorm = _first(_runmean(t, 3) < T_DORMANT, float(len(t)))
        f['осень_длина'] = dorm
        f['осень_гдд'] = float(np.nansum(np.clip(t[:int(dorm)], 0, None)))
    gr = g('гр', 'сред', 9, 1, 1, 1)
    if gr is not None:
        f['осень_закалка'] = _count((gr >= 0) & (gr <= T_HARDEN))
    s = g('почв', 'сред', 9, 1, 10, 1)
    if s is not None:
        f['осень_влага'] = _nan(np.nanmean, s)
    p = g('ос', 'сум', 9, 1, 10, 1)
    if p is not None:
        f['осень_ос'] = float(np.nansum(p))
    t1, p1 = g('t', 'сред', 9, 1, 10, 1), p
    if t1 is not None and p1 is not None:
        f['осень_гтк'] = _gtk(t1, p1)
    nd = g('ndvi', 'сред', 11, 11, 12, 1)
    if nd is not None:
        f['осень_ndvi'] = _nan(np.nanmean, nd)

    # --- Зима: перезимовка -------------------------------------------------------------
    grm = g('гр', 'мин', 12, 1, 3, 1)
    grs = g('гр', 'сред', 12, 1, 3, 1)
    sn = g('снег', 'сред', 12, 1, 3, 1)
    snx = g('снег', 'макс', 12, 1, 3, 1)
    if grm is not None:
        f['зима_грунт_мин'] = _nan(np.nanmin, grm)
        f['зима_дней_м10'] = _count(grm < GR_KILL)
    if grs is not None:
        warm = grs > T_DEHARDEN
        f['зима_разакалка'] = _count(warm[1:] & ~warm[:-1])
    if sn is not None:
        f['зима_снег30'] = _count(sn >= SNOW_DEEP)
    if snx is not None:
        f['зима_снег_макс'] = _nan(np.nanmax, snx)
    if grm is not None and sn is not None:
        f['зима_зимобой'] = _count((grm < GR_KILL) & (sn < SNOW_BARE))
    if grs is not None and sn is not None:
        f['зима_корка'] = _count((sn > SNOW_THIN) & (grs > 0))

    # --- Весна: отрастание и возвратные заморозки --------------------------------------
    spr = g('t', 'сред', 2, 1, 6, 1)
    sprm = g('t', 'мин', 2, 1, 6, 1)
    if spr is not None:
        start = _first(_runmean(spr, 5) > T_SPRING, float(len(spr)))
        f['весна_отраст'] = start
        if sprm is not None:
            a, b = int(start), min(int(start) + 30, len(sprm))
            f['весна_заморозок'] = _nan(np.nanmin, sprm[a:b])
            f['весна_дней_мороз'] = _count(sprm[a:b] < T_SPRING_FROST)
    s = g('почв', 'сред', 4, 1, 5, 1)
    if s is not None:
        f['весна_влага'] = _nan(np.nanmean, s)

    # --- Трубка (май): закладка колоса -------------------------------------------------
    s = g('почв', 'сред', 5, 1, 6, 1)
    if s is not None:
        f['труб_влага'] = _nan(np.nanmean, s)
    s = g('почв', 'мин', 5, 1, 6, 1)
    if s is not None:
        f['труб_влага_мин'] = _nan(np.nanmin, s)
    nd = g('ndvi', 'сред', 5, 1, 6, 1)
    if nd is not None:
        f['труб_ndvi'] = _nan(np.nanmean, nd)
    t1, p1 = g('t', 'сред', 5, 1, 6, 1), g('ос', 'сум', 5, 1, 6, 1)
    if t1 is not None and p1 is not None:
        f['труб_гтк'] = _gtk(t1, p1)
    tm = g('t', 'макс', 5, 1, 6, 1)
    if tm is not None:
        f['труб_жара31'] = _count(tm >= HEAT_ANTH)

    # --- Колошение и цветение (1-20 июня): число зёрен ---------------------------------
    t1, p1 = g('t', 'сред', 6, 1, 6, 21), g('ос', 'сум', 6, 1, 6, 21)
    if t1 is not None and p1 is not None:
        f['колош_гтк'] = _gtk(t1, p1)
    tm = g('t', 'макс', 6, 1, 6, 21)
    if tm is not None:
        f['колош_жара31'] = _count(tm >= HEAT_ANTH)
    s = g('почв', 'мин', 6, 1, 6, 21)
    if s is not None:
        f['колош_влага_мин'] = _nan(np.nanmin, s)
    nd = g('ndvi', 'сред', 6, 1, 6, 21)
    if nd is not None:
        f['колош_ndvi_макс'] = _nan(np.nanmax, nd)

    # --- Налив (21 июня - 20 июля): масса зерна ----------------------------------------
    t1, p1 = g('t', 'сред', 6, 21, 7, 21), g('ос', 'сум', 6, 21, 7, 21)
    if t1 is not None and p1 is not None:
        f['налив_гтк'] = _gtk(t1, p1)
    if p1 is not None:
        f['налив_сухостой'] = _longest(np.nan_to_num(p1, nan=0.0) < DRY_MM)
    tm = g('t', 'макс', 6, 21, 7, 21)
    if tm is not None:
        f['налив_жара33'] = _count(tm >= HEAT_FILL)
        rhm = g('вл', 'мин', 6, 21, 7, 21)
        if rhm is not None:
            f['налив_суховей'] = _count((tm >= SUKH_T) & (rhm <= SUKH_RH))
    nd = g('ndvi', 'сред', 6, 11, 7, 21)
    if nd is not None:
        f['налив_ndvi_спад'] = _slope(nd)

    # --- Вегетация целиком (апрель - 20 июля) ------------------------------------------
    t1, p1 = g('t', 'сред', 4, 1, 7, 21), g('ос', 'сум', 4, 1, 7, 21)
    if t1 is not None and p1 is not None:
        f['вегет_гтк'] = _gtk(t1, p1)
    tm = g('t', 'макс', 4, 1, 7, 21)
    if tm is not None:
        f['вегет_жара31'] = _count(tm >= HEAT_ANTH)
    return f


def build(rl, mask_parm):
    """Таблица признаков по всей панели. -> DataFrame с territory_id, year и признаками."""
    keys = rl[['territory_id', 'year']].drop_duplicates()
    rows, t0 = [], time.time()
    for i, (tid, grp) in enumerate(keys.groupby('territory_id'), 1):
        years = sorted(set(grp['year'].astype(int)))
        need = sorted(set(years) | {y - 1 for y in years})
        cache = district_daily(tid, need, mask_parm)
        for yr in years:
            f = features(cache, int(yr))
            f.update({'territory_id': tid, 'year': int(yr)})
            rows.append(f)
        if i % 100 == 0:
            print(f'    районов {i}, {time.time() - t0:.0f} c', flush=True)
    t = pd.DataFrame(rows)
    cols = ['territory_id', 'year'] + [c for c in SIGNS if c in t.columns]
    return t[cols]


# --------------------------------------------------------------------------- #
# Проверка
# --------------------------------------------------------------------------- #
MIN_PER_DISTRICT = 3     # меньше трёх лет — норму района считать не из чего
MIN_PER_YEAR = 30        # меньше тридцати районов в году — корреляция не считается
PREV_YEARS = 4           # норма района по последним N ПРОШЛЫМ годам: столько есть и в работе
PREV_MIN = 2             # меньше двух прошлых лет — отклонение не считаем

# Когда признак становится известен: окно закрылось — значение есть. Это и есть
# заблаговременность прогноза, а не свободный параметр.
LEAD = {'осень': '1 июня', 'зима': '1 июня', 'весна': '1 июня', 'труб': '1 июня',
        'колош': '21 июня', 'налив': 'уборка', 'вегет': 'уборка'}
LEADS = ('1 июня', '21 июня', 'уборка')


def lead_of(name):
    """Срок, к которому признак известен."""
    return LEAD[name.split('_')[0]]


def known_by(cols, lead):
    """Признаки, известные к сроку lead."""
    i = LEADS.index(lead)
    return [c for c in cols if LEADS.index(lead_of(c)) <= i]


def zscore(t, cols, era):
    """z-оценка признаков по району в пределах эпохи; вне эпохи NaN.

    Норма района только по годам своей эпохи: иначе знание о проверочных годах протекает
    через среднее и разброс района.
    """
    g = t.loc[era].groupby('territory_id')[cols]
    mu, sd, n = g.transform('mean'), g.transform('std'), g.transform('count')
    z = (t.loc[era, cols] - mu) / sd.where(sd > 1e-12)
    out = pd.DataFrame(np.nan, index=t.index, columns=cols)
    out.loc[era, cols] = z.where(n >= MIN_PER_DISTRICT)
    return out.to_numpy(float)


def prevnorm(t, cols, era):
    """Отклонение признака от средней района по ПРЕДЫДУЩИМ годам, в единицах разброса эпохи.

    Отличие от ``zscore`` принципиальное для прогноза. Там норма района считается по всем
    годам эпохи, то есть в неё входят и годы ПОСЛЕ оцениваемого — в работе такой нормы нет: в
    году Y известны только годы до Y. Здесь норма честная, по прошлым годам района, ровно то,
    что даёт условие задачи «известно несколько урожайностей и их ряды». Масштаб (деление на
    разброс) берётся по эпохе обучения, иначе масштаб проверочных лет протекал бы в модель.
    """
    v = t[cols].to_numpy(float)
    tid, yr = t['territory_id'].to_numpy(), t['year'].to_numpy(int)
    out = np.full_like(v, np.nan)
    for s in np.unique(tid):
        k = np.flatnonzero(tid == s)
        k = k[np.argsort(yr[k])]
        for i in range(len(k)):
            p = v[k[max(0, i - PREV_YEARS):i]]
            if p.size == 0:
                continue
            n = np.isfinite(p).sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mu = np.nanmean(p, axis=0)
            out[k[i]] = np.where(n >= PREV_MIN, v[k[i]] - mu, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sd = np.nanstd(out[era], axis=0)
    return out / np.where(sd > 1e-12, sd, np.nan)


def per_year_r(v, yr, era, y):
    """Корреляция с остатком внутри каждого года эпохи. -> массив по годам."""
    out = []
    for g in sorted(set(yr[era])):
        m = (yr == g) & np.isfinite(v) & np.isfinite(y)
        if m.sum() < MIN_PER_YEAR or np.std(v[m]) < 1e-12:
            continue
        out.append(float(np.corrcoef(v[m], y[m])[0, 1]))
    return np.array(out)


def composite(Z, pick, sign):
    """Равновесная сумма признаков с заданными знаками; пропуск = норма района."""
    a = np.where(np.isfinite(Z[:, pick]), Z[:, pick], 0.0) * sign
    return a.mean(axis=1)


def ridge(Z, y, lam):
    """Гребневая регрессия на заполненных нулями z-оценках. -> вектор весов."""
    X = np.where(np.isfinite(Z), Z, 0.0)
    m = np.isfinite(y)
    X, yy = X[m], y[m]
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ yy)


def centre_in_year(v, yr):
    """Предсказание минус его средняя по году.

    Зачем. Модель учится на остатке, нормированном по району, и внутри года упорядочивает
    районы. Уровня года она не знает: «каким выдался год» — связь более сильная, и
    нецентрированное предсказание вбирает её и начинает перехлёстывать. Прежняя линейная
    поправка в центнерах портила модель именно по этой причине (см. ``curve_autoencoder``).
    """
    out = np.full_like(v, np.nan)
    for g in sorted(set(yr)):
        m = (yr == g) & np.isfinite(v)
        if m.sum() >= MIN_PER_YEAR:
            out[m] = v[m] - v[m].mean()
    return out


def calib(v, y, era):
    """Масштаб предсказания по эпохе ОБУЧЕНИЯ: наклон остатка на предсказание."""
    m = era & np.isfinite(v) & np.isfinite(y)
    den = float(np.sum(v[m] ** 2))
    return float(np.sum(v[m] * y[m]) / den) if den > 1e-12 else 0.0


def pick_lam(Z, y, yr, era, grid):
    """Гребень по годам ЭПОХИ ОБУЧЕНИЯ: год вон, остальные учат. -> (lam, r, прогнозы).

    Прогнозы отложенных лет нужны не только для выбора гребня. По ним берётся масштаб
    поправки: наклон, посчитанный «в себя», завышен вдвое-втрое, и применённый к другой
    эпохе он перехлёстывает, то есть ухудшает RMSE даже при верном знаке связи.
    """
    years = sorted(set(yr[era]))
    best = (grid[0], -9.0, None)
    for lam in grid:
        v = np.full(len(y), np.nan)
        for g in years:
            tr = era & (yr != g)
            w = ridge(Z[tr], y[tr], lam)
            m = era & (yr == g)
            v[m] = np.where(np.isfinite(Z[m]), Z[m], 0.0) @ w
        rs = [float(r[0]) for g in years
              for r in [per_year_r(v, yr, era & (yr == g), y)] if len(r)]
        if rs and np.mean(rs) > best[1]:
            best = (lam, float(np.mean(rs)), v)
    return best


def evaluate(rl, t, split):
    """Проверка литературных признаков: по одному, равновесной суммой и гребнем по эпохам."""
    d = rl[['territory_id', 'year', 'ост', 'sd', 'ост_norm']].merge(
        t, on=['territory_id', 'year'], how='inner')
    cols = [c for c in SIGNS if c in d.columns]
    yr = d['year'].to_numpy(int)
    tid = d['territory_id'].to_numpy()
    y = d['ост_norm'].to_numpy(float)
    early, late = yr <= split, yr > split
    all_era = np.ones(len(d), bool)
    print(f'строк {len(d)}, районов {len(set(tid))}, признаков {len(cols)}')
    print(f'ранняя {yr.min()}-{split}: {early.sum()} строк; '
          f'поздняя {split + 1}-{yr.max()}: {late.sum()} строк')

    ZA, ZE, ZL = (zscore(d, cols, e) for e in (all_era, early, late))
    rng = np.random.default_rng(5)
    ysh = y.copy()
    for s in np.unique(tid):
        k = np.flatnonzero(tid == s)
        ysh[k] = y[rng.permutation(k)]

    rows = []
    for j, c in enumerate(cols):
        ra = per_year_r(ZA[:, j], yr, all_era, y)
        re = per_year_r(ZE[:, j], yr, early, y)
        rlate = per_year_r(ZL[:, j], yr, late, y)
        sh = per_year_r(ZA[:, j], yr, all_era, ysh)
        she = per_year_r(ZE[:, j], yr, early, ysh)
        shl = per_year_r(ZL[:, j], yr, late, ysh)
        exp = SIGNS[c]
        стойкий = (len(re) and len(rlate) and np.sign(re.mean()) == np.sign(rlate.mean())
                   and abs(re.mean()) > np.abs(she).mean()
                   and abs(rlate.mean()) > np.abs(shl).mean())
        rows.append({'признак': c, 'знак_ожид': exp, 'покрытие': int(np.isfinite(ZA[:, j]).sum()),
                     'r_18лет': round(float(ra.mean()), 4) if len(ra) else np.nan,
                     'лет': len(ra),
                     'знак_держится': int((np.sign(ra) == np.sign(ra.mean())).sum()),
                     'как_ждали': ('да' if exp and np.sign(ra.mean()) == exp
                                   else ('-' if not exp else 'НЕТ')) if len(ra) else '',
                     'r_ранняя': round(float(re.mean()), 4) if len(re) else np.nan,
                     'r_поздняя': round(float(rlate.mean()), 4) if len(rlate) else np.nan,
                     'r_шум': round(float(np.abs(sh).mean()), 4) if len(sh) else np.nan,
                     'стойкий': 'да' if стойкий else ''})
    tab = pd.DataFrame(rows).sort_values('r_18лет', key=lambda s: -s.abs())
    print('\n=== признаки по одному (z-оценка по району, корреляция внутри года) ===')
    print(tab.to_string(index=False))
    st = tab[tab['стойкий'] == 'да']
    print('\n=== стойкие: знак один в обеих эпохах и обе выше своего шумового пола ===')
    print(st[['признак', 'знак_ожид', 'как_ждали', 'r_ранняя', 'r_поздняя',
              'r_18лет']].to_string(index=False) if len(st) else '  нет')

    core = [c for c in CORE if c in cols]
    pick = np.array([cols.index(c) for c in core])
    sign = np.array([SIGNS[c] for c in core], float)
    print(f'\n=== равновесная сумма ядра ({len(core)} признаков, знаки из литературы) ===')
    sums = []
    for Z, era, nm in ((ZA, all_era, 'все 18 лет'), (ZE, early, f'{yr.min()}-{split}'),
                       (ZL, late, f'{split + 1}-{yr.max()}')):
        r = per_year_r(composite(Z, pick, sign), yr, era, y)
        rs = per_year_r(composite(Z, pick, sign), yr, era, ysh)
        print('  %-12s r %+.4f (лет %d, плюс в %d), на перемешанном остатке %+.4f'
              % (nm, r.mean(), len(r), int((r > 0).sum()), rs.mean()))
        sums.append({'что': 'ядро', 'эпоха': nm, 'r': round(float(r.mean()), 4),
                     'лет': len(r), 'r_шум': round(float(rs.mean()), 4)})

    print('\n=== отбор ВНУТРИ литературного набора: выбираем на одной эпохе, сумма на другой ===')
    print('кандидатов %d, а не две сотни, поэтому плата за отбор мала; сравнивать с шумом'
          % len(cols))
    for nk in (4, 8):
        for Zs, sel, Zt, tst, lbl in ((ZE, early, ZL, late, 'ранняя -> поздняя'),
                                      (ZL, late, ZE, early, 'поздняя -> ранняя')):
            line = []
            for tgt, nm in ((y, 'данные'), (ysh, 'шум')):
                sc = np.array([per_year_r(Zs[:, j], yr, sel, tgt).mean()
                               if len(per_year_r(Zs[:, j], yr, sel, tgt)) else np.nan
                               for j in range(len(cols))])
                ok = np.flatnonzero(np.isfinite(sc))
                pk = ok[np.argsort(-np.abs(sc[ok]))][:nk]
                sg = np.sign(sc[pk])
                r = per_year_r(composite(Zt, pk, sg), yr, tst, tgt)
                line.append((nm, r.mean(), len(r), int((r > 0).sum()),
                             [cols[j] for j in pk]))
            print('  k=%d %-18s данные r %+.4f (плюс в %d из %d), шум %+.4f'
                  % (nk, lbl, line[0][1], line[0][3], line[0][2], line[1][1]))
            if nk == 4:
                print('      отобраны: ' + ', '.join(line[0][4]))
            sums.append({'что': f'отбор k={nk}', 'норма': 'по эпохе', 'срок': '', 'эпоха': lbl,
                         'признаков': nk, 'r': round(float(line[0][1]), 4), 'лет': line[0][2],
                         'r_шум': round(float(line[1][1]), 4)})

    sd = d['sd'].to_numpy(float)
    ost = d['ост'].to_numpy(float)
    grid = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

    def run_ridge(Zs, sel, Zt, tst, lbl, use, tag):
        """Гребень на подмножестве признаков: учим на одной эпохе, проверяем на другой."""
        j = np.array([cols.index(c) for c in use])
        lam, r_in, vloo = pick_lam(Zs[:, j], y, yr, sel, grid)
        w = ridge(Zs[np.ix_(sel, j)], y[sel], lam)
        v = np.where(np.isfinite(Zt[:, j]), Zt[:, j], 0.0) @ w
        r = per_year_r(v, yr, tst, y)
        vc = centre_in_year(v, yr)
        k = calib(centre_in_year(vloo, yr), y, sel)
        m = tst & np.isfinite(vc) & np.isfinite(ost) & np.isfinite(sd)
        line = float(np.sqrt(np.mean(ost[m] ** 2)))
        raw = float(np.sqrt(np.mean((ost[m] - v[m] * sd[m]) ** 2)))
        cal = float(np.sqrt(np.mean((ost[m] - k * vc[m] * sd[m]) ** 2)))
        rsh = per_year_r(v, yr, tst, ysh)
        print('    %-18s признаков %2d, гребень %4.0f; на отборе r %+.4f, НА ПРОВЕРКЕ '
              'r %+.4f (плюс в %d из %d), шум %+.4f'
              % (lbl, len(use), lam, r_in, r.mean(), int((r > 0).sum()), len(r), rsh.mean()))
        print('        RMSE линии %.3f; как есть %.3f (ΔRMSE %+.3f), с центрированием по году '
              'и масштабом %.2f: %.3f (ΔRMSE %+.3f ц/га)'
              % (line, raw, line - raw, k, cal, line - cal))
        sums.append({'что': 'гребень', 'норма': tag, 'срок': '', 'эпоха': lbl,
                     'признаков': len(use), 'r': round(float(r.mean()), 4), 'лет': len(r),
                     'r_шум': round(float(rsh.mean()), 4),
                     'ΔRMSE_как_есть': round(line - raw, 4),
                     'ΔRMSE': round(line - cal, 4)})
        return r.mean()

    print('\n=== гребень: учим на одной эпохе, проверяем на другой ===')
    print('норма района по эпохе (не из работы: в неё входят и будущие годы)')
    for Zs, sel, Zt, tst, lbl in ((ZE, early, ZL, late, 'ранняя -> поздняя'),
                                  (ZL, late, ZE, early, 'поздняя -> ранняя')):
        run_ridge(Zs, sel, Zt, tst, lbl, cols, 'по эпохе')

    print('\nнорма района по прошлым годам (как в работе), по срокам прогноза')
    for sel, tst, lbl in ((early, late, 'ранняя -> поздняя'), (late, early, 'поздняя -> ранняя')):
        Zp = prevnorm(d, cols, sel)
        for lead in LEADS:
            use = known_by(cols, lead)
            print('  срок %s:' % lead)
            run_ridge(Zp, sel, Zp, tst, lbl, use, 'по прошлым')
            sums[-1]['срок'] = lead
    return tab, pd.DataFrame(sums)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='Литературные признаки на панели 18 лет')
    ap.add_argument('--culture', required=True)
    ap.add_argument('--split', type=int, default=2012, help='последний год ранней эпохи')
    ap.add_argument('--rebuild', action='store_true', help='пересчитать таблицу признаков')
    ap.add_argument('--eval', action='store_true', help='прогнать проверку')
    a = ap.parse_args()

    cu = P.culture_row(a.culture)
    _hdoy, task = P.harvest_doy(cu['culture_id'])
    src = os.path.join(config.REPORTS_DIR, 'peak_residual', task + '_early')
    if not os.path.exists(os.path.join(src, 'residuals.csv')):
        raise SystemExit(f'нет {src}/residuals.csv; сначала peak_residual_screen.py --early')
    rl = pd.read_csv(os.path.join(src, 'residuals.csv'), sep=';', decimal=',',
                     encoding='utf-8-sig')
    pk = pd.read_csv(os.path.join(config.REPORTS_DIR, 'peak_regression', task + '_early',
                                  'peaks.csv'), sep=';', decimal=',', encoding='utf-8-sig')
    mask_parm = str(pk['маска'].mode().iloc[0])

    outdir = os.path.join(config.REPORTS_DIR, 'agro_panel', task)
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, 'features.csv')
    if os.path.exists(fpath) and not a.rebuild:
        t = pd.read_csv(fpath, sep=';', decimal=',', encoding='utf-8-sig')
        print(f'признаки: беру из {fpath}, {t.shape}')
    else:
        print(f'{cu["title"]}: маска {mask_parm}, строк панели {len(rl)}')
        t = build(rl, mask_parm)
        t.to_csv(fpath, sep=';', decimal=',', index=False, encoding='utf-8-sig')
        print(f'признаки: {t.shape} -> {fpath}')

    if a.eval:
        tab, sums = evaluate(rl, t, a.split)
        tab.to_csv(os.path.join(outdir, 'single.csv'), sep=';', decimal=',', index=False,
                   encoding='utf-8-sig')
        sums.to_csv(os.path.join(outdir, 'summary.csv'), sep=';', decimal=',', index=False,
                    encoding='utf-8-sig')
        print(f'\nотчёт -> {outdir}')


if __name__ == '__main__':
    main()
