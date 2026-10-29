"""Признаки из агрономической литературы: не «что можно посчитать», а «что считают важным».

Отличие от ``series_features``. Там ряд резюмировался по фазам развития, но набор величин
выбирал я: средние, минимумы, максимумы, счётчики — всё, что осмысленно для агронома. Здесь
обратный порядок. Сначала по литературе выписаны **механизмы**, которыми исследователи
объясняют формирование урожая озимой пшеницы, затем каждый механизм переведён в число,
вычислимое из наших каналов, и только это число попадает в пространство признаков. Величина,
за которой нет механизма, сюда не входит, сколько бы корреляции она ни показывала.

Зачем так. Независимых наблюдений в задаче около 32 (четверть дисперсии отклонения урожайности
— один общий сдвиг на год, а лет всего восемь). Против 32 наблюдений 347 признаков безнадёжны,
и отбирать их перебором значит подгонять. Литература — единственный источник отбора, который
не смотрит в наши данные, то есть не тратит эти 32 наблюдения.

Механизмы и их источники
========================

**Осень: уход в зиму.** Закалка начинается, когда температура на глубине узла кущения опускается
ниже ~10 °C, и требует 4–6 недель (K-State Extension). Накопленная сумма тепла от всходов до
покоя должна превысить ~100 °C·сут при базе 0 °C, а начало покоя определяют как трое суток со
средней ниже ~3 °C (Lithuanian sowing-date study, Atmosphere 2023). Избыточное осеннее развитие
при тёплой осени, напротив, снижает морозостойкость (Autumn growth and cold hardening). Запас
продуктивной влаги в почве на посев — классический предиктор отечественной агрометеорологии.

**Зима: перезимовка.** Критерий гибели — температура на глубине узла кущения ниже критической
в течение 2–3 суток; для закалённой пшеницы называют −15…−17 °C (Агрохимцентр «Ставропольский»),
при 2–4 побегах до −17…−22 °C. Снег высотой 15 см поднимает температуру в зоне узла до
−7…−10 °C даже при морозе ниже −30 °C, поэтому вреден мороз именно **без снега**. Белорусская
методика прогноза перезимовки прямо использует продолжительность периода со снегом 30 см и
более. Отдельный механизм — разакалка: в оттепель растения теряют морозостойкость и должны
закаливаться заново, а снег на талой почве даёт выпревание и ледяную корку.

**Весна: отрастание и возвратные заморозки.** После выхода в трубку точка роста поднимается в
стебель, и лёд в ней образуется уже при −2…−4 °C — то есть тот же мороз, безвредный в кущении,
весной губителен (глав-агроном, физиология криостресса).

**Трубка — колошение: критическое окно числа зёрен.** Вода в фазы выхода в трубку и цветения
определяет урожай сильнее, чем в любой другой период (Anthesis and booting: two critical stages
to water stress). Методика Гидрометцентра строит прогноз на двух предикторах: запас продуктивной
влаги в метровом слое и число стеблей на м² весной, на фазы выхода в трубку и колошения
(обзор агрометеорологических прогнозов). Число стеблей мы не наблюдаем — его спутниковый
аналог NDVI, и для него майское значение и колошение называют сильнейшим окном (MODIS по
Шаньдуну даёт прогноз за 40 дней до уборки; Sentinel-2 по Литве — начало мая). Переувлажнение
около цветения вредит сильнее, чем в налив (critical period for waterlogging).

**Налив: масса зерна.** Порог повреждения жарой около 31 °C в цветение и 33–37 °C в налив
(Porter & Gawith; Niwas & Khichar), причём потери определяются накопленной дозой дней выше
порога, а не пиком. Суховей по ГОСТ — температура выше 25 °C при относительной влажности ниже
30 % и ветре от 5 м/с. Ускоренное старение листа («запал») видно как быстрый спад NDVI.

**Увлажнение в целом.** Гидротермический коэффициент Селянинова ГТК = 10·ΣP / Σt при t>10 —
стандартный показатель отечественной агрометеорологии; ГТК ниже 1 означает недостаточное
увлажнение (АгроАтлас).

Что из этого мы посчитать НЕ можем
==================================

- **Ветра нет** среди каналов, поэтому суховей считается по двум признакам из трёх.
- **Суточных минимумов и максимумов нет**, только средние за сутки. Литературные пороги заданы
  по максимуму (31 °C в цветение), поэтому они переведены на шкалу суточных средних: проверка
  по данным показала, что средняя ≥ 25 °C встречается у 85 % примеров, а ≥ 30 °C — у 32 %, и
  граница 25 °C по средней соответствует дню с максимумом около 31 °C.
- **Влага только в слое 0–10 см**, а классический предиктор — метровый слой. Это суррогат.
- **Критическая температура узла кущения недостижима**: порог −15 °C на глубине 10 см
  срабатывает у 0.9 % примеров, −17 °C у 0.2 %. Поэтому вымерзание считается по градации
  −8 / −10 / −12 °C: температура на 10 см демпфирована относительно узла на 2–3 см, и мороз
  −10 °C на десяти сантиметрах соответствует примерно критическому на узле. Заодно это
  содержательный вывод: прямое вымерзание в нашей выборке — редкое событие.

Кэш: ``workspace/datasets/<task>/agro_features.csv`` (ключ — territory_id и год).

    python code/agro_features.py --task winter_wheat_raw_v2 --report
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

import config
import explore_models as E

# Ось сезона: 0 = 1 сентября года Y−1, 323 = 21 июля года Y.
SEP1, DEC1, JAN1, MAR1, APR1, MAY1, JUN1, JUN21, JUL21 = 0, 91, 122, 181, 212, 242, 273, 293, 323
SEASON_LEN = JUL21 + 1

# Пороги: каждый взят из литературы и проверен на частоту срабатывания в наших данных.
T_DORMANT = 3.0        # начало покоя: трое суток со средней ниже этого (Atmosphere 2023)
T_HARDEN = 10.0        # закалка идёт, пока грунт между 0 и этим (K-State Extension)
T_DEHARDEN = 5.0       # оттепель, снимающая закалку
GR_KILL = -10.0        # мороз на глубине 10 см как аналог критического на узле кущения
GR_KILL_SOFT = -8.0
GR_KILL_HARD = -12.0
SNOW_BARE = 0.05       # нет снежного одеяла, м
SNOW_DEEP = 0.30       # мощный снег (предиктор белорусской методики), м
SNOW_THIN = 0.02       # снег на талой почве -> ледяная корка, м
T_SPRING = 5.0         # возобновление вегетации: пятидневная средняя выше этого
T_SPRING_FROST = -2.0  # лёд в точке роста после выхода в трубку
HEAT_DAY = 25.0        # день жары по суточной средней (аналог максимума ~31 °C)
SUKHOVEI_T = 22.0      # суховей по ГОСТ без ветра: температура…
SUKHOVEI_RH = 50.0     # …и относительная влажность
DRY_MM = 1.0           # сухой день, мм
SOIL_WET = 44.0       # верх шкалы влаги 0–10 см (максимум 46.6), для справки

CHANNELS = {'t': 'mean_temp', 'ос': 'mean_prec', 'вл': 'mean_rh', 'снег': 'mean_snod',
            'почв': 'mean_soilw10', 'грунт': 'mean_tmpgr10', 'ndvi': 'ndvi'}

# Ядро: механизмы, которые в литературе названы определяющими чаще прочих.
CORE = ('осень_гдд', 'осень_влага', 'зима_грунт_мин', 'зима_снег30', 'зима_зимобой',
        'весна_заморозок', 'труб_влага', 'труб_ndvi', 'колош_гтк', 'колош_жара',
        'налив_суховей', 'налив_ndvi_спад')

FEATURE_SETS = ('agro', 'agro_d', 'agro_all', 'core', 'core_d', 'core_all')


def _nanmin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmin(a, axis=1)


def _nanmax(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmax(a, axis=1)


def _nanmean(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(a, axis=1)


def _count(mask):
    """Число дней, где условие выполнено (NaN считается невыполнением)."""
    return np.count_nonzero(mask, axis=1).astype(np.float32)


def _longest_run(mask):
    out = np.zeros(mask.shape[0], dtype=np.float32)
    cur = np.zeros(mask.shape[0], dtype=np.float32)
    for t in range(mask.shape[1]):
        cur = np.where(mask[:, t], cur + 1.0, 0.0)
        np.maximum(out, cur, out=out)
    return out


def _slope(a):
    """Наклон по дням, ед./сут, по конечным значениям строки. -> (N,)."""
    n, t = a.shape
    x = np.arange(t, dtype=np.float64)
    ok = np.isfinite(a)
    cnt = ok.sum(axis=1)
    av = np.where(ok, a, 0.0)
    sx = np.where(ok, x, 0.0).sum(axis=1)
    sy = av.sum(axis=1)
    sxx = np.where(ok, x * x, 0.0).sum(axis=1)
    sxy = (av * np.where(ok, x, 0.0)).sum(axis=1)
    den = cnt * sxx - sx * sx
    out = np.full(n, np.nan)
    good = (cnt >= 5) & (den != 0)
    out[good] = ((cnt * sxy - sx * sy)[good] / den[good])
    return out


def _gtk(t, p):
    """ГТК Селянинова: 10·ΣP / Σ(t−10) при t>10. Σ по дням окна."""
    heat = np.nansum(np.clip(t - 10.0, 0, None), axis=1)
    return 10.0 * np.nansum(p, axis=1) / np.maximum(heat, 1.0)


def _first_true(mask):
    """Индекс первого True в строке, иначе число столбцов. -> (N,)."""
    has = mask.any(axis=1)
    idx = mask.argmax(axis=1).astype(np.float32)
    return np.where(has, idx, float(mask.shape[1]))


def _runmean(a, k):
    """Скользящее среднее по k дням, выровненное по левому краю окна. -> (N, T−k+1)."""
    c = np.cumsum(np.where(np.isfinite(a), a, 0.0), axis=1)
    c = np.concatenate([np.zeros((a.shape[0], 1)), c], axis=1)
    return (c[:, k:] - c[:, :-k]) / float(k)


def season_arrays(B, names, tgt_block, cutoff=None):
    """Каналы сезона на оси «день от 1 сентября». -> {короткое имя: (N, 324)} либо {}.

    Сезон склеен из блока ``tgt_block−1`` (1 сен – 31 дек) и ``tgt_block`` (1 янв – 21 июл).
    ``cutoff`` — день года в целевом блоке, после которого данных нет: дальше NaN.
    """
    idx = {nm: i for i, nm in enumerate(names)}
    if tgt_block < 1:
        return {}
    size = B.shape[3]
    out = {}
    for short, full in CHANNELS.items():
        if full not in idx:
            continue
        autumn = B[:, idx[full], tgt_block - 1, 243:size]            # 1 сен – 31 дек
        target = B[:, idx[full], tgt_block, 0:JUL21 - JAN1 + 1]      # 1 янв – 21 июл
        a = np.concatenate([autumn, target], axis=1).astype(np.float64)
        if a.shape[1] != SEASON_LEN:
            raise ValueError(f'ось сезона {a.shape[1]} вместо {SEASON_LEN}')
        if cutoff:
            a[:, JAN1 + int(cutoff):] = np.nan
        out[short] = a
    return out


def _win(a, start, stop):
    """Отрезок оси сезона; None, если он целиком за отсечкой (все значения NaN)."""
    s = a[:, start:stop]
    if s.shape[1] == 0 or not np.isfinite(s).any():
        return None
    return s


def agro_features(w):
    """Признаки по механизмам литературы. w — выход ``season_arrays``. -> {имя: (N,)}."""
    if not w:
        return {}
    f = {}
    n = next(iter(w.values())).shape[0]

    # --- Осень: длина вегетации, закалка, уход в зиму -------------------------------------
    if 't' in w:
        aut = _win(w['t'], SEP1, JAN1)
        if aut is not None:
            # Начало покоя: первый из трёх подряд суток со средней ниже +3 °C.
            cold = _runmean(aut, 3) < T_DORMANT
            # День ухода в покой он же длина осенней вегетации от 1 сентября.
            dorm = _first_true(cold)
            f['осень_длина'] = dorm.astype(np.float32)
            day = np.arange(aut.shape[1])[None, :]
            before = day < dorm[:, None]
            f['осень_гдд'] = np.nansum(np.where(before, np.clip(aut, 0, None), 0.0), axis=1)
    if 'грунт' in w:
        aut = _win(w['грунт'], SEP1, JAN1)
        if aut is not None:
            f['осень_закалка'] = _count((aut >= 0) & (aut <= T_HARDEN))
    if 'почв' in w:
        s = _win(w['почв'], SEP1, SEP1 + 30)
        if s is not None:
            f['осень_влага'] = _nanmean(s)
    if 'ос' in w:
        s = _win(w['ос'], SEP1, SEP1 + 30)
        if s is not None:
            f['осень_ос'] = np.nansum(s, axis=1)
    if 'ndvi' in w:
        s = _win(w['ndvi'], DEC1 - 20, DEC1)
        if s is not None:
            f['осень_ndvi_уход'] = _nanmean(s)

    # --- Зима: перезимовка ---------------------------------------------------------------
    gr = _win(w['грунт'], DEC1, MAR1) if 'грунт' in w else None
    sn = _win(w['снег'], DEC1, MAR1) if 'снег' in w else None
    if gr is not None:
        f['зима_грунт_мин'] = _nanmin(gr)
        f['зима_дней_м8'] = _count(gr < GR_KILL_SOFT)
        f['зима_дней_м12'] = _count(gr < GR_KILL_HARD)
        # Разакалка: сколько раз грунт поднимался выше +5 °C после мороза и снова падал.
        warm = gr > T_DEHARDEN
        f['зима_разакалка'] = _count(warm[:, 1:] & ~warm[:, :-1])
    if 't' in w:
        at = _win(w['t'], DEC1, MAR1)
        if at is not None:
            f['зима_t_мин'] = _nanmin(at)
    if sn is not None:
        f['зима_снег30'] = _count(sn >= SNOW_DEEP)
        f['зима_снег_макс'] = _nanmax(sn)
    if gr is not None and sn is not None:
        f['зима_зимобой'] = _count((gr < GR_KILL) & (sn < SNOW_BARE))
        f['зима_корка'] = _count((sn > SNOW_THIN) & (gr > 0))

    # --- Весна: отрастание и возвратные заморозки ----------------------------------------
    if 't' in w:
        spr = _win(w['t'], JAN1 + 31, JUN1)      # с 1 февраля по 31 мая
        if spr is not None:
            start = _first_true(_runmean(spr, 5) > T_SPRING)
            f['весна_отраст'] = start.astype(np.float32)
            day = np.arange(spr.shape[1])[None, :]
            after = (day >= start[:, None]) & (day < start[:, None] + 30)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                f['весна_заморозок'] = np.nanmin(np.where(after, spr, np.nan), axis=1)
            f['весна_дней_мороз'] = _count(after & (spr < T_SPRING_FROST))
    if 'почв' in w:
        s = _win(w['почв'], APR1, MAY1)
        if s is not None:
            f['весна_влага'] = _nanmean(s)

    # --- Трубка (май): закладка колоса ---------------------------------------------------
    if 'почв' in w:
        s = _win(w['почв'], MAY1, JUN1)
        if s is not None:
            f['труб_влага'] = _nanmean(s)
            f['труб_влага_мин'] = _nanmin(s)
    if 'ndvi' in w:
        s = _win(w['ndvi'], MAY1, JUN1)
        if s is not None:
            f['труб_ndvi'] = _nanmean(s)
    if 't' in w and 'ос' in w:
        t, p = _win(w['t'], MAY1, JUN1), _win(w['ос'], MAY1, JUN1)
        if t is not None and p is not None:
            f['труб_гтк'] = _gtk(t, p)
    if 't' in w:
        t = _win(w['t'], MAY1, JUN1)
        if t is not None:
            f['труб_жара'] = _count(t >= HEAT_DAY)

    # --- Колошение и цветение (1–21 июня): число зёрен -----------------------------------
    if 'ndvi' in w:
        s = _win(w['ndvi'], JUN1, JUN21)
        if s is not None:
            f['колош_ndvi_макс'] = _nanmax(s)
    if 't' in w and 'ос' in w:
        t, p = _win(w['t'], JUN1, JUN21), _win(w['ос'], JUN1, JUN21)
        if t is not None and p is not None:
            f['колош_гтк'] = _gtk(t, p)
    if 't' in w:
        t = _win(w['t'], JUN1, JUN21)
        if t is not None:
            f['колош_жара'] = _count(t >= HEAT_DAY)
    if 'почв' in w:
        s = _win(w['почв'], JUN1, JUN21)
        if s is not None:
            f['колош_влага_мин'] = _nanmin(s)
            # Переувлажнение около цветения вредит сильнее, чем в налив, но порога на наших
            # данных не поставить: влага 0–10 см в июне не доходит до верха шкалы ни разу
            # (99-й процентиль 38.4 при максимуме шкалы 46.6). Поэтому не счётчик дней выше
            # порога, а сама верхняя граница — нелинейность пусть ищет модель.
            f['колош_влага_макс'] = _nanmax(s)

    # --- Налив (22 июня – 21 июля): масса зерна ------------------------------------------
    if 't' in w:
        t = _win(w['t'], JUN21, JUL21 + 1)
        if t is not None:
            f['налив_жара'] = _count(t >= HEAT_DAY)
            if 'вл' in w:
                rh = _win(w['вл'], JUN21, JUL21 + 1)
                f['налив_суховей'] = _count((t >= SUKHOVEI_T) & (rh <= SUKHOVEI_RH))
            if 'ос' in w:
                p = _win(w['ос'], JUN21, JUL21 + 1)
                f['налив_гтк'] = _gtk(t, p)
                f['налив_сухостой'] = _longest_run(p < DRY_MM)
    if 'ndvi' in w:
        s = _win(w['ndvi'], JUN1 + 10, JUL21 + 1)     # спад от пика к уборке
        if s is not None:
            f['налив_ndvi_спад'] = _slope(s)

    out = {}
    for k, v in f.items():
        v = np.asarray(v, dtype=np.float32)
        out[k] = v if v.shape == (n,) else np.full(n, np.nan, dtype=np.float32)
    return out


def build_table(Xb, names, meta, ids, cutoff=None, with_delta=True):
    """Таблица агрономических признаков: сезон, прошлый сезон и их разности."""
    size = int(meta['time_rows_size'])
    n_years = Xb.shape[2] // size
    if bool(meta.get('concat_newest_first', True)):
        raise ValueError('ось сезона рассчитана на порядок блоков от старого к новому')
    B = Xb.reshape(Xb.shape[0], Xb.shape[1], n_years, size)
    tgt = n_years - 1

    cur = agro_features(season_arrays(B, names, tgt, cutoff))
    cols = dict(cur)
    if with_delta:
        prev = agro_features(season_arrays(B, names, tgt - 1, None))
        for k, v in cur.items():
            if k in prev:
                cols['Δ' + k] = (v - prev[k]).astype(np.float32)
    table = pd.DataFrame(cols)
    table.insert(0, 'year', ids['year'].to_numpy(dtype=int))
    table.insert(0, 'territory_id', (ids['id_region'].astype(int).astype(str) + '_'
                                     + ids['id_district'].astype(int).astype(str)).to_numpy())
    return table


def cache_path(task, cutoff=None):
    tag = '' if cutoff in (None, 0) else f'_cut{int(cutoff)}'
    return os.path.join(config.DATASETS_DIR, task, f'agro_features{tag}.csv')


def table_for(task, cutoff=None, rebuild=False, with_delta=True):
    path = cache_path(task, cutoff)
    if os.path.exists(path) and not rebuild:
        return pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
    Xb, _y, _ph, ids, meta, names = E.load_dataset(task)
    table = build_table(Xb, list(names), meta, ids, cutoff, with_delta)
    table.to_csv(path, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    return table


def select_columns(table, kind='agro_all'):
    """Имена признаков набора: сезон, разности или всё; «core» — только опорные механизмы."""
    feats = [c for c in table.columns if c not in ('territory_id', 'year')]
    if kind.startswith('core'):
        feats = [c for c in feats if c.lstrip('Δ') in CORE]
    if kind in ('agro', 'core'):
        return [c for c in feats if not c.startswith('Δ')]
    if kind in ('agro_d', 'core_d'):
        return [c for c in feats if c.startswith('Δ')]
    return feats


def main():
    ap = argparse.ArgumentParser(description='Признаки по механизмам агрономической литературы')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--cutoff', type=int, default=0)
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()

    cutoff = a.cutoff or None
    table = table_for(a.task, cutoff, a.rebuild)
    for kind in FEATURE_SETS:
        print(f'{kind:10s} признаков {len(select_columns(table, kind))}')
    print(f'кэш: {cache_path(a.task, cutoff)}')
    if not a.report:
        return

    feats = select_columns(table, 'agro_all')
    print()
    print('--- разброс и пропуски')
    d = table[feats]
    rep = pd.DataFrame({'среднее': d.mean(), 'σ': d.std(), 'мин': d.min(), 'макс': d.max(),
                        'пропусков': d.isna().mean()})
    print(rep[~rep.index.str.startswith('Δ')].round(2).to_string())

    dead = rep[(rep['σ'] == 0) | rep['σ'].isna()]
    if len(dead):
        print()
        print('МЁРТВЫЕ (нулевой разброс):', list(dead.index))

    plan = pd.read_csv(os.path.join(config.DATASETS_DIR, a.task, 'plan.csv'),
                       encoding='utf-8-sig')
    plan['territory_id'] = plan['territory_id'].astype(str)
    key = ['territory_id', 'year']
    m = table.merge(plan[key + ['productive', 'historical_productive']], on=key, how='inner')
    m = m[np.isfinite(m['productive']) & np.isfinite(m['historical_productive'])]
    dev = (m['productive'] / m['historical_productive'].clip(lower=1.0) - 1.0) * 100
    cor = {}
    for c in feats:
        v = m[c]
        if v.notna().sum() > 100 and v.std() > 0:
            cor[c] = float(np.corrcoef(v.fillna(v.mean()), dev)[0, 1])
    top = pd.Series(cor).sort_values(key=abs, ascending=False)
    print()
    print(f'--- связь с отклонением урожайности (n={len(m)}), 25 сильнейших по |r|:')
    print(top.head(25).round(3).to_string())
    print()
    print('ВНИМАНИЕ: корреляции по всему датасету, то есть по годам и районам вместе. '
          'Это проверка знаков на здравый смысл, а не оценка пригодности для прогноза.')


if __name__ == '__main__':
    main()
