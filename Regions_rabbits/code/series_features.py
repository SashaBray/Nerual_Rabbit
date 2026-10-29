"""Резюмирование временных рядов: сезон -> набор агрономически осмысленных чисел.

Зачем. Все прежние попытки (разделы 4–10 протокола) кормили модель сырым рядом: 12 каналов ×
1095 дней = 13 140 чисел на 872 обучающих примера. Ни нейросеть на 12 млн весов, ни на 32
тысячи, ни аддитивная по каналам, ни автоэнкодер со сжатием до 8 чисел на канал на честном
сплите эталон не обошли. Здесь размерность сокращается не обучением, а **руками**: ряд
сворачивается в несколько десятков величин, каждая из которых что-то значит для агронома —
сумма осадков за налив, число дней с морозом без снега, гидротермический коэффициент
Селянинова за май.

Это не «ещё одна архитектура», а другой класс гипотез. Нейросеть ищет форму кривой, которую
мы не умеем назвать; резюме проверяет ровно те величины, которые агрономия считает важными.
Если сигнала нет и в них — отрицательный результат становится куда сильнее, потому что
«плохо искали» уже не возразить: искали именно там, где положено.

**Календарь.** Проверено на данных: индекс 0 блока = 1 января, 364 = 31 декабря (высота снега
0.18–0.22 м в январе и ноль с мая по октябрь, пик NDVI в июне, аккумуляторы сбрасываются в
январе). Блоки идут от старого к новому (``concat_newest_first: false``), целевой — последний.
Сезон озимой пшеницы урожая Y поэтому **разрезан между блоками**: посев и осеннее кущение в
сентябре–декабре блока Y−1, зимовка на стыке, вегетация и уборка в январе–июле блока Y.

**Окна** — по фазам развития озимой пшеницы:

===========  ====================================  ==========================================
Окно         Календарь                             Что решается в эту фазу
===========  ====================================  ==========================================
``осень``    1 сен – 30 ноя года Y−1               всходы, кущение, уход в зиму
``зима``     1 дек Y−1 – 28 фев Y                  перезимовка, выпад от морозов и льда
``весна``    1 мар – 30 апр Y                      отрастание, возобновление вегетации
``труб``     1 – 31 мая Y                          выход в трубку, закладка колоса
``колош``    1 – 21 июн Y                          колошение и цветение
``налив``    22 июн – 21 июл Y                     налив зерна, формирование массы
``сезон``    1 сен Y−1 – 21 июл Y                  весь сезон целиком
===========  ====================================  ==========================================

**Прошлый сезон.** Те же окна, сдвинутые на год назад (1 сен Y−2 – 21 июл Y−1), лежат целиком
в блоках 0 и 1, то есть доступны. Поэтому к каждому признаку считается разность «этот сезон
минус прошлый» (``Δ``): урожайность предсказывается в отклонениях от скользящей средней, а
отклонение естественно сравнивать с тем, каким был предыдущий сезон. Разность честна по
построению — только прошлое.

**Заблаговременность.** ``cutoff`` отрезает всё в целевом блоке позже указанного дня года:
``--cutoff 151`` — прогноз на 31 мая, когда налива ещё не было. Окна целиком за отсечкой дают
NaN, частично задетые считаются по доступной части. Прошлый сезон отсечка не трогает: он к
моменту прогноза давно известен.

    python code/series_features.py --task winter_wheat_raw_v2            # собрать и закэшировать
    python code/series_features.py --task winter_wheat_raw_v2 --report   # проверка на глаз

Кэш: ``workspace/datasets/<task>/season_features.csv`` (ключ — territory_id и год).
"""
import argparse
import os

import numpy as np
import pandas as pd

import config
import explore_models as E

# Начало месяцев в не-високосном году, 0-based: 1 янв = 0, 1 сен = 243.
MONTH_START = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
JAN, MAR, MAY, JUN, JUL, SEP, DEC = (MONTH_START[i] for i in (0, 2, 4, 5, 6, 8, 11))

# Окно — список отрезков (сдвиг блока относительно целевого, начало, конец-исключая).
WINDOWS = {
    'осень': [(-1, SEP, DEC)],                       # 1 сен – 30 ноя
    'зима': [(-1, DEC, 365), (0, JAN, MAR)],         # 1 дек – 28 фев
    'весна': [(0, MAR, MAY)],                        # 1 мар – 30 апр
    'труб': [(0, MAY, JUN)],                         # 1 – 31 мая
    'колош': [(0, JUN, JUN + 21)],                   # 1 – 21 июн
    'налив': [(0, JUN + 21, JUL + 21)],              # 22 июн – 21 июл
    'сезон': [(-1, SEP, 365), (0, JAN, JUL + 21)],   # весь сезон
}

# Какие каналы нужны и под какими короткими именами они ходят внутри.
CHANNELS = {'t': 'mean_temp', 'ос': 'mean_prec', 'вл': 'mean_rh', 'давл': 'mean_p',
            'снег': 'mean_snod', 'покр': 'mean_snowc', 'почв': 'mean_soilw10',
            'грунт': 'mean_tmpgr10', 'ndvi': 'ndvi'}

T_BASE = 5.0        # биологический ноль озимой пшеницы, °C
FROST = -5.0        # день с морозом
HARD_FROST = -10.0  # критический мороз для зимобоя
HEAT = 25.0         # день с жарой
DRY_MM = 1.0        # сухой день по осадкам, мм
DRY_RH = 45.0       # сухой день по влажности воздуха, %
SNOW_M = 0.05       # день со снежным покровом, м


def _longest_run(mask):
    """Длина самой длинной серии True в каждой строке (N, T) -> (N,)."""
    out = np.zeros(mask.shape[0], dtype=np.float32)
    cur = np.zeros(mask.shape[0], dtype=np.float32)
    for t in range(mask.shape[1]):
        cur = np.where(mask[:, t], cur + 1.0, 0.0)
        np.maximum(out, cur, out=out)
    return out


def _segment(B, ch_idx, seg, tgt_block, cutoff):
    """Вырезать отрезок окна из (N, ch, block, day). -> (N, dt) или None, если за отсечкой."""
    shift, a, b = seg
    block = tgt_block + shift
    if block < 0:
        return None
    if shift == 0 and cutoff is not None:
        b = min(b, int(cutoff))
        if b <= a:
            return None
    return B[:, ch_idx, block, a:b]


def _window_arrays(B, idx, window, tgt_block, cutoff):
    """Склеить отрезки окна по каждому нужному каналу. -> {короткое имя: (N, T)} либо {}."""
    out = {}
    for short, full in CHANNELS.items():
        if full not in idx:
            continue
        parts = [p for p in (_segment(B, idx[full], s, tgt_block, cutoff)
                             for s in WINDOWS[window]) if p is not None]
        if parts:
            out[short] = np.concatenate(parts, axis=1)
    return out


def _window_features(w, prefix):
    """Признаки одного окна по склеенным каналам. -> {имя: (N,)}."""
    f = {}
    n = None
    for a in w.values():
        n = a.shape[0]
        break
    if n is None:
        return f

    if 't' in w:
        t = w['t']
        f['t_ср'] = t.mean(axis=1)
        f['t_мин'] = t.min(axis=1)
        f['t_макс'] = t.max(axis=1)
        f['гдд'] = np.clip(t - T_BASE, 0, None).sum(axis=1)
        f['дни_мороз'] = (t < FROST).sum(axis=1).astype(np.float32)
        f['дни_жара'] = (t > HEAT).sum(axis=1).astype(np.float32)
    if 'ос' in w:
        p = w['ос']
        f['ос_сум'] = p.sum(axis=1)
        f['ос_макс'] = p.max(axis=1)
        f['дни_сух'] = (p < DRY_MM).sum(axis=1).astype(np.float32)
        f['сухостой'] = _longest_run(p < DRY_MM)
    if 't' in w and 'ос' in w:
        # Гидротермический коэффициент Селянинова: 10·ΣP / Σt при t>10. Классический
        # показатель увлажнения вегетационного периода в отечественной агрометеорологии.
        heat = np.clip(w['t'], 10.0, None).sum(axis=1) - 10.0 * w['t'].shape[1]
        f['гтк'] = 10.0 * w['ос'].sum(axis=1) / np.maximum(heat, 1.0)
    if 'вл' in w:
        f['вл_ср'] = w['вл'].mean(axis=1)
        f['дни_сух_возд'] = (w['вл'] < DRY_RH).sum(axis=1).astype(np.float32)
    if 'давл' in w:
        f['давл_ср'] = w['давл'].mean(axis=1)
    if 'снег' in w:
        f['снег_ср'] = w['снег'].mean(axis=1)
        f['снег_макс'] = w['снег'].max(axis=1)
        f['снег_дней'] = (w['снег'] > SNOW_M).sum(axis=1).astype(np.float32)
    if 'покр' in w:
        f['покр_ср'] = w['покр'].mean(axis=1)
    if 'почв' in w:
        f['почв_ср'] = w['почв'].mean(axis=1)
        f['почв_мин'] = w['почв'].min(axis=1)
    if 'грунт' in w:
        f['грунт_ср'] = w['грунт'].mean(axis=1)
        f['грунт_мин'] = w['грунт'].min(axis=1)
    if 'ndvi' in w:
        v = w['ndvi']
        f['ndvi_ср'] = v.mean(axis=1)
        f['ndvi_макс'] = v.max(axis=1)
    if 't' in w and 'снег' in w:
        # Зимобой: мороз без снежного одеяла. Главная причина гибели озимых в этой зоне.
        f['зимобой'] = ((w['t'] < HARD_FROST) & (w['снег'] < SNOW_M)).sum(axis=1).astype(np.float32)
        f['оттепели'] = (w['t'] > 0).sum(axis=1).astype(np.float32)
    return {f'{prefix}_{k}': v.astype(np.float32) for k, v in f.items()}


def _season_extras(B, idx, tgt_block, cutoff):
    """Признаки, которые считаются по сезону целиком, а не по окну."""
    f = {}
    if 'ndvi' not in idx:
        return f
    parts = [p for p in (_segment(B, idx['ndvi'], s, tgt_block, cutoff)
                         for s in WINDOWS['сезон']) if p is not None]
    if not parts:
        return f
    v = np.concatenate(parts, axis=1)
    f['сезон_ndvi_инт'] = v.sum(axis=1)
    f['сезон_ndvi_день_макс'] = v.argmax(axis=1).astype(np.float32)
    # Скорость весеннего отрастания: наклон NDVI с 1 марта по 31 мая, ед./сут.
    spring = [p for p in (_segment(B, idx['ndvi'], (0, MAR, JUN), tgt_block, cutoff),) if p is not None]
    if spring and spring[0].shape[1] >= 5:
        s = spring[0]
        x = np.arange(s.shape[1], dtype=np.float32)
        xc = x - x.mean()
        f['сезон_ndvi_рост_весна'] = (s - s.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
    return {k: v.astype(np.float32) for k, v in f.items()}


def season_features(B, names, tgt_block, cutoff=None):
    """Все признаки одного сезона. B — (N, ch, block, day). -> {имя: (N,)}."""
    idx = {nm: i for i, nm in enumerate(names)}
    out = {}
    for window in WINDOWS:
        w = _window_arrays(B, idx, window, tgt_block, cutoff)
        out.update(_window_features(w, window))
    out.update(_season_extras(B, idx, tgt_block, cutoff))
    return out


def build_table(Xb, names, meta, ids, cutoff=None, with_delta=True):
    """Таблица признаков по всем примерам: сезон, прошлый сезон и их разности."""
    size = int(meta['time_rows_size'])
    n_years = Xb.shape[2] // size
    B = Xb.reshape(Xb.shape[0], Xb.shape[1], n_years, size)
    newest_first = bool(meta.get('concat_newest_first', True))
    tgt = 0 if newest_first else n_years - 1
    if newest_first:
        raise ValueError('окна рассчитаны на порядок блоков от старого к новому '
                         '(concat_newest_first: false)')

    cur = season_features(B, names, tgt, cutoff)
    cols = dict(cur)
    if with_delta:
        prev = season_features(B, names, tgt - 1, None)   # прошлый сезон известен целиком
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
    return os.path.join(config.DATASETS_DIR, task, f'season_features{tag}.csv')


def table_for(task, cutoff=None, rebuild=False, with_delta=True):
    """Таблица признаков задачи, с кэшем на диске."""
    path = cache_path(task, cutoff)
    if os.path.exists(path) and not rebuild:
        return pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
    Xb, _y, _ph, ids, meta, names = E.load_dataset(task)
    table = build_table(Xb, list(names), meta, ids, cutoff, with_delta)
    table.to_csv(path, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    return table


FEATURE_SETS = ('base', 'delta', 'full')


def select_columns(table, kind='full'):
    """Имена признаков нужного набора: только сезон, только разности или всё."""
    feats = [c for c in table.columns if c not in ('territory_id', 'year')]
    if kind == 'base':
        return [c for c in feats if not c.startswith('Δ')]
    if kind == 'delta':
        return [c for c in feats if c.startswith('Δ')]
    return feats


def main():
    ap = argparse.ArgumentParser(description='Резюмирование рядов в агрономические признаки')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--cutoff', type=int, default=0,
                    help='день года в целевом блоке, после которого данные не берутся '
                         '(0 — весь сезон; 151 — прогноз на 31 мая)')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--report', action='store_true', help='проверка признаков на глаз')
    a = ap.parse_args()

    cutoff = a.cutoff or None
    table = table_for(a.task, cutoff, a.rebuild)
    feats = select_columns(table, 'full')
    base = select_columns(table, 'base')
    print(f'{a.task}: {len(table)} примеров, признаков {len(feats)} '
          f'({len(base)} по сезону + {len(feats) - len(base)} разностей)')
    print(f'кэш: {cache_path(a.task, cutoff)}')
    if not a.report:
        return

    nan = table[feats].isna().mean()
    if (nan > 0).any():
        print()
        print('--- с пропусками (окно за отсечкой):')
        print(nan[nan > 0].round(3).to_string())

    print()
    print('--- разброс по окнам (среднее ± σ по датасету)')
    for window in WINDOWS:
        cols = [c for c in base if c.startswith(window + '_')]
        if not cols:
            continue
        print(f'{window}: {len(cols)} признаков')
        for c in cols[:4]:
            v = table[c]
            print(f'    {c:28s} {v.mean():9.2f} ± {v.std():7.2f}')

    plan = pd.read_csv(os.path.join(config.DATASETS_DIR, a.task, 'plan.csv'),
                       encoding='utf-8-sig')
    key = ['territory_id', 'year']
    plan['territory_id'] = plan['territory_id'].astype(str)
    m = table.merge(plan[key + ['productive', 'historical_productive']], on=key, how='inner')
    m = m[np.isfinite(m['productive']) & np.isfinite(m['historical_productive'])]
    dev = (m['productive'] / m['historical_productive'].clip(lower=1.0) - 1.0) * 100
    cor = {c: float(np.corrcoef(m[c].fillna(m[c].mean()), dev)[0, 1]) for c in feats}
    top = pd.Series(cor).dropna().sort_values(key=abs, ascending=False)
    print()
    print(f'--- связь с отклонением урожайности (n={len(m)}), 20 сильнейших по |r|:')
    print(top.head(20).round(3).to_string())
    print()
    print('ВНИМАНИЕ: эти корреляции посчитаны по всему датасету, то есть по годам и районам '
          'вместе.\nОни показывают, что признаки не пустые, но НЕ являются честной оценкой: '
          'в них сидит\nи различие районов, и различие лет. Честная проверка — лес на фолдах.')


if __name__ == '__main__':
    main()
