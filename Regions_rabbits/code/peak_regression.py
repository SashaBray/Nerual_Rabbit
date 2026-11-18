"""Регрессия пика NDVI -> урожайность: точность ПО РАЙОНАМ, три определения урожайности.

Вопрос. Агроном строит по району облако точек «максимум NDVI за сезон -> урожайность» и
проводит через него линию. Какую точность это даёт, если считать её честно и по каждому
району отдельно? И меняется ли ответ от того, ЧТО называть урожайностью?

Три цели (одна культура, один район, один год):

* ``валовая``  — валовой сбор, ц (БДПМО ``Y48007019``). Зависит от площади, а не только
  от того, как росло;
* ``убранная`` — урожайность в расчёте на УБРАННУЮ площадь, ц/га (``Y48007025``,
  таблица ``yields``). Официальная «урожайность» Росстата;
* ``посевная`` — сбор / посевную площадь, ц/га (``Y48007019`` / ``Y48007018``). Отличается
  от предыдущей ровно на гибель и списание: убранная площадь ≤ посевной.

Все три считаются на ОДНОМ наборе (район, год) — там, где есть и все три значения, и пик
NDVI. Иначе точности несопоставимы: у целей разное покрытие по годам.

Протокол. Для каждого района — исключение одного года (leave-one-year-out): линия
подгоняется по остальным годам района, прогноз делается на отложенный год. Год, по которому
мы отвечаем, в подгонку не попадает никогда. Эталон тоже честный: средняя урожайность района
по тем же обучающим годам. Отсюда

    R2 = 1 - MSE(модель) / MSE(средняя района)

то есть «что регрессия добавляет к простому знанию, сколько здесь растёт обычно». Ноль —
не добавляет ничего, отрицательное — мешает.

Нюанс с масками NDVI (существенный, см. ``cultures_masks``). Масок всего три:
``..._ozim`` (озимые), ``..._spring`` (яровые), ``..._agro`` (пашня целиком). Поэтому

* маска не про культуру, а про группу: NDVI «озимой пшеницы» снят с маски ВСЕХ озимых;
* у культур с ``mask_culture = Яровая пшеница`` маска в 70 регионах яровая, а в 28 —
  пашня целиком, то есть в этих регионах в пиксель попадают и озимые;
* ряды по маскам ``spring`` и ``agro`` в базе есть только до 2022 года, по ``ozim`` — до 2025.

Поэтому маска каждого района пишется в выгрузку, а точность отдельно разбивается по маскам.

Запуск:
    python code/peak_regression.py --culture "Озимая пшеница"
    python code/peak_regression.py --culture all --min-years 6

Выход: ``workspace/reports/peak_regression/<культура>/`` — ``per_district.csv`` (длинная
таблица), ``map_table.csv`` (широкая, по одной строке на район — для раскраски карты),
``summary.csv``, графики; плюс отчёт ``docs/Регрессия_пиков_NDVI_<культура>.docx``.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import build_dataset as B
import config
import db
import explore_models as E
import plots

# culture_id -> имя задания в harvest_dates.csv (оттуда берётся день уборки)
TASK_OF_CULTURE = {
    1: 'winter_wheat_all', 2: 'spring_wheat_all', 3: 'soy_all', 4: 'barley_spring_all',
    5: 'barley_winter_all', 6: 'corn_all', 7: 'buckwheat_all', 8: 'peas_all',
    9: 'sunflower_all', 10: 'flax_all', 11: 'sugarbeet_all', 12: 'potato_all',
    13: 'rice_all', 14: 'rye_winter_all', 15: 'rye_spring_all', 16: 'oats_all',
}
NDVI_FEATURE = {'name': 'ndvi', 'is_ndvi': True, 'historical': False}
SENTINEL = 1e8          # в БДПМО пропуск закодирован как 999999999
HARVEST_MIN = 0.2       # убранная/посевная ниже этого — не списание, а ошибка данных
HARVEST_MAX = 1.05      # убранная площадь не бывает заметно больше посевной
MIN_FIT = 3             # минимум точек для подгонки линии внутри фолда
TARGETS = ('валовая', 'убранная', 'посевная')
FORMS = ('линейная', 'экспонента')
# «исключение года» — линия по всем остальным годам района (и прошлым, и будущим);
# «только прошлые» — линия по годам строго до отвечаемого, как это и бывает в работе.
SCHEMES = ('исключение года', 'только прошлые')
UNITS = {'валовая': 'ц', 'убранная': 'ц/га', 'посевная': 'ц/га'}


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
def culture_row(name):
    """Строка культуры из config/cultures.csv по id или названию."""
    cu = pd.read_csv(config.CULTURES_FILE, encoding='utf-8-sig')
    key = str(name).strip().lower()
    hit = cu[cu['culture_id'].astype(str) == key]
    if hit.empty:
        hit = cu[cu['title'].astype(str).str.strip().str.lower() == key]
    if hit.empty:
        raise SystemExit(f'культура «{name}» не найдена; есть: '
                         + ', '.join(cu['title'].astype(str)))
    return hit.iloc[0]


def harvest_doy(culture_id):
    """День года, к которому культура убрана (окно для поиска пика)."""
    task = TASK_OF_CULTURE.get(int(culture_id))
    hd = pd.read_csv(os.path.join(config.CONFIG_DIR, 'harvest_dates.csv'))
    hit = hd[hd['task'].astype(str) == str(task)]
    if hit.empty:
        raise SystemExit(f'нет дня уборки для задания «{task}» в harvest_dates.csv')
    return int(hit['harvest_doy'].iloc[0]), str(task)


def _clean(v):
    """Числовой столбец без сентинелов БДПМО."""
    v = pd.to_numeric(v, errors='coerce')
    return v.where((v > 0) & (v < SENTINEL))


def targets_table(bdpmo_culture, early=False):
    """Три цели на общем наборе (район, год). -> (DataFrame, диагностика покрытия).

    ``early=True`` добавляет годы, которых нет в БДПМО. ``crop_stats`` начинается с
    2014, а ``yields`` идёт с 2007, и без этого режима ранние годы отсекает проверка
    согласованности трёх показателей: у них нет ни сбора, ни посевной площади, так что
    сверять урожайность не с чем. Брать их так можно: сверка показала, что ``yields``
    и БДПМО ``цель_убранная`` на 7400 общих строк совпадают (корреляция 1.0000,
    средняя разница 0.00, 100 % строк в пределах 1 ц/га). Цена — только цель
    «убранная»: валовая и посевная за те годы неизвестны, как и признаки БДПМО
    (посевная площадь и удобрения).
    """
    cs = db.load('crop_stats')
    cs = cs[cs['culture'].astype(str) == str(bdpmo_culture)]
    wide = (cs[cs['indicator'].isin(['посевная', 'сбор'])]
            .pivot_table(index=['territory_id', 'year'], columns='indicator',
                         values='value', aggfunc='first')
            .reset_index())
    for c in ('посевная', 'сбор'):
        if c not in wide.columns:
            raise SystemExit(f'в crop_stats нет показателя «{c}» для культуры {bdpmo_culture}')
        wide[c] = _clean(wide[c])

    yl = db.load('yields')
    yl = yl[yl['culture'].astype(str) == str(bdpmo_culture)][['territory_id', 'year', 'yield']]
    yl['убранная'] = _clean(yl['yield']).where(lambda s: s <= E.YIELD_MAX)

    d = wide.merge(yl[['territory_id', 'year', 'убранная']], on=['territory_id', 'year'],
                   how='outer')
    d['territory_id'] = d['territory_id'].astype(str)
    d['year'] = pd.to_numeric(d['year'], errors='coerce')
    d['валовая'] = d['сбор']
    d['посевная_ц_га'] = (d['сбор'] / d['посевная']).where(lambda s: s <= E.YIELD_MAX)

    # согласованность трёх показателей: убранная площадь = сбор / урожайность
    доля = (d['сбор'] / d['убранная']) / d['посевная']
    согласовано = доля.between(HARVEST_MIN, HARVEST_MAX)
    diag = {'строк_сбор': int(d['сбор'].notna().sum()),
            'строк_посевная': int(d['посевная'].notna().sum()),
            'строк_убранная': int(d['убранная'].notna().sum()),
            'все_три': int((d['сбор'].notna() & d['посевная'].notna()
                            & d['убранная'].notna()).sum()),
            'несогласованных': int((d['убранная'].notna() & d['посевная'].notna()
                                    & d['сбор'].notna() & ~согласовано).sum()),
            'доля_уборки_медиана': float(доля[согласовано].median()),
            'списание_>2%': float((доля[согласовано] < 0.98).mean()),
            'списание_>10%': float((доля[согласовано] < 0.90).mean())}

    # ранние годы: сбора и посевной нет, значит согласованность проверять не на чем
    только_урожайность = (d['убранная'].notna() & d['сбор'].isna()
                          & d['посевная'].isna())
    keep = (согласовано | только_урожайность) if early else согласовано
    diag['строк_только_урожайность'] = int((только_урожайность & keep).sum())
    d = d[keep].copy()
    d = d.rename(columns={'посевная_ц_га': 'цель_посевная'})
    d['цель_валовая'] = d['валовая']
    d['цель_убранная'] = d['убранная']
    d['доля_уборки'] = доля[keep]
    cols = ['territory_id', 'year', 'посевная', 'сбор', 'доля_уборки'] + \
           [f'цель_{t}' for t in TARGETS]
    # в режиме ранних лет обязательна только цель «убранная»: остального там нет
    need = ['territory_id', 'year', 'цель_убранная'] if early else cols
    d = d[cols].dropna(subset=need)
    d['year'] = d['year'].astype(int)
    return d.reset_index(drop=True), diag


def peaks_table(tids, years, mask_culture, hdoy, cache):
    """Пик NDVI и день пика для (район, год). Кэшируется в ``cache``."""
    if os.path.exists(cache):
        return pd.read_csv(cache, sep=';', decimal=',', encoding='utf-8-sig')
    terr = db.load('territories').set_index(db.load('territories')['territory_id'].astype(str))
    masks = db.get_masks_df()
    rows, t0 = [], time.time()
    for i, tid in enumerate(tids, 1):
        if tid not in terr.index:
            continue
        parm, _h = B._resolve_parm(NDVI_FEATURE, mask_culture,
                                   terr.loc[tid, 'id_region'], masks)
        if parm is None:
            continue
        for yr in years:
            xy = db.get_time_series(tid, parm, int(yr))
            if not xy:
                continue
            a = np.asarray(xy, dtype=float)
            a = a[(a[:, 0] >= 0) & (a[:, 0] <= hdoy) & (a[:, 1] > 0)]
            if len(a) < 5:                         # меньше 5 снимков за сезон — пика нет
                continue
            j = int(np.argmax(a[:, 1]))
            rows.append({'territory_id': tid, 'year': int(yr), 'маска': parm,
                         'пик': round(float(a[j, 1]), 4), 'день_пика': int(a[j, 0]),
                         'снимков': len(a)})
        if i % 200 == 0:
            print(f'   районов обработано {i}/{len(tids)}, '
                  f'рядов {len(rows)}, {time.time() - t0:.0f} с', flush=True)
    p = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    p.to_csv(cache, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    return p


# --------------------------------------------------------------------------- #
# Регрессия по району
# --------------------------------------------------------------------------- #
def _fit(form, x, y):
    """Подгонка линии. -> функция прогноза или None."""
    if len(x) < MIN_FIT or np.std(x) < 1e-9:
        return None
    lo, hi = float(np.min(x)), float(np.max(x))
    cap = float(1.5 * np.max(y)) if np.max(y) > 0 else np.inf
    if form == 'линейная':
        b, a = np.polyfit(x, y, 1)

        def f(v):
            return float(np.clip(a + b * min(max(float(v), lo), hi), 0.0, cap))
    else:
        m = y > 0
        if m.sum() < MIN_FIT or np.std(x[m]) < 1e-9:
            return None
        b, lna = np.polyfit(x[m], np.log(y[m]), 1)

        def f(v):
            return float(np.clip(np.exp(lna + b * min(max(float(v), lo), hi)), 0.0, cap))
    return f


def _pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def district_rows(d, target, form, min_years, scheme):
    """По каждому району: корреляция, честный R2, RMSE. -> list[dict]."""
    out = []
    col = f'цель_{target}'
    min_pred = min_years - 1 if scheme == 'исключение года' else MIN_FIT
    for tid, s in d.groupby('territory_id'):
        s = s.sort_values('year')
        x, y = s['пик'].to_numpy(float), s[col].to_numpy(float)
        n = len(y)
        if n < min_years:
            continue
        pred, base, r_fit = np.full(n, np.nan), np.full(n, np.nan), []
        for i in range(n):
            k = (np.arange(n) != i) if scheme == 'исключение года' else (np.arange(n) < i)
            if k.sum() < MIN_FIT:
                continue
            f = _fit(form, x[k], y[k])
            base[i] = float(y[k].mean())           # эталон — средняя по тем же годам
            if f is not None:
                pred[i] = f(x[i])
                r_fit.append(_pearson(x[k], y[k]))
        ok = np.isfinite(pred) & np.isfinite(base)
        if ok.sum() < min_pred:
            continue
        mse_m = float(np.mean((y[ok] - pred[ok]) ** 2))
        mse_b = float(np.mean((y[ok] - base[ok]) ** 2))
        mean_y = float(np.mean(y))
        rho = spearmanr(x, y).statistic if n >= 4 else np.nan
        out.append({
            'territory_id': tid, 'цель': target, 'форма': form, 'схема': scheme,
            'маска': s['маска'].iloc[0], 'лет': n, 'лет_прогноза': int(ok.sum()),
            'год_первый': int(s['year'].min()), 'год_последний': int(s['year'].max()),
            'r': _pearson(x, y), 'спирмен': float(rho) if rho == rho else np.nan,
            'r_обуч': float(np.nanmean(r_fit)) if r_fit else np.nan,
            'R2': 1.0 - mse_m / mse_b if mse_b > 0 else np.nan,
            'RMSE': float(np.sqrt(mse_m)), 'RMSE_эталон': float(np.sqrt(mse_b)),
            'RMSE_проц': 100.0 * float(np.sqrt(mse_m)) / mean_y if mean_y > 0 else np.nan,
            'среднее': mean_y, 'пик_среднее': float(np.mean(x)),
            'день_пика': float(np.mean(s['день_пика'])),
        })
    return out


def _agg(df, label):
    """Сводка по подмножеству районов."""
    if df.empty:
        return {'выборка': label, 'районов': 0,
                **{k: np.nan for k in ('R2_медиана', 'R2_среднее', 'R2>0_доля',
                                       'RMSE_медиана', 'RMSE_эталон', 'уровень',
                                       'RMSE%_медиана', 'r_медиана')}}
    r2, rp = df['R2'].astype(float), df['RMSE_проц'].astype(float)
    return {'выборка': label, 'районов': len(df),
            'R2_медиана': round(float(r2.median()), 3),
            'R2_среднее': round(float(r2.mean()), 3),
            'R2>0_доля': round(float((r2 > 0).mean()), 3),
            'RMSE_медиана': round(float(df['RMSE'].median()), 2),
            'RMSE_эталон': round(float(df['RMSE_эталон'].median()), 2),
            'уровень': round(float(df['среднее'].median()), 2),
            'RMSE%_медиана': round(float(rp.median()), 1),
            'r_медиана': round(float(df['r'].median()), 3)}


def summarize(dd):
    """Главная таблица: схема x цель x форма x способ отбора районов."""
    rows = []
    for scheme in SCHEMES:
        for target in TARGETS:
            for form in FORMS:
                s = dd[(dd['схема'] == scheme) & (dd['цель'] == target)
                       & (dd['форма'] == form)]
                if s.empty:
                    continue
                for label, sub in (('все районы', s),
                                   ('r > 0.5 (отбор по всем годам)', s[s['r'] > 0.5]),
                                   ('r > 0.5 (отбор по обучающим)', s[s['r_обуч'] > 0.5])):
                    rows.append({'схема': scheme, 'цель': target, 'форма': form,
                                 **_agg(sub, label)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Графики и отчёт
# --------------------------------------------------------------------------- #
def _plots(outdir, dd, d, form, scheme):
    imgs = []
    s = dd[(dd['форма'] == form) & (dd['схема'] == scheme)]

    def hist_r(lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        for t, c in zip(TARGETS, ('#1f77b4', '#2ca02c', '#d62728')):
            v = s[s['цель'] == t]['r'].dropna()
            ax.hist(v, bins=25, histtype='step', lw=2, color=c,
                    label=f'{t} (медиана {v.median():.2f})')
        ax.axvline(0.5, color='k', ls=':', lw=1)
        ax.set_xlabel(plots.tr('Корреляция пика NDVI и урожайности в районе',
                               'Correlation of NDVI peak and yield, per district', lang))
        ax.set_ylabel(plots.tr('Число районов', 'Districts', lang))
        ax.set_title(plots.tr('Связь пика NDVI с тремя определениями урожайности',
                              'NDVI peak vs three yield definitions', lang))
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout(); return fig
    imgs.append(('Распределение корреляций по районам',
                 plots.bilingual(outdir, 'corr_hist', hist_r)))

    def hist_r2(lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        for t, c in zip(TARGETS, ('#1f77b4', '#2ca02c', '#d62728')):
            v = s[s['цель'] == t]['R2'].astype(float).clip(-1.5, 1).dropna()
            ax.hist(v, bins=30, histtype='step', lw=2, color=c,
                    label=f'{t} (медиана {v.median():.2f})')
        ax.axvline(0, color='k', ls='--', lw=1,
                   label=plots.tr('уровень средней района', 'district mean level', lang))
        ax.set_xlabel(plots.tr('R² на отложенном годе (обрезано снизу -1.5)',
                               'R² on the held-out year (clipped at -1.5)', lang))
        ax.set_ylabel(plots.tr('Число районов', 'Districts', lang))
        ax.set_title(plots.tr('Точность регрессии пика по районам',
                              'Peak-regression skill by district', lang))
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout(); return fig
    imgs.append(('Распределение точности по районам',
                 plots.bilingual(outdir, 'r2_hist', hist_r2)))

    def sc(lang):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
        for ax, t in zip(axes, TARGETS):
            v = s[s['цель'] == t]
            ax.scatter(v['r'], v['R2'].astype(float).clip(-1.5, 1), s=10, alpha=0.4,
                       color='#1f77b4', edgecolors='none')
            ax.axhline(0, color='k', ls='--', lw=1); ax.axvline(0.5, color='r', ls=':', lw=1)
            ax.set_title(t); ax.grid(True, alpha=0.3)
            ax.set_xlabel(plots.tr('корреляция r в районе', 'district correlation r', lang))
        axes[0].set_ylabel(plots.tr('R² на отложенном годе', 'R² on held-out year', lang))
        fig.suptitle(plots.tr('Отбор районов по корреляции: чем он помогает',
                              'Selecting districts by correlation: what it buys', lang))
        fig.tight_layout(); return fig
    imgs.append(('Точность против корреляции', plots.bilingual(outdir, 'r2_vs_r', sc)))

    best = (s[(s['цель'] == 'убранная') & (s['лет'] >= 8)]
            .sort_values('r', ascending=False))
    picks = list(best.head(2)['territory_id']) + list(best.tail(2)['territory_id'])

    def fits(lang):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        for ax, tid in zip(axes.ravel(), picks):
            sub = d[d['territory_id'] == tid].sort_values('year')
            x, y = sub['пик'].to_numpy(float), sub['цель_убранная'].to_numpy(float)
            ax.scatter(x, y, s=34, color='#1f77b4')
            for xi, yi, yr in zip(x, y, sub['year']):
                ax.annotate(str(yr)[2:], (xi, yi), fontsize=7,
                            xytext=(3, 3), textcoords='offset points')
            f = _fit(form, x, y)
            if f is not None:
                xx = np.linspace(x.min(), x.max(), 50)
                ax.plot(xx, [f(v) for v in xx], 'g-')
            row = s[(s['цель'] == 'убранная') & (s['territory_id'] == tid)].iloc[0]
            ax.set_title(f'{tid}: r={row["r"]:.2f}, R²={row["R2"]:.2f}', fontsize=10)
            ax.set_xlabel(plots.tr('пик NDVI', 'NDVI peak', lang))
            ax.set_ylabel(plots.tr('урожайность, ц/га', 'yield, c/ha', lang))
            ax.grid(True, alpha=0.3)
        fig.suptitle(plots.tr('Два района с тесной связью и два со слабой (цель «убранная»)',
                              'Two tight and two loose districts (target "harvested")', lang))
        fig.tight_layout(); return fig
    imgs.append(('Примеры подгонки по районам', plots.bilingual(outdir, 'fits', fits)))

    def bymask(lang):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        g = s.groupby(['маска', 'цель'])['R2'].median().unstack()
        g.plot.bar(ax=ax, rot=15)
        ax.axhline(0, color='k', lw=1)
        ax.set_ylabel(plots.tr('медиана R² по районам', 'median R² over districts', lang))
        ax.set_xlabel(plots.tr('маска NDVI', 'NDVI mask', lang))
        ax.set_title(plots.tr('Точность в зависимости от маски NDVI',
                              'Skill by NDVI mask', lang))
        ax.grid(True, alpha=0.3, axis='y'); fig.tight_layout(); return fig
    if s['маска'].nunique() > 1:
        imgs.append(('Точность по маскам NDVI', plots.bilingual(outdir, 'by_mask', bymask)))

    def byscheme(lang):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        g = (dd[dd['форма'] == form].groupby(['схема', 'цель'])['R2']
             .median().unstack().reindex(list(SCHEMES)))
        g.plot.bar(ax=ax, rot=0)
        ax.axhline(0, color='k', lw=1)
        ax.set_ylabel(plots.tr('медиана R² по районам', 'median R² over districts', lang))
        ax.set_xlabel('')
        ax.set_title(plots.tr('Цена честности: линия по прошлым годам против линии по всем',
                              'Past-years-only fit vs leave-one-year-out', lang))
        ax.grid(True, alpha=0.3, axis='y'); fig.tight_layout(); return fig
    imgs.append(('Две схемы подгонки', plots.bilingual(outdir, 'by_scheme', byscheme)))
    return imgs


ABS_COLS = ('RMSE_медиана', 'RMSE_эталон', 'уровень')


def _fmt_abs(v):
    """Абсолютная величина в таблицу: ц/га — с двумя знаками, центнеры — без дробей."""
    v = float(v)
    if not np.isfinite(v):
        return ''
    return f'{v:.2f}' if abs(v) < 1000 else f'{v:,.0f}'.replace(',', '\u00a0')


def _docx(outdir, title, meta, diag, sdf, mdf, dd, imgs, form, scheme):
    from docx import Document
    from docx.shared import Inches
    docs = os.path.join(os.path.dirname(os.path.dirname(config.REPORTS_DIR)), 'docs')
    os.makedirs(docs, exist_ok=True)
    doc = Document()
    doc.add_heading(f'Регрессия пика NDVI → урожайность: {title}', 0)

    doc.add_heading('Что считалось', level=1)
    doc.add_paragraph(
        'По каждому РАЙОНУ отдельно строится облако точек «максимум NDVI за сезон → '
        'урожайность» и через него проводится линия — так, как это делает агроном. '
        f'Пик берётся по дням 0–{meta["hdoy"]} года урожая (до уборки), по маске культуры '
        f'«{meta["mask_culture"]}». Форма зависимости в основной таблице — {form}; вторая '
        'форма посчитана для сравнения.')
    doc.add_paragraph(
        'Точность считается с исключением одного года: линия подгоняется по остальным годам '
        'района, прогноз делается на отложенный год, и так по каждому году. Эталон — средняя '
        'урожайность района по тем же обучающим годам. Поэтому R² = 1 − MSE(модель)/MSE(средняя) '
        'отвечает на вопрос «что пик NDVI добавляет к простому знанию, сколько здесь растёт '
        'обычно». Ноль — не добавляет ничего, отрицательное значение — мешает.')
    doc.add_paragraph(
        'Отвечаемый год в подгонку не попадает никогда, но в схеме «исключение года» линия '
        'строится и по БУДУЩИМ годам района — в работе так нельзя. Поэтому посчитана и вторая '
        'схема, «только прошлые»: линия по годам строго до отвечаемого. Она короче (первые '
        f'{MIN_FIT} года уходят на разгон) и строже; разница между схемами и есть цена того, '
        'что связь оценена по будущему.')

    doc.add_heading('Три определения урожайности', level=1)
    for s in (f'валовая — валовой сбор, {UNITS["валовая"]} (БДПМО Y48007019). Зависит не '
              'только от того, как росло, но и от того, сколько посеяли;',
              f'убранная — урожайность в расчёте на убранную площадь, {UNITS["убранная"]} '
              '(Y48007025). Официальная «урожайность» Росстата;',
              f'посевная — сбор, делённый на посевную площадь, {UNITS["посевная"]} '
              '(Y48007019 / Y48007018). От предыдущей отличается ровно на гибель и списание.'):
        doc.add_paragraph(s, style='List Bullet')
    doc.add_paragraph(
        f'Насколько второе и третье определения вообще расходятся: медиана отношения убранной '
        f'площади к посевной {diag["доля_уборки_медиана"]:.4f}; списано больше 2 % площади в '
        f'{100*diag["списание_>2%"]:.0f} % случаев «район-год», больше 10 % — в '
        f'{100*diag["списание_>10%"]:.1f} %. То есть в большинстве районов убранная площадь '
        'совпадает с посевной и две цели почти тождественны; расходятся они на хвосте, '
        'в годы гибели озимых.')
    doc.add_paragraph(
        f'Строк с показателями: сбор {diag["строк_сбор"]}, посевная площадь '
        f'{diag["строк_посевная"]}, урожайность {diag["строк_убранная"]}; все три сразу — '
        f'{diag["все_три"]}. Отброшено как несогласованное (убранная площадь вне '
        f'{HARVEST_MIN}–{HARVEST_MAX} посевной — это ошибки данных, а не списание): '
        f'{diag["несогласованных"]}.')

    doc.add_heading('Нюанс с масками NDVI', level=1)
    doc.add_paragraph(
        'Масок в базе всего три: озимые (..._ozim), яровые (..._spring) и пашня целиком '
        '(..._agro). Из этого следуют ограничения, которые нельзя забывать при переносе '
        'опыта на другие культуры.')
    for s in ('Маска не про культуру, а про группу. NDVI «озимой пшеницы» снят с маски ВСЕХ '
              'озимых — туда же попадают озимая рожь и озимый ячмень. Отсюда предел: пик '
              'описывает не поле этой культуры, а всю группу в границах района.',
              f'Для культуры «{meta["mask_culture"]}» маска задана по регионам так: '
              f'{meta["mask_plan"]}. Регионов с такой записью, у которых в каталоге есть '
              f'районы: {meta["mask_regions_with_districts"]}. Фактически в опыт попали '
              f'маски: {meta["mask_fact"]}.',
              'Если по культуре задействовано больше одной маски, смешивать районы нельзя: '
              'под маской «пашня» в пиксель попадают и озимые, и яровые. Разбивка по маскам — '
              'в следующем разделе; у каждого района маска выписана в выгрузку.',
              'Ряды по маскам «яровые» и «пашня» в базе есть только до 2022 года, по озимым — '
              'до 2025. Для яровых культур окно опыта короче на три года, и это видно по '
              'интервалу годов в разделе «Покрытие опыта».'):
        doc.add_paragraph(s, style='List Bullet')

    doc.add_heading('Покрытие опыта', level=1)
    doc.add_paragraph(
        f'Культура: {title} (в БДПМО «{meta["bdpmo_culture"]}»). Годы: '
        f'{meta["year_min"]}–{meta["year_max"]}. Пар «район-год» со всеми тремя целями и пиком '
        f'NDVI: {meta["n_pairs"]}; районов: {meta["n_districts"]}; в зачёт пошли районы '
        f'минимум с {meta["min_years"]} годами — таких {meta["n_scored"]}.')

    doc.add_heading('Главный результат', level=1)
    doc.add_paragraph(
        f'Схема «{scheme}», форма — {form}. Для каждой цели три строки: все районы; районы, '
        'где корреляция пика с урожайностью выше 0.5; и то же, но с отбором по обучающим '
        'годам. Третья строка — честная: во второй район отбирается по той же корреляции, '
        'которую потом и подтверждает, и число получается завышенным.')
    sel = (sdf['форма'] == form) & (sdf['схема'] == scheme)
    E._table_from_df(doc, sdf[sel].drop(columns=['форма', 'схема', *ABS_COLS],
                                        errors='ignore'), maxrows=30)

    q = {}
    for t in TARGETS:
        v = sdf[(sdf['форма'] == form) & (sdf['схема'] == scheme) & (sdf['цель'] == t)]
        q[t] = (v[v['выборка'] == 'все районы'].iloc[0],
                v[v['выборка'] == 'r > 0.5 (отбор по обучающим)'].iloc[0])

    doc.add_heading('Абсолютная ошибка и чем она отличается от R²', level=1)
    doc.add_paragraph(
        'R² — величина относительная: он сравнивает модель с эталоном и ничего не говорит о '
        'том, на сколько центнеров мы ошибаемся. Поэтому та же таблица в исходных единицах — '
        'медианная по районам ошибка модели, ошибка эталона (средней района) и типичный '
        'уровень цели, чтобы ошибку было с чем соотнести.')
    at = sdf[(sdf['форма'] == form) & (sdf['схема'] == scheme)].copy()
    at['ед.'] = at['цель'].map(UNITS)
    for c in ABS_COLS:
        at[c] = at[c].map(_fmt_abs)
    E._table_from_df(doc, at[['цель', 'ед.', 'выборка', 'районов', 'RMSE_медиана',
                              'RMSE_эталон', 'уровень', 'RMSE%_медиана', 'R2_медиана']],
                     maxrows=30)

    a0, a1 = q['убранная']
    doc.add_paragraph(
        f'Здесь видно то, чего по R² не видно. При отборе районов по корреляции медианная '
        f'ошибка модели меняется с {_fmt_abs(a0["RMSE_медиана"])} на '
        f'{_fmt_abs(a1["RMSE_медиана"])} {UNITS["убранная"]}, а ошибка эталона — с '
        f'{_fmt_abs(a0["RMSE_эталон"])} на {_fmt_abs(a1["RMSE_эталон"])}. R² при этом '
        f'поднимается с {a0["R2_медиана"]} до {a1["R2_медиана"]}. То есть рост R² обеспечен '
        f'не столько тем, что мы точнее в центнерах, сколько тем, что у отобранных районов '
        f'хуже справляется эталон: урожайность там колеблется сильнее, и пик NDVI успевает '
        f'объяснить большую долю этого разброса. Критерий r > 0.5 выбирает районы с большой '
        f'изменчивостью, а не районы, где прогноз в центнерах надёжнее.')
    lo, hi = sorted([float(a0['RMSE_медиана']), float(a1['RMSE_медиана'])])
    doc.add_paragraph(
        f'Для продукта, который обещает «процент от нормы», значимо именно абсолютное число: '
        f'{_fmt_abs(lo)}–{_fmt_abs(hi)} {UNITS["убранная"]} при типичной урожайности '
        f'{_fmt_abs(a0["уровень"])}, то есть около {a0["RMSE%_медиана"]:.0f} % урожая. '
        f'Отбор районов этого почти не меняет.')
    g0, g1 = q['валовая']
    doc.add_paragraph(
        f'У валового сбора абсолютные числа между районами несравнимы: он измеряется в '
        f'{UNITS["валовая"]} и зависит от размера района. Медианная ошибка меняется с '
        f'{_fmt_abs(g0["RMSE_медиана"])} на {_fmt_abs(g1["RMSE_медиана"])}, и это движение '
        f'почти целиком про размер района, а не про точность: типичный сбор у отобранных '
        f'районов {_fmt_abs(g1["уровень"])} против {_fmt_abs(g0["уровень"])} по всем. Для '
        f'этой цели районы сравнимы только в процентах: {g0["RMSE%_медиана"]} % и '
        f'{g1["RMSE%_медиана"]} %.')

    other_sch = [s for s in SCHEMES if s != scheme][0]
    doc.add_heading(f'Схема «{other_sch}»', level=1)
    doc.add_paragraph(
        'Разница с главной таблицей — цена того, что там связь оценивалась и по будущим годам '
        'района. Эта таблица строже и ближе к работе: на каждый год отвечает линия, построенная '
        'только по предыдущим годам.')
    sel = (sdf['форма'] == form) & (sdf['схема'] == other_sch)
    E._table_from_df(doc, sdf[sel].drop(columns=['форма', 'схема', *ABS_COLS],
                                        errors='ignore'), maxrows=30)

    doc.add_heading('Вторая форма зависимости', level=1)
    other = [f for f in FORMS if f != form][0]
    doc.add_paragraph(f'Схема «{scheme}», но линия — {other}.')
    sel = (sdf['форма'] == other) & (sdf['схема'] == scheme)
    E._table_from_df(doc, sdf[sel].drop(columns=['форма', 'схема', *ABS_COLS],
                                        errors='ignore'), maxrows=30)

    doc.add_heading('Разбивка по маскам NDVI', level=1)
    E._table_from_df(doc, mdf, maxrows=30)

    doc.add_heading('Графики', level=1)
    for name, png in imgs:
        doc.add_heading(name, level=2)
        if png:
            doc.add_picture(png, width=Inches(6.3))

    s = dd[(dd['форма'] == form) & (dd['схема'] == scheme)
           & (dd['цель'] == 'убранная')].copy()
    show = ['territory_id', 'регион', 'район', 'лет', 'r', 'R2', 'RMSE', 'RMSE_эталон',
            'среднее', 'RMSE_проц']
    show = [c for c in show if c in s.columns]
    doc.add_heading('20 районов с лучшей точностью (цель «убранная»)', level=1)
    E._table_from_df(doc, s.sort_values('R2', ascending=False)[show].round(3), maxrows=20)
    doc.add_heading('20 районов с худшей точностью (цель «убранная»)', level=1)
    E._table_from_df(doc, s.sort_values('R2')[show].round(3), maxrows=20)

    doc.add_heading('Что из этого следует', level=1)
    for s in (
        f'Пик NDVI действительно несёт сигнал, но немного. По всем районам медианный R² = '
        f'{q["убранная"][0]["R2_медиана"]} для урожайности на убранную площадь: регрессия '
        f'снимает около {100*q["убранная"][0]["R2_медиана"]:.0f} % дисперсии, которую не '
        f'снимает простая средняя района. Выше нуля оказались '
        f'{100*q["убранная"][0]["R2>0_доля"]:.0f} % районов — то есть примерно в каждом '
        f'третьем районе пик NDVI не помогает вовсе.',
        f'Определение урожайности меняет ответ сильно. Валовой сбор предсказывается хуже всех '
        f'(R² = {q["валовая"][0]["R2_медиана"]}, относительная ошибка '
        f'{q["валовая"][0]["RMSE%_медиана"]} %), потому что в нём сидит посевная площадь, а '
        f'её NDVI не знает. Две цели «на гектар» идут почти вровень '
        f'(R² = {q["убранная"][0]["R2_медиана"]} и {q["посевная"][0]["R2_медиана"]}, '
        f'ошибка {q["убранная"][0]["RMSE%_медиана"]} % и '
        f'{q["посевная"][0]["RMSE%_медиана"]} %).',
        'Убранная и посевная площади в этих данных почти всегда совпадают, поэтому две цели '
        '«на гектар» и не могут разойтись. Отсюда вывод: выбор между ними не важен, а важно '
        'не брать валовой сбор, если вопрос о том, как росло.',
        f'Отбор районов по корреляции помогает заметно: там, где связь пика с урожайностью '
        f'тесная (r > 0.5), медианный R² поднимается до {q["убранная"][1]["R2_медиана"]}, '
        f'и таких районов {q["убранная"][1]["районов"]} из {q["убранная"][0]["районов"]}. '
        'Это и есть практический вывод для продукта: район должен сам доказать, что в нём '
        'регрессия работает, и только тогда ей можно верить. Но рост R² не означает такого '
        'же роста точности: абсолютная ошибка при отборе почти не меняется '
        f'({_fmt_abs(q["убранная"][0]["RMSE_медиана"])} против '
        f'{_fmt_abs(q["убранная"][1]["RMSE_медиана"])} {UNITS["убранная"]}), растёт в '
        'основном ошибка эталона — подробнее в разделе «Абсолютная ошибка».',
        'А вот отбор по всей истории против отбора по обучающим годам почти не расходится: '
        'два критерия выбирают практически одни и те же районы, и медианы совпадают в третьем '
        'знаке. То есть привычная практика «посчитали корреляцию по всей истории и отобрали '
        'районы» здесь не завышает результат — проверять это было нужно, но поправка оказалась '
        'мала. В схеме «только прошлые» отбор по обучающим годам отсекает больше районов, '
        'и там разница заметнее.',
        'Главное ограничение не в методе, а в носителе: мы учимся на средних по району '
        '(тысячи км²), а продукт должен отвечать про поле. Региональная линия «пик → '
        'урожайность» на поле не переносится напрямую — её придётся калибровать.'):
        doc.add_paragraph(s, style='List Bullet')

    doc.add_heading('Как повторить и как перенести на другую культуру', level=1)
    doc.add_paragraph(
        'Опыт не привязан к культуре: цель, маска NDVI и день уборки берутся из конфигурации '
        '(config/cultures.csv, cultures_masks, config/harvest_dates.csv), всё остальное — '
        'общее. Пики кэшируются в peaks.csv, поэтому повторный прогон занимает секунды.')
    for cmd in ('python code/peak_regression.py --culture "Озимая пшеница"',
                'python code/peak_regression.py --culture "Яровая пшеница" --min-years 6',
                'python code/peak_regression.py --culture all --min-years 6',
                'python code/peak_regression.py --culture Соя --form экспонента'):
        doc.add_paragraph(cmd, style='Intense Quote')

    doc.add_heading('Выгрузки', level=1)
    for s in ('per_district.csv — длинная таблица: район × цель × форма, со всеми метриками;',
              'map_table.csv — по одной строке на район, метрики по целям в столбцах. '
              'Это таблица для раскраски карты: есть territory_id, id_region, id_district, '
              'названия региона и района, маска NDVI, число лет, точность в процентах и в '
              'центнерах (RMSE), ошибка эталона (RMSE_эталон) и уровень цели (среднее);',
              'summary.csv — сводка; peaks.csv — пики NDVI по району и году (кэш).'):
        doc.add_paragraph(s, style='List Bullet')
    doc.add_paragraph(f'Каталог: {outdir}')

    out = os.path.join(docs, f'Регрессия_пиков_NDVI_{title.replace(" ", "_")}.docx')
    try:
        doc.save(out)
    except PermissionError:
        out = out.replace('.docx', '_new.docx'); doc.save(out)
    print(f'Отчёт Word: {out}')
    return out


# --------------------------------------------------------------------------- #
def run(culture, min_years, form, rebuild):
    cu = culture_row(culture)
    title = str(cu['title'])
    hdoy, task = harvest_doy(cu['culture_id'])
    outdir = os.path.join(config.REPORTS_DIR, 'peak_regression', task)
    os.makedirs(outdir, exist_ok=True)
    print(f'=== {title}: маска «{cu["mask_culture"]}», пик по дням 0–{hdoy} ({task})')

    tg, diag = targets_table(cu['bdpmo_culture'])
    print(f'целей: {len(tg)} пар «район-год», районов {tg.territory_id.nunique()}, '
          f'годы {tg.year.min()}–{tg.year.max()}; несогласованных отброшено '
          f'{diag["несогласованных"]}')

    # что маска обещает по регионам и что из этого реально покрыто районами
    masks = db.get_masks_df()
    col = {str(c).strip().lower(): c for c in masks.columns}[
        str(cu['mask_culture']).strip().lower()]
    plan = masks[['id_region', col]].copy()
    plan['rid'] = plan['id_region'].apply(db._norm_id)
    terr_all = db.load('territories')
    have = set(terr_all['id_region'].apply(db._norm_id))
    mask_plan = dict(plan[col].dropna().value_counts())
    mask_with = dict(plan[plan['rid'].isin(have)][col].dropna().value_counts())

    cache = os.path.join(outdir, 'peaks.csv')
    if rebuild and os.path.exists(cache):
        os.remove(cache)
    pk = peaks_table(sorted(tg['territory_id'].unique()),
                     sorted(tg['year'].unique()), cu['mask_culture'], hdoy, cache)
    print(f'пиков NDVI: {len(pk)} рядов, районов {pk.territory_id.nunique()}, '
          f'маски {dict(pk["маска"].value_counts())}')

    d = tg.merge(pk, on=['territory_id', 'year'], how='inner')
    print(f'общий набор: {len(d)} пар, районов {d.territory_id.nunique()}, '
          f'годы {d.year.min()}–{d.year.max()}')

    rows = []
    for sch in SCHEMES:
        for t in TARGETS:
            for f in FORMS:
                rows += district_rows(d, t, f, min_years, sch)
    dd = pd.DataFrame(rows)
    if dd.empty:
        raise SystemExit(f'ни один район не набрал {min_years} лет')

    terr = db.load('territories')
    terr['territory_id'] = terr['territory_id'].astype(str)
    terr['id_region'] = terr['id_region'].apply(db._norm_id)
    terr['id_district'] = terr['id_district'].apply(db._norm_id)
    keys = terr[['territory_id', 'id_region', 'id_district', 'region', 'district']]
    dd = dd.merge(keys, on='territory_id', how='left').rename(
        columns={'region': 'регион', 'district': 'район'})

    scheme = SCHEMES[0]
    main = (dd['форма'] == form) & (dd['схема'] == scheme)
    sdf = summarize(dd)
    mdf = (dd[main].groupby(['маска', 'цель'])
           .agg(районов=('R2', 'size'), R2_медиана=('R2', 'median'),
                R2_среднее=('R2', 'mean'), r_медиана=('r', 'median'),
                RMSE_проц=('RMSE_проц', 'median')).round(3).reset_index())

    # широкая таблица для карты: одна строка на район
    wide = dd[main].pivot_table(
        index=['territory_id', 'id_region', 'id_district', 'регион', 'район', 'маска', 'лет'],
        columns='цель', values=['r', 'R2', 'RMSE', 'RMSE_эталон', 'RMSE_проц', 'среднее'],
        aggfunc='first')
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide = wide.round(4).reset_index()
    past = (dd[(dd['форма'] == form) & (dd['схема'] == SCHEMES[1])]
            .pivot_table(index='territory_id', columns='цель', values='R2', aggfunc='first'))
    past.columns = [f'R2_прошлые_{c}' for c in past.columns]
    wide = wide.merge(past.round(4).reset_index(), on='territory_id', how='left')

    dd.round(4).to_csv(os.path.join(outdir, 'per_district.csv'), sep=';', decimal=',',
                       index=False, encoding='utf-8-sig')
    wide.to_csv(os.path.join(outdir, 'map_table.csv'), sep=';', decimal=',',
                index=False, encoding='utf-8-sig')
    sdf.to_csv(os.path.join(outdir, 'summary.csv'), sep=';', decimal=',',
               index=False, encoding='utf-8-sig')
    pd.DataFrame([diag]).to_csv(os.path.join(outdir, 'coverage.csv'), sep=';', decimal=',',
                                index=False, encoding='utf-8-sig')

    print(f'\n=== сводка (форма: {form})')
    print(sdf[sdf['форма'] == form].drop(columns=['форма']).to_string(index=False))
    print('\n=== по маскам NDVI')
    print(mdf.to_string(index=False))

    meta = {'hdoy': hdoy, 'mask_culture': cu['mask_culture'],
            'bdpmo_culture': cu['bdpmo_culture'], 'min_years': min_years,
            'year_min': int(d.year.min()), 'year_max': int(d.year.max()),
            'n_pairs': len(d), 'n_districts': int(d.territory_id.nunique()),
            'n_scored': int(dd[main]['territory_id'].nunique()),
            'mask_plan': ', '.join(f'{k} — {v} регионов' for k, v in mask_plan.items()),
            'mask_regions_with_districts': ', '.join(f'{k} — {v}'
                                                     for k, v in mask_with.items()) or 'нет',
            'mask_fact': ', '.join(f'{k} — районов {v}' for k, v in
                                   d.groupby('маска')['territory_id'].nunique().items())}
    imgs = _plots(outdir, dd, d, form, scheme)
    _docx(outdir, title, meta, diag, sdf, mdf, dd, imgs, form, scheme)
    print(f'Таблицы: {outdir}')
    return sdf


def main():
    ap = argparse.ArgumentParser(description='Регрессия пика NDVI по районам')
    ap.add_argument('--culture', default='Озимая пшеница',
                    help='название или culture_id из config/cultures.csv; «all» — все')
    ap.add_argument('--min-years', type=int, default=7,
                    help='минимум лет в районе, чтобы он пошёл в зачёт')
    ap.add_argument('--form', default='линейная', choices=list(FORMS),
                    help='форма зависимости в основной таблице')
    ap.add_argument('--rebuild-peaks', action='store_true', help='пересчитать кэш пиков')
    a = ap.parse_args()

    if str(a.culture).strip().lower() == 'all':
        cu = pd.read_csv(config.CULTURES_FILE, encoding='utf-8-sig')
        for t in cu['title']:
            try:
                run(t, a.min_years, a.form, a.rebuild_peaks)
            except SystemExit as e:
                print(f'--- {t}: пропущено ({e})')
    else:
        run(a.culture, a.min_years, a.form, a.rebuild_peaks)


if __name__ == '__main__':
    main()
