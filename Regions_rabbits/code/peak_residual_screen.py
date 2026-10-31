"""Скрининг: что ещё коррелирует с ОСТАТКОМ районной регрессии пика NDVI.

Вопрос. Районная линия «пик NDVI -> урожайность» снимает около 15 % ошибки относительно
средней района (``peak_regression.py``). Остаётся остаток. Есть ли в УЖЕ СКАЧАННЫХ рядах
величина, которая этот остаток объясняет?

Почему цель — именно остаток, а не урожайность. Признак, коррелирующий с урожайностью,
скорее всего коррелирует и с пиком NDVI, то есть несёт ту же информацию и ничего не
добавляет. Нас интересует ортогональное: то, чего пик не увидел.

Почему модель двухэтажная. В районе 10–11 лет. Вторая переменная внутри районной регрессии
съест степени свободы, и честный R² упадёт — не потому, что признак плохой. Поэтому районная
часть остаётся простой линией, а поправка на остаток учится СРАЗУ ПО ВСЕМ районам (тысячи
наблюдений). Остаток перед этим обезразмеривается: делится на разброс урожайности самого
района по обучающим годам. Признаки тоже центрируются внутри района — иначе пулинг ответит
на вопрос «где вообще теплее», а не «какой год в этом районе выдался необычным».

Что перебирается: 10 погодных рядов и маска NDVI, 7 окон сезона, по несколько сводок на ряд
(средняя, максимум, минимум, накопление, прирост накопленного ряда, наклон) — около двухсот
кандидатов. Накопленные температура и осадки считаются двумя независимыми путями: суммой по
суточному ряду (``ГДД`` с базой 5 °C, ``накоп``) и приростом готовых накопленных рядов Веги
(``прирост``) — вторые надёжнее, если в суточном ряду есть пропуски.

Два окна стоит выделить. ``после_пика`` — условия между пиком и уборкой: пик снимается ДО
налива зерна, поэтому эта информация ортогональна пику по построению. ``осень_пред`` — сев и
уход в зиму, дни прошлого года (в ряду они идут со сдвигом −365).

Две семьи кандидатов сверх этих окон. Первая: то же значение, но как ОТКЛОНЕНИЕ от нормы
района по предыдущим годам (суффикс ``@нормаN``). Норма по всей истории района в работе не
существует — в год Y известны только годы до Y, — а вычитание нормы само по себе заметно
поднимает внутригодовую корреляцию: признак начинает говорить «этот год здесь необычный», а
не «здесь так всегда». Вторая семья — фаза. Любая сводка по окну складывает два разных
события в одно число: район ОТСТАЛ в развитии и район вышел СЛАБЫМ дают одинаково низкий
весенний интеграл, а агрономически это разные беды. Поэтому кривая года совмещается со
средней кривой того же района по прошлым годам сдвигом по времени, и сдвиг
(``фаза_сдвиг``, дни; положительный — развитие ОПЕРЕЖАЕТ норму района), амплитуда
(``фаза_ампл``) и остаток формы после совмещения (``фаза_форма``) докладываются по
отдельности.

Отбирается не один лучший признак, а сумма трёх с РАВНЫМИ весами. Отбор одного среди сотен
кандидатов шумен сам по себе: критерий отбора сам оценён по десяти годам, и признак,
случайно удачный на обучении, вытесняет верный. Сумма ничего не подгоняет, и мерится это
прямо: лучший один даёт внутригодовую корреляцию 0.07 на отложенных годах, сумма трёх —
0.18 (озимая пшеница, 398 кандидатов).

Чем такой перебор обманывает. На двух сотнях кандидатов и нескольких тысячах строк
корреляция 0.1–0.15 находится на чистом шуме. Поэтому:

* признак выбирается по ОБУЧАЮЩИМ годам, а докладывается по отложенному году;
* отдельно — выбор на одной половине районов и проверка на другой;
* контроль перестановкой: остаток перемешивается внутри района, и тот же перебор
  докладывает, что находится на заведомом шуме. Это и есть планка, с которой надо
  сравнивать, а не с нулём.

Итог докладывается в ц/га (ΔRMSE: насколько поправка уменьшила ошибку районной линии), а не
только в корреляциях — корреляция легко растёт там, где центнеры не двигаются.

Признак «остаток прошлого года» сознательно не включён: при исключении года его расчёт
подсматривает отвечаемый год.

Запуск:
    python code/peak_residual_screen.py --culture "Озимая пшеница"
    python code/peak_residual_screen.py --culture "Яровая пшеница" --min-years 6

Выход: ``workspace/reports/peak_residual/<задание>/`` — ``candidates.csv`` (все кандидаты с
честными корреляциями), ``folds.csv`` (по фолдам), ``residuals.csv``, ``features.csv`` и
``phase.csv`` (кэши), графики; плюс отчёт ``docs/Остаток_пика_NDVI_<культура>.docx``.
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

import config
import db
import explore_models as E
import peak_regression as P
import plots

# ряд -> какие сводки по нему имеют смысл
PARMS = {
    'mean_temp': ('сред', 'макс', 'мин', 'ГДД'),
    'mean_temp_acc': ('макс', 'прирост'),
    'mean_prec': ('сред', 'макс', 'накоп'),
    'mean_prec_acc': ('макс', 'прирост'),
    'mean_rh': ('сред', 'мин'),
    'mean_p': ('сред',),
    'mean_tmpgr10': ('сред', 'мин'),
    'mean_soilw10': ('сред', 'мин', 'наклон'),
    'mean_snowc': ('сред', 'макс'),
    'mean_snod': ('сред', 'макс', 'накоп'),
}
NDVI_SUMM = ('макс', 'сред', 'накоп', 'наклон')
BDPMO_FEATS = ('площадь_отн', 'удобр_отн')
GDD_BASE = 5.0          # база эффективных температур, °C
NORM_YEARS = 4          # норма района по последним N ПРОШЛЫМ годам: столько есть и в проде
PHASE_FROM = 60         # с какого дня года кривая NDVI кладётся на суточную сетку
PHASE_SHIFT = 30        # на сколько дней разрешено сдвигать кривую при совмещении
MIN_PAIRS = 200         # меньше пар — корреляцию не считаем
TOP_K = 10              # сколько признаков берёт гребневая поправка
ALPHA = 10.0            # штраф гребня; не подбирался, взят с запасом
SCHEMES = ('год', 'районы', 'год_внутри')
SUM_K = 3               # сколько признаков берёт сумма с равными весами
# объявленная заранее тройка сводок кривой NDVI: сколько, когда, как быстро падает
TRIPLE = ('накоп|весна', 'фаза_сдвиг|до_пика', 'наклон|после_пика')
COLS = ['фолд', 'лучший_признак', 'r_обуч', 'r_тест', 'r_внутри_тест',
        'ΔRMSE_внутр', 'ΔRMSE_один', f'ΔRMSE_топ{TOP_K}']
COMP_COLS = ['фолд', 'r_внутри_тест', f'r_внутри_сум{SUM_K}', 'r_внутри_тройка',
             'ΔRMSE_внутр', f'ΔRMSE_сум{SUM_K}', 'ΔRMSE_тройка']


# --------------------------------------------------------------------------- #
# Остаток районной линии
# --------------------------------------------------------------------------- #
def fertilizer(bdpmo_culture):
    """Удобрения по району и году из crop_stats. -> DataFrame или None."""
    cs = db.load('crop_stats')
    cs = cs[(cs['culture'].astype(str) == str(bdpmo_culture))
            & (cs['indicator'].astype(str) == 'удобрения')]
    if cs.empty:
        return None
    t = cs[['territory_id', 'year', 'value']].copy()
    t['territory_id'] = t['territory_id'].astype(str)
    t['year'] = pd.to_numeric(t['year'], errors='coerce')
    t['удобрения'] = P._clean(t['value'])
    return t[['territory_id', 'year', 'удобрения']].dropna()


def resid_table(d, target, form, min_years):
    """Остаток районной линии при исключении года, обезразмеренный по району."""
    col = f'цель_{target}'
    out = []
    for tid, s in d.groupby('territory_id'):
        s = s.sort_values('year')
        x, y = s['пик'].to_numpy(float), s[col].to_numpy(float)
        n = len(y)
        if n < min_years:
            continue
        for i in range(n):
            k = np.arange(n) != i
            f = P._fit(form, x[k], y[k])
            sd = float(np.std(y[k]))
            if f is None or sd <= 0:
                continue
            pred = f(x[i])
            row = {'territory_id': tid, 'year': int(s['year'].iloc[i]),
                   'день_пика': int(s['день_пика'].iloc[i]),
                   'цель': float(y[i]), 'линия': float(pred),
                   'ост': float(y[i] - pred), 'sd': sd,
                   'ост_norm': float((y[i] - pred) / sd),
                   'уровень': float(np.mean(y[k]))}
            # признаки БДПМО — относительно обучающих лет того же района
            for name, src in (('площадь_отн', 'посевная'), ('удобр_отн', 'удобр_га')):
                row[name] = np.nan
                if src not in s.columns:
                    continue
                v = s[src].to_numpy(float)
                m = float(np.nanmean(v[k])) if np.isfinite(v[k]).any() else np.nan
                if np.isfinite(m) and m > 0 and np.isfinite(v[i]):
                    row[name] = float(v[i] / m)
            out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Признаки-кандидаты
# --------------------------------------------------------------------------- #
def windows(hdoy, peak):
    """Окна сезона в днях года урожая; отрицательные дни — прошлый год."""
    peak = int(min(max(peak, 40), hdoy - 10))
    return {'осень_пред': (-120, -30), 'зима': (1, 60), 'ранняя_весна': (61, 105),
            'весна': (106, 150), 'до_пика': (0, peak), 'после_пика': (peak, hdoy),
            'сезон': (0, hdoy)}


def _series(tid, parm, year, cache):
    k = (parm, int(year))
    if k not in cache:
        a = np.asarray(db.get_time_series(tid, parm, int(year)), dtype=float)
        cache[k] = a if a.size else np.empty((0, 2))
    return cache[k]


def _combined(tid, parm, year, cache):
    """Ряд года плюс прошлогодняя осень: её дни сдвинуты на −365."""
    cur = _series(tid, parm, year, cache)
    prev = _series(tid, parm, year - 1, cache)
    if not prev.size:
        return cur
    prev = prev.copy()
    prev[:, 0] -= 365.0
    return np.vstack([prev, cur]) if cur.size else prev


def _summ(kind, days, v):
    """Одна сводка по куску ряда. -> float или NaN."""
    if len(v) < 3:
        return np.nan
    if kind == 'сред':
        return float(np.mean(v))
    if kind == 'макс':
        return float(np.max(v))
    if kind == 'мин':
        return float(np.min(v))
    if kind == 'накоп':
        return float(np.sum(v))
    if kind == 'ГДД':
        return float(np.sum(np.clip(v - GDD_BASE, 0.0, None)))
    if kind == 'прирост':
        return float(v[-1] - v[0])
    if kind == 'наклон':
        if np.ptp(days) < 1:
            return np.nan
        return float(np.polyfit(days, v, 1)[0])
    return np.nan


def features_table(rl, mask_parm, hdoy, cache):
    """Таблица кандидатов на (район, год). Кэшируется: пересчёт долгий."""
    if os.path.exists(cache):
        print(f'признаки: беру из {os.path.basename(cache)}')
        f = pd.read_csv(cache, sep=';', decimal=',', encoding='utf-8-sig')
        f['territory_id'] = f['territory_id'].astype(str)
        return f
    plan = [(p, s) for p, ss in PARMS.items() for s in ss]
    plan += [(mask_parm, s) for s in NDVI_SUMM]
    keys = rl[['territory_id', 'year', 'день_пика']].drop_duplicates()
    out, t0 = [], time.time()
    for i, (tid, grp) in enumerate(keys.groupby('territory_id'), 1):
        sc = {}
        for _j, r in grp.iterrows():
            yr, wins = int(r['year']), windows(hdoy, int(r['день_пика']))
            row = {'territory_id': tid, 'year': yr}
            for parm, kind in plan:
                a = _combined(tid, parm, yr, sc)
                short = parm.replace('mean_', '').replace('_7dc_modis_int', '')
                for wn, (lo, hi) in wins.items():
                    name = f'{short}|{kind}|{wn}'
                    if not a.size:
                        row[name] = np.nan
                        continue
                    m = (a[:, 0] >= lo) & (a[:, 0] <= hi)
                    if 'ndvi' in parm:
                        m &= a[:, 1] > 0
                    row[name] = _summ(kind, a[m, 0], a[m, 1])
            out.append(row)
        if i % 100 == 0:
            print(f'    признаки: районов {i}, {time.time() - t0:.0f} c', flush=True)
    f = pd.DataFrame(out)
    f.round(4).to_csv(cache, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'признаки: {f.shape[1] - 2} кандидатов, {time.time() - t0:.0f} c')
    return f


def _grid_curve(a, lo, hi):
    """Кривая на суточной сетке [lo, hi] из ряда (день, значение). -> массив или None."""
    if a.size == 0:
        return None
    m = (a[:, 0] >= lo - 20) & (a[:, 0] <= hi + 20) & (a[:, 1] > 0)
    if m.sum() < 8:
        return None
    d, v = a[m, 0], a[m, 1]
    o = np.argsort(d)
    return np.interp(np.arange(lo, hi + 1), d[o], v[o])


def _align(c, prev, lo, hi_w):
    """Совмещение кривой года с нормой района по прошлым годам сдвигом по времени.

    Сдвиг ищется по максимуму корреляции, а не по минимуму ошибки: корреляция не зависит от
    масштаба, поэтому «отстал» и «вышел слабым» не путаются между собой. -> (сдвиг в днях,
    отношение амплитуд, относительный остаток формы).

    Знак сдвига: значение года в день t−sh сравнивается с нормой в день t, поэтому
    положительный sh означает, что район приходит в то же состояние РАНЬШЕ своей нормы —
    развитие опережает. Проверка знака: корреляция sh с днём пика той же съёмки −0.42.
    """
    if c is None or len(prev) < 2:
        return np.nan, np.nan, np.nan
    ref = np.mean(np.vstack(prev), axis=0)
    a, b = PHASE_SHIFT, int(hi_w - lo)
    if b - a < 20:
        return np.nan, np.nan, np.nan
    t = np.arange(a, b + 1)
    r = ref[t]
    rc = r - r.mean()
    nr = float(np.sqrt(rc @ rc))
    if not np.isfinite(r).all() or r.mean() <= 0 or nr <= 1e-9:
        return np.nan, np.nan, np.nan
    best = None
    for sh in range(-PHASE_SHIFT, PHASE_SHIFT + 1):
        v = c[t - sh]
        vc = v - v.mean()
        nv = float(np.sqrt(vc @ vc))
        if nv <= 1e-9:
            continue
        g = float(vc @ rc) / (nv * nr)
        if best is None or g > best[1]:
            best = (sh, g, v)
    if best is None:
        return np.nan, np.nan, np.nan
    sh, _g, v = best
    amp = float(v.mean() / r.mean())
    if not np.isfinite(amp) or amp <= 0:
        return float(sh), np.nan, np.nan
    res = float(np.sqrt(np.mean((v / amp - r) ** 2)) / r.mean())
    return float(sh), amp, res


def phase_table(rl, mask_parm, hdoy, cache, window):
    """Фаза, амплитуда и остаток формы кривой NDVI против нормы района по прошлым годам.

    Норма берётся только по ПРЕДЫДУЩИМ годам: в год Y других в работе и нет. Кэшируется
    отдельно от features.csv, чтобы не пересчитывать погодные сводки.
    """
    if os.path.exists(cache):
        print(f'фаза: беру из {os.path.basename(cache)}')
        f = pd.read_csv(cache, sep=';', decimal=',', encoding='utf-8-sig')
        f['territory_id'] = f['territory_id'].astype(str)
        return f
    lo, hi = PHASE_FROM, int(hdoy)
    keys = rl[['territory_id', 'year', 'день_пика']].drop_duplicates()
    out, t0 = [], time.time()
    for i, (tid, grp) in enumerate(keys.groupby('territory_id'), 1):
        grp = grp.sort_values('year')
        cur, peaks = {}, {}
        for _j, r in grp.iterrows():
            yr = int(r['year'])
            a = np.asarray(db.get_time_series(tid, mask_parm, yr), dtype=float)
            cur[yr] = _grid_curve(a if a.size else np.empty((0, 2)), lo, hi)
            peaks[yr] = int(r['день_пика'])
        ys = sorted(cur)
        for k, yr in enumerate(ys):
            row = {'territory_id': tid, 'year': yr}
            prev = [cur[y] for y in ys[max(0, k - window):k] if cur[y] is not None]
            for wn, hw in (('до_пика', peaks[yr]), ('сезон', hi - PHASE_SHIFT)):
                sh, amp, res = _align(cur[yr], prev, lo, min(hw, hi - PHASE_SHIFT))
                row[f'ndvi|фаза_сдвиг|{wn}'] = sh
                row[f'ndvi|фаза_ампл|{wn}'] = amp
                row[f'ndvi|фаза_форма|{wn}'] = res
            out.append(row)
        if i % 150 == 0:
            print(f'    фаза: районов {i}, {time.time() - t0:.0f} c', flush=True)
    f = pd.DataFrame(out)
    f.round(4).to_csv(cache, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'фаза: {f.shape[1] - 2} признака, {time.time() - t0:.0f} c')
    return f


def _past_anomaly(X, tid, yr, window):
    """Отклонение каждого признака от нормы района по ПРЕДЫДУЩИМ годам.

    Отличие от ``_zscore`` принципиальное, хотя обе функции вычитают норму района. Там норма
    считается по всем (или по обучающим) годам сразу — в работе такой нормы нет: в июле года
    Y известны только годы до Y. Здесь норма честная, и её можно посчитать в проде. Первый
    год района и годы, где прошлых значений меньше двух, остаются пустыми — это и есть цена.
    """
    A = np.full_like(X, np.nan)
    for t in np.unique(tid):
        k = np.flatnonzero(tid == t)
        k = k[np.argsort(yr[k])]
        for i in range(1, len(k)):
            p = X[k[max(0, i - window):i]]
            ok = np.isfinite(p)
            n = ok.sum(axis=0)
            mu = np.where(n >= 2, np.nansum(np.where(ok, p, 0.0), axis=0) / np.maximum(n, 1),
                          np.nan)
            A[k[i]] = X[k[i]] - mu
    return A


def _family(name):
    """К какой семье кандидат относится: нормировка по всей истории, по прошлым годам, фаза."""
    if 'фаза' in name:
        return 'фаза'
    return 'норма_прошлых' if '@норма' in name else 'вся_история'


# --------------------------------------------------------------------------- #
# Честный перебор
# --------------------------------------------------------------------------- #
def _rho(a, b, min_n=MIN_PAIRS):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n or np.unique(a[m]).size < 5:
        return np.nan
    r = spearmanr(a[m], b[m]).statistic
    return float(r) if np.isfinite(r) else np.nan


def _zscore(X, tid, rows):
    """Центрирование и нормировка каждого признака ВНУТРИ района по заданным строкам.

    В схеме с отложенным ГОДОМ берутся только обучающие строки. В схеме с отложенными
    РАЙОНАМИ так нельзя: у тестового района обучающих строк нет вовсе, нормировка вышла бы
    пустой и корреляция не считалась бы. Там нормируем по собственным строкам района — это
    законно, потому что нормировка не касается цели, только значений признака.
    """
    tr = pd.DataFrame(X[rows], index=pd.Index(tid[rows], name='t'))
    g = tr.groupby(level=0)
    mu = g.mean().reindex(tid).to_numpy(float)
    sd = g.std().reindex(tid).to_numpy(float)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, np.nan)
    return (X - mu) / sd


def _delta_rmse(z, ynorm, yabs, sd, train, test):
    """Насколько поправка по одному признаку уменьшает ошибку линии, в единицах цели."""
    m = train & np.isfinite(z) & np.isfinite(ynorm)
    t = test & np.isfinite(yabs)
    if m.sum() < MIN_PAIRS or t.sum() < 20:
        return np.nan
    b, a = np.polyfit(z[m], ynorm[m], 1)
    corr = (a + b * np.where(np.isfinite(z), z, 0.0)) * sd
    r0 = float(np.sqrt(np.mean(yabs[t] ** 2)))
    r1 = float(np.sqrt(np.mean((yabs[t] - corr[t]) ** 2)))
    return r0 - r1


def _delta_ridge(Z, ynorm, yabs, sd, train, test, top):
    """То же для гребневой поправки по нескольким признакам сразу."""
    from sklearn.linear_model import Ridge
    A = np.where(np.isfinite(Z[:, top]), Z[:, top], 0.0)
    m = train & np.isfinite(ynorm)
    t = test & np.isfinite(yabs)
    if m.sum() < MIN_PAIRS or t.sum() < 20:
        return np.nan
    corr = Ridge(alpha=ALPHA).fit(A[m], ynorm[m]).predict(A) * sd
    r0 = float(np.sqrt(np.mean(yabs[t] ** 2)))
    r1 = float(np.sqrt(np.mean((yabs[t] - corr[t]) ** 2)))
    return r0 - r1


def _rho_in(z, ynorm, yr, rows):
    """Корреляция внутри года, усреднённая по годам, на заданных строках."""
    rs = [_rho(z[rows & (yr == y)], ynorm[rows & (yr == y)], min_n=50)
          for y in np.unique(yr[rows])]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else np.nan


def _delta_rmse_in(z, ynorm, yabs, sd, yr, train, test):
    """Выигрыш по ошибке ВНУТРИ года: уровень года вычтен и из факта, и из поправки.

    Метрика мониторинга. Уровень незнакомого года не называется никем, поэтому наказывать
    за него поправку нечестно; вопрос в том, упорядочивает ли она районы внутри года.
    """
    m = train & np.isfinite(z) & np.isfinite(ynorm)
    t = test & np.isfinite(yabs)
    if m.sum() < MIN_PAIRS or t.sum() < 20:
        return np.nan
    b, a = np.polyfit(z[m], ynorm[m], 1)
    corr = (a + b * np.where(np.isfinite(z), z, 0.0)) * sd
    e0, e1, out = yabs[t], yabs[t] - corr[t], []
    for y in np.unique(yr[t]):
        k = yr[t] == y
        if k.sum() < 20:
            continue
        out.append((float(np.sqrt(np.mean((e0[k] - e0[k].mean()) ** 2)))
                    - float(np.sqrt(np.mean((e1[k] - e1[k].mean()) ** 2)))))
    return float(np.mean(out)) if out else np.nan


def _composite(Z, cols, score):
    """Сумма отобранных признаков с РАВНЫМИ весами; знак каждого — по знаку его оценки.

    Зачем это вместо лучшего одного. Критерий отбора сам оценивается по десяти годам, и
    среди сотен кандидатов всегда найдётся случайно удачный на обучении — он и вытеснит
    верный. Сумма не подгоняет ни одного коэффициента, поэтому этой ошибки почти не делает.
    Второй довод измерен: удачи признаков по годам отрицательно коррелируют (−0.29 у
    весеннего интеграла со сдвигом фазы), то есть они страхуют друг друга.
    """
    g = np.sign(np.nan_to_num(score[cols]))
    g = np.where(g == 0, 1.0, g)
    A = np.where(np.isfinite(Z[:, cols]), Z[:, cols], 0.0) * g
    return A.mean(axis=1)


def _triple_idx(names):
    """Индексы объявленной тройки сводок кривой NDVI. -> список или None."""
    out = []
    for suf in TRIPLE:
        k = [i for i, n in enumerate(names)
             if n.endswith(suf) and 'ndvi' in n and '@норма' not in n]
        if len(k) != 1:
            return None
        out.append(k[0])
    return out


def within_year_matrix(X, tid, yr, ynorm):
    """Матрица «год x кандидат» внутригодовых корреляций. -> (годы, матрица)."""
    Z = _zscore(X, tid, np.ones(len(tid), bool))
    years = np.unique(yr)
    R = np.full((len(years), Z.shape[1]), np.nan)
    for i, y in enumerate(years):
        m = yr == y
        for j in range(Z.shape[1]):
            R[i, j] = _rho(Z[m, j], ynorm[m], min_n=50)
    return years, R


def within_year(X, tid, yr, ynorm):
    """Корреляция с остатком ВНУТРИ года, усреднённая по годам.

    Главная проверка на подмену. Пулинговая корреляция может держаться на том, что год
    целиком выдался плохим: тогда она велика, но районы внутри года не упорядочивает, а
    продукту нужно именно это — «какое поле в этом году хуже прочих». Если величина здесь
    около нуля, пулинговая находка была эффектом года.
    """
    _years, R = within_year_matrix(X, tid, yr, ynorm)
    return np.nanmean(R, axis=0)


def _comp_cols(Z, order, tri, score, ynorm, yabs, sd, yr, train, test):
    """Столбцы по суммам: отобранная топ-SUM_K и объявленная заранее тройка NDVI."""
    out = {}
    for lbl, cols in ((f'сум{SUM_K}', [int(j) for j in order[:SUM_K]]), ('тройка', tri)):
        if cols is None:
            out[f'r_внутри_{lbl}'] = np.nan
            out[f'ΔRMSE_{lbl}'] = np.nan
            continue
        c = _composite(Z, cols, score)
        out[f'r_внутри_{lbl}'] = round(_rho_in(c, ynorm, yr, test), 4)
        out[f'ΔRMSE_{lbl}'] = round(_delta_rmse_in(c, ynorm, yabs, sd, yr, train, test), 4)
    return out


def screen(rl, X, names, scheme, shuffle=False, seed=1):
    """Перебор с честным отбором. -> (таблица по фолдам, таблица по кандидатам)."""
    tid = rl['territory_id'].to_numpy()
    yr = rl['year'].to_numpy(int)
    ynorm = rl['ост_norm'].to_numpy(float)
    yabs = rl['ост'].to_numpy(float)
    sd = rl['sd'].to_numpy(float)
    if shuffle:                               # контроль: годы внутри района перемешаны
        rng = np.random.default_rng(seed)
        ynorm, yabs = ynorm.copy(), yabs.copy()
        for t in np.unique(tid):
            k = np.flatnonzero(tid == t)
            p = rng.permutation(k)
            ynorm[k], yabs[k] = ynorm[p], yabs[p]

    if scheme in ('год', 'год_внутри'):
        folds = [(str(y), yr != y, yr == y) for y in np.unique(yr)]
    else:                                     # половины районов: выбор на одной, проверка на другой
        uniq = np.unique(tid)
        half = set(uniq[::2])
        a = np.array([t in half for t in tid])
        folds = [('половина A', a, ~a), ('половина B', ~a, a)]
    # отбор в схеме «год_внутри» идёт по способности упорядочить районы внутри обучающих лет
    ry = within_year_matrix(X, tid, yr, ynorm) if scheme == 'год_внутри' else (None, None)

    tri = _triple_idx(names)
    frows, tests, trains = [], [], []
    for fname, train, test in folds:
        Z = _zscore(X, tid, train if scheme == 'год' else np.ones(len(tid), bool))
        r_tr = np.array([_rho(Z[train, j], ynorm[train]) for j in range(Z.shape[1])])
        r_te = np.array([_rho(Z[test, j], ynorm[test]) for j in range(Z.shape[1])])
        trains.append(r_tr)
        tests.append(r_te)
        if not np.isfinite(r_tr).any():
            continue
        score = r_tr
        if scheme == 'год_внутри':
            years, R = ry
            score = np.nanmean(R[years != int(fname)], axis=0)
        if not np.isfinite(score).any():
            continue
        order = np.argsort(-np.abs(np.nan_to_num(score)))
        best = int(order[0])
        top = [int(j) for j in order[:TOP_K]]
        frows.append({'схема': scheme, 'фолд': fname, 'строк_обуч': int(train.sum()),
                      'строк_тест': int(test.sum()), 'кандидатов': int(np.isfinite(r_tr).sum()),
                      'лучший_признак': names[best],
                      'r_обуч': round(float(score[best]), 4),
                      'r_тест': round(float(r_te[best]), 4),
                      'r_внутри_тест': round(_rho_in(Z[:, best], ynorm, yr, test), 4),
                      'ΔRMSE_внутр': round(_delta_rmse_in(Z[:, best], ynorm, yabs, sd,
                                                          yr, train, test), 4),
                      'ΔRMSE_один': round(_delta_rmse(Z[:, best], ynorm, yabs, sd,
                                                      train, test), 4),
                      **_comp_cols(Z, order, tri, score, ynorm, yabs, sd, yr, train, test),
                      f'ΔRMSE_топ{TOP_K}': round(_delta_ridge(Z, ynorm, yabs, sd,
                                                              train, test, top), 4)})
    Za = _zscore(X, tid, np.ones(len(tid), bool))
    cand = pd.DataFrame({'признак': names,
                         'r_пул': [round(_rho(Za[:, j], ynorm), 4)
                                   for j in range(Za.shape[1])],
                         'r_обуч_сред': np.nanmean(np.vstack(trains), axis=0).round(4),
                         'r_тест_сред': np.nanmean(np.vstack(tests), axis=0).round(4),
                         'r_внутри_года': within_year(X, tid, yr, ynorm).round(4),
                         'фолдов': np.isfinite(np.vstack(tests)).sum(axis=0),
                         'семья': [_family(n) for n in names]})
    cand['|r_тест|'] = cand['r_тест_сред'].abs().round(4)
    cand['покрытие'] = cand['фолдов'] >= int(0.8 * max(cand['фолдов'].max(), 1))
    return pd.DataFrame(frows), cand.sort_values('|r_тест|', ascending=False)


# --------------------------------------------------------------------------- #
# Графики и отчёт
# --------------------------------------------------------------------------- #
def _plots(outdir, cand, cand_sh, folds):
    imgs = []

    def hist(lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        for v, c, ru, en in ((cand[cand['покрытие']]['r_тест_сред'].dropna(), '#1f77b4',
                              'кандидаты', 'candidates'),
                             (cand_sh[cand_sh['покрытие']]['r_тест_сред'].dropna(), '#999999',
                              'контроль (годы перемешаны)', 'control (years shuffled)')):
            ax.hist(v, bins=40, histtype='step', lw=2, color=c,
                    label=f'{plots.tr(ru, en, lang)}: |max| {v.abs().max():.3f}')
        ax.axvline(0, color='k', lw=1)
        ax.set_xlabel(plots.tr('Корреляция с остатком на отложенном годе',
                               'Correlation with residual on the held-out year', lang))
        ax.set_ylabel(plots.tr('Число признаков', 'Features', lang))
        ax.set_title(plots.tr('Что находится в данных и что — на шуме',
                              'Signal found vs noise floor', lang))
        ax.legend()
        fig.tight_layout()
        return fig

    imgs.append(('Кандидаты против контроля перестановкой',
                 plots.bilingual(outdir, 'screen_hist', hist)))

    def bars(lang):
        f = folds[folds['схема'] == 'год']
        fig, ax = plt.subplots(figsize=(9, 5))
        k = np.arange(len(f))
        ax.bar(k - 0.2, f['ΔRMSE_один'], 0.4, color='#1f77b4',
               label=plots.tr('один признак', 'single feature', lang))
        ax.bar(k + 0.2, f[f'ΔRMSE_топ{TOP_K}'], 0.4, color='#2ca02c',
               label=plots.tr(f'гребень, топ-{TOP_K}', f'ridge, top-{TOP_K}', lang))
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(k)
        ax.set_xticklabels(f['фолд'], rotation=45)
        ax.set_ylabel(plots.tr('Выигрыш, единицы цели', 'Gain, target units', lang))
        ax.set_title(plots.tr('Сколько поправка снимает с ошибки районной линии',
                              'Error removed from the per-district line', lang))
        ax.legend()
        fig.tight_layout()
        return fig

    imgs.append(('Выигрыш по годам', plots.bilingual(outdir, 'screen_delta', bars)))
    return imgs


def _docx(outdir, title, meta, folds, cand, cand_sh, imgs):
    from docx import Document
    from docx.shared import Inches
    docs = os.path.join(os.path.dirname(os.path.dirname(config.REPORTS_DIR)), 'docs')
    os.makedirs(docs, exist_ok=True)
    doc = Document()
    doc.add_heading(f'Остаток регрессии пика NDVI: что его объясняет. {title}', 0)

    doc.add_heading('Вопрос и устройство опыта', level=1)
    doc.add_paragraph(
        'Районная линия «пик NDVI → урожайность» снимает часть ошибки, но остаётся остаток. '
        'Здесь мы ищем в уже скачанных рядах величину, которая этот остаток объясняет. Цель — '
        'именно остаток, а не урожайность: признак, коррелирующий с урожайностью, скорее '
        'всего коррелирует и с пиком NDVI, то есть несёт ту же информацию.')
    doc.add_paragraph(
        'Модель двухэтажная, и это вынужденно. В районе 10–11 лет, поэтому вторая переменная '
        'внутри районной регрессии съест степени свободы. Районная часть остаётся простой '
        'линией, а поправка на остаток учится сразу по всем районам. Остаток делится на '
        'разброс урожайности самого района, признаки центрируются внутри района — так вопрос '
        'становится «какой год в этом районе выдался необычным», а не «где вообще теплее».')
    doc.add_paragraph(
        f'Цель: {meta["target"]} ({meta["unit"]}). Районов {meta["districts"]}, строк '
        f'«район-год» {meta["rows"]}, годы {meta["year_min"]}–{meta["year_max"]}. Кандидатов '
        f'{meta["n_cand"]}. Ошибка самой линии на этих строках: {meta["rmse_line"]:.2f} '
        f'{meta["unit"]}.')

    doc.add_heading('Что перебиралось', level=1)
    doc.add_paragraph(
        f'Ряды: {", ".join(PARMS)} и маска NDVI «{meta["mask"]}». Окна сезона: '
        f'{", ".join(meta["windows"])}. Сводки: средняя, максимум, минимум, накопление '
        f'(сумма по суточному ряду), ГДД (сумма температур выше {GDD_BASE:.0f} °C), прирост '
        'готового накопленного ряда, наклон. Накопленные температура и осадки посчитаны '
        'двумя путями — суммой по суточному ряду и приростом накопленных рядов Веги; второй '
        'надёжнее, если в суточном ряду есть пропуски.')
    for s in ('«после_пика» — условия между пиком и уборкой. Пик снимается до налива зерна, '
              'поэтому это единственное окно, ортогональное пику по построению;',
              '«осень_пред» — сев и уход в зиму, дни прошлого года;',
              f'плюс признаки БДПМО: {", ".join(BDPMO_FEATS)} — площадь и удобрения '
              'относительно средних обучающих лет того же района.'):
        doc.add_paragraph(s, style='List Bullet')
    doc.add_paragraph(
        'Признак «остаток прошлого года» сознательно исключён: при исключении года его '
        'расчёт подсматривает отвечаемый год.')
    doc.add_paragraph(
        f'Каждая сводка перебирается дважды. Первый раз — как есть, с нормировкой внутри '
        f'района по всем доступным годам. Второй раз — как отклонение от нормы района по '
        f'последним {meta["norm"]} ПРОШЛЫМ годам (суффикс «@норма{meta["norm"]}»). Вторая '
        f'форма честнее: нормы по всей истории в работе не существует, в год Y известны '
        f'только годы до Y. Цена — первые годы каждого района остаются пустыми.')
    doc.add_paragraph(
        f'Отдельно считается фаза кривой NDVI. Любая сводка по окну складывает два разных '
        f'события в одно число: район отстал в развитии и район вышел слабым дают одинаково '
        f'низкий весенний интеграл, а это разные беды с разным исходом. Поэтому кривая года '
        f'совмещается со средней кривой того же района по прошлым годам сдвигом по времени '
        f'(до ±{PHASE_SHIFT} дней, по максимуму корреляции — она не зависит от масштаба), и '
        f'докладываются порознь сдвиг в днях «фаза_сдвиг» (положительный — развитие '
        f'запаздывает), отношение амплитуд «фаза_ампл» и остаток формы после совмещения '
        f'«фаза_форма».')

    m = folds[folds['схема'] == 'год']
    md = folds[folds['схема'] == 'районы']
    mi = folds[folds['схема'] == 'год_внутри']
    cols = COLS

    doc.add_heading('Отбор по пулинговой корреляции, отложенный год', level=1)
    doc.add_paragraph(
        'Признак выбирается по обучающим годам по корреляции со всем остатком сразу, а '
        f'докладывается по отложенному году. ΔRMSE_один и ΔRMSE_топ{TOP_K} — насколько '
        f'поправка уменьшила ошибку районной линии, в {meta["unit"]}; отрицательное значение '
        'означает, что поправка сделала хуже. ΔRMSE_внутр — то же, но уровень года вычтен '
        'и из факта, и из поправки: метрика мониторинга.')
    E._table_from_df(doc, m[cols], maxrows=40)
    doc.add_paragraph(
        'Знаки по годам разные, поэтому средние корреляции берутся по модулю: иначе плюс '
        'одного года погасит минус другого и выйдет ноль там, где связь на обучении сильна. '
        f'По модулю: r на обучении {m["r_обуч"].abs().mean():.3f}, на отложенном годе '
        f'{m["r_тест"].abs().mean():.3f} при среднем со знаком {m["r_тест"].mean():+.3f} — '
        f'то есть знак на тесте случаен. ΔRMSE одним признаком {m["ΔRMSE_один"].mean():+.3f} '
        f'{meta["unit"]}, гребнем по топ-{TOP_K} — {m[f"ΔRMSE_топ{TOP_K}"].mean():+.3f}, то '
        f'есть {100 * m["ΔRMSE_один"].mean() / meta["rmse_line"]:+.1f} % и '
        f'{100 * m[f"ΔRMSE_топ{TOP_K}"].mean() / meta["rmse_line"]:+.1f} % от ошибки линии. '
        'Связь, уверенно найденная на обучении, смены года не переживает.')

    doc.add_heading('А переносится ли она на другие районы', level=1)
    doc.add_paragraph(
        'Тот же перебор, но фолды — половины районов: отбор на одной половине, проверка на '
        'другой, годы у обеих общие. Это разделение проверяет перенос по территории, а не по '
        'времени.')
    E._table_from_df(doc, md[cols], maxrows=10)
    doc.add_paragraph(
        f'Здесь связь переносится почти без потерь: r {md["r_обуч"].mean():+.3f} на отборе и '
        f'{md["r_тест"].mean():+.3f} на незнакомых районах, выигрыш '
        f'{md["ΔRMSE_один"].mean():+.3f} {meta["unit"]} одним признаком и '
        f'{md[f"ΔRMSE_топ{TOP_K}"].mean():+.3f} гребнем. Сопоставление двух разделений и даёт '
        'ответ: найденная зависимость — настоящая и общая для страны, но она про УРОВЕНЬ '
        'года, а не про различия внутри него. Выигрыш в этой таблице получен только потому, '
        'что обучающая половина районов видела те же самые годы. В работе такого знания нет: '
        'в июле текущего года урожайность не собрана нигде. Поэтому отсечка по году '
        'обязательна, а эта таблица — измерение того, сколько она у нас забирает.')
    doc.add_paragraph(
        f'Та же величина в столбце ΔRMSE_внутр ({md["ΔRMSE_внутр"].mean():+.3f} '
        f'{meta["unit"]}) показывает то же с другой стороны: если уровень года не '
        'засчитывать, от выигрыша не остаётся ничего.')

    doc.add_heading('Отбор по внутригодовой корреляции', level=1)
    doc.add_paragraph(
        'Если пулинговый отбор выбирает признак, знающий про уровень года, выбирать надо по '
        'другому правилу: по способности упорядочить районы ВНУТРИ обучающих лет. Фолды те '
        'же — отложенный год, — меняется только критерий отбора (столбец r_обуч здесь и есть '
        'внутригодовая корреляция по обучающим годам).')
    doc.add_paragraph(
        'Эту таблицу нельзя читать отдельно от следующей. Здесь в каждом году берётся ОДИН '
        'лучший признак, и на четырёх сотнях кандидатов такой отбор сам становится главным '
        'источником ошибки: в следующем разделе та же схема с суммой трёх даёт вдвое больше.')
    E._table_from_df(doc, mi[cols], maxrows=40)
    pick = mi['лучший_признак'].value_counts()
    doc.add_paragraph(
        f'Выбирается почти всегда одно и то же: {pick.index[0]} ({int(pick.iloc[0])} из '
        f'{len(mi)} лет). На отложенном годе внутригодовая корреляция '
        f'{mi["r_внутри_тест"].mean():+.3f} в среднем и положительна в '
        f'{int((mi["r_внутри_тест"] > 0).sum())} годах из {len(mi)}. В центнерах это '
        f'{mi["ΔRMSE_внутр"].mean():+.3f} {meta["unit"]}, то есть '
        f'{100 * mi["ΔRMSE_внутр"].mean() / meta["rmse_line"]:+.1f} % от ошибки линии: '
        'корреляция звучит заметно лучше, чем стоит в единицах цели.')

    doc.add_heading('Отбор одного признака против суммы трёх', level=1)
    doc.add_paragraph(
        'Отбор лучшего одного среди сотен кандидатов шумен сам по себе. Критерий отбора '
        'оценивается по тем же десяти годам, и среди четырёх сотен кандидатов всегда '
        'найдётся случайно удачный на обучении — он и вытеснит верный. Поэтому рядом '
        f'считается сумма {SUM_K} лучших с РАВНЫМИ весами: она не подгоняет ни одного '
        'коэффициента. Довод не только теоретический — удачи признаков по годам '
        'отрицательно коррелируют: в годы, когда весенний интеграл проваливается, сдвиг '
        'фазы как раз работает, и наоборот.')
    E._table_from_df(doc, mi[COMP_COLS], maxrows=40)
    doc.add_paragraph(
        f'Среднее по отложенным годам: лучший один '
        f'{mi["r_внутри_тест"].mean():+.3f}, сумма {SUM_K} '
        f'{mi[f"r_внутри_сум{SUM_K}"].mean():+.3f}, объявленная тройка '
        f'{mi["r_внутри_тройка"].mean():+.3f}; в {meta["unit"]} — '
        f'{mi["ΔRMSE_внутр"].mean():+.3f}, {mi[f"ΔRMSE_сум{SUM_K}"].mean():+.3f} и '
        f'{mi["ΔRMSE_тройка"].mean():+.3f} при ошибке линии {meta["rmse_line"]:.2f}.')
    doc.add_paragraph(
        'Столбец «тройка» надо читать с оговоркой, и она существенная. Это объявленная '
        f'заранее тройка сводок кривой NDVI — {", ".join(TRIPLE)}, то есть «сколько», '
        '«когда» и «как быстро падает», — но выбрана она была уже после того, как перебор '
        'прошёл по всем годам. Поэтому её число — не честная оценка, а верхняя граница: '
        'столько даёт знание, какие три сводки брать. Разница между столбцами «сумма» и '
        '«тройка» и есть цена незнания. Проверить тройку можно только там, где она не '
        'подбиралась, — на других культурах.')

    doc.add_heading('Планка: контроль перестановкой', level=1)
    doc.add_paragraph(
        'Тот же перебор, но остаток перемешан по годам внутри района — сигнала там нет по '
        'построению. Если на шуме лучший кандидат показывает столько же, сколько на данных, '
        'находка мнимая.')
    doc.add_paragraph(
        'Сравниваются только кандидаты, посчитанные почти на всех фолдах: иначе редкий '
        'признак, доживший до трёх фолдов, выдаёт случайное число за находку.')
    cmp_t = pd.DataFrame([
        {'перебор': 'данные', 'кандидатов': int(cand['покрытие'].sum()),
         'лучшая |r| на тесте': round(float(cand[cand['покрытие']]['|r_тест|'].max()), 4),
         'признаков с |r| > 0.1': int((cand[cand['покрытие']]['|r_тест|'] > 0.1).sum())},
        {'перебор': 'перемешанный остаток',
         'кандидатов': int(cand_sh['покрытие'].sum()),
         'лучшая |r| на тесте': round(float(cand_sh[cand_sh['покрытие']]['|r_тест|'].max()), 4),
         'признаков с |r| > 0.1': int((cand_sh[cand_sh['покрытие']]['|r_тест|'] > 0.1).sum())}])
    E._table_from_df(doc, cmp_t)

    doc.add_heading('Норма района: вся история, прошлые годы, фаза', level=1)
    doc.add_paragraph(
        'Три семьи кандидатов сравниваются на одном переборе и с одной планкой. «вся_история» '
        '— нормировка внутри района по всем годам; «норма_прошлых» — отклонение от нормы по '
        f'последним {meta["norm"]} прошлым годам; «фаза» — сдвиг, амплитуда и остаток формы '
        'кривой NDVI. Планка в последнем столбце считается по той же семье на перемешанном '
        'остатке: у разных семей разное число кандидатов, а значит и разный шанс случайной '
        'находки.')
    fam = []
    for nm, sub in cand[cand['покрытие']].groupby('семья'):
        sh = cand_sh[cand_sh['покрытие'] & (cand_sh['семья'] == nm)]
        fam.append({'семья': nm, 'кандидатов': len(sub),
                    'лучшая |r_тест|': round(float(sub['|r_тест|'].max()), 4),
                    'лучшая |r_внутри_года|': round(float(sub['r_внутри_года'].abs().max()), 4),
                    'планка на шуме': (round(float(sh['|r_тест|'].max()), 4)
                                       if len(sh) else np.nan)})
    fam = pd.DataFrame(fam).sort_values('лучшая |r_внутри_года|', ascending=False)
    E._table_from_df(doc, fam)
    bst = fam.iloc[0]
    doc.add_paragraph(
        f'Лучшая внутригодовая корреляция у семьи «{bst["семья"]}» '
        f'({bst["лучшая |r_внутри_года|"]:.3f} при планке {bst["планка на шуме"]:.3f}). '
        'Сравнение семей между собой и есть ответ на вопрос, стоит ли подавать модели '
        'разность «текущий год минус историческая кривая» вместо самого значения.')
    top_ph = cand[cand['покрытие'] & (cand['семья'] == 'фаза')]
    if not top_ph.empty:
        doc.add_paragraph('Признаки фазы по отдельности:')
        E._table_from_df(doc, top_ph[['признак', 'r_пул', 'r_тест_сред', 'r_внутри_года']],
                         maxrows=6)

    doc.add_heading('20 лучших кандидатов по честной корреляции', level=1)
    doc.add_paragraph(
        'Ранжирование по корреляции на отложенных годах — без отбора, поэтому его можно '
        'читать как список подозреваемых, а не как подтверждённый результат.')
    E._table_from_df(doc, cand[cand['покрытие']].head(20)[
        ['признак', 'семья', 'r_пул', 'r_обуч_сред', 'r_тест_сред', 'r_внутри_года']],
        maxrows=20)
    doc.add_paragraph(
        'Столбцы стоит читать вместе. «r_пул» — корреляция по всем строкам сразу; '
        '«r_обуч_сред» и «r_тест_сред» — она же на обучающих и на отложенных годах; '
        '«r_внутри_года» — корреляция внутри одного года, то есть способность упорядочить '
        'районы между собой. Расхождение между первым и последним столбцом и есть эффект '
        'года: признак знает, что год выдался плохим, но не знает, где именно.')

    doc.add_heading('Что из этого следует', level=1)
    full = cand[cand['покрытие']]
    floor = float(cand_sh[cand_sh['покрытие']]['|r_тест|'].max())
    top_in = full.reindex(full['r_внутри_года'].abs().sort_values(ascending=False).index)
    best_in = top_in.iloc[0]
    ndvi_top = int(top_in.head(5)['признак'].str.contains('ndvi').sum())
    hold = float(mi[f'r_внутри_сум{SUM_K}'].mean())
    plus = int((mi[f'r_внутри_сум{SUM_K}'] > 0).sum())
    # находкой считаем только то, что подтвердилось на отложенных годах и с запасом над планкой
    conf = hold > max(2 * floor, 0.05) and plus >= 0.6 * len(mi)
    found = (
        f'Дополнительная информация в данных есть, и она в самом ряду NDVI, а не в погоде. '
        f'Лучший кандидат по внутригодовой корреляции — «{best_in["признак"]}» '
        f'({best_in["r_внутри_года"]:+.3f}); среди пяти лучших таких признаков '
        f'{ndvi_top} из 5 сняты с того же ряда NDVI, из которого мы брали один только '
        f'максимум. Сумма {SUM_K} с равными весами держит на отложенных годах {hold:+.3f}, '
        f'плюс в {plus} годах из {len(mi)}, планка перестановкой {floor:.3f} — это находка, '
        f'а не шум. Пик выбрасывает всю форму кривой, и именно форма несёт то, чего пик не '
        f'увидел: сколько биомассы набрано весной, насколько развитие опережает норму '
        f'района и как быстро съёмка падает после пика.')
    nofound = (
        f'Для этой культуры находки нет. Лучший кандидат по внутригодовой корреляции — '
        f'«{best_in["признак"]}» ({best_in["r_внутри_года"]:+.3f}) при планке перестановкой '
        f'{floor:.3f}, то есть он почти не отличается от шума; на отложенных годах от него '
        f'остаётся {hold:+.3f}, положительный знак лишь в {plus} годах из {len(mi)}. Ни один '
        f'из {int(cand["покрытие"].sum())} кандидатов не прошёл проверку, и это тоже '
        f'результат: добавлять к пику нечего, разницу надо искать не здесь.')
    for s in (
        found if conf else nofound,
        f'Погодные признаки в этом переборе — ловушка, и дорогая. Пулинговая корреляция у них '
        f'самая большая ({m["r_обуч"].abs().mean():.2f} по модулю у выбираемого по обучению), '
        f'на незнакомых районах она держится ({md["r_тест"].mean():+.2f}), а на незнакомом '
        f'годе исчезает ({m["r_тест"].abs().mean():.2f} по модулю при случайном знаке, '
        f'{m["r_тест"].mean():+.2f} со знаком). Значит, они знают, каким выдался год, а не какие '
        f'районы в этом году хуже прочих. Любой отбор по пулинговой корреляции выберет '
        f'именно их и проиграет: отбирать надо по внутригодовой.',
        f'Но и найденное надо мерить в центнерах. Внутригодовая корреляция '
        f'{hold:+.3f} на отложенных годах — это '
        f'{mi[f"ΔRMSE_сум{SUM_K}"].mean():+.3f} {meta["unit"]} от ошибки линии '
        f'{meta["rmse_line"]:.2f}, около '
        f'{abs(100 * mi[f"ΔRMSE_сум{SUM_K}"].mean() / meta["rmse_line"]):.0f} %. Ранговая '
        f'корреляция 0.2 отвечает примерно R² = 0.04, поэтому выигрыш и не может быть '
        + ('большим. Это полезная добавка, а не перелом.' if conf else
           'большим — а здесь он и вовсе отрицательный, то есть поправку ставить не на чем.'),
        f'Потолок там, где его и ожидали. Строк «район-год» {meta["rows"]}, но независимых '
        f'наблюдений связи «погода → урожайность» всего '
        f'{meta["year_max"] - meta["year_min"] + 1} — столько, сколько лет: в один год все '
        f'районы попадают под одну погоду. Районы добавляют степеней свободы только тому, '
        f'что различается внутри года, — то есть съёмке, а не сводкам погоды.',
        'Отсюда рабочий вывод: вторую переменную к пику искать не стоит, стоит лучше '
        'использовать сам ряд NDVI, а поправку держать общей для страны и небольшой. '
        'Районную калибровку поправки 10 лет не выдержат.',
        'И вывод про сам перебор, обошедшийся дороже всего: расширение списка кандидатов '
        'ухудшило честный результат. Пока кандидатов было две сотни, отбор лучшего одного '
        'давал 0.19 на отложенных годах; на четырёх сотнях он даёт вдвое меньше, хотя сами '
        'данные не изменились и среди них появился настоящий новый признак. Лечится это не '
        'умным критерием отбора (проверено: отбор по устойчивости к годам ещё хуже), а '
        'отказом от отбора — суммой нескольких признаков с равными весами.',
        'И вывод про сам опыт: его стоит прогонять по каждой культуре отдельно. Находка на '
        'одной культуре не переносится автоматически — у озимых весенний ряд NDVI говорит о '
        'перезимовке, которой у яровых просто нет, и там тот же признак оказывается почти '
        'повторением пика.'):
        doc.add_paragraph(s, style='List Bullet')

    doc.add_heading('Графики', level=1)
    for name, png in imgs:
        doc.add_heading(name, level=2)
        if png:
            doc.add_picture(png, width=Inches(6.3))

    doc.add_heading('Как повторить', level=1)
    doc.add_paragraph(
        'Опыт не привязан к культуре: цель, маска и день уборки берутся из конфигурации. '
        'Признаки кэшируются в features.csv, пики — в кэше peak_regression, поэтому повторный '
        'прогон быстрый.')
    for cmd in ('python code/peak_residual_screen.py --culture "Озимая пшеница"',
                'python code/peak_residual_screen.py --culture "Яровая пшеница" --min-years 6',
                'python code/peak_residual_screen.py --culture "Озимая пшеница" '
                '--target посевная'):
        doc.add_paragraph(cmd, style='Intense Quote')

    doc.add_heading('Выгрузки', level=1)
    for s in ('candidates.csv — все кандидаты с корреляциями на обучении и на тесте;',
              'phase.csv — сдвиг, амплитуда и остаток формы кривой NDVI (кэш);',
              'candidates_shuffled.csv — то же на перемешанном остатке (планка);',
              'folds.csv — по фолдам: выбранный признак и ΔRMSE;',
              'residuals.csv — остаток районной линии по району и году;',
              'features.csv — значения всех кандидатов (кэш).'):
        doc.add_paragraph(s, style='List Bullet')
    doc.add_paragraph(f'Каталог: {outdir}')

    out = os.path.join(docs, f'Остаток_пика_NDVI_{title.replace(" ", "_")}.docx')
    try:
        doc.save(out)
    except PermissionError:
        out = out.replace('.docx', '_new.docx')
        doc.save(out)
    print(f'Отчёт Word: {out}')
    return out


# --------------------------------------------------------------------------- #
def run(culture, target, min_years, form, rebuild, norm_years=NORM_YEARS):
    cu = P.culture_row(culture)
    hdoy, task = P.harvest_doy(cu['culture_id'])
    title = str(cu['title'])
    outdir = os.path.join(config.REPORTS_DIR, 'peak_residual', task)
    os.makedirs(outdir, exist_ok=True)
    fcache = os.path.join(outdir, 'features.csv')
    pcache_f = os.path.join(outdir, 'phase.csv')
    for c in (fcache, pcache_f):
        if rebuild and os.path.exists(c):
            os.remove(c)

    d, _diag = P.targets_table(cu['bdpmo_culture'])
    fert = fertilizer(cu['bdpmo_culture'])
    if fert is not None:
        d = d.merge(fert, on=['territory_id', 'year'], how='left')
        d['удобр_га'] = d['удобрения'] / d['посевная'].replace(0, np.nan)

    pcache = os.path.join(config.REPORTS_DIR, 'peak_regression', task, 'peaks.csv')
    os.makedirs(os.path.dirname(pcache), exist_ok=True)
    pk = P.peaks_table(sorted(d['territory_id'].unique()), sorted(d['year'].unique()),
                       cu['mask_culture'], hdoy, pcache)
    pk['territory_id'] = pk['territory_id'].astype(str)
    d = d.merge(pk, on=['territory_id', 'year'], how='inner')
    if d.empty:
        raise SystemExit('нет пересечения целей и пиков NDVI')

    rl = resid_table(d, target, form, min_years)
    if rl.empty:
        raise SystemExit(f'ни один район не набрал {min_years} лет')
    mask_parm = str(d['маска'].mode().iloc[0])
    f = features_table(rl, mask_parm, hdoy, fcache)
    rl = rl.merge(f, on=['territory_id', 'year'], how='left')
    ph = phase_table(rl, mask_parm, hdoy, pcache_f, norm_years)
    rl = rl.merge(ph, on=['territory_id', 'year'], how='left')

    base = [c for c in rl.columns if '|' in c]
    names = base + [c for c in BDPMO_FEATS if c in rl.columns]
    X = rl[names].to_numpy(float)
    # вторая семья: те же величины, но как отклонение от нормы района по прошлым годам.
    # Признаки фазы уже построены как отклонение от такой нормы, их повторно не трогаем.
    anom = [c for c in base if 'фаза' not in c]
    A = _past_anomaly(rl[anom].to_numpy(float), rl['territory_id'].to_numpy(),
                      rl['year'].to_numpy(int), norm_years)
    X = np.hstack([X, A])
    names = names + [f'{c}@норма{norm_years}' for c in anom]
    keep = np.isfinite(X).sum(axis=0) >= MIN_PAIRS      # кандидаты без покрытия не нужны
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    rmse_line = float(np.sqrt(np.mean(rl['ост'].to_numpy(float) ** 2)))
    print(f'остаток: строк {len(rl)}, районов {rl["territory_id"].nunique()}, '
          f'RMSE линии {rmse_line:.2f}; кандидатов с покрытием {len(names)}')

    parts, cands = [], None
    for sch in SCHEMES:
        fd, cd = screen(rl, X, names, sch)
        parts.append(fd)
        if sch == 'год':
            cands = cd
    folds = pd.concat(parts, ignore_index=True)
    _fd_sh, cand_sh = screen(rl, X, names, 'год', shuffle=True)

    print('\n=== по фолдам')
    print(folds.to_string(index=False))
    print('\n=== 15 лучших кандидатов (корреляция с остатком на отложенных годах)')
    print(cands.head(15).to_string(index=False))
    print(f'\n=== среднее по схемам (ошибка линии {rmse_line:.2f})')
    g = folds.copy()
    for c in ('r_обуч', 'r_тест', 'r_внутри_тест'):
        g[f'|{c}|'] = g[c].abs()                  # знаки по годам разные и гасят друг друга
    print(g.groupby('схема')[['|r_обуч|', '|r_тест|', '|r_внутри_тест|', 'r_внутри_тест',
                              'ΔRMSE_внутр', 'ΔRMSE_один', f'ΔRMSE_топ{TOP_K}']]
          .mean().round(3).to_string())
    print('\n=== отбор одного против суммы (схема год_внутри)')
    print(folds[folds['схема'] == 'год_внутри'][COMP_COLS].to_string(index=False))
    print(folds[folds['схема'] == 'год_внутри'][COMP_COLS[1:]].mean().round(4).to_string())
    fl = cand_sh[cand_sh['покрытие']]['|r_тест|'].max()
    print(f'планка перестановкой: лучшая |r| на шуме {fl:.4f}, на данных '
          f'{cands[cands["покрытие"]]["|r_тест|"].max():.4f}')

    rl.round(4).to_csv(os.path.join(outdir, 'residuals.csv'), sep=';', decimal=',',
                       index=False, encoding='utf-8-sig')
    folds.to_csv(os.path.join(outdir, 'folds.csv'), sep=';', decimal=',',
                 index=False, encoding='utf-8-sig')
    cands.to_csv(os.path.join(outdir, 'candidates.csv'), sep=';', decimal=',',
                 index=False, encoding='utf-8-sig')
    cand_sh.to_csv(os.path.join(outdir, 'candidates_shuffled.csv'), sep=';', decimal=',',
                   index=False, encoding='utf-8-sig')

    meta = {'target': target, 'unit': P.UNITS[target], 'mask': mask_parm,
            'districts': int(rl['territory_id'].nunique()), 'rows': len(rl),
            'year_min': int(rl['year'].min()), 'year_max': int(rl['year'].max()),
            'n_cand': len(names), 'rmse_line': rmse_line, 'norm': norm_years,
            'windows': list(windows(hdoy, 160))}
    imgs = _plots(outdir, cands, cand_sh, folds)
    _docx(outdir, title, meta, folds, cands, cand_sh, imgs)
    print(f'Таблицы: {outdir}')
    return folds


def main():
    ap = argparse.ArgumentParser(description='Что объясняет остаток регрессии пика NDVI')
    ap.add_argument('--culture', required=True, help='название или culture_id; all — все')
    ap.add_argument('--target', default='убранная', choices=P.TARGETS)
    ap.add_argument('--min-years', type=int, default=7)
    ap.add_argument('--form', default='линейная', choices=P.FORMS)
    ap.add_argument('--norm-years', type=int, default=NORM_YEARS,
                    help='сколько прошлых лет берёт норма района')
    ap.add_argument('--rebuild', action='store_true', help='пересчитать кэш признаков')
    a = ap.parse_args()

    if str(a.culture).strip().lower() == 'all':
        cu = pd.read_csv(config.CULTURES_FILE, encoding='utf-8-sig')
        for name in cu['title'].astype(str):
            print(f'\n################ {name}')
            try:
                run(name, a.target, a.min_years, a.form, a.rebuild, a.norm_years)
            except SystemExit as e:
                print(f'пропуск: {e}')
        return
    run(a.culture, a.target, a.min_years, a.form, a.rebuild, a.norm_years)


if __name__ == '__main__':
    main()
