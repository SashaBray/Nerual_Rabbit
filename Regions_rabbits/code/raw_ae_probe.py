"""Разбор кода автоэнкодера: на что он опирается и не переоткрыл ли литературу.

Зачем отдельный модуль. ``raw_autoencoder`` даёт числа переноса, но положительный перенос
без разбора — чёрный ящик, и ему нельзя верить. Здесь берётся ТО ЕДИНСТВЕННОЕ плечо, что
устояло на трёх затравках (суточная сетка, обучение на поздней эпохе, проверка на ранней),
и задаются три вопроса.

1. **Откуда сигнал.** Норма веса гребня и вклад в разброс предсказания по каналам. Сумма
   вкладов может превышать итоговый разброс: тогда каналы гасят друг друга и сигнал лежит
   в их разнице, а не в каждом по себе.
2. **На что это похоже.** Корреляция предсказания с каждым из 34 литературных признаков
   ``agro_panel`` внутри года на ПРОВЕРОЧНОЙ эпохе. Признак центрируется по району, чтобы
   сравнивались отклонения, а не уровни районов.
3. **Один сигнал или два.** Литературный гребень и объединённый гребень на том же плече.
   Если код — переоткрытая литература, объединение прибавки не даст, а связь двух
   предсказаний будет высокой. Это прямая проверка, а не рассуждение.

Нормировка литературных признаков повторяет ``agro_panel``: каждая эпоха z-оценивается сама
по себе (вне своей эпохи норма района протекала бы в проверку), пропуски заменяются нулём —
«район как обычно». Требовать все 34 значения нельзя: часть окон в иные годы не считается,
и полный набор остаётся лишь у 474 строк из 9679, поэтому порог — половина признаков.

Запуск:
    python code/raw_ae_probe.py --culture "Озимая пшеница"
    python code/raw_ae_probe.py --culture "Озимая пшеница" --res 6h --seeds 0
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

import agro_panel as G
import config
import peak_regression as P
import raw_autoencoder as R

TOP_FEATURES = 12        # сколько литературных признаков показать в разборе
NEED_LIT = 0.5           # доля литературных признаков, обязанных быть заполненными


def era_matrix(d, cols, early, late):
    """z-оценки литературных признаков: каждая эпоха нормирована сама по себе.

    Так же, как в ``agro_panel``: обучающие строки берут норму своей эпохи, проверочные —
    своей. Иначе среднее и разброс района, посчитанные на годах обучения, сами несут
    сведения о проверочных годах.
    """
    ze = G.zscore(d, cols, pd.Series(early, index=d.index))
    zl = G.zscore(d, cols, pd.Series(late, index=d.index))
    return np.where(late[:, None], zl, ze)


def arm(Z, y, yr, sel, tst, name, need=1.0):
    """Гребень на матрице Z, настройка по отложенным годам обучения. -> (прогноз, r).

    ``need`` — какая доля столбцов обязана быть заполнена. Для кодов это единица: код либо
    есть целиком, либо его нет. Для литературных признаков порог ниже (см. модуль).
    """
    fin = np.isfinite(Z)
    zok = fin.mean(axis=1) >= need
    Zf = np.where(fin, Z, 0.0)
    fit = sel & np.isfinite(y) & zok
    lam, r_in = R.ridge_loo(Zf, y, yr, fit)
    w = R.ridge_fit(Zf, np.where(fit, y, np.nan), lam)
    v = np.where(zok, Zf @ w, np.nan)
    r = R.per_year_r(v, yr, tst & np.isfinite(y), y)
    print('  %-16s гребень %6.0f, отложенные %+.4f, проверка %+.4f (плюс в %d из %d)'
          % (name, lam, r_in, r.mean(), int((r > 0).sum()), len(r)))
    return v, float(r.mean()), w


def channels(Z, w, v, tst, code):
    """Вклад каждого канала в предсказание: норма веса и разброс своей доли."""
    print('\n--- откуда сигнал: вес гребня и вклад в разброс по каналам ---')
    for j, c in enumerate(R.CHANNELS):
        sl = slice(j * code, (j + 1) * code)
        part = np.where(np.isfinite(Z[:, sl]).all(axis=1), Z[:, sl] @ w[sl], np.nan)
        print('  %-14s |вес| %.4f, вклад в разброс %.4f'
              % (c.replace('mean_', ''), np.linalg.norm(w[sl]), np.nanstd(part[tst])))
    print('  всего разброс предсказания %.4f' % np.nanstd(v[tst]))


def like_what(d, cols, v, yr, tst):
    """На что похоже предсказание: связь с литературными признаками внутри года."""
    print('\n--- на что похоже: корреляция внутри года, ПРОВЕРОЧНАЯ эпоха ---')
    out = []
    for c in cols:
        a = pd.Series(d[c].to_numpy(float))
        g = a.groupby(d['territory_id']).transform(lambda s: s - s.mean()).to_numpy()
        rr = R.per_year_r(g, yr, tst, v)
        if len(rr):
            out.append((c, float(rr.mean())))
    for c, rr in sorted(out, key=lambda t: -abs(t[1]))[:TOP_FEATURES]:
        print('  %-18s r %+.4f' % (c, rr))


def main():
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    warnings.filterwarnings('ignore', message='Degrees of freedom')
    ap = argparse.ArgumentParser(description='Разбор кода автоэнкодера сырых рядов')
    ap.add_argument('--culture', required=True)
    ap.add_argument('--split', type=int, default=2012, help='последний год ранней эпохи')
    ap.add_argument('--res', default='daily', choices=('6h', 'daily'))
    ap.add_argument('--code', type=int, default=R.CODE)
    ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args()

    cu = P.culture_row(a.culture)
    _hdoy, task = P.harvest_doy(cu['culture_id'])
    src = os.path.join(config.REPORTS_DIR, 'peak_residual', task + '_early', 'residuals.csv')
    if not os.path.exists(src):
        raise SystemExit('нет %s; сначала peak_residual_screen.py --early' % src)
    rl = pd.read_csv(src, sep=';', decimal=',', encoding='utf-8-sig')
    keys = rl[['territory_id', 'year']].reset_index(drop=True)
    tid, yr = keys['territory_id'].to_numpy(), keys['year'].to_numpy(int)
    early, late = yr <= a.split, yr > a.split
    era_of = np.where(early, 0, 1)
    ost = R.drop_district_era(rl['ост_norm'].to_numpy(float), tid, era_of)
    y = R.centre_in_year(ost, yr)

    cdir = os.path.join(config.DATASETS_DIR, 'raw_ae')
    A, cover = R.build_cube(keys, R.SUB[a.res], os.path.join(cdir, '%s_%s.npz' % (task, a.res)))
    ok = (cover >= R.MIN_COVER).all(axis=1)
    for s in np.unique(tid):
        for e in (0, 1):
            k = np.flatnonzero((tid == s) & (era_of == e))
            if k.size < R.MIN_PER_DISTRICT:
                ok[k] = False
    An = R.anomalies(A, tid, era_of)
    del A
    X = np.ascontiguousarray(np.where(np.isfinite(An), An, 0.0), dtype=np.float32)
    del An
    # учим на ПОЗДНЕЙ эпохе: только это плечо устояло на трёх затравках, в остальных
    # разбирать нечего
    sel, tst = late & ok, early & ok

    agp = os.path.join(config.REPORTS_DIR, 'agro_panel', task, 'features.csv')
    if not os.path.exists(agp):
        raise SystemExit('нет %s; сначала agro_panel.py --rebuild' % agp)
    f = pd.read_csv(agp, sep=';', decimal=',', encoding='utf-8-sig')
    d = keys.merge(f, on=['territory_id', 'year'], how='left')
    cols = [c for c in f.columns if c not in ('territory_id', 'year')]
    Zl = era_matrix(d, cols, early, late)
    print('%s: строк %d, литературных признаков %d, заполнено не меньше половины у %d'
          % (cu['title'], len(keys), len(cols),
             int((np.isfinite(Zl).mean(axis=1) >= NEED_LIT).sum())))

    print('\n--- плечи ---')
    vl, rlit, _wl = arm(Zl, y, yr, sel, tst, 'литература', NEED_LIT)
    res, first = [], True
    for sd in [int(v) for v in str(a.seeds).split(',') if v.strip()]:
        ae = R.AE(len(R.CHANNELS), a.code, X.shape[2], R.DEPTH[a.res])
        x, ch = R.flat(X, sel)
        R.train_ae(ae, x, ch, sd)
        Z = R.codes_of(ae, X, ok)
        va, ra, wa = arm(Z, y, yr, sel, tst, 'код з%d' % sd)
        _vb, rb, _wb = arm(np.column_stack([Z, Zl]), y, yr, sel, tst,
                           'код+литер з%d' % sd, NEED_LIT)
        both = np.isfinite(va) & np.isfinite(vl) & tst
        cc = float(pd.Series(va[both]).groupby(yr[both]).corr(pd.Series(vl[both])).mean())
        print('    связь кода с литературой внутри года на проверке %+.3f' % cc)
        res.append((sd, ra, rb, cc))
        if first:
            channels(np.where(np.isfinite(Z), Z, np.nan), wa, va, tst, a.code)
            like_what(d, cols, va, yr, tst)
            first = False

    print('\n--- один сигнал или два ---')
    print('литература одна %+.4f' % rlit)
    for sd, ra, rb, cc in res:
        print('затравка %d: код %+.4f, код+литература %+.4f, прибавка %+.4f, связь %+.3f'
              % (sd, ra, rb, rb - max(ra, rlit), cc))


if __name__ == '__main__':
    main()
