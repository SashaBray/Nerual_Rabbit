"""Переносятся ли найденные признаки между эпохами: отбор на одних годах, проверка на других.

Зачем нужен отдельный опыт. Перебор остатка пика (``peak_residual_screen.py``) отбирает
признаки и считает результат на одних и тех же годах, поэтому его лучшие числа — верхняя
оценка, а не проверка. До 2026-11-18 это нельзя было разделить: независимых лет было 11, и
отрезать половину было не на чем. После докачки 2007-2013 (см. ``download_6h.py``) лет стало
18, и появились две эпохи сравнимого веса:

* **ранняя** 2007-2012 — 2078 районо-лет, около 350 районов в год;
* **поздняя** 2013-2024 — 7601 районо-лет, около 630 районов в год.

Как считается. Признак приводится к z-оценке по своему району — так же, как в переборе,
потому что модель отвечает на вопрос «этот район в этом году лучше или хуже своего
обычного». Норма района берётся ТОЛЬКО по годам своей эпохи: иначе знание о проверочных
годах протекает в отбор через среднее и разброс района. Дальше на эпохе отбора берутся
``K`` признаков с наибольшей средней корреляцией по годам, из них складывается сумма с
равными весами и знаками эпохи отбора, и эта сумма проверяется на другой эпохе.

Контроль обязателен: тот же отбор прогоняется на перемешанном остатке (годы внутри района
переставлены). Сравнивать результат надо не с нулём, а с этим контролем — отбор лучшего из
двух сотен кандидатов сам по себе даёт заметное число.

Запуск (нужен отчёт прогона с ``--early``):
    python code/epoch_transfer.py --culture "Озимая пшеница"
    python code/epoch_transfer.py --culture "Яровая пшеница" --k 6 --split 2012
"""
import argparse
import os

import numpy as np
import pandas as pd

import config
import peak_regression as P

MIN_PER_DISTRICT = 3      # меньше трёх лет — норму района считать не из чего
MIN_PER_YEAR = 30         # меньше тридцати районов в году — корреляция не считается
MIN_ROWS = 200            # признак без покрытия в обеих эпохах сравнивать нельзя


def load(culture):
    """Остатки прогона с ``--early``: там и признаки, и цель, и годы 2007-2024."""
    cu = P.culture_row(culture)
    _hdoy, task = P.harvest_doy(cu['culture_id'])
    path = os.path.join(config.REPORTS_DIR, 'peak_residual', task + '_early',
                        'residuals.csv')
    if not os.path.exists(path):
        raise SystemExit(f'нет {path}; сначала прогнать peak_residual_screen.py --early')
    return pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig'), str(cu['title'])


def zscore(rl, cols, era):
    """z-оценка признаков по району в пределах эпохи; вне эпохи NaN."""
    g = rl.loc[era].groupby('territory_id')[cols]
    mu, sd, n = g.transform('mean'), g.transform('std'), g.transform('count')
    z = (rl.loc[era, cols] - mu) / sd.where(sd > 1e-12)
    out = pd.DataFrame(np.nan, index=rl.index, columns=cols)
    out.loc[era, cols] = z.where(n >= MIN_PER_DISTRICT)
    return out.to_numpy(float)


def per_year_r(v, yr, era, target):
    """Корреляция с остатком ВНУТРИ каждого года эпохи."""
    out = []
    for g in sorted(set(yr[era])):
        m = (yr == g) & np.isfinite(v) & np.isfinite(target)
        if m.sum() < MIN_PER_YEAR or np.std(v[m]) < 1e-12:
            continue
        out.append(float(np.corrcoef(v[m], target[m])[0, 1]))
    return np.array(out)


def composite(Z, pick, sign):
    """Сумма отобранных признаков с равными весами; знак каждого — из эпохи отбора."""
    a = np.where(np.isfinite(Z[:, pick]), Z[:, pick], 0.0) * sign
    return a.mean(axis=1)


def transfer(cols, yr, Zs, sel, Zt, tst, target, k):
    """Отобрать k признаков на одной эпохе, проверить сумму на другой."""
    score = np.full(len(cols), np.nan)
    for j in range(len(cols)):
        if min(np.isfinite(Zs[:, j]).sum(), np.isfinite(Zt[:, j]).sum()) < MIN_ROWS:
            continue
        r = per_year_r(Zs[:, j], yr, sel, target)
        if len(r):
            score[j] = r.mean()
    ok = np.flatnonzero(np.isfinite(score))
    if len(ok) < k:
        raise SystemExit('слишком мало признаков с покрытием в обеих эпохах')
    pick = ok[np.argsort(-np.abs(score[ok]))][:k]
    sign = np.sign(score[pick])
    return (pick, sign, score,
            per_year_r(composite(Zs, pick, sign), yr, sel, target),
            per_year_r(composite(Zt, pick, sign), yr, tst, target))


def main():
    ap = argparse.ArgumentParser(description='Перенос признаков между эпохами')
    ap.add_argument('--culture', required=True)
    ap.add_argument('--split', type=int, default=2012, help='последний год ранней эпохи')
    ap.add_argument('--k', type=int, default=4, help='сколько признаков берёт сумма')
    a = ap.parse_args()

    rl, title = load(a.culture)
    cols = [c for c in rl.columns if '|' in c]
    tid, yr = rl['territory_id'].to_numpy(), rl['year'].to_numpy(int)
    y = rl['ост_norm'].to_numpy(float)
    early, late = yr <= a.split, yr > a.split
    if min(early.sum(), late.sum()) < 500:
        raise SystemExit('одна из эпох слишком мала для сравнения')
    print(f'{title}: строк {len(rl)}, районов {len(set(tid))}, кандидатов {len(cols)}')
    print(f'ранняя {yr[early].min()}-{a.split}: {early.sum()} строк, '
          f'{len(set(tid[early]))} районов; поздняя {a.split + 1}-{yr[late].max()}: '
          f'{late.sum()} строк, {len(set(tid[late]))} районов')

    ZE, ZL = zscore(rl, cols, early), zscore(rl, cols, late)
    rng = np.random.default_rng(5)
    ysh = y.copy()
    for t in np.unique(tid):
        k = np.flatnonzero(tid == t)
        ysh[k] = y[rng.permutation(k)]

    rows = []
    for tgt, nm in ((y, 'данные'), (ysh, 'шум')):
        for Zs, sel, Zt, tst, lbl in ((ZE, early, ZL, late, 'ранняя -> поздняя'),
                                      (ZL, late, ZE, early, 'поздняя -> ранняя')):
            pick, sign, score, r_sel, r_tst = transfer(cols, yr, Zs, sel, Zt, tst,
                                                       tgt, a.k)
            if nm == 'данные':
                print(f'\n{lbl}: отобраны')
                for j, s in zip(pick, sign):
                    print('   %-34s знак %+d, на отборе r %+.4f' % (cols[j], s, score[j]))
            rows.append({'что': nm, 'направление': lbl,
                         'r_отбор': round(float(r_sel.mean()), 4),
                         'r_проверка': round(float(r_tst.mean()), 4),
                         'лет_проверки': len(r_tst),
                         'знак_совпал': int((r_tst > 0).sum())})
    t = pd.DataFrame(rows)
    print('\n=== перенос суммы %d признаков ===' % a.k)
    print(t.to_string(index=False))
    д = t[t['что'] == 'данные']['r_проверка'].mean()
    ш = t[t['что'] == 'шум']['r_проверка'].mean()
    print(f'\nсредняя проверка: на данных {д:+.4f}, на шуме {ш:+.4f}')
    print('вывод: перенос ' + ('есть' if д > ш + 0.02 else 'НЕ отличим от шума'))


if __name__ == '__main__':
    main()
