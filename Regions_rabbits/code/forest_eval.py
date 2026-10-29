"""Классические методы на резюмированных признаках: случайный лес и соседи по цеху.

Шаг назад от нейросетей. На 872 обучающих примерах лес имеет два преимущества перед сетью,
и оба существенны именно здесь:

1. Ему не нужна ранняя остановка по валидации. Вся морока с «тихим и громким провалом»
   (разделы 7 и 9 протокола) происходила из того, что сеть выбирала эпоху по валидации, а
   валидация то помогала, то вредила. У леса нет эпох: он усредняет деревья, и чем их больше,
   тем стабильнее, без риска переобучиться по числу деревьев.
2. Он сам показывает, что берёт. Важность по перестановке (permutation importance) на
   валидации — честная оценка: признак перемешивается, смотрим, насколько испортился прогноз
   на данных, которых модель не видела.

Признаки — из ``series_features.py`` (окна по фазам развития, разности с прошлым сезоном).

Протокол тот же, что у сетей, чтобы цифры были сопоставимы: фолды из ``folds.py``, цель —
отклонение от скользящей средней района в процентах, метрики на абсолютной шкале, эталон —
сама скользящая средняя. Гиперпараметры выбираются по валидационной части фолда, тест
смотрится один раз.

    python code/forest_eval.py --name ww_forest --scheme ww_blocks
    python code/forest_eval.py --name ww_forest_lad --scheme ww_ladder \\
           --folds 2019|random 2019|double 2020|random 2020|double

Отчёты: ``runs.csv`` (строка на модель и фолд) и ``importance.csv`` (важность признаков).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import agro_features as AF
import config
import ensemble_single as ES
import explore_models as E
import folds as FD
import run_experiment as R
import series_features as SF

# Наборы признаков живут в двух модулях и в двух таблицах: имена окон у них общие
# («зима_зимобой» есть и там и там), поэтому таблицы не сливаются, а выбираются по набору.
FEATURE_MODULE = {**{fs: SF for fs in SF.FEATURE_SETS}, **{fs: AF for fs in AF.FEATURE_SETS}}

RUN_COLUMNS = ['task', 'fold', 'model', 'features', 'cutoff', 'seed', 'n_feat', 'n_train',
               'n_val', 'n_test', 'select', 'params', 'shrink', 'sel_rmse', 'val_rmse',
               'base_val_rmse', 'test_rmse', 'test_r2', 'base_rmse', 'base_r2', 'minutes',
               'finished_at']
IMP_COLUMNS = ['fold', 'model', 'features', 'feature', 'важность', 'σ']
YEAR_COLUMNS = ['fold', 'model', 'features', 'select', 'seed', 'test_year', 'n',
                'test_rmse', 'base_rmse', 'выигрыш']

MODELS = ('rf', 'et', 'hgb', 'ridge')


def _append(path, row, columns):
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, sep=';', decimal=',', index=False, encoding='utf-8-sig',
              mode='a' if os.path.exists(path) else 'w', header=not os.path.exists(path))


SHRINK = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SELECT = ('val', 'group', 'both')


def grid(model):
    """Небольшая сетка гиперпараметров."""
    if model in ('rf', 'et'):
        return [{'min_samples_leaf': lf, 'max_features': mf}
                for lf in (1, 3, 8, 20) for mf in ('sqrt', 0.3)]
    if model == 'hgb':
        return [{'max_leaf_nodes': n, 'learning_rate': lr, 'min_samples_leaf': 20}
                for n in (7, 15, 31) for lr in (0.03, 0.08)]
    if model == 'ridge':
        return [{'alpha': al} for al in np.logspace(-1, 5, 13)]
    return [{}]


def build(model, params, seed):
    if model == 'rf':
        return RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1, **params)
    if model == 'et':
        return ExtraTreesRegressor(n_estimators=400, random_state=seed, n_jobs=-1, **params)
    if model == 'hgb':
        return HistGradientBoostingRegressor(max_iter=400, random_state=seed,
                                             early_stopping=False, **params)
    return make_pipeline(StandardScaler(), Ridge(random_state=None, **params))


def oof_by_group(model, params, seed, X, fit_y, groups, n_splits=5):
    """Прогноз вне обучения с группировкой по РАЙОНУ: имитирует условия теста.

    Отбор гиперпараметров по валидационной части фолда для блочной схемы нечестен в одну
    сторону: валидация — это другой ГОД тех же районов, поэтому вознаграждает запоминание
    района, а в тесте районы чужие. Группировка по району в обучающей части воспроизводит
    именно это условие и при этом не касается ни валидации, ни теста.
    """
    uniq = np.unique(groups)
    k = int(min(n_splits, len(uniq)))
    if k < 2:
        return None
    oof = np.full(len(X), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=k).split(X, fit_y, groups=groups):
        m = build(model, params, seed)
        m.fit(X[tr_i], fit_y[tr_i])
        oof[te_i] = m.predict(X[te_i])
    return oof


def shrink_rmse(pred_dev, y_true, ph_true, clip):
    """RMSE на абсолютной шкале при каждом коэффициенте доверия λ. -> {λ: rmse}."""
    out = {}
    for lam in SHRINK:
        p = ES.to_absolute(np.asarray(pred_dev, float) * lam, ph_true, clip)
        out[lam] = float(E.metrics(y_true, p)['RMSE'])
    return out


def align(table, tid, years, columns):
    """Матрица признаков в порядке строк датасета. -> (X, доля найденных)."""
    key = pd.MultiIndex.from_arrays([table['territory_id'].astype(str),
                                     table['year'].astype(int)])
    t = table.set_index(key)
    want = pd.MultiIndex.from_arrays([pd.Index(tid).astype(str), pd.Index(years).astype(int)])
    sub = t.reindex(want)
    return sub[columns].to_numpy(dtype=np.float64), float(sub[columns[0]].notna().mean())


def main():
    ap = argparse.ArgumentParser(description='Лес и классика на резюмированных признаках')
    ap.add_argument('--name', default='ww_forest')
    ap.add_argument('--scheme', default='ww_blocks')
    ap.add_argument('--task', default='winter_wheat_raw_v2')
    ap.add_argument('--folds', nargs='*', default=None)
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--features', nargs='+', default=['full'],
                    choices=list(SF.FEATURE_SETS) + list(AF.FEATURE_SETS))
    ap.add_argument('--cutoff', type=int, default=0,
                    help='заблаговременность: день года, после которого данные не берутся')
    ap.add_argument('--select', default='both', choices=list(SELECT),
                    help='чем выбирать гиперпараметры и λ: val — валидационный год фолда, '
                         'group — районная кросс-проверка внутри обучения, both — среднее')
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=5, help='повторов в важности по перестановке')
    ap.add_argument('--no-importance', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    cutoff = a.cutoff or None
    scheme = FD.read_definition(a.scheme)
    fold_names = a.folds or sorted(str(x) for x in scheme['fold'].unique())
    specs = [FD.fold_spec(scheme, f) for f in fold_names]
    out = os.path.join(config.REPORTS_DIR, 'experiments', a.name)
    os.makedirs(out, exist_ok=True)
    runs_path = os.path.join(out, 'runs.csv')
    imp_path = os.path.join(out, 'importance.csv')
    years_path = os.path.join(out, 'years.csv')
    done = set()
    if os.path.exists(runs_path):
        prev = pd.read_csv(runs_path, sep=';', decimal=',', encoding='utf-8-sig')
        if 'select' not in prev.columns:
            prev['select'] = 'val'
        done = set(zip(prev['fold'].astype(str), prev['model'].astype(str),
                       prev['features'].astype(str), prev['select'].astype(str),
                       prev['seed'].astype(int)))

    plan = [(str(s['fold']), m, fs, a.select, sd) for s in specs for m in a.models
            for fs in a.features for sd in range(1, a.seeds + 1)]
    todo = [p for p in plan if p not in done]
    print(f'Опыт «{a.name}»: фолдов {len(specs)}, моделей {len(a.models)}, '
          f'наборов признаков {len(a.features)}, seed {a.seeds} -> прогонов {len(plan)}, '
          f'осталось {len(todo)}')
    print(f'   схема {a.scheme}; отсечка {cutoff or "весь сезон"}; отбор по «{a.select}»; '
          f'цель — отклонение от скользящей средней, допускается усадка λ')
    if a.dry_run:
        return
    with open(os.path.join(out, 'experiment.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': a.name, 'scheme': a.scheme, 'task': a.task, 'models': a.models,
                   'features': a.features, 'cutoff': a.cutoff, 'folds': fold_names,
                   'seeds': a.seeds, 'target': 'ratio', 'select': a.select,
                   'shrink': list(SHRINK)}, f, ensure_ascii=False, indent=2)

    tables = {m: m.table_for(a.task, cutoff) for m in set(FEATURE_MODULE[fs] for fs in a.features)}
    ds = R.Dataset(a.task, 'raw', 0)
    years = ds.ids['year'].to_numpy(dtype=int)
    table_by_set = {fs: tables[FEATURE_MODULE[fs]] for fs in a.features}
    cols_by_set = {fs: FEATURE_MODULE[fs].select_columns(table_by_set[fs], fs)
                   for fs in a.features}
    t_start = time.time()

    for i, (fold, model, fs, select, seed) in enumerate(todo, 1):
        t0 = time.time()
        spec = next(s for s in specs if str(s['fold']) == fold)
        _Xn, y, ph, tr, va, te = ds.fold_arrays(spec)
        columns = cols_by_set[fs]
        X, found = align(table_by_set[fs], ds.tid, years, columns)
        if found < 0.999:
            print(f'   ВНИМАНИЕ: признаки найдены лишь для {found:.1%} примеров')
        med = np.nanmedian(X[tr], axis=0)                 # пропуски — медианой ОБУЧЕНИЯ
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med)
        keep = X[tr].std(axis=0) > 0                      # постоянные на обучении — выбросить
        X, kept = X[:, keep], [c for c, k in zip(columns, keep) if k]
        fit_y, clip = ES.ratio_target(y, ph, tr)
        groups = ds.tid[tr]

        best = None
        for params in grid(model):
            score = {}
            if select in ('val', 'both'):
                m = build(model, params, seed)
                m.fit(X[tr], fit_y[tr])
                score['val'] = shrink_rmse(m.predict(X[va]), y[va], ph[va], clip)
            if select in ('group', 'both'):
                oof = oof_by_group(model, params, seed, X[tr], fit_y[tr], groups)
                if oof is None:
                    continue
                ok = np.isfinite(oof)
                score['group'] = shrink_rmse(oof[ok], y[tr][ok], ph[tr][ok], clip)
            for lam in SHRINK:
                vals = [sc[lam] for sc in score.values()]
                sel = float(np.mean(vals))
                if best is None or sel < best[0]:
                    best = (sel, params, lam)
        sel_rmse, best_params, best_lam = best

        final = build(model, best_params, seed)
        final.fit(X[tr], fit_y[tr])
        pred = ES.to_absolute(final.predict(X[te]) * best_lam, ph[te], clip)
        pred_va = ES.to_absolute(final.predict(X[va]) * best_lam, ph[va], clip)
        mt, base = E.metrics(y[te], pred), E.metrics(y[te], ph[te])
        mv, base_v = E.metrics(y[va], pred_va), E.metrics(y[va], ph[va])
        _append(runs_path, {
            'task': a.task, 'fold': fold, 'model': model, 'features': fs, 'cutoff': a.cutoff,
            'seed': seed, 'n_feat': len(kept), 'n_train': len(tr), 'n_val': len(va),
            'n_test': len(te), 'select': select,
            'params': json.dumps(best_params, ensure_ascii=False), 'shrink': best_lam,
            'sel_rmse': round(sel_rmse, 4), 'val_rmse': round(float(mv['RMSE']), 4),
            'base_val_rmse': round(float(base_v['RMSE']), 4),
            'test_rmse': round(float(mt['RMSE']), 4), 'test_r2': round(float(mt['R2']), 4),
            'base_rmse': round(float(base['RMSE']), 4), 'base_r2': round(float(base['R2']), 4),
            'minutes': round((time.time() - t0) / 60, 1),
            'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, RUN_COLUMNS)
        print(f'[{i}/{len(todo)}] {fold}, {model}, {fs}, seed {seed}: признаков {len(kept)}, '
              f'{best_params}, λ={best_lam} -> отбор {sel_rmse:.2f}, '
              f'тест RMSE {mt["RMSE"]:.2f} / R² {mt["R2"]:.3f} (эталон {base["RMSE"]:.2f}), '
              f'{(time.time() - t0) / 60:.1f} мин', flush=True)

        # Разбивка тестовой ошибки по году. Средний выигрыш по фолду считается по всем его
        # тестовым годам сразу, и уровни лет частично гасят друг друга; лесенка на одиночных
        # годах выигрыша не находит. Эти строки разводят два утверждения на одних прогонах.
        for yr in np.unique(years[te]):
            sub = years[te] == yr
            if sub.sum() < 20:                            # слишком мало, RMSE не оценивается
                continue
            r_mod = float(E.metrics(y[te][sub], pred[sub])['RMSE'])
            r_base = float(E.metrics(y[te][sub], ph[te][sub])['RMSE'])
            _append(years_path, {
                'fold': fold, 'model': model, 'features': fs, 'select': select, 'seed': seed,
                'test_year': int(yr), 'n': int(sub.sum()), 'test_rmse': round(r_mod, 4),
                'base_rmse': round(r_base, 4),
                'выигрыш': round(r_base - r_mod, 4)}, YEAR_COLUMNS)

        if not a.no_importance:
            # Важность по перестановке на ВАЛИДАЦИИ: данные, которых модель не видела, но и
            # не тест. Так таблица важностей не становится скрытым подглядыванием в тест.
            r = permutation_importance(final, X[va], fit_y[va], n_repeats=a.repeats,
                                       random_state=seed, n_jobs=-1)
            order = np.argsort(-r.importances_mean)[:40]
            for j in order:
                _append(imp_path, {'fold': fold, 'model': model, 'features': fs,
                                   'feature': kept[j],
                                   'важность': round(float(r.importances_mean[j]), 4),
                                   'σ': round(float(r.importances_std[j]), 4)}, IMP_COLUMNS)

    if not todo:
        print('нечего считать: всё уже есть в runs.csv')
        return
    print(f'\nвсего {(time.time() - t_start) / 60:.1f} мин')

    d = pd.read_csv(runs_path, sep=';', decimal=',', encoding='utf-8-sig')
    print()
    print('=== RMSE по модели и набору признаков (среднее по фолдам)')
    p = d.pivot_table(index=['model', 'features'],
                      values=['test_rmse', 'base_rmse', 'shrink'], aggfunc='mean')
    p['выигрыш'] = p['base_rmse'] - p['test_rmse']
    print(p.round(3).sort_values('test_rmse').to_string())
    print()
    print('=== по фолдам')
    print(d.pivot_table(index='fold', columns=['model', 'features'],
                        values='test_rmse').round(2).to_string())
    if os.path.exists(imp_path) and not a.no_importance:
        imp = pd.read_csv(imp_path, sep=';', decimal=',', encoding='utf-8-sig')
        print()
        print('=== важнейшие признаки (по перестановке на валидации, среднее по фолдам)')
        s = imp.groupby('feature')['важность'].agg(['mean', 'count'])
        print(s[s['count'] >= 2].sort_values('mean', ascending=False).head(25).round(3).to_string())


if __name__ == '__main__':
    main()
