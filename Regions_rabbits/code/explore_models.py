"""Разведочный анализ датасета + базовые ML-модели (Linear / Tree / RandomForest).

Запуск:
    python code/explore_models.py rye_winter_all
    python code/explore_models.py rye_winter_all --length 10 --lengths 3,5,10,20,50,100

Что делает:
1. Находит датасет задания (``workspace/datasets/<task>``: matrix.csv, scalar.csv, meta.json).
2. Каждый пример = матрица (признак × время). ДОБАВЛЯЕТ строку, целиком заполненную
   скользящим средним урожайности (``historical_productive`` из scalar) — как ещё один признак.
3. Перемешивает и делит на train / test / val = 70% / 15% / 15%.
4. По TRAIN считает параметры нормировки (z-score на признак) и сохраняет их в файл
   (``normalization.json``) — их же используют будущие модели. Нормирует все выборки.
5. Сжимает временные ряды усреднением до L точек (по умолчанию свип L=3..100, график RMSE(L)).
   Матрица разворачивается в вектор -> вход классических моделей. Таргет — абсолютная
   урожайность (выход НЕ нормируется).
6. Метрики MSE/RMSE/R²/Pearson, графики (scatter pred↔real, гистограммы), важность признаков.
7. Таблицы погрешностей на test и val (id_region, id_district, year, real, pred, откл., откл.²).
8. Авто-отчёт ``report_<task>.docx`` со всеми графиками и таблицами.

Артефакты: ``workspace/reports/experiments/<task>/``.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import config
import plots

DEFAULT_LENGTHS = [3, 5, 8, 12, 18, 25, 40, 60, 80, 100]
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15
# Режим сплита: 'random' (случайно по примерам, 70/15/15) или
# 'year' (leave-one-year-out: test=последний год, val=предпоследний, train=ранние).
# Сплит по годам убирает «память района» (район не делится между train и test по разным годам)
# и проверяет настоящий прогноз будущего года.
SPLIT_MODE = 'random'
YIELD_MAX = 5000.0   # ц/га; больше — сентинел пропуска БДПМО (999999999) или мусор


# --------------------------------------------------------------------------- #
# Загрузка датасета
# --------------------------------------------------------------------------- #
def load_dataset(task):
    """Вернуть (X, y, prod_hist, ids, meta, feature_names).

    X — (n, n_features, time_len), y — реальная урожайность (productive),
    prod_hist — скользящее среднее (historical_productive),
    ids — DataFrame[id_region, id_district, year].
    """
    ddir = config.dataset_dir(task)
    meta_path = os.path.join(ddir, 'meta.json')
    if not os.path.exists(meta_path):
        raise SystemExit(f'Нет датасета {task}: {meta_path} не найден. Соберите его build_dataset.py.')
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    n_features = int(meta['n_features'])
    time_len = int(meta['time_rows_size']) * int(meta['duration_years'])
    feature_names = list(meta.get('features') or [f'f{i}' for i in range(n_features)])

    # scalar.csv (числовой, без заголовка): id_region, id_district, year, productive, hist_productive
    scalar = np.loadtxt(os.path.join(ddir, 'scalar.csv'))
    if scalar.ndim == 1:
        scalar = scalar.reshape(1, -1)
    ids = pd.DataFrame({
        'id_region': scalar[:, 0].astype(int),
        'id_district': scalar[:, 1].astype(int),
        'year': scalar[:, 2].astype(int),
    })
    y = scalar[:, 3].astype(np.float64)
    prod_hist = scalar[:, 4].astype(np.float64)

    # matrix.csv: строка = пример, длина n_features*time_len (разделитель — пробел)
    mat = pd.read_csv(os.path.join(ddir, 'matrix.csv'), sep=' ', header=None,
                      dtype=np.float32, engine='c').dropna(axis=1, how='all').values
    width = n_features * time_len
    if mat.shape[1] != width:
        raise SystemExit(f'Ширина matrix.csv {mat.shape[1]} != n_features*time_len {width}.')
    if mat.shape[0] != len(y):
        raise SystemExit(f'Строк matrix {mat.shape[0]} != scalar {len(y)}.')
    X = mat.reshape(mat.shape[0], n_features, time_len)
    if np.isnan(X).any():                       # подстраховка (паддинг — нули, NaN не ожидается)
        print(f'      ! в matrix.csv {int(np.isnan(X).sum())} NaN — заменяю нулями')
        X = np.nan_to_num(X, nan=0.0)
    return X, y, prod_hist, ids, meta, feature_names


def drop_corrupt(Xb, y, prod_hist, ids):
    """Отбросить битые примеры (misalignment в build: значения чужих признаков попали в
    позиции другого — напр. давление ~1000 в строке ndvi). Робастно: по каждому признаку
    медиана/MAD, пример битый, если у какого-то признака >2% точек вылетают за 50·MAD.
    """
    F = Xb.shape[1]
    # Потолок по признаку = 5× от 95-перцентиля его ПИКОВ по примерам (max по времени,
    # только среди ненулевых пиков). Устойчиво к загрязнению (битые строки — крайний верх,
    # 95-перцентиль их не видит) и к прерывистым признакам (снег/осадки: потолок берётся
    # по реальному пику, а не по медиане). Пример битый, если ЛЮБОЙ признак выбивает потолок
    # (значение чужого признака — давление/temp_acc — попало в позицию мелкого).
    rowmax = Xb.max(axis=2)                      # (n, F): пик каждого признака на пример
    ceil = np.empty(F)
    for f in range(F):
        nz = rowmax[:, f][rowmax[:, f] > 0]
        ceil[f] = 5.0 * (np.percentile(nz, 95) if nz.size else 0.0) + 1e-9
    corrupt = (rowmax > ceil.reshape(1, F)).any(axis=1)
    keep = ~corrupt
    return (Xb[keep], y[keep], prod_hist[keep], ids[keep].reset_index(drop=True),
            int(corrupt.sum()))


def add_prod_hist_row(X, prod_hist, feature_names):
    """Добавить признак-строку, целиком заполненную скользящим средним урожайности."""
    n, _f, t = X.shape
    extra = np.repeat(prod_hist.reshape(n, 1, 1), t, axis=2).astype(X.dtype)  # (n,1,t)
    X2 = np.concatenate([X, extra], axis=1)
    return X2, feature_names + ['prod_hist_ma']


# --------------------------------------------------------------------------- #
# Сплит, нормировка, сжатие
# --------------------------------------------------------------------------- #
def split_indices(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * SPLIT_VAL))
    n_test = int(round(n * SPLIT_TEST))
    val = idx[:n_val]
    test = idx[n_val:n_val + n_test]
    train = idx[n_val + n_test:]
    return train, test, val


def split_by_year(years, seed):
    """Leave-one-year-out: test = последний год, val = предпоследний, train = ранние.

    Внутри каждой группы порядок перемешивается seed-ом (для воспроизводимости),
    но разбиение задаётся годом, а не случайно — район не «перетекает» между train и test.
    """
    years = np.asarray(years)
    uniq = np.sort(np.unique(years))
    if len(uniq) < 3:
        raise ValueError(f'Для сплита по годам нужно >=3 года, есть {len(uniq)}: {uniq}')
    test_y, val_y = uniq[-1], uniq[-2]
    rng = np.random.default_rng(seed)
    test = rng.permutation(np.where(years == test_y)[0])
    val = rng.permutation(np.where(years == val_y)[0])
    train = rng.permutation(np.where(years < val_y)[0])
    return train, test, val, (int(uniq[0]), int(val_y), int(test_y))


def fit_normalization(X_train, feature_names):
    """Per-feature z-score по train: среднее/СКО по всем примерам и точкам времени."""
    mean = X_train.mean(axis=(0, 2))
    std = X_train.std(axis=(0, 2))
    std = np.where(std < 1e-8, 1.0, std)
    params = {'features': feature_names,
              'mean': mean.astype(float).tolist(),
              'std': std.astype(float).tolist()}
    return params


def apply_normalization(X, params):
    mean = np.asarray(params['mean'], dtype=np.float32).reshape(1, -1, 1)
    std = np.asarray(params['std'], dtype=np.float32).reshape(1, -1, 1)
    return (X - mean) / std


def downsample(X, length):
    """Усреднить временную ось до ``length`` точек: (n,F,T) -> (n,F,length)."""
    groups = np.array_split(np.arange(X.shape[2]), length)
    return np.stack([X[:, :, g].mean(axis=2) for g in groups], axis=2)


def to_vectors(Xc):
    """(n,F,L) -> (n, F*L). Колонка f*L+b — признак f, бин b."""
    return Xc.reshape(Xc.shape[0], -1)


# --------------------------------------------------------------------------- #
# Модели и метрики
# --------------------------------------------------------------------------- #
def make_models(trees, seed):
    return {
        'linear': LinearRegression(),
        'tree': DecisionTreeRegressor(random_state=seed),
        'forest': RandomForestRegressor(n_estimators=trees, n_jobs=-1, random_state=seed),
    }


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float('nan')
    return {'MSE': float(mse), 'RMSE': rmse, 'R2': float(r2), 'R_pearson': r}


# --------------------------------------------------------------------------- #
# Свип по длине L
# --------------------------------------------------------------------------- #
def length_sweep(Xn_tr, y_tr, Xn_val, y_val, lengths, trees, seed, outdir):
    rows = []
    for L in lengths:
        Xtr = to_vectors(downsample(Xn_tr, L))
        Xv = to_vectors(downsample(Xn_val, L))
        for name, mdl in make_models(trees, seed).items():
            mdl.fit(Xtr, y_tr)
            m = metrics(y_val, mdl.predict(Xv))
            rows.append({'length': L, 'model': name, **m})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(outdir, 'lengths_sweep.csv'), index=False, encoding='utf-8-sig')

    def draw(lang):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ['linear', 'tree', 'forest']:
            sub = sweep[sweep.model == name].sort_values('length')
            ax.plot(sub.length, sub.RMSE, marker='o', label=name)
        ax.set_xlabel(plots.tr('Длина сжатого ряда L', 'Compressed series length L', lang))
        ax.set_ylabel(plots.tr('RMSE (валидация)', 'RMSE (validation)', lang))
        ax.set_title(plots.tr('Влияние длины сжатия на RMSE', 'Effect of compression length on RMSE', lang))
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    path = plots.bilingual(outdir, 'lengths_sweep', draw)
    # лучшая L по средней RMSE моделей
    best_L = int(sweep.groupby('length').RMSE.mean().idxmin())
    return sweep, best_L, path


# --------------------------------------------------------------------------- #
# Графики и таблицы финального прогона
# --------------------------------------------------------------------------- #
def scatter_plot(y_true, y_pred, name, outdir):
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))

    def draw(lang):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(y_true, y_pred, s=8, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        ax.set_xlabel(plots.tr('Реальная урожайность', 'Actual yield', lang))
        ax.set_ylabel(plots.tr('Предсказанная урожайность', 'Predicted yield', lang))
        ax.set_title(plots.tr(f'{name}: предсказание vs реальность (test)',
                              f'{name}: predicted vs actual (test)', lang))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    return plots.bilingual(outdir, f'scatter_{name}', draw)


def hist_plot(y_true, y_pred, name, outdir):
    bins = np.linspace(float(min(y_true.min(), y_pred.min())),
                       float(max(y_true.max(), y_pred.max())), 40)

    def draw(lang):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(y_true, bins=bins, alpha=0.5, label=plots.tr('реальная', 'actual', lang))
        ax.hist(y_pred, bins=bins, alpha=0.5, label=plots.tr('предсказанная', 'predicted', lang))
        ax.set_xlabel(plots.tr('Урожайность', 'Yield', lang))
        ax.set_ylabel(plots.tr('Частота', 'Frequency', lang))
        ax.set_title(plots.tr(f'{name}: распределение (test)', f'{name}: distribution (test)', lang))
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    return plots.bilingual(outdir, f'hist_{name}', draw)


def aggregate_importance(values, feature_names, length):
    """Свернуть важность по колонкам (F*L) к исходным признакам (сумма по L бинам)."""
    arr = np.asarray(values).reshape(len(feature_names), length)
    per = arr.sum(axis=1)
    df = pd.DataFrame({'feature': feature_names, 'importance': per})
    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def error_table(ids_split, y_true, y_pred):
    dev = y_true - y_pred
    return pd.DataFrame({
        'id_region': ids_split['id_region'].values,
        'id_district': ids_split['id_district'].values,
        'year': ids_split['year'].values,
        'real_yield': np.round(y_true, 3),
        'pred_yield': np.round(y_pred, 3),
        'deviation': np.round(dev, 3),
        'deviation_sq': np.round(dev ** 2, 3),
    })


# --------------------------------------------------------------------------- #
# DOCX-отчёт
# --------------------------------------------------------------------------- #
def build_docx(task, meta, length, best_L, n_split, metrics_df, imp_lin, imp_rf,
               sweep_png, model_imgs, outdir):
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading(f'Эксперимент ML: {task}', level=0)
    doc.add_paragraph(f'Культура: {meta.get("culture_title", "")} '
                      f'(маска: {meta.get("mask_culture", "")})')
    doc.add_paragraph(f'Признаков: {meta["n_features"]} (+1 строка скользящего среднего '
                      f'урожайности); длина ряда: {meta["time_rows_size"]}×'
                      f'{meta["duration_years"]} точек.')
    doc.add_paragraph(f'Разбиение: train={n_split[0]}, test={n_split[1]}, val={n_split[2]} '
                      f'(70/15/15). Длина сжатия L={length} '
                      f'(лучшая по свипу: {best_L}). Выход НЕ нормируется.')

    doc.add_heading('Метрики моделей', level=1)
    _table_from_df(doc, metrics_df)

    doc.add_heading('Влияние длины сжатия L на RMSE', level=1)
    doc.add_picture(sweep_png, width=Inches(6))

    for name in ['linear', 'tree', 'forest']:
        doc.add_heading(f'Модель: {name}', level=1)
        for img in model_imgs.get(name, []):
            doc.add_picture(img, width=Inches(5.5))

    doc.add_heading('Важность признаков — линейная модель (|вес|)', level=1)
    _table_from_df(doc, imp_lin)
    doc.add_heading('Важность признаков — случайный лес', level=1)
    _table_from_df(doc, imp_rf)

    path = os.path.join(outdir, f'report_{task}.docx')
    doc.save(path)
    return path


def _table_from_df(doc, df, maxrows=40):
    df = df.head(maxrows)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = 'Light Grid Accent 1'
    for j, c in enumerate(df.columns):
        t.rows[0].cells[j].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(df.columns):
            v = row[c]
            cells[j].text = f'{v:.4g}' if isinstance(v, (int, float, np.floating)) else str(v)


# --------------------------------------------------------------------------- #
# Подготовка данных (общая для разведки и нейросети)
# --------------------------------------------------------------------------- #
def prepare_data(task, seed):
    """Загрузить, очистить, разбить и нормировать датасет (как в разведке).

    Возвращает dict: Xn (n,F,T нормированная, со строкой prod_hist), y (абсолютная урожайность,
    НЕ нормируется), ids (id_region/id_district/year), индексы tr/te/va (70/15/15 по seed),
    norm (параметры нормировки по train), feature_names, meta, counts.
    """
    Xb, y, prod_hist, ids, meta, feature_names = load_dataset(task)
    keep = ~np.isnan(y)                                    # без метки урожайности
    dropped = int((~keep).sum())
    Xb, y, prod_hist = Xb[keep], y[keep], prod_hist[keep]
    ids = ids[keep].reset_index(drop=True)
    ok = (y >= 0) & (y <= YIELD_MAX) & (np.isnan(prod_hist) | (prod_hist <= YIELD_MAX))
    n_outlier = int((~ok).sum())                          # сентинел/выброс урожайности
    Xb, y, prod_hist = Xb[ok], y[ok], prod_hist[ok]
    ids = ids[ok].reset_index(drop=True)
    Xb, y, prod_hist, ids, n_corrupt = drop_corrupt(Xb, y, prod_hist, ids)
    n = Xb.shape[0]
    split_years = None
    if SPLIT_MODE == 'year':
        tr, te, va, split_years = split_by_year(ids['year'].values, seed)
    else:
        tr, te, va = split_indices(n, seed)
    med = float(np.nanmedian(prod_hist[tr]))              # prod_hist пуст у ранних лет
    prod_hist = np.where(np.isnan(prod_hist), med, prod_hist)
    X, feature_names = add_prod_hist_row(Xb, prod_hist, feature_names)
    norm = fit_normalization(X[tr], feature_names)
    Xn = apply_normalization(X, norm)
    return {'Xn': Xn, 'y': y, 'ids': ids, 'tr': tr, 'te': te, 'va': va, 'norm': norm,
            'feature_names': feature_names, 'meta': meta, 'split_mode': SPLIT_MODE,
            'split_years': split_years,
            'counts': {'n': n, 'dropped': dropped, 'n_outlier': n_outlier,
                       'n_corrupt': n_corrupt, 'prod_hist_median': med}}


# --------------------------------------------------------------------------- #
# Главный сценарий
# --------------------------------------------------------------------------- #
def run(task, lengths, length, trees, seed):
    outdir = os.path.join(config.REPORTS_DIR, 'experiments', task)
    os.makedirs(outdir, exist_ok=True)
    err_dir = os.path.join(outdir, 'error_tables')
    os.makedirs(err_dir, exist_ok=True)

    print(f'[1/6] Загрузка и подготовка датасета {task}…')
    P = prepare_data(task, seed)
    Xn, y, ids = P['Xn'], P['y'], P['ids']
    tr, te, va = P['tr'], P['te'], P['va']
    norm, feature_names, meta, c = P['norm'], P['feature_names'], P['meta'], P['counts']
    n_split = (len(tr), len(te), len(va))
    print(f'      примеров с меткой: {c["n"]} (без урожайности: {c["dropped"]}, '
          f'выбросов урожайности: {c["n_outlier"]}, битых/misalignment: {c["n_corrupt"]}); '
          f'признаков с prod_hist: {Xn.shape[1]}, точек времени: {Xn.shape[2]}; '
          f'сплит {n_split} (70/15/15)')
    with open(os.path.join(outdir, 'normalization.json'), 'w', encoding='utf-8') as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    Xn_tr, Xn_te, Xn_va = Xn[tr], Xn[te], Xn[va]
    y_tr, y_te, y_va = y[tr], y[te], y[va]

    print(f'[4/6] Свип длины L={lengths}…')
    sweep, best_L, sweep_png = length_sweep(Xn_tr, y_tr, Xn_va, y_va, lengths, trees, seed, outdir)
    L = length or best_L
    print(f'      лучшая L по валидации: {best_L}; для отчёта L={L}')

    print(f'[5/6] Финальное обучение при L={L}…')
    Xtr = to_vectors(downsample(Xn_tr, L))
    Xte = to_vectors(downsample(Xn_te, L))
    Xva = to_vectors(downsample(Xn_va, L))

    metric_rows = []
    model_imgs = {}
    imp_lin = imp_rf = None
    for name, mdl in make_models(trees, seed).items():
        mdl.fit(Xtr, y_tr)
        p_te, p_va = mdl.predict(Xte), mdl.predict(Xva)
        metric_rows.append({'model': name, 'split': 'test', **metrics(y_te, p_te)})
        metric_rows.append({'model': name, 'split': 'val', **metrics(y_va, p_va)})
        model_imgs[name] = [scatter_plot(y_te, p_te, name, outdir),
                            hist_plot(y_te, p_te, name, outdir)]
        error_table(ids.iloc[te], y_te, p_te).to_csv(
            os.path.join(err_dir, f'errors_{name}_test.csv'), index=False, encoding='utf-8-sig')
        error_table(ids.iloc[va], y_va, p_va).to_csv(
            os.path.join(err_dir, f'errors_{name}_val.csv'), index=False, encoding='utf-8-sig')
        if name == 'linear':
            imp_lin = aggregate_importance(np.abs(mdl.coef_), feature_names, L)
            imp_lin.to_csv(os.path.join(outdir, 'importance_linear.csv'),
                           index=False, encoding='utf-8-sig')
        if name == 'forest':
            imp_rf = aggregate_importance(mdl.feature_importances_, feature_names, L)
            imp_rf.to_csv(os.path.join(outdir, 'importance_forest.csv'),
                          index=False, encoding='utf-8-sig')

    # нижний BASELINE: лес ТОЛЬКО на скользящем среднем урожайности (prod_hist), 1 признак.
    # Показывает «пол» — сколько метрик даёт одна инерция урожайности без остальных параметров.
    ph_idx = feature_names.index('prod_hist_ma')
    Xb_tr = Xn_tr[:, ph_idx, 0].reshape(-1, 1)       # prod_hist константа по времени -> берём 1 точку
    Xb_te = Xn_te[:, ph_idx, 0].reshape(-1, 1)
    Xb_va = Xn_va[:, ph_idx, 0].reshape(-1, 1)
    base = RandomForestRegressor(n_estimators=trees, n_jobs=-1, random_state=seed)
    base.fit(Xb_tr, y_tr)
    metric_rows.append({'model': 'baseline_prodhist', 'split': 'test',
                        **metrics(y_te, base.predict(Xb_te))})
    metric_rows.append({'model': 'baseline_prodhist', 'split': 'val',
                        **metrics(y_va, base.predict(Xb_va))})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(outdir, 'metrics.csv'), index=False, encoding='utf-8-sig')
    print(metrics_df.to_string(index=False))

    print('[6/6] Сборка docx-отчёта…')
    rep = build_docx(task, meta, L, best_L, n_split, metrics_df, imp_lin, imp_rf,
                     sweep_png, model_imgs, outdir)
    print(f'\nГотово. Артефакты: {outdir}')
    print(f'  отчёт: {rep}')
    print(f'  таблицы погрешностей: {err_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Разведочный анализ + базовые ML-модели по датасету.')
    ap.add_argument('task', help='имя задания/датасета, напр. rye_winter_all')
    ap.add_argument('--lengths', default=None,
                    help='список L через запятую для свипа (по умолч. 3..100)')
    ap.add_argument('--length', type=int, default=None,
                    help='L для финального отчёта (по умолч. — лучшая по свипу)')
    ap.add_argument('--trees', type=int, default=200, help='деревьев в случайном лесу')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    lengths = ([int(x) for x in a.lengths.split(',')] if a.lengths else DEFAULT_LENGTHS)
    run(a.task, lengths, a.length, a.trees, a.seed)
