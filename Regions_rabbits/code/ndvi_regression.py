"""Классическая регрессия по NDVI — прогноз урожайности озимой пшеницы.

Метод (как у агрономов): для каждого РАЙОНА по всем годам сопоставляются (максимум NDVI весеннего
пика; урожайность). Максимум берётся в ПЕРВОЙ половине года (у озимых два пика — берём весенний,
до уборки). Облако точек аппроксимируется ЭКСПОНЕНТОЙ  урожай = a·exp(b·maxNDVI)  (через лог-линейную
регрессию ln(y) = ln(a) + b·maxNDVI). Для расчётного года maxNDVI подставляется в уравнение -> прогноз.

Данные: winter_wheat_honest (NDVI целевого года, канал 0). Сплит train/test/val 70/15/15 (как обычно).
Подгонка экспоненты — на TRAIN-точках района; прогноз и метрики — на TEST и VAL. Если у района мало
TRAIN-точек (<3) или нет вариации NDVI — откат к средней урожайности района (иначе — глобальной).

Запуск:  python code/ndvi_regression.py
Выход:   workspace/reports/ndvi_regression/ (+ docs/Классическая_регрессия_NDVI.docx)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import explore_models as E
import plots
import train_honest as TH

HALF = 182          # первая половина года (весенний пик, до уборки ~210)
MIN_FIT = 3         # минимум train-точек района для подгонки экспоненты
SEED = 42


def fit_exp(x, yv):
    """Лог-линейная подгонка y = a·exp(b·x). Возврат (ln a, b) или None."""
    m = yv > 0
    x, yv = x[m], yv[m]
    if len(x) < MIN_FIT or np.std(x) < 1e-6:
        return None
    b, lna = np.polyfit(x, np.log(yv), 1)
    return float(lna), float(b)


def pred_exp(fit, x):
    return float(np.clip(np.exp(fit[0] + fit[1] * x), 0, E.YIELD_MAX))


def pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    outdir = os.path.join(config.REPORTS_DIR, 'ndvi_regression')
    os.makedirs(outdir, exist_ok=True)
    z, meta = TH.load_honest('winter_wheat_honest')
    raw, y, ids = z['raw'], z['y'].astype(float), z['ids']
    ny = int(meta['duration_years'])
    keep = ~np.isnan(y) & (y >= 0) & (y <= E.YIELD_MAX)
    raw, y, ids = raw[keep], y[keep], ids[keep]
    mx = raw[:, 0, ny - 1, :HALF].max(axis=1)            # максимум NDVI весеннего пика целевого года
    ok = mx > 0.0                                        # отбрасываем примеры без NDVI
    raw, y, ids, mx = raw[ok], y[ok], ids[ok], mx[ok]
    reg, dist = ids[:, 0], ids[:, 1]
    terr = reg * 100000 + dist
    n = len(y)
    tr, te, va = TH.split_random(n, SEED)
    in_tr = np.zeros(n, bool); in_tr[tr] = True
    in_te = np.zeros(n, bool); in_te[te] = True
    in_va = np.zeros(n, bool); in_va[va] = True
    split_of = np.empty(n, object); split_of[tr] = 'train'; split_of[te] = 'test'; split_of[va] = 'val'
    print(f'Примеров {n}; районов {len(set(terr))}, регионов {len(set(reg))}; '
          f'сплит train/test/val = {len(tr)}/{len(te)}/{len(va)}')

    glob_mean = float(y[tr].mean())
    pred = np.full(n, np.nan)
    method = np.empty(n, object)
    drows = []                                           # по районам
    for t in sorted(set(terr.tolist())):
        sel = terr == t
        x_all, y_all = mx[sel], y[sel]
        tr_m = sel & in_tr
        x_tr, y_tr = mx[tr_m], y[tr_m]
        fit = fit_exp(x_tr, y_tr)
        dmean = float(y_tr.mean()) if tr_m.sum() else glob_mean
        # прогноз для всех точек района (нужен на test/val)
        for i in np.where(sel)[0]:
            if fit is not None:
                pred[i] = pred_exp(fit, mx[i]); method[i] = 'exp'
            else:
                pred[i] = dmean; method[i] = 'fallback'
        r = pearson(x_all, y_all)
        ridx = np.where(sel)[0][0]
        te_m, va_m = sel & in_te, sel & in_va
        mse_te = float(np.mean((y[te_m] - pred[te_m]) ** 2)) if te_m.sum() else np.nan
        mse_va = float(np.mean((y[va_m] - pred[va_m]) ** 2)) if va_m.sum() else np.nan
        drows.append({'id_region': int(reg[ridx]), 'id_district': int(dist[ridx]),
                      'n_всего': int(sel.sum()), 'n_train': int(tr_m.sum()),
                      'corr_ndvi_yield': round(r, 3) if r == r else np.nan,
                      'exp_a': round(np.exp(fit[0]), 2) if fit else np.nan,
                      'exp_b': round(fit[1], 3) if fit else np.nan,
                      'подгонка': 'exp' if fit else 'откат',
                      'n_test': int(te_m.sum()), 'MSE_test': round(mse_te, 1) if mse_te == mse_te else np.nan,
                      'n_val': int(va_m.sum()), 'MSE_val': round(mse_va, 1) if mse_va == mse_va else np.nan})
    ddf = pd.DataFrame(drows)

    # ----- метрики -----
    def mset(idx):
        return E.metrics(y[idx], pred[idx])

    overall = []
    for nm, idx in (('test', te), ('val', va)):
        m = mset(idx)
        cov = float((method[idx] == 'exp').mean())
        overall.append({'выборка': nm, 'n': len(idx), 'MSE': round(m['MSE'], 2), 'RMSE': round(m['RMSE'], 2),
                        'R2': round(m['R2'], 3), 'Пирсон': round(m['R_pearson'], 3),
                        'доля_exp': round(cov, 3)})
    odf = pd.DataFrame(overall)
    print('\n=== ОБЩИЕ метрики ==='); print(odf.to_string(index=False))

    # по регионам (test и val)
    rrows = []
    for rg in sorted(set(reg.tolist())):
        row = {'id_region': int(rg)}
        for nm, idx in (('test', te), ('val', va)):
            sub = idx[reg[idx] == rg]
            if len(sub) >= 3:
                m = mset(sub)
                row[f'n_{nm}'] = len(sub); row[f'R2_{nm}'] = round(m['R2'], 3); row[f'RMSE_{nm}'] = round(m['RMSE'], 2)
            else:
                row[f'n_{nm}'] = len(sub); row[f'R2_{nm}'] = np.nan; row[f'RMSE_{nm}'] = np.nan
        rsel = ddf['id_region'] == rg
        row['ср_corr_районов'] = round(float(ddf.loc[rsel, 'corr_ndvi_yield'].mean()), 3)
        rrows.append(row)
    rdf = pd.DataFrame(rrows).sort_values('R2_test', ascending=False, na_position='last')

    ddf.to_csv(os.path.join(outdir, 'per_district.csv'), index=False, encoding='utf-8-sig')
    rdf.to_csv(os.path.join(outdir, 'per_region.csv'), index=False, encoding='utf-8-sig')
    odf.to_csv(os.path.join(outdir, 'overall.csv'), index=False, encoding='utf-8-sig')

    imgs = _plots(outdir, mx, y, pred, te, terr, reg, dist, in_tr, ddf)
    _docx(outdir, odf, rdf, ddf, imgs)
    print(f'\nКорреляция NDVI↔урожай по районам: медиана {ddf.corr_ndvi_yield.median():.2f}, '
          f'доля |r|>0.5: {(ddf.corr_ndvi_yield.abs() > 0.5).mean():.2f}')
    print(f'Готово: {outdir}')


def _plots(outdir, mx, y, pred, te, terr, reg, dist, in_tr, ddf):
    imgs = []

    # 1) распределение корреляций по районам
    def hist(lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(ddf['corr_ndvi_yield'].dropna(), bins=30, color='#2ca02c', alpha=0.8)
        ax.axvline(float(ddf['corr_ndvi_yield'].median()), color='red', ls='--',
                   label=plots.tr(f'медиана {ddf.corr_ndvi_yield.median():.2f}',
                                  f'median {ddf.corr_ndvi_yield.median():.2f}', lang))
        ax.set_xlabel(plots.tr('Корреляция maxNDVI ↔ урожайность (по району)',
                               'Correlation maxNDVI ↔ yield (per district)', lang))
        ax.set_ylabel(plots.tr('Число районов', 'Districts', lang))
        ax.set_title(plots.tr('Связь весеннего максимума NDVI и урожайности по районам',
                              'Spring-max NDVI vs yield correlation by district', lang))
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout(); return fig
    imgs.append(('Распределение корреляций по районам', plots.bilingual(outdir, 'corr_hist', hist)))

    # 2) предсказание vs реальность на тесте
    def sc(lang):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        lo, hi = 0, float(max(y[te].max(), pred[te].max()))
        ax.scatter(y[te], pred[te], s=12, alpha=0.4, color='#1f77b4', edgecolors='none')
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x')
        m = E.metrics(y[te], pred[te])
        ax.set_title(plots.tr(f'Тест: R²={m["R2"]:.3f}, RMSE={m["RMSE"]:.2f}',
                              f'Test: R²={m["R2"]:.3f}, RMSE={m["RMSE"]:.2f}', lang))
        ax.set_xlabel(plots.tr('Реальная урожайность, ц/га', 'Actual yield, c/ha', lang))
        ax.set_ylabel(plots.tr('Прогноз по NDVI, ц/га', 'NDVI prediction, c/ha', lang))
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout(); return fig
    imgs.append(('Прогноз vs реальность (тест)', plots.bilingual(outdir, 'pred_vs_actual_test', sc)))

    # 3) примеры подгонки экспоненты для 4 районов (с достаточным числом точек): 2 высокая r, 2 низкая
    cand = ddf[(ddf['n_всего'] >= 6) & ddf['corr_ndvi_yield'].notna()].sort_values('corr_ndvi_yield')
    picks = list(cand.tail(2).itertuples()) + list(cand.head(2).itertuples())

    def fits(lang):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, row in zip(axes.ravel(), picks):
            t = row.id_region * 100000 + row.id_district
            sel = terr == t
            xs, ys, istr = mx[sel], y[sel], in_tr[sel]
            ax.scatter(xs[istr], ys[istr], color='#1f77b4', s=30,
                       label=plots.tr('train', 'train', lang))
            ax.scatter(xs[~istr], ys[~istr], color='#ff7f0e', s=30,
                       label=plots.tr('test/val', 'test/val', lang))
            f = fit_exp(xs[istr], ys[istr])
            if f:
                xx = np.linspace(xs.min(), xs.max(), 50)
                ax.plot(xx, [pred_exp(f, v) for v in xx], 'g-',
                        label=plots.tr('экспонента', 'exponential', lang))
            ax.set_title(f'рег {row.id_region} р-н {row.id_district}: r={row.corr_ndvi_yield}')
            ax.set_xlabel('maxNDVI'); ax.set_ylabel(plots.tr('урожай, ц/га', 'yield, c/ha', lang))
            ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
        fig.suptitle(plots.tr('Примеры: экспоненциальная регрессия урожай↔maxNDVI по районам',
                              'Examples: exponential yield↔maxNDVI regression by district', lang))
        fig.tight_layout(); return fig
    imgs.append(('Примеры подгонки экспоненты', plots.bilingual(outdir, 'example_fits', fits)))
    return imgs


def _docx(outdir, odf, rdf, ddf, imgs):
    from docx import Document
    from docx.shared import Inches
    # .../Regions_rabbits/docs (как в make_docs.py): REPORTS_DIR=workspace/reports -> вверх до проекта
    docs = os.path.join(os.path.dirname(os.path.dirname(config.REPORTS_DIR)), 'docs')
    os.makedirs(docs, exist_ok=True)
    doc = Document()
    doc.add_heading('Классическая регрессия по NDVI — озимая пшеница', 0)
    doc.add_paragraph(
        'Простая модель, широко применяемая специалистами. Для каждого РАЙОНА по всем годам строится '
        'связь «максимум NDVI весеннего пика → урожайность». Максимум NDVI берётся в первой половине '
        'года (у озимых два пика; весенний — до уборки ~конец июля). Облако точек аппроксимируется '
        'ЭКСПОНЕНТОЙ урожай = a·exp(b·maxNDVI) (через лог-линейную регрессию). Для расчётного года '
        'maxNDVI подставляется в уравнение района — получается прогноз.')
    doc.add_paragraph(
        f'Данные: winter_wheat_honest, NDVI целевого года (канал 0), весенний максимум по дням 0–{HALF}. '
        f'Сплит train/test/val = 70/15/15 (как обычно). Экспонента подгоняется на TRAIN-точках района '
        f'(минимум {MIN_FIT}); при нехватке точек/отсутствии вариации NDVI — откат к средней урожайности '
        'района. Корреляция и метрики — на TEST и VAL.')

    doc.add_heading('Общие метрики', level=1)
    E._table_from_df(doc, odf)
    doc.add_paragraph(
        '«доля_exp» — доля примеров, спрогнозированных самой экспонентой (а не откатом к среднему). '
        'Для сравнения: «пол» только по средней урожайности давал MSE≈84; свёрточная сеть ≈27; '
        'ансамбль ≈25 (на том же датасете и сплите).')

    doc.add_heading('Метрики по регионам (сортировка по R² на тесте)', level=1)
    E._table_from_df(doc, rdf.round(3), maxrows=60)

    doc.add_heading('Корреляция maxNDVI ↔ урожайность по районам', level=1)
    cc = ddf['corr_ndvi_yield'].dropna()
    doc.add_paragraph(
        f'Районов с подгонкой: {len(ddf)}; медиана корреляции {cc.median():.2f}, средняя {cc.mean():.2f}; '
        f'доля районов с |r|>0.5: {(cc.abs() > 0.5).mean()*100:.0f}%, c |r|>0.7: {(cc.abs() > 0.7).mean()*100:.0f}%. '
        'Полная таблица по каждому району (id_region, id_district, n, корреляция, коэффициенты a,b) — '
        'в файле per_district.csv. Ниже — районы с самой сильной и самой слабой связью.')
    top = ddf.dropna(subset=['corr_ndvi_yield']).sort_values('corr_ndvi_yield', ascending=False)
    doc.add_heading('Топ-15 районов по корреляции', level=2)
    E._table_from_df(doc, top.head(15))
    doc.add_heading('15 районов с самой слабой связью', level=2)
    E._table_from_df(doc, top.tail(15))

    doc.add_heading('Графики', level=1)
    for title, png in imgs:
        doc.add_heading(title, level=2)
        if png:
            doc.add_picture(png, width=Inches(6.3))

    doc.add_heading('Комментарии', level=1)
    for s in [
        'Модель полностью локальна: каждый район описывается своей экспонентой урожай↔maxNDVI, '
        'без общих параметров. Это её сила (учёт специфики района) и слабость (мало точек на район — '
        '7–8 лет, из них ~5 в train).',
        'Связь NDVI↔урожай сильно различается по районам: где она тесная (|r| высок), прогноз надёжен; '
        'где облако точек размыто — экспонента почти не помогает, и метрики хуже.',
        'Метод даёт быстрый интерпретируемый прогноз по одному показателю (весенний максимум NDVI), '
        'но уступает многомерным моделям (свёрточная сеть, ансамбль), использующим всю динамику '
        'погоды и вегетации.']:
        doc.add_paragraph(s, style='List Bullet')

    out = os.path.join(docs, 'Классическая_регрессия_NDVI.docx')
    try:
        doc.save(out)
    except PermissionError:
        out = os.path.join(docs, 'Классическая_регрессия_NDVI_new.docx'); doc.save(out)
    print(f'Отчёт: {out}')


if __name__ == '__main__':
    main()
