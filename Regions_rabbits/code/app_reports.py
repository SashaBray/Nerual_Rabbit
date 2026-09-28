"""Формирование подробных отчётов по прогнозу: таблицы + Word с графиками.

Содержимое отчёта:
  * текущие прогнозы выбранными моделями на запрашиваемый год;
  * история прогнозов из БД для этого (район, культура, год) — если делались раньше;
  * УТОЧНЁННЫЕ метрики модели для данного района: бэктест по предыдущим годам
    (прогноз -> сравнение с фактом -> MAE/RMSE/смещение), таблица и график факт/прогноз.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import app_geo as G
import app_predict as AP
import app_store as ASt
import config
import db
import explore_models as E
import plots

POINT_MODELS = ['nn', 'rf', 'linear', 'ndvi', 'ensemble']


def territory_label(id_region, id_district):
    t = db.get_territory(db.territory_id(id_region, id_district))
    if not t:
        return f'регион {id_region}, район {id_district}'
    reg = t.get('region') or id_region
    dist = t.get('district') or id_district
    return f'{reg} — {dist}'


def backtest(id_region, id_district, culture, models, last_k=6):
    """Прогнозы по прошлым годам с известным фактом -> таблица и метрики по моделям."""
    cfg = AP.resolve_culture(culture)
    tid = db.territory_id(id_region, id_district)
    yields = {y: v for y, v in db.get_yields(tid, cfg['bdpmo_culture']).items()
              if v is not None and v == v}
    years = sorted(yields)[-last_k:]
    rows = []
    for y in years:
        r = AP.predict_for_territory(id_region, id_district, culture, y, models)
        rec = {'год': y, 'факт': round(float(yields[y]), 2)}
        for m in models:
            v = r['values'].get(m)
            rec[m] = round(v, 2) if v is not None else None
        rows.append(rec)
    bt = pd.DataFrame(rows)
    met = []
    for m in models:
        if m not in bt:
            continue
        d = bt.dropna(subset=[m])
        if len(d) < 2:
            met.append({'модель': AP.token_label(m), 'n': len(d), 'MAE': None, 'RMSE': None, 'смещение': None})
            continue
        err = d[m].values - d['факт'].values
        met.append({'модель': AP.token_label(m), 'n': len(d),
                    'MAE': round(float(np.mean(np.abs(err))), 2),
                    'RMSE': round(float(np.sqrt(np.mean(err ** 2))), 2),
                    'смещение': round(float(np.mean(err)), 2)})
    return bt, pd.DataFrame(met)


def current_predictions_df(id_region, id_district, culture, year, models):
    r = AP.predict_for_territory(id_region, id_district, culture, year, models)
    rows = [{'модель': AP.token_label(m), 'прогноз, ц/га': (round(r['values'][m], 2)
             if r['values'].get(m) is not None else None), 'примечание': r['notes'].get(m)}
            for m in models]
    return pd.DataFrame(rows), r


def _backtest_chart(bt, models, outdir, slug):
    def draw(lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(bt['год'], bt['факт'], 'k-o', lw=2, label=plots.tr('факт', 'actual', lang))
        for m in models:
            if m in bt:
                ax.plot(bt['год'], bt[m], '--.', label=AP.token_label(m))
        ax.set_xlabel(plots.tr('Год', 'Year', lang)); ax.set_ylabel(plots.tr('Урожайность, ц/га', 'Yield, c/ha', lang))
        ax.set_title(plots.tr('Бэктест: факт и прогнозы по годам', 'Backtest: actual vs predictions', lang))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout(); return fig
    return plots.bilingual(outdir, f'backtest_{slug}', draw)


def _history_chart(hist, outdir, slug):
    def draw(lang):
        fig, ax = plt.subplots(figsize=(9, 5))
        for m, g in hist.groupby('model_name'):
            ax.plot(pd.to_datetime(g['made_at']), g['value'], '-o', label=str(m))
        ax.set_xlabel(plots.tr('Когда сделан прогноз', 'Prediction time', lang))
        ax.set_ylabel(plots.tr('Прогноз, ц/га', 'Forecast, c/ha', lang))
        ax.set_title(plots.tr('История прогнозов на этот год', 'Forecast history for this year', lang))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout(); return fig
    return plots.bilingual(outdir, f'history_{slug}', draw)


def generate_report(id_region, id_district, culture, year, models=None, outpath=None, last_k=6):
    """Сформировать Word-отчёт. Возвращает (путь, dict с таблицами)."""
    from docx import Document
    from docx.shared import Inches
    models = models or ['nn', 'rf', 'linear', 'ndvi', 'ensemble']
    outdir = os.path.join(config.REPORTS_DIR, 'app_forecasts')
    os.makedirs(outdir, exist_ok=True)
    tid = db.territory_id(id_region, id_district)
    slug = f'{tid}_{culture}_{year}'.replace(' ', '_')
    label = territory_label(id_region, id_district)

    cur_df, r = current_predictions_df(id_region, id_district, culture, year, models)
    bt, met = backtest(id_region, id_district, culture, models, last_k)
    hist = ASt.prediction_history(tid, culture, year)

    doc = Document()
    doc.add_heading(f'Прогноз урожайности — {culture}', 0)
    doc.add_paragraph(f'Территория: {label} (id {tid}). Прогнозируемый год: {year}. '
                      f'Средняя урожайность (история): '
                      f'{round(r["prod_hist"],1) if r["prod_hist"] else "—"} ц/га.')

    doc.add_heading('Текущий прогноз', level=1)
    E._table_from_df(doc, cur_df)

    doc.add_heading('Уточнённые метрики модели для этого района (бэктест прошлых лет)', level=1)
    doc.add_paragraph(f'Для последних {last_k} лет с известным фактом построены прогнозы и сопоставлены '
                      'с реальной урожайностью; ниже — ошибки (MAE/RMSE) и систематическое смещение.')
    E._table_from_df(doc, met)
    doc.add_heading('Факт и прогнозы по годам', level=2)
    E._table_from_df(doc, bt)
    if len(bt) >= 2:
        doc.add_picture(_backtest_chart(bt, models, outdir, slug), width=Inches(6.3))

    doc.add_heading('История прогнозов на этот год', level=1)
    if hist is not None and not hist.empty and len(hist) >= 1:
        hcols = ['made_at', 'model_name', 'value', 'автор', 'comment']
        E._table_from_df(doc, hist[[c for c in hcols if c in hist]])
        if hist['made_at'].nunique() >= 2:
            doc.add_picture(_history_chart(hist.dropna(subset=['value']), outdir, slug), width=Inches(6.3))
    else:
        doc.add_paragraph('Ранее прогнозы на этот год для данной территории не делались.')

    out = outpath or os.path.join(outdir, f'forecast_{slug}.docx')
    out = _save_docx(doc, out)
    return out, {'current': cur_df, 'backtest': bt, 'metrics': met, 'history': hist}


def _district_name(reg, dist):
    if dist is None:
        return '(регион целиком)'
    t = db.get_territory(db.territory_id(reg, dist))
    return str(t.get('district')) if t and t.get('district') else str(dist)


def _territory_backtest(reg, dist, culture, models, last_k):
    """Бэктест ансамбля по последним годам с фактом -> dict {mse,rmse,mae,r2,pearson,n}."""
    tid = db.territory_id(reg, dist)
    cfg = AP.resolve_culture(culture)
    yields = {y: v for y, v in db.get_yields(tid, cfg['bdpmo_culture']).items() if v is not None and v == v}
    years = sorted(yields)
    if last_k:
        years = years[-int(last_k):]
    preds, acts = [], []
    for y in years:
        r = AP.predict_for_territory(reg, dist, culture, y, models=list(models) + ['ensemble'],
                                     ensemble_of=list(models))
        v = r['values'].get('ensemble')
        if v is not None:
            preds.append(float(v)); acts.append(float(yields[y]))
    none = {'mse': None, 'rmse': None, 'mae': None, 'r2': None, 'pearson': None, 'n': len(preds)}
    if len(preds) < 2:
        return none
    p = np.array(preds); a = np.array(acts); err = p - a
    mse = float(np.mean(err ** 2))
    sstot = float(np.sum((a - a.mean()) ** 2))
    pear = (float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 1e-9 and np.std(a) > 1e-9 else None)
    return {'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(np.abs(err))),
            'r2': (1 - float(np.sum(err ** 2)) / sstot) if sstot > 1e-9 else None,
            'pearson': pear, 'n': len(preds)}


DEFAULT_METRIC_DEPTH = 6


def store_territory_metrics(reg, dist, culture, models, batch_id, account_id=None,
                            last_k=DEFAULT_METRIC_DEPTH):
    """Посчитать уточнённые MSE/R² территории (бэктест) и сохранить в БД — вызывается при прогнозе.

    Тогда отчёт по партии просто читает метрики из БД, без повторного бэктеста (быстро).
    """
    pts = [m for m in models if m != 'ensemble'] or list(models)
    mt = _territory_backtest(reg, dist, culture, pts, last_k)
    db.upsert_territory_metrics(db.territory_id(reg, dist), culture, batch_id, mt['mse'], mt['r2'],
                                last_k=last_k, account_id=account_id,
                                rmse=mt['rmse'], mae=mt['mae'], pearson=mt['pearson'])
    return mt


def model_accuracy(culture, sample=40, last_k=5, progress=None, store=True):
    """Точность моделей культуры: бэктест по выборке районов × последним годам с фактом.

    Для каждой модели агрегирует пары (прогноз, факт) и считает MSE/RMSE/R².
    Возвращает DataFrame [модель, mkey, n, MSE, RMSE, R²]; по умолчанию сохраняет в БД.
    """
    import app_geo as _G
    cfg = AP.resolve_culture(culture)
    models = list(AP.available_models(culture))
    # Состав ансамбля — свежая версия каждого типа, как по умолчанию в интерфейсе. Проверять
    # вхождение токена в ('nn','rf',...) нельзя: токен модели-файла равен ИМЕНИ ПАПКИ
    # ('winter_wheat_all_nn_v1'), поэтому старое условие оставляло в ансамбле одну ndvi —
    # и «точность ансамбля» совпадала с точностью NDVI-регрессии.
    comps, _seen = [], set()
    for m in models:
        if m == 'ensemble':
            continue
        kind, _v = AP.parse_token(m)
        if kind in _seen:
            continue
        _seen.add(kind)
        comps.append(m)

    terrs = []
    for rid, _name in _G.list_regions():
        for did in sorted(AP.districts_with_data(rid, culture)):
            terrs.append((rid, did))
    terrs.sort()
    if sample and len(terrs) > sample:                     # равномерная выборка районов
        step = len(terrs) / float(sample)
        terrs = [terrs[int(i * step)] for i in range(sample)]

    acc = {m: {'p': [], 'a': []} for m in models}
    for n, (rid, did) in enumerate(terrs, 1):
        if progress:
            progress(n, len(terrs))
        tid = db.territory_id(rid, did)
        yld = {y: v for y, v in db.get_yields(tid, cfg['bdpmo_culture']).items() if v is not None and v == v}
        years = sorted(yld)[-int(last_k):] if last_k else sorted(yld)
        for y in years:
            try:
                r = AP.predict_for_territory(rid, did, culture, y, models=models, ensemble_of=comps)
            except Exception:                              # noqa: BLE001 — район пропускаем
                continue
            for m in models:
                v = r['values'].get(m)
                if v is not None and v == v:
                    acc[m]['p'].append(float(v)); acc[m]['a'].append(float(yld[y]))

    rows = []
    for m in models:
        p = np.array(acc[m]['p']); a = np.array(acc[m]['a'])
        if len(p) >= 2:
            err = p - a
            mse = float(np.mean(err ** 2)); rmse = float(np.sqrt(mse))
            sstot = float(np.sum((a - a.mean()) ** 2))
            r2 = (1 - float(np.sum(err ** 2)) / sstot) if sstot > 1e-9 else None
        else:
            mse = rmse = r2 = None
        rows.append({'модель': AP.token_label(m), 'mkey': m, 'n': int(len(p)),
                     'MSE': round(mse, 2) if mse is not None else None,
                     'RMSE': round(rmse, 2) if rmse is not None else None,
                     'R²': round(r2, 2) if r2 is not None else None})
        if store:
            db.upsert_model_metrics(culture, m, len(p), mse, rmse, r2, sample=sample, last_k=last_k)
    return pd.DataFrame(rows)


def batch_table(batch_id, last_k=6, progress=None):
    """Табличная часть отчёта (гибрид старой версии и образца): по строке на территорию.

    Колонки: №, id_country, region, district, culture, id_region, id_district, Average productivity,
    <год> = урожайность из архива (по столбцу на год, глубина last_k), <модель>_predict_<год>,
    dispersion, R2 score, Mean squared error, Root mean squared error, Mean absolute error,
    Pearson correlation, Разница.
    """
    bp = ASt.batch_predictions(batch_id)
    if bp.empty:
        raise ValueError('В партии нет прогнозов.')
    culture = str(bp['culture'].dropna().iloc[0])
    cfg = AP.resolve_culture(culture)
    bp = bp.copy(); bp['mkey'] = bp['model_name'].apply(lambda s: str(s).split(':')[-1])
    model_keys = sorted(set(bp['mkey'].astype(str)), key=AP.token_sort_key)
    years = sorted(int(y) for y in bp['predict_year'].dropna().unique())
    pred_cols = [f'{m}_predict_{y}' for m in model_keys for y in years]
    stored = db.get_batch_metrics(batch_id)
    terrs = sorted(bp['territory_id'].unique())

    rnd3 = lambda v: round(float(v), 3) if v is not None and v == v else None
    recs = []; year_set = set()
    for i, tid in enumerate(terrs, 1):                     # 1-й проход: данные по территориям
        if progress:
            progress(i, len(terrs))
        rg, dd = str(tid).split('_'); reg = int(rg); dist = None if dd == 'nan' else int(float(dd))
        sub = bp[bp['territory_id'] == tid]
        t = db.get_territory(tid) or {}
        yraw = {int(y): float(v) for y, v in db.get_yields(tid, cfg['bdpmo_culture']).items()
                if v is not None and v == v}
        avg = float(np.mean(list(yraw.values()))) if yraw else None
        disp = float(np.var(list(yraw.values()))) if len(yraw) >= 2 else None
        ky = sorted(yraw)[-int(last_k):] if last_k else sorted(yraw)
        yld = {y: round(yraw[y], 2) for y in ky}; year_set.update(ky)
        preds = {}
        for m in model_keys:
            for y in years:
                hit = sub[(sub['mkey'] == m) & (sub['predict_year'] == y)]
                v = hit['value'].iloc[0] if not hit.empty else None
                preds[f'{m}_predict_{y}'] = round(float(v), 2) if v is not None and v == v else None
        _missing = lambda v: v is None or (isinstance(v, float) and v != v)   # None или NaN
        st = stored.get(str(tid), {})
        if any(_missing(st.get(k)) for k in ('rmse', 'mae', 'pearson')):
            pts = [m for m in model_keys if m != 'ensemble'] or model_keys
            try:                                          # старые партии: добираем метрики ансамбля бэктестом
                mt = _territory_backtest(reg, dist, culture, pts, DEFAULT_METRIC_DEPTH)
                st = {k: (mt.get(k) if _missing(st.get(k)) else st.get(k))
                      for k in ('mse', 'rmse', 'mae', 'r2', 'pearson')}
            except Exception:                             # noqa: BLE001
                pass
        recs.append({'reg': reg, 'dist': dist, 'id_country': t.get('id_country'),
                     'region': G.region_name(reg), 'district': _district_name(reg, dist),
                     'avg': avg, 'disp': disp, 'yld': yld, 'preds': preds, 'st': st})

    year_cols = sorted(year_set)
    if last_k:
        year_cols = year_cols[-int(last_k):]              # общая глубина по году

    rows = []
    for n, rec in enumerate(recs, 1):
        row = {'№': n, 'id_country': rec['id_country'], 'region': rec['region'],
               'district': rec['district'], 'culture': culture,
               'id_region': rec['reg'], 'id_district': ('' if rec['dist'] is None else rec['dist']),
               'Average productivity': round(rec['avg'], 2) if rec['avg'] is not None else None}
        for y in year_cols:
            row[str(y)] = rec['yld'].get(y)               # урожайность за год из архива (пусто, если нет)
        row.update(rec['preds'])
        st = rec['st']
        row['dispersion'] = round(rec['disp'], 2) if rec['disp'] is not None else None
        row['R2 score'] = rnd3(st.get('r2'))
        row['Mean squared error'] = rnd3(st.get('mse'))
        row['Root mean squared error'] = rnd3(st.get('rmse'))
        row['Mean absolute error'] = rnd3(st.get('mae'))
        row['Pearson correlation'] = rnd3(st.get('pearson'))
        rows.append(row)

    cols = (['№', 'id_country', 'region', 'district', 'culture', 'id_region', 'id_district',
             'Average productivity'] + [str(y) for y in year_cols] + pred_cols
            + ['dispersion', 'R2 score', 'Mean squared error', 'Root mean squared error',
               'Mean absolute error', 'Pearson correlation'])
    return pd.DataFrame(rows, columns=cols), culture


def _safe_dir(name):
    """Безопасное имя папки: убираем символы, недопустимые в путях Windows."""
    s = ''.join('_' if c in '<>:"/\\|?*' else c for c in str(name)).strip().strip('.')
    return s or 'report'


# --------------------------------------------------------------------------- #
# Запись минимального .xlsx без сторонних библиотек (один лист, inline-строки/числа).
# --------------------------------------------------------------------------- #
def _xlsx_col(n):
    s = ''; n += 1
    while n:
        n, r = divmod(n - 1, 26); s = chr(65 + r) + s
    return s


def _xlsx_esc(t):
    return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _write_xlsx(df, path, sheet_name='Sheet1'):
    """Записать DataFrame в .xlsx (числа Excel покажет в локали — с запятой). Без openpyxl."""
    import zipfile
    cols = list(df.columns)

    def cell(ci, ri, val):
        ref = f'{_xlsx_col(ci)}{ri}'
        if val is None or (isinstance(val, float) and val != val) or val == '':
            return ''
        if isinstance(val, bool):
            val = int(val)
        if isinstance(val, (int, float)):
            return f'<c r="{ref}"><v>{val}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{_xlsx_esc(val)}</t></is></c>'

    rows_xml = ['<row r="1">' + ''.join(cell(ci, 1, c) for ci, c in enumerate(cols)) + '</row>']
    for ri, (_, row) in enumerate(df.iterrows(), start=2):
        rows_xml.append(f'<row r="{ri}">' + ''.join(cell(ci, ri, row[c]) for ci, c in enumerate(cols)) + '</row>')
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetData>' + ''.join(rows_xml) + '</sheetData></worksheet>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
          '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>')
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          f'<sheets><sheet name="{_xlsx_esc(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>')
    wbr = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
           '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
           '</Relationships>')
    tmp = path + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ct)
        z.writestr('_rels/.rels', rels)
        z.writestr('xl/workbook.xml', wb)
        z.writestr('xl/_rels/workbook.xml.rels', wbr)
        z.writestr('xl/worksheets/sheet1.xml', sheet)
    try:
        os.replace(tmp, path); return path
    except PermissionError:                               # файл открыт — сохраняем рядом
        alt = path.replace('.xlsx', '_new.xlsx'); os.replace(tmp, alt); return alt


# --------------------------------------------------------------------------- #
# Графики -> две папки: <outdir>/png и <outdir>/pdf
# --------------------------------------------------------------------------- #
def _save_fig(fig, outdir, name):
    pngdir = os.path.join(outdir, 'png'); pdfdir = os.path.join(outdir, 'pdf')
    os.makedirs(pngdir, exist_ok=True); os.makedirs(pdfdir, exist_ok=True)
    png = os.path.join(pngdir, name + '.png')
    fig.savefig(png, dpi=120, bbox_inches='tight')
    fig.savefig(os.path.join(pdfdir, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
# NDVI-регрессия территории (Word-отчёт по образцу)
# --------------------------------------------------------------------------- #
def _exp_fit(x, y):
    m = y > 0
    if m.sum() < 3 or np.std(x[m]) < 1e-6:
        return None
    b, lna = np.polyfit(x[m], np.log(y[m]), 1)
    return float(lna), float(b)


def _fit_quality(actual, pred):
    actual = np.asarray(actual, float); err = actual - np.asarray(pred, float)
    sstot = float(np.sum((actual - actual.mean()) ** 2))
    q1, q3 = np.percentile(err, [25, 75])
    return {'r2': (1 - float(np.sum(err ** 2)) / sstot) if sstot > 1e-9 else None,
            'mae': float(np.mean(np.abs(err))), 'std': float(np.std(err)), 'iqr': float(q3 - q1)}


def _ndvi_analysis(reg, dist, culture, target_year):
    """Данные NDVI-регрессии территории + ряды NDVI/тепла/осадков (расчётный год и многолетнее
    среднее). Только экспоненциальная модель. None — если данных по NDVI/урожайности мало."""
    cfg = AP.resolve_culture(culture)
    tid = db.territory_id(reg, dist)
    parm = AP._ndvi_parm(cfg, reg)
    if parm is None:
        return None
    yields = {int(y): float(v) for y, v in db.get_yields(tid, cfg['bdpmo_culture']).items()
              if v is not None and v == v}
    pts = []
    for y, yv in yields.items():
        mx = AP._max_ndvi(cfg, parm, tid, y)
        if mx is not None and mx > 0:
            pts.append((y, mx, yv))
    if len(pts) < 3:
        return None
    pts.sort()
    yrs = [p[0] for p in pts]
    x = np.array([p[1] for p in pts]); yv = np.array([p[2] for p in pts])
    mx_t = AP._max_ndvi(cfg, parm, tid, int(target_year))

    ef = _exp_fit(x, yv)                                   # только экспоненциальная модель (п.3)
    exp = None
    if ef:
        exp = _fit_quality(yv, np.clip(np.exp(ef[0] + ef[1] * x), 0, E.YIELD_MAX))
        exp.update(f=f'{np.exp(ef[0]):.2f}*({np.exp(ef[1]):.2f})^x',
                   pred=(round(float(np.clip(np.exp(ef[0] + ef[1] * mx_t), 0, E.YIELD_MAX)), 3)
                         if mx_t is not None else None))
    corr = (float(np.corrcoef(x, yv)[0, 1]) if np.std(x) > 1e-9 and np.std(yv) > 1e-9 else None)

    cur = lambda p: db.get_time_series_array(tid, p, int(target_year), 365, 'zeros')
    clim = lambda p: db.mean_over_years(tid, p, int(target_year), 40, 365, 'zeros')
    none_if_empty = lambda a: (a if getattr(a, 'shape', [0])[0] else None)
    cum_pos = lambda a: (np.cumsum(np.clip(a, 0, None)) if none_if_empty(a) is not None else None)
    return {'years': yrs, 'x': x, 'y': yv, 'mx_t': mx_t, 'corr': corr, 'exp': exp, 'ef': ef,
            'mean_yield': float(np.mean(yv)),
            'ndvi_cur': none_if_empty(cur(parm)), 'ndvi_clim': none_if_empty(clim(parm)),
            'heat_cur': cum_pos(cur('mean_temp')), 'heat_clim': cum_pos(clim('mean_temp')),
            'prec_cur': cum_pos(cur('mean_prec')), 'prec_clim': cum_pos(clim('mean_prec'))}


def _model_backtests(reg, dist, culture, models, ens_components, target_year, last_k):
    """Для каждой выбранной модели: прогнозы по прошлым годам (бэктест) и уточнённые метрики."""
    cfg = AP.resolve_culture(culture); tid = db.territory_id(reg, dist)
    yields = {int(y): float(v) for y, v in db.get_yields(tid, cfg['bdpmo_culture']).items()
              if v is not None and v == v}
    hist = sorted(y for y in yields if y != int(target_year))[-int(last_k):]
    res = {}
    for m in models:
        preds = {}
        for y in hist:
            try:
                if m == 'ensemble':
                    rr = AP.predict_for_territory(reg, dist, culture, y, models=['ensemble'],
                                                  ensemble_of=ens_components)
                else:
                    rr = AP.predict_for_territory(reg, dist, culture, y, models=[m])
                v = rr['values'].get(m)
            except Exception:                             # noqa: BLE001
                v = None
            if v is not None and v == v:
                preds[y] = float(v)
        ys = [y for y in hist if y in preds]
        if len(ys) >= 2:
            p = np.array([preds[y] for y in ys]); a = np.array([yields[y] for y in ys]); err = p - a
            mse = float(np.mean(err ** 2)); sst = float(np.sum((a - a.mean()) ** 2))
            met = {'MSE': round(mse, 3), 'RMSE': round(float(np.sqrt(mse)), 3),
                   'MAE': round(float(np.mean(np.abs(err))), 3),
                   'R²': round(1 - float(np.sum(err ** 2)) / sst, 3) if sst > 1e-9 else None,
                   'Пирсон': (round(float(np.corrcoef(p, a)[0, 1]), 3)
                              if np.std(p) > 1e-9 and np.std(a) > 1e-9 else None)}
        else:
            met = {'MSE': None, 'RMSE': None, 'MAE': None, 'R²': None, 'Пирсон': None}
        res[m] = {'preds': preds, 'metrics': met}
    return res, yields


def _fig_corr(y_years, y_vals, n_years, n_vals, target_year, target_estimated, outdir, name):
    """Урожайность и max NDVI по годам на двух осях. Линия NDVI включает расчётный год."""
    fig, ax = plt.subplots(figsize=(9, 5)); ax2 = ax.twinx()
    l1 = ax.plot(y_years, y_vals, 'b-o', ms=4, label='урожайность, ц/га')
    l2 = ax2.plot(n_years, n_vals, 'g-s', ms=4, label='max NDVI')
    if target_year in n_years:                            # выделить точку расчётного года
        ax2.scatter([target_year], [n_vals[list(n_years).index(target_year)]],
                    color='green', marker='*', s=140, zorder=5,
                    label=('NDVI расч. года (оценка)' if target_estimated else 'NDVI расч. года'))
    ax.set_xlabel('Год'); ax.set_ylabel('Урожайность, ц/га', color='b'); ax2.set_ylabel('max NDVI', color='g')
    ax.set_title('Урожайность и максимум NDVI по годам'); ax.grid(True, alpha=0.3)
    hs = l1 + l2; ax.legend(hs, [h.get_label() for h in hs], fontsize=8)
    fig.tight_layout(); return _save_fig(fig, outdir, name)


def _fig_two_series(cur, clim, target_year, title, ylabel, outdir, name, color='#d62728'):
    """Ряд расчётного года + многолетнее среднее (по дням года). Цвет линий — по типу величины."""
    if cur is None and clim is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    if cur is not None:
        ax.plot(range(len(cur)), cur, '-', color=color, label=f'{target_year} год')
    if clim is not None:                                   # среднемноголетняя — тот же цвет, пунктир, бледнее
        ax.plot(range(len(clim)), clim, '--', color=color, alpha=0.5, label='среднее за прошлые годы')
    ax.set_xlabel('День года'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    return _save_fig(fig, outdir, name)


def _fig_exp(an, outdir, name):
    x, yv = an['x'], an['y']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, yv, s=28, color='#1f77b4', label='данные')
    if an['ef']:
        xs = np.linspace(float(x.min()), float(x.max()), 60)
        ax.plot(xs, np.clip(np.exp(an['ef'][0] + an['ef'][1] * xs), 0, E.YIELD_MAX), 'g-', label='экспонента')
    ax.set_xlabel('max NDVI'); ax.set_ylabel('Урожайность, ц/га')
    ax.set_title('Экспоненциальная модель аппроксимации'); ax.grid(True, alpha=0.3)
    ax.legend(); fig.tight_layout(); return _save_fig(fig, outdir, name)


def _fig_backtest(bts, yields, target_year, target_preds, outdir, name):
    """Урожайность/год: факт из архива + прогнозы моделей по прошлым годам + прогноз на целевой год."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ya = sorted(yields)
    ax.plot(ya, [yields[y] for y in ya], 'k-o', lw=2, label='урожайность (архив)')
    for m, d in bts.items():
        xs = sorted(d['preds'])
        lbl = AP.token_label(m)
        if xs:
            ax.plot(xs, [d['preds'][y] for y in xs], '--.', label=lbl)
        tv = target_preds.get(m)
        if tv is not None and tv == tv:
            ax.scatter([target_year], [tv], marker='X', s=70, zorder=5)
    ax.axvline(target_year, color='gray', ls=':', lw=1)
    ax.set_xlabel('Год'); ax.set_ylabel('Урожайность, ц/га')
    ax.set_title('Тестирование моделей на прошлых годах и прогноз')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7); fig.tight_layout()
    return _save_fig(fig, outdir, name)


def _fig_history(tid, culture, target_year, outdir, name):
    """Время/прогноз: как менялись прогнозы на этот год, если делались раньше."""
    h = ASt.prediction_history(tid, culture, int(target_year))
    if h is None or h.empty:
        return None
    h = h.dropna(subset=['value'])
    if h.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for mn, g in h.groupby('model_name'):
        g = g.sort_values('made_at')
        ax.plot(pd.to_datetime(g['made_at']), g['value'], '-o', ms=4,
                label=AP.token_label(str(mn).split(':')[-1]))
    ax.set_xlabel('Когда сделан прогноз'); ax.set_ylabel('Прогноз, ц/га')
    ax.set_title(f'История прогнозов на {target_year} год')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7); fig.autofmt_xdate(); fig.tight_layout()
    return _save_fig(fig, outdir, name)


def _gost(doc):
    """Базовое оформление по ГОСТ: Times New Roman 14, полуторный интервал, выравнивание по ширине,
    абзацный отступ; заголовки — по центру, полужирные."""
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    nm = 'Times New Roman'
    st = doc.styles['Normal']
    st.font.name = nm; st.font.size = Pt(14)
    st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), nm)
    pf = st.paragraph_format
    pf.line_spacing = 1.5; pf.first_line_indent = Cm(1.25)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; pf.space_after = Pt(0)
    for h in ('Title', 'Heading 1', 'Heading 2'):
        try:
            s = doc.styles[h]
        except KeyError:
            continue
        s.font.name = nm; s.font.bold = True
        s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), nm)
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.space_before = Pt(12); s.paragraph_format.space_after = Pt(6)


def _add_fig(doc, path, caption):
    """Рисунок по центру + подпись по центру + пустой абзац-отступ (None — пропустить)."""
    from docx.shared import Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    if not path:
        return
    doc.add_picture(path, width=Inches(5.7))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent = Cm(0)
    doc.add_paragraph()                                   # отступ после рисунка


def _add_table(doc, df, title=None):
    """Заголовок (слева) + таблица по центру + пустой абзац-отступ."""
    from docx.shared import Cm, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    if title:
        p = doc.add_paragraph(title)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT; p.paragraph_format.first_line_indent = Cm(0)
    E._table_from_df(doc, df)
    tbl = doc.tables[-1]; tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in tbl.rows:                                   # узкие колонки: без красной строки, шрифт 11
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.first_line_indent = Cm(0)
                for run in par.runs:
                    run.font.size = Pt(11)
    doc.add_paragraph()                                   # отступ после таблицы


def batch_report(batch_id, last_k=6, outdir=None, progress=None):
    """Отчёт по образцам: Excel-таблица (формат шаблона) + Word по территориям.

    Word на каждую территорию: динамика урожай↔maxNDVI, NDVI (год+среднее), накопленные тепло и
    осадки (год+среднее), экспоненциальная модель и её параметры, таблица прогнозов и уточнённых
    метрик выбранных моделей, бэктест моделей по годам и история прогнозов.
    Графики — в подпапках <outdir>/png и <outdir>/pdf. Возвращает (xlsx, docx, таблица)."""
    from docx import Document
    from docx.shared import Inches
    outdir = outdir or os.path.join(config.REPORTS_DIR, 'app_forecasts')
    os.makedirs(outdir, exist_ok=True)
    table, culture = batch_table(batch_id, last_k)
    base = f'batch_{batch_id}_{culture}'.replace(' ', '_')
    xlsx_path = _write_xlsx(table, os.path.join(outdir, base + '.xlsx'))

    bp = ASt.batch_predictions(batch_id).copy()
    bp['mkey'] = bp['model_name'].apply(lambda s: str(s).split(':')[-1])
    model_keys = sorted(set(bp['mkey'].astype(str)), key=AP.token_sort_key)
    ens_components = [m for m in model_keys if m != 'ensemble']
    years = sorted(int(y) for y in bp['predict_year'].dropna().unique())
    target_year = max(years) if years else None
    info = ASt.list_forecast_batches(); meta = info[info['batch_id'] == str(batch_id)]

    doc = Document()
    _gost(doc)
    doc.add_heading(culture, 0)
    if not meta.empty:
        m = meta.iloc[0]
        doc.add_paragraph(f'Партия: {batch_id}. Сделан: {m["когда"]}, автор: {m["автор"]}. '
                          f'Территорий: {int(m["территорий"])}. Расчётный год: {target_year}. '
                          f'Табличные данные — в файле «{os.path.basename(xlsx_path)}».')

    total = len(table)
    for i, (_, trow) in enumerate(table.iterrows(), 1):
        if progress:
            progress(i, total)
        if i > 1:
            doc.add_page_break()                          # каждый субъект — с новой страницы
        reg = int(trow['id_region']); dd = trow['id_district']
        dist = None if dd == '' else int(dd)
        name = trow['region'] if dist is None else f'{trow["region"]} / {trow["district"]}'
        doc.add_heading(str(name), level=1)
        slug = _safe_dir(f'{batch_id}_{reg}_{dd or "reg"}')
        tid = db.territory_id(reg, dist)
        target_preds = {m: trow.get(f'{m}_predict_{target_year}') for m in model_keys}

        an = _ndvi_analysis(reg, dist, culture, target_year)
        if an is not None:
            yrs = an['years']
            doc.add_paragraph(f'Выборка для регрессии: {len(yrs)}. Данные с {max(yrs)} по {min(yrs)} год.')
            if an['corr'] is not None:
                doc.add_paragraph(f'Коэффициент корреляции урожая ({culture}) и максимального значения '
                                  f'вегетационного индекса NDVI: {an["corr"]:.3f}')
            # NDVI расчётного года: измеренный, иначе оценка = среднее max NDVI за последние 4 года
            mx_plot = an['mx_t']; estimated = False
            if mx_plot is None and len(an['x']):
                mx_plot = float(np.mean(an['x'][-4:])); estimated = True
            if an['mx_t'] is not None:
                doc.add_paragraph(f'Максимальное значение NDVI, измеренное в {target_year} году: {an["mx_t"]:.3f}')
            elif mx_plot is not None:
                doc.add_paragraph(f'Значение NDVI на {target_year} год неизвестно; использовано среднее '
                                  f'за последние годы: {mx_plot:.3f}')
            # (1) динамика урожай↔maxNDVI по годам (NDVI включает расчётный год)
            add_t = mx_plot is not None and target_year not in yrs
            n_years = list(yrs) + ([target_year] if add_t else [])
            n_vals = list(an['x']) + ([mx_plot] if add_t else [])
            _add_fig(doc, _fig_corr(yrs, an['y'], n_years, n_vals, target_year, estimated, outdir, f'{slug}_corr'),
                     'Рис. Динамика урожайности и максимальных значений NDVI по годам.')
            # (2) NDVI года + многолетнее среднее
            _add_fig(doc, _fig_two_series(an['ndvi_cur'], an['ndvi_clim'], target_year,
                     f'NDVI: {target_year} год и многолетнее среднее', 'NDVI', outdir, f'{slug}_ndvi', '#2ca02c'),
                     f'Рис. Вегетационный индекс NDVI в {target_year} году и среднемноголетний.')
            # (4) накопленная теплота
            _add_fig(doc, _fig_two_series(an['heat_cur'], an['heat_clim'], target_year,
                     'Накопленная положительная температура', '∑ t⁺, °C·сут', outdir, f'{slug}_heat', '#d62728'),
                     'Рис. Накопленная сумма положительных температур (год и среднемноголетняя).')
            # (5) накопленные осадки
            _add_fig(doc, _fig_two_series(an['prec_cur'], an['prec_clim'], target_year,
                     'Накопленные осадки', '∑ осадков, мм', outdir, f'{slug}_prec', '#1f77b4'),
                     'Рис. Накопленные осадки (год и среднемноголетние).')
            # (3) экспоненциальная модель + параметры
            _add_fig(doc, _fig_exp(an, outdir, f'{slug}_exp'),
                     'Рис. Экспоненциальная модель аппроксимации зависимости урожайности от max NDVI.')
            if an['exp']:
                e = an['exp']
                _add_table(doc, pd.DataFrame([{'f(x)': e['f'],
                    'R^2': round(e['r2'], 3) if e['r2'] is not None else None,
                    'MAE': round(e['mae'], 3), 'STD': round(e['std'], 3), 'IQR': round(e['iqr'], 3),
                    'Ожидаемая урожайность': e['pred']}]),
                    'Таблица – Параметры аппроксимации (экспоненциальная модель).')
        else:
            doc.add_paragraph('Недостаточно данных NDVI/урожайности для регрессии по этой территории.')

        # (6) таблица прогнозов и уточнённых метрик выбранных моделей
        bts, yields = _model_backtests(reg, dist, culture, model_keys, ens_components, target_year, last_k)
        mrows = []
        for m in model_keys:
            met = bts[m]['metrics']; tv = target_preds.get(m)
            mrows.append({'модель': AP.token_label(m),
                          f'прогноз {target_year}, ц/га': round(float(tv), 2) if tv is not None and tv == tv else None,
                          'MSE': met['MSE'], 'RMSE': met['RMSE'], 'MAE': met['MAE'],
                          'R²': met['R²'], 'Пирсон': met['Пирсон']})
        _add_table(doc, pd.DataFrame(mrows), 'Таблица – Прогнозы и уточнённые метрики моделей.')
        # (7) тестирование моделей по годам
        if yields:
            _add_fig(doc, _fig_backtest(bts, yields, target_year, target_preds, outdir, f'{slug}_backtest'),
                     'Рис. Тестирование моделей на прошлых годах и прогноз на расчётный год.')
        # (8) история прогнозов на этот год
        _add_fig(doc, _fig_history(tid, culture, target_year, outdir, f'{slug}_history'),
                 'Рис. Как менялись прогнозы на расчётный год (по времени).')
        if an is not None:
            doc.add_paragraph(f'Средняя урожайность за прошлые годы: {an["mean_yield"]:.3f}')

    docx_path = _save_docx(doc, os.path.join(outdir, base + '.docx'))
    return xlsx_path, docx_path, table


def _save_docx(doc, out):
    """Атомарное сохранение: пишем во временный файл и подменяем целевой -> частичных/битых .docx не бывает."""
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    tmp = out + '.tmp'
    doc.save(tmp)                                          # полный файл во временный
    try:
        os.replace(tmp, out)                               # атомарная замена
        return out
    except PermissionError:                                # целевой открыт в Word — сохраняем рядом
        alt = out.replace('.docx', '_new.docx')
        os.replace(tmp, alt)
        return alt
