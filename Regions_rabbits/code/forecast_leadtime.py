"""Эксперимент: как падает точность нейросети при ЗАБЛАГОВРЕМЕННОМ прогнозе.

Матрица примера = [целевой год | год−1 | год−2 | …] (concat_newest_first). Прогноз «сейчас»
делается в какой-то день целевого года: дни ДО этой даты — реальные, дни ПОСЛЕ (будущее) —
заменяются усреднённым МНОГОЛЕТНИМ графиком района (среднее по блокам прошлых лет — климатология).
Чем раньше дата прогноза, тем больше хвоста заменено -> тем «заблаговременнее» прогноз.

Сдвигаем дату прогноза по целевому году и меряем RMSE/R²/MSE на тесте. Результат — таблица и график
зависимости точности от заблаговременности.

Запуск:
    python code/forecast_leadtime.py winter_wheat_all_nn/v1     # конкретная версия
    python code/forecast_leadtime.py winter_wheat_all           # последняя версия группы <task>_nn

Выход: workspace/reports/forecast/<model>/ — leadtime.csv, leadtime.png, report_*.docx
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
import explore_models as E
import plots
import train_nn as T


def resolve_model(arg):
    """arg -> относительный путь модели 'group/vN' (берёт последнюю версию, если дана группа/task)."""
    base = config.MODELS_DIR
    direct = os.path.join(base, *arg.split('/'))
    if os.path.exists(os.path.join(direct, 'model.pt')):
        return arg.replace('\\', '/')
    group = arg if arg.endswith('_nn') else f'{arg}_nn'
    gdir = os.path.join(base, group)
    if os.path.isdir(gdir):
        vs = [d for d in os.listdir(gdir)
              if d.startswith('v') and d[1:].isdigit()
              and os.path.exists(os.path.join(gdir, d, 'model.pt'))]
        if vs:
            return f'{group}/' + max(vs, key=lambda d: int(d[1:]))
    raise SystemExit(f'Не нашёл модель для "{arg}" в {base}')


def load_model(rel):
    mdir = os.path.join(config.MODELS_DIR, *rel.split('/'))
    arch = json.load(open(os.path.join(mdir, 'architecture.json'), encoding='utf-8'))
    det = json.load(open(os.path.join(mdir, 'details.json'), encoding='utf-8'))
    model, *_ = T.make_model(arch['arch_name'], arch['in_channels'], arch['in_length'])
    model.load_state_dict(torch.load(os.path.join(mdir, 'model.pt'), map_location='cpu'))
    model.eval()
    return model, arch, det


def run(arg, step_days):
    rel = resolve_model(arg)
    model, arch, det = load_model(rel)
    task, seed = det['task'], int(det.get('seed', 42))
    size = int(det['time_rows_size'])                 # длина одного года (точек)
    print(f'Модель: {rel} (культура {task}); готовлю данные…')

    P = E.prepare_data(task, seed)
    Xn = P['Xn'].astype(np.float32)
    y = P['y'].astype(np.float32)
    te = P['te']
    F, Tlen = Xn.shape[1], Xn.shape[2]
    n_years = Tlen // size
    if n_years < 2:
        raise SystemExit('Нужно >=2 лет в ряду для климатологии (среднего по прошлым годам).')

    # индекс блока ЦЕЛЕВОГО года зависит от порядка склейки (порядко-независимо):
    newest_first = bool(_dataset_meta(task).get('concat_newest_first', True))
    tgt_idx = 0 if newest_first else n_years - 1      # newest_first: первый блок; иначе последний
    tgt0 = tgt_idx * size                             # начало целевого блока в плоском ряду

    # климатология типичного года (среднее по прошлым блокам), РАСТЯНУТАЯ на весь ряд (n_years лет)
    blocks = Xn.reshape(Xn.shape[0], F, n_years, size)
    clim_year = np.delete(blocks, tgt_idx, axis=2).mean(axis=2)        # (n, F, size)
    clim_full = np.tile(clim_year, (1, 1, n_years))                   # (n, F, n_years*size)

    Xte, yte, clim_te = Xn[te], y[te], clim_full[te]
    H = _harvest_doy(task)                            # день года уборки в целевом году
    harvest_abs = tgt0 + H                            # абсолютный день уборки во всём ряду
    print(f'      уборка ~день года {H} (абс. {harvest_abs}); заблаговременность по ВСЕМУ ряду '
          f'({n_years} г., помесячно), шаг {step_days} дн.')

    # «сейчас» = абс. день a; известно 0..a, всё ПОСЛЕ -> климатология. lead = harvest_abs - a.
    leads = sorted(set(range(0, harvest_abs + 1, step_days)) | {harvest_abs})
    rows = []
    for lead_d in leads:
        a = max(0, min(Tlen, harvest_abs - lead_d))
        Xm = Xte.copy()
        if a < Tlen:
            Xm[:, :, a:] = clim_te[:, :, a:]
        pred = T.predict(model, Xm, 'cpu')
        m = E.metrics(yte, pred)
        rows.append({'days_before_harvest': lead_d, 'months_before': round(lead_d / 30.44, 1),
                     'cutoff_abs_day': a, **m})
    df = pd.DataFrame(rows).sort_values('days_before_harvest').reset_index(drop=True)

    outdir = os.path.join(config.REPORTS_DIR, 'forecast', rel.replace('/', '_'))
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, 'leadtime.csv'), index=False, encoding='utf-8-sig')
    culture = _culture(task)

    # графики ошибки (RMSE и MSE) + R² от «дней до уборки» (ось инвертирована: время → к уборке вправо)
    def _fig(metric, color):
        def draw(lang):
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.plot(df.days_before_harvest, df[metric], 'o-', color=color, label=metric)
            ax1.set_xlabel(plots.tr('Дней до уборки (время →, ближе к уборке)',
                                    'Days before harvest (time →, toward harvest)', lang))
            ax1.set_ylabel(metric, color=color); ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.plot(df.days_before_harvest, df.R2, 's-', color='tab:blue', label='R²')
            ax2.set_ylabel('R²', color='tab:blue'); ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax1.invert_xaxis()                        # 0 (уборка) СПРАВА, рано — слева (ось общая для twinx)
            ax1.set_title(plots.tr(f'Заблаговременность ({metric}) — {culture}, уборка ~д.{H}',
                                   f'Lead time ({metric}) — {task}, harvest ~DOY {H}', lang))
            fig.tight_layout()
            return fig
        return draw

    png_rmse = plots.bilingual(outdir, 'leadtime_rmse', _fig('RMSE', 'tab:red'))
    png_mse = plots.bilingual(outdir, 'leadtime_mse', _fig('MSE', 'tab:orange'))

    _docx(rel, task, det, df, [png_rmse, png_mse], outdir)

    base = df[df.days_before_harvest == 0].iloc[0]
    print(df[['days_before_harvest', 'months_before', 'cutoff_abs_day', 'MSE', 'RMSE', 'R2']].to_string(index=False))
    print(f'\nВ день уборки (lead=0): R²={base.R2:.3f}, RMSE={base.RMSE:.3f}, MSE={base.MSE:.3f}')
    print(f'Артефакты: {outdir}')
    return rel, task, culture, df


def _dataset_meta(task):
    p = os.path.join(config.dataset_dir(task), 'meta.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}


def _culture(task):
    return _dataset_meta(task).get('culture_title', task)


_HARVEST = None


def _harvest_doy(task, default=240):
    """День года уборки культуры из workspace/config/harvest_dates.csv (по task)."""
    global _HARVEST
    if _HARVEST is None:
        p = os.path.join(config.CONFIG_DIR, 'harvest_dates.csv')
        _HARVEST = {}
        if os.path.exists(p):
            d = pd.read_csv(p, encoding='utf-8-sig')
            _HARVEST = {str(r['task']): int(r['harvest_doy']) for _, r in d.iterrows()}
    if task not in _HARVEST:
        print(f'      ! нет даты уборки для {task} — беру день {default}; добавьте в harvest_dates.csv')
    return _HARVEST.get(task, default)


def run_all(step_days):
    rels = []
    for g in sorted(os.listdir(config.MODELS_DIR)):
        if not os.path.isdir(os.path.join(config.MODELS_DIR, g)):
            continue
        try:
            rels.append(resolve_model(g))
        except SystemExit:
            pass
    results = []
    for rel in rels:
        print(f'\n===== {rel} =====')
        results.append(run(rel, step_days))
    aggregate(results)


def aggregate_from_saved():
    """Пересобрать сводку из уже сохранённых leadtime.csv (без перерасчёта предсказаний)."""
    import re
    fdir = os.path.join(config.REPORTS_DIR, 'forecast')
    results = []
    for d in sorted(os.listdir(fdir)):
        p = os.path.join(fdir, d, 'leadtime.csv')
        if d == '_summary' or not os.path.exists(p):
            continue
        m = re.match(r'(.+)_v\d+$', d)
        group = m.group(1) if m else d
        task = group[:-3] if group.endswith('_nn') else group
        results.append((d, task, _culture(task), pd.read_csv(p)))
    if results:
        aggregate(results)


def aggregate(results):
    outdir = os.path.join(config.REPORTS_DIR, 'forecast', '_summary')
    os.makedirs(outdir, exist_ok=True)

    # сводный график R² от «дней до уборки» — ось инвертирована (к уборке = вправо)
    def draw_all(lang):
        fig, ax = plt.subplots(figsize=(11, 6))
        for _rel, task, culture, df in results:
            d = df.sort_values('days_before_harvest')
            ax.plot(d.days_before_harvest, d.R2, marker='.', label=culture if lang == 'ru' else task)
        ax.invert_xaxis()
        ax.set_xlabel(plots.tr('Дней до уборки (время →, ближе к уборке)',
                               'Days before harvest (time →, toward harvest)', lang))
        ax.set_ylabel('R² (test)')
        ax.set_title(plots.tr('Заблаговременность прогноза по культурам',
                              'Forecast lead time by culture', lang))
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2, loc='lower left')
        fig.tight_layout()
        return fig
    png_r2 = plots.bilingual(outdir, 'leadtime_all', draw_all)

    # сводный график MSE — лог-шкала Y (MSE между культурами различается на порядки)
    def draw_all_mse(lang):
        fig, ax = plt.subplots(figsize=(11, 6))
        for _rel, task, culture, df in results:
            d = df.sort_values('days_before_harvest')
            ax.plot(d.days_before_harvest, d.MSE, marker='.', label=culture if lang == 'ru' else task)
        ax.set_yscale('log'); ax.invert_xaxis()
        ax.set_xlabel(plots.tr('Дней до уборки (время →, ближе к уборке)',
                               'Days before harvest (time →, toward harvest)', lang))
        ax.set_ylabel(plots.tr('MSE (test, лог. шкала)', 'MSE (test, log scale)', lang))
        ax.set_title(plots.tr('Заблаговременность прогноза по культурам (MSE)',
                              'Forecast lead time by culture (MSE)', lang))
        ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=7, ncol=2, loc='upper left')
        fig.tight_layout()
        return fig
    png_mse = plots.bilingual(outdir, 'leadtime_all_mse', draw_all_mse)

    # сводная таблица: R² за N дней до уборки
    targets = [0, 90, 180, 365, 545, 730]            # уборка, 3/6/12/18/24 мес. до неё
    rows = []
    for _rel, task, culture, df in results:
        rec = {'culture': culture, 'task': task}
        for t in targets:
            i = (df.days_before_harvest - t).abs().idxmin()
            rec[f'R2_d{t}'] = round(float(df.loc[i, 'R2']), 3)
        rec['drop_0_to_730'] = round(rec['R2_d0'] - rec['R2_d730'], 3)
        rows.append(rec)
    tab = pd.DataFrame(rows).sort_values('R2_d0', ascending=False).reset_index(drop=True)
    tab.to_csv(os.path.join(outdir, 'leadtime_summary.csv'), index=False, encoding='utf-8-sig')

    _docx_all(tab, [png_r2, png_mse], outdir)
    print('\n=== Сводка: R² за N дней ДО уборки ===')
    print(tab.to_string(index=False))
    print(f'\nСводные артефакты: {outdir}')


def _docx_all(tab, pngs, outdir):
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    doc.add_heading('Заблаговременность прогноза — сводка по культурам', 0)
    doc.add_paragraph('Тест на ТЕСТОВОЙ выборке. Заблаговременность считается относительно ДАТЫ УБОРКИ '
                      'культуры (workspace/config/harvest_dates.csv). Прогноз за N дней до уборки: '
                      'известна часть сезона до даты прогноза, остальное заменено многолетним средним '
                      'графиком района. R2_d0 — в день уборки; R2_d730 — за 2 года (≈24 мес.) до неё.')
    E._table_from_df(doc, tab)
    for png in pngs:
        if png:
            doc.add_picture(png, width=Inches(6.5))
    doc.save(os.path.join(outdir, 'leadtime_summary.docx'))


def _docx(rel, task, det, df, pngs, outdir):
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    doc.add_heading('Заблаговременность прогноза', 0)
    doc.add_paragraph(f'Модель: {rel} — культура {task} (arch={det.get("arch")}, '
                      f'lr={det.get("lr")}, batch={det.get("batch")}).')
    doc.add_paragraph('Заблаговременность считается относительно ДАТЫ УБОРКИ. Прогноз за N дней до '
                      'уборки: известна часть сезона до даты прогноза, остальное заменено многолетним '
                      'средним графиком района. Чем ближе к уборке (правее по оси), тем выше точность.')
    base = df[df.days_before_harvest == 0].iloc[0]
    far = df.iloc[(df.days_before_harvest - 365).abs().idxmin()]
    doc.add_paragraph(f'В день уборки: R²={base.R2:.3f}, RMSE={base.RMSE:.3f}, MSE={base.MSE:.3f}. '
                      f'За ~{int(far.days_before_harvest)} дней до уборки: R²={far.R2:.3f}, '
                      f'RMSE={far.RMSE:.3f}.')
    E._table_from_df(doc, df.round(3))
    for png in pngs:
        if png:
            doc.add_picture(png, width=Inches(6.0))
    p = os.path.join(outdir, f'report_{rel.replace("/", "_")}.docx')
    doc.save(p)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Тест заблаговременности прогноза нейросети.')
    ap.add_argument('model', help='модель: group/vN, group, task или "all" (все сети)')
    ap.add_argument('--step-days', type=int, default=30, help='шаг сетки дат прогноза, дней (по умолч. ~месяц)')
    a = ap.parse_args()
    if a.model == 'all':
        run_all(a.step_days)
    elif a.model == 'resummary':
        aggregate_from_saved()
    else:
        run(a.model, a.step_days)
