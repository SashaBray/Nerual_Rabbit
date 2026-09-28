"""Выгрузка прогноза (партии) из БД в самостоятельную папку-артефакт.

Партия прогноза живёт в БД (таблицы ``predictions`` / ``territory_metrics``), а поделиться
ею надо целиком: таблицы, метаданные, отчёт. Модуль собирает один стандартный каталог —
его можно заархивировать и отправить коллеге, и он читается без самой программы:

    <Прогноз ...>/
        README.txt              — что внутри и как читать (обычный текст)
        meta.json               — машинные метаданные партии (формат, модели, автор, файлы)
        predictions.csv         — сводная таблица прогноза (';' и десятичная запятая, для Excel)
        predictions.xlsx        — та же таблица книгой Excel
        predictions_raw.csv     — строки прогнозов как они лежат в БД
        territory_metrics.csv   — уточнённые метрики территорий партии
        models.csv              — какие модели участвовали: версия, заметка, точность
        yields_used.csv         — урожайности, на которых построен прогноз (с источником)
        report.docx             — Word-отчёт по территориям (если включён)
        png/, pdf/              — графики отчёта

Состав фиксирован: ``FORMAT`` + ``FORMAT_VERSION`` в ``meta.json`` позволяют отличить версию
формата, если состав когда-то изменится.
"""
import json
import os
import shutil
from datetime import datetime

import pandas as pd

import app_predict as P
import app_reports as R
import app_store as ASt
import app_yields as Y
import config
import db

FORMAT = 'nerual-rabbit-forecast-export'
FORMAT_VERSION = 1

# Имя файла -> описание (идёт и в README, и в meta.json)
FILE_NOTES = {
    'predictions.csv': 'сводная таблица прогноза: территории, урожайности прошлых лет, '
                       'прогнозы моделей, метрики (разделитель «;», десятичная запятая)',
    'predictions.xlsx': 'та же сводная таблица книгой Excel',
    'predictions_raw.csv': 'прогнозы как они хранятся в БД (по строке на модель/год/территорию)',
    'territory_metrics.csv': 'уточнённые MSE/RMSE/MAE/R²/Пирсон по территориям партии',
    'models.csv': 'модели партии: версия, дата, заметка о версии, точность по бэктесту',
    'yields_used.csv': 'урожайности, использованные при расчёте, с указанием источника',
    'outcomes.csv': 'план и итог партии: по строке на запланированный прогноз — рассчитан ли и почему нет',
    'report.docx': 'Word-отчёт: по главе на территорию (графики, регрессия, бэктест)',
    'meta.json': 'метаданные партии в машинном виде',
    'README.txt': 'это описание',
}


def exports_dir():
    """Каталог выгрузок по умолчанию (workspace/exports), создаётся при первом обращении."""
    d = getattr(config, 'EXPORTS_DIR', os.path.join(config.WORKSPACE, 'exports'))
    os.makedirs(d, exist_ok=True)
    return d


def suggest_folder_name(batch_id, culture=None, made_at=None):
    """Предлагаемое имя папки выгрузки: «Прогноз <культура> <дата> <batch_id>»."""
    if culture is None or made_at is None:
        info = ASt.list_forecast_batches()
        hit = info[info['batch_id'].astype(str) == str(batch_id)] if not info.empty else info
        if not hit.empty:
            culture = culture or str(hit.iloc[0].get('культура') or '')
            made_at = made_at or str(hit.iloc[0].get('когда') or '')
    day = str(made_at or '')[:10]
    return R._safe_dir(' '.join(x for x in ('Прогноз', culture, day, str(batch_id)) if x))


def _batch_info(batch_id):
    """Строка сводки партии из БД как dict (или пустой dict)."""
    info = ASt.list_forecast_batches()
    if info.empty:
        return {}
    hit = info[info['batch_id'].astype(str) == str(batch_id)]
    return hit.iloc[0].to_dict() if not hit.empty else {}


def _models_frame(batch_id, culture):
    """Состав моделей партии: токен, название, папка, дата, заметка, точность."""
    bp = ASt.batch_predictions(batch_id)
    tokens = sorted({str(m).split(':')[-1] for m in bp['model_name'].dropna()}, key=P.token_sort_key)
    metrics = db.get_model_metrics(culture)
    rnd = lambda v: None if v is None or v != v else round(float(v), 3)
    rows = []
    for tok in tokens:
        met = metrics.get((str(culture), tok), {})
        folder = P.model_dir(tok)
        rows.append({
            'токен': tok,
            'модель': P.token_label(tok),
            'тип': P.parse_token(tok)[0],
            'папка модели': os.path.basename(folder) if folder else '',
            'создана': P.model_created_at(tok),
            'комментарий': P.get_model_comment(tok),
            'из метаданных': P.model_autonote(tok),
            'MSE': rnd(met.get('mse')), 'RMSE': rnd(met.get('rmse')), 'R²': rnd(met.get('r2')),
            'оценка n': met.get('n'),
        })
    return pd.DataFrame(rows)


def _yields_frame(batch_id, culture):
    """Урожайности территорий партии с источником значения (БД источников / ручной ввод)."""
    import app_geo as G

    bp = ASt.batch_predictions(batch_id)
    cfg = P.resolve_culture(culture)
    cols = ['territory_id', 'регион', 'район', 'культура', 'год', 'урожайность, ц/га', 'источник']
    rows = []
    for tid in sorted(bp['territory_id'].unique()):
        reg, dist = str(tid).split('_')
        did = None if dist == 'nan' else int(float(dist))
        name_r = G.region_name(int(reg))
        name_d = R._district_name(int(reg), did)
        for year, (value, src) in sorted(db.get_yields_detailed(tid, cfg['bdpmo_culture']).items()):
            if value is None or value != value:
                continue
            rows.append({'territory_id': tid, 'регион': name_r, 'район': name_d,
                         'культура': cfg['bdpmo_culture'], 'год': year,
                         'урожайность, ц/га': round(float(value), 3),
                         'источник': 'ручной ввод' if src == Y.SRC_USER else 'база источников'})
    return pd.DataFrame(rows, columns=cols)               # пустая — но с заголовками


def _csv(df, path):
    """CSV в русской локали Excel: ';' как разделитель, ',' как десятичный, BOM."""
    df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
    return path


def _outcome_line(outcome):
    """Строка итога партии для README (или пояснение, что учёта ещё не было)."""
    if not outcome:
        return 'сведений о плане нет (партия сделана до учёта причин нерасчёта)'
    return f'рассчитано {outcome["done"]} из {outcome["planned"]}; причины нерасчёта — outcomes.csv'


def _readme_text(meta):
    """Человекочитаемое описание папки (README.txt)."""
    lines = [
        'ПРОГНОЗ УРОЖАЙНОСТИ — выгрузка из программы Nerual Rabbit',
        '=' * 60,
        '',
        f'Культура:        {meta["culture"]}',
        f'Расчётные годы:  {", ".join(str(y) for y in meta["predict_years"])}',
        f'Территорий:      {meta["territories"]}   прогнозов: {meta["predictions"]}',
        f'Партия (id):     {meta["batch_id"]}',
        f'Сделан:          {meta["made_at"]}   автор: {meta["author"]}',
        f'Комментарий:     {meta["comment"] or "—"}',
        f'Выгружено:       {meta["exported_at"]}   кем: {meta["exported_by"]}',
        f'Итог:            {_outcome_line(meta.get("outcome"))}',
        '',
        'МОДЕЛИ',
        '-' * 60,
    ]
    for m in meta['models']:
        head = f'  {m["модель"]}'
        if m.get('создана'):
            head += f' (сохранена {m["создана"]})'
        lines.append(head)
        if m.get('комментарий'):
            lines.append(f'      заметка: {m["комментарий"]}')
        if m.get('из метаданных'):
            lines.append(f'      параметры: {m["из метаданных"]}')
        acc = ', '.join(f'{k} {m[k]}' for k in ('MSE', 'RMSE', 'R²') if m.get(k) is not None)
        if acc:
            lines.append(f'      точность по бэктесту: {acc}')
    lines += ['', 'ФАЙЛЫ', '-' * 60]
    for f in meta['files']:
        lines.append(f'  {f["name"]:<22} — {f["note"]}')
    lines += [
        '',
        'КАК ЧИТАТЬ',
        '-' * 60,
        '  Урожайность везде в ц/га. Столбцы вида <модель>_predict_<год> — прогноз этой модели',
        '  на этот год; столбцы-годы (2019, 2020, ...) — фактическая урожайность из архива.',
        '  Метрики (MSE/RMSE/MAE/R²/Пирсон) посчитаны по прошлым годам той же территории:',
        '  это оценка точности прогноза именно здесь, а не по стране в целом.',
        '  Ансамбль — среднее перечисленных в meta.json моделей.',
        '  В yields_used.csv видно, какие урожайности взяты из источников (ЕМИСС, архив БДПМО),',
        '  а какие введены оператором вручную.',
        '',
        f'  Формат выгрузки: {meta["format"]} v{meta["format_version"]}.',
    ]
    return '\n'.join(lines)


def export_batch(batch_id, out_parent=None, folder_name=None, last_k=6, with_report=True,
                 make_zip=True, exported_by=None, progress=None, message=None):
    """Собрать папку-артефакт с прогнозом партии (и, по желанию, ZIP рядом).

    ``progress(pct)`` — 0..100, ``message(text)`` — комментарий к текущему шагу.
    Возвращает dict: ``dir``, ``zip``, ``files``, ``meta``, ``table``.
    """
    say = message or (lambda _t: None)
    step = progress or (lambda _p: None)

    info = _batch_info(batch_id)
    culture = str(info.get('культура') or '')
    if not culture:                                        # сводка не нашлась — берём из самих прогнозов
        bp0 = ASt.batch_predictions(batch_id)
        if bp0.empty:
            raise ValueError(f'В партии {batch_id} нет прогнозов.')
        culture = str(bp0['culture'].dropna().iloc[0])

    out_parent = out_parent or exports_dir()
    outdir = os.path.join(out_parent, R._safe_dir(folder_name or suggest_folder_name(
        batch_id, culture, info.get('когда'))))
    os.makedirs(outdir, exist_ok=True)
    say(f'Папка выгрузки: {outdir}')

    # --- таблицы (быстрая часть) ---
    step(5); say('Сводная таблица прогноза…')
    if with_report:
        step(10); say('Отчёт по территориям (графики и Word) — это самая долгая часть…')

        def rep_progress(i, total):
            step(10 + int(70 * i / max(total, 1)))
            say(f'   территория {i} из {total}')

        xlsx_path, docx_path, table = R.batch_report(batch_id, last_k=last_k, outdir=outdir,
                                                     progress=rep_progress)
        os.replace(xlsx_path, os.path.join(outdir, 'predictions.xlsx'))
        os.replace(docx_path, os.path.join(outdir, 'report.docx'))
    else:
        table, _cult = R.batch_table(batch_id, last_k=last_k)
        R._write_xlsx(table, os.path.join(outdir, 'predictions.xlsx'))
        docx_path = None

    step(82); say('Таблицы CSV и метаданные…')
    _csv(table, os.path.join(outdir, 'predictions.csv'))

    bp = ASt.batch_predictions(batch_id)
    _csv(bp, os.path.join(outdir, 'predictions_raw.csv'))

    tm = db.load('territory_metrics')
    if not tm.empty and 'batch_id' in tm:
        tm = tm[tm['batch_id'].astype(str) == str(batch_id)]
    _csv(tm, os.path.join(outdir, 'territory_metrics.csv'))

    models = _models_frame(batch_id, culture)
    _csv(models, os.path.join(outdir, 'models.csv'))

    yl = _yields_frame(batch_id, culture)
    _csv(yl, os.path.join(outdir, 'yields_used.csv'))

    import app_preflight as PF                             # план и итог партии (если учёт уже вёлся)
    oc = db.load(PF.OUTCOMES_TABLE)
    oc = oc[oc['batch_id'].astype(str) == str(batch_id)] if not oc.empty and 'batch_id' in oc else oc
    has_outcomes = not oc.empty
    if has_outcomes:
        _csv(oc.assign(модель=oc['model'].map(P.token_label))[
            ['label', 'year', 'модель', 'planned', 'status', 'reason']].rename(columns={
                'label': 'территория', 'year': 'год', 'planned': 'по плану',
                'status': 'итог', 'reason': 'почему не рассчитан'}),
            os.path.join(outdir, 'outcomes.csv'))
    outcome_summary, _reasons, _failed = PF.batch_outcomes(batch_id)

    years = sorted(int(y) for y in bp['predict_year'].dropna().unique())
    manual = int((yl['источник'] == 'ручной ввод').sum()) if not yl.empty else 0
    names = (['predictions.csv', 'predictions.xlsx', 'predictions_raw.csv', 'territory_metrics.csv',
              'models.csv', 'yields_used.csv'] + (['outcomes.csv'] if has_outcomes else [])
             + (['report.docx'] if with_report else []) + ['meta.json', 'README.txt'])
    meta = {
        'format': FORMAT, 'format_version': FORMAT_VERSION,
        'batch_id': str(batch_id), 'culture': culture, 'predict_years': years,
        'made_at': str(info.get('когда') or ''), 'author': str(info.get('автор') or ''),
        'comment': str(info.get('комментарий') or ''),
        'territories': int(bp['territory_id'].nunique()), 'predictions': int(len(bp)),
        'yields_rows': int(len(yl)), 'yields_manual': manual,
        'yield_unit': 'ц/га', 'metrics_depth_years': last_k,
        'models': models.to_dict('records'),
        'outcome': outcome_summary,                        # сколько рассчитано из запланированного
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'exported_by': str(exported_by or ''),
        'app': 'Nerual Rabbit',
        'files': [{'name': n, 'note': FILE_NOTES.get(n, '')} for n in names],
    }
    with open(os.path.join(outdir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(_readme_text(meta))

    # размеры файлов — уже после записи всех
    for item in meta['files']:
        path = os.path.join(outdir, item['name'])
        item['bytes'] = os.path.getsize(path) if os.path.exists(path) else 0

    zip_path = None
    if make_zip:
        step(92); say('Архив ZIP…')
        zip_path = shutil.make_archive(outdir, 'zip', root_dir=os.path.dirname(outdir),
                                       base_dir=os.path.basename(outdir))
    step(100); say('Выгрузка готова')
    return {'dir': outdir, 'zip': zip_path, 'files': meta['files'], 'meta': meta, 'table': table}
