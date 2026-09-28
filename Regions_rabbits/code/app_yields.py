"""Проверка наличия урожайностей и ручной ввод недостающих (перед прогнозом).

Раздел отвечает на вопрос «какие данные об урожайности уже есть по культуре в выбранных
регионах и чего не хватает», и позволяет вписать недостающие значения руками. Ручной ввод
хранится в ОТДЕЛЬНОЙ таблице БД ``user_yields``: загрузчики источников (ЕМИСС, архив БДПМО)
переписывают ``yields`` целиком, а введённое оператором должно переживать переподгрузку и
работать, когда сервисы-источники недоступны (см. ``db.get_yields`` — там значения
пользователя накладываются поверх источника).

Собственных файлов модуль не создаёт — только таблицы БД.
"""
import app_geo as G
import app_predict as P
import db

SRC_SOURCE = 'source'      # значение пришло из источника (таблица yields)
SRC_USER = 'user'          # значение введено оператором (таблица user_yields)


def culture_key(culture_title):
    """Название культуры в интерфейсе -> ключ культуры в таблицах урожайности (bdpmo_culture)."""
    return P.resolve_culture(culture_title)['bdpmo_culture']


def _rows_for(id_regions, include_districts):
    """Список территорий таблицы: [(territory_id, id_region, id_district|None, подпись)]."""
    names = dict(G.list_regions())
    out = []
    for rid in sorted(set(int(r) for r in id_regions), key=lambda r: names.get(r, str(r))):
        rname = names.get(rid, str(rid))
        out.append((db.territory_id(rid, None), rid, None, f'{rname} — регион целиком'))
        if include_districts:
            for did, dname in G.list_districts(rid):
                out.append((db.territory_id(rid, did), rid, did, f'{rname} / {dname}'))
    return out


def availability(id_regions, culture_title, year_from, year_to, include_districts=True,
                 only_gaps=False):
    """Что известно об урожайности культуры по территориям и годам.

    Возвращает dict:
      ``culture``  — ключ культуры в БД (bdpmo_culture);
      ``years``    — список годов (столбцы таблицы);
      ``rows``     — [{'territory_id','id_region','id_district','label','filled','user'}];
      ``values``   — {(territory_id, year): (значение, источник)}, источник — ``'source'``/``'user'``;
      ``stats``    — {'cells','filled','user','gaps'}.

    ``only_gaps`` оставляет в таблице лишь территории, где в диапазоне лет есть пропуски.
    Чтение векторное (по всей таблице разом), а не по одной территории — иначе на сотнях
    районов раздел открывался бы десятки секунд.
    """
    culture = culture_key(culture_title)
    year_from, year_to = int(year_from), int(year_to)
    if year_to < year_from:
        year_from, year_to = year_to, year_from
    years = list(range(year_from, year_to + 1))

    rows = _rows_for(id_regions, include_districts)
    tids = {r[0] for r in rows}

    values = {}
    for table, src in (('yields', SRC_SOURCE), ('user_yields', SRC_USER)):
        df = db.load(table)
        if df.empty:
            continue
        sub = df[(df['culture'].astype(str) == str(culture))
                 & (df['territory_id'].astype(str).isin(tids))]
        if sub.empty:
            continue
        for tid, year, val in zip(sub['territory_id'], sub['year'], sub['yield']):
            try:
                y, v = int(year), float(val)
            except (TypeError, ValueError):
                continue
            if v != v or not (year_from <= y <= year_to):     # NaN или вне диапазона
                continue
            values[(str(tid), y)] = (v, src)                  # user_yields читается вторым -> перекрывает

    out_rows, filled_all, user_all = [], 0, 0
    for tid, rid, did, label in rows:
        filled = sum(1 for y in years if (tid, y) in values)
        user = sum(1 for y in years if values.get((tid, y), (None, None))[1] == SRC_USER)
        if only_gaps and filled == len(years):
            continue
        out_rows.append({'territory_id': tid, 'id_region': rid, 'id_district': did,
                         'label': label, 'filled': filled, 'user': user})
        filled_all += filled
        user_all += user

    cells = len(out_rows) * len(years)
    return {'culture': culture, 'culture_title': culture_title, 'years': years,
            'rows': out_rows, 'values': values,
            'stats': {'cells': cells, 'filled': filled_all, 'user': user_all,
                      'gaps': cells - filled_all}}


def save_manual(edits, culture, account_id=None, comment=None):
    """Записать/удалить значения оператора.

    ``edits`` — список ``(territory_id, year, value)``; ``value is None`` означает
    «убрать моё значение» (строка из ``user_yields`` удаляется, значение источника,
    если оно есть, остаётся). Возвращает ``{'written': n, 'deleted': k}``.
    """
    records, drop = [], []
    for tid, year, value in edits or []:
        if value is None:
            drop.append((str(tid), str(culture), int(year)))
        else:
            records.append({'territory_id': str(tid), 'culture': str(culture),
                            'year': int(year), 'yield': float(value)})
    deleted = db.delete_user_yields(drop)
    written = db.upsert_user_yields(records, account_id=account_id, comment=comment)
    return {'written': written, 'deleted': deleted}


def parse_value(text):
    """Строка из ячейки таблицы -> число или None (пусто). Запятая допускается как разделитель."""
    s = str(text or '').strip().replace(',', '.').replace(' ', '')
    if s == '':
        return None
    return float(s)                                           # ValueError ловит вызывающая сторона


def to_dataframe(av):
    """Свод в широкую таблицу (территория × годы) — для выгрузки в CSV/просмотра."""
    import pandas as pd

    recs = []
    for r in av['rows']:
        rec = {'территория': r['label'], 'territory_id': r['territory_id']}
        for y in av['years']:
            hit = av['values'].get((r['territory_id'], y))
            rec[str(y)] = None if hit is None else hit[0]
        rec['заполнено'] = f"{r['filled']}/{len(av['years'])}"
        rec['вручную'] = r['user']
        recs.append(rec)
    return pd.DataFrame(recs)
