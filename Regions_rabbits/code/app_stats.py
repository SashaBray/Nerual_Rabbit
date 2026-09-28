"""Подгрузка свежей региональной статистики урожайности с ЕМИСС (fedstat) в БД.

Связывает `fedstat.py` (веб-парсер) с проектом: сопоставляет имена регионов/культур со
справочниками ЕМИСС, тянет ряды по годам и пишет их в таблицу ``yields`` на уровень региона
(территория ``<id_region>_nan``). Урожайность за конкретный год берётся только если она
реально есть у ЕМИСС (см. защиту от подмены в fedstat.fetch_yield/ fetch_series).
"""
import app_geo as G
import app_predict as P
import db
import fedstat as FS


def fedstat_culture_name(culture_title):
    """Имя культуры проекта -> ключ в Culture_Dictionary ЕМИСС (или None)."""
    cd = FS._load_dict('Culture_Dictionary.json')
    try:
        bdpmo = P.resolve_culture(culture_title)['bdpmo_culture']
    except Exception:                                     # noqa: BLE001
        bdpmo = None
    for name in (bdpmo, culture_title):
        if name and name in cd:
            return name
    return None


def fedstat_region_name(id_region):
    """Имя региона проекта -> ключ в Region_Dictionary ЕМИСС (или None)."""
    name = G.region_name(id_region)
    rd = FS._load_dict('Region_Dictionary.json')
    return name if name in rd else None


def load_fresh_yields(id_regions, culture_title, progress=None, force_ipv4=True):
    """Подгрузить с ЕМИСС урожайность регионов по культуре и записать в БД (yields, <reg>_nan).

    Возвращает dict: ok, reason?, rows (по строке на регион: регион/статус/лет/последний год),
    all_records — что записано. Пишет в БД одним пакетом (быстро).
    """
    bdpmo = None
    try:
        bdpmo = P.resolve_culture(culture_title)['bdpmo_culture']
    except Exception:                                     # noqa: BLE001
        pass
    fed_cult = fedstat_culture_name(culture_title)
    if fed_cult is None or bdpmo is None:
        return {'ok': False, 'reason': f'культура «{culture_title}» не найдена в справочнике ЕМИСС',
                'rows': [], 'written': 0}

    id_regions = sorted(set(int(r) for r in id_regions))
    rows, records = [], []
    total = len(id_regions)

    def report(i, msg):
        if progress:
            try:
                progress(i, total, msg)                   # новый вид: с комментарием
            except TypeError:
                progress(i, total)                        # совместимость со старым (i, total)

    for i, rid in enumerate(id_regions, 1):
        rname = G.region_name(rid)
        report(i - 1, f'Запрос к ЕМИСС: {rname}…')      # что делаем сейчас
        fed_reg = fedstat_region_name(rid)
        if fed_reg is None:
            rows.append({'регион': rname, 'статус': 'нет в справочнике ЕМИСС', 'лет': 0, 'последний год': None})
            report(i, f'{rname}: нет в справочнике ЕМИСС — пропуск')
            continue
        try:
            series = FS.fetch_series(fed_reg, culture=fed_cult, force_ipv4=force_ipv4)
        except Exception as exc:                          # noqa: BLE001
            rows.append({'регион': rname, 'статус': f'ошибка: {type(exc).__name__}', 'лет': 0, 'последний год': None})
            report(i, f'{rname}: ошибка запроса ({type(exc).__name__})')
            continue
        tid = db.territory_id(rid, None)                  # региональный уровень: <id_region>_nan
        for year, yld in series.items():
            records.append({'territory_id': tid, 'culture': bdpmo, 'year': int(year), 'yield': float(yld)})
        if series:
            rows.append({'регион': rname, 'статус': 'загружено', 'лет': len(series), 'последний год': max(series)})
            report(i, f'{rname}: получено {len(series)} лет (последний {max(series)})')
        else:
            rows.append({'регион': rname, 'статус': 'нет данных / сеть', 'лет': 0, 'последний год': None})
            report(i, f'… {rname}: данных нет или запрос не прошёл')

    report(total, f'Запись в БД: {len(records)} значений…')
    written = db.upsert_yields_bulk(records)
    return {'ok': True, 'rows': rows, 'written': written, 'culture': bdpmo}
