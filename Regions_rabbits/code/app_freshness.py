"""Свежесть временных рядов: для ТЕКУЩЕГО года проверяем, не устарели ли ряды в БД,
и при необходимости докачиваем свежие версии из Vega (по ukey активной учётки).

Прошлые годы считаются полными (не трогаем). Для текущего года, если последний день ряда
отстаёт от сегодняшнего больше чем на ``max_gap_days``, ряд перезапрашивается у Vega.
"""
import datetime

import build_dataset as B
import db
import tasks as tasks_mod
from vega import VegaClient

MAX_GAP_DAYS = 16          # NDVI/метео — декадные/недельные; >16 дней без обновления считаем устаревшим


def series_last_day(tid, parm, year):
    xy = db.get_time_series(tid, parm, int(year))
    return max((int(d) for d, _ in xy), default=None)


def is_stale(tid, parm, year, max_gap_days=MAX_GAP_DAYS):
    """True, если ряд текущего года отсутствует или давно не обновлялся."""
    today = datetime.date.today()
    if int(year) != today.year:
        return False
    last = series_last_day(tid, parm, year)
    if last is None:
        return True
    return (today.timetuple().tm_yday - last) > max_gap_days


def refresh_series(client, id_region, id_district, parm, year):
    """Перезапросить ряд у Vega и записать в оверлей. (ok, status)."""
    tid = db.territory_id(id_region, id_district)
    if str(id_district).strip().lower() not in ('', 'nan'):
        uid = db.current_vega_uid(tid)
        if uid is None:
            return False, 'no_uid'
        payload, ok = client.district(parm, year, uid); used = uid
    else:
        payload, ok = client.region(parm, year, id_region); used = id_region
    if ok and payload:
        try:
            xy = (payload['data']['1'].get('xy') or [])
        except (KeyError, TypeError):
            xy = []
        if xy:
            db.put_time_series(tid, parm, year, xy, vega_uid=used)
            return True, 'refreshed'
    return False, 'empty'


def ensure_fresh(cfg, id_region, id_district, year, ukey=None, max_gap_days=MAX_GAP_DAYS):
    """Если year — текущий, проверить и докачать устаревшие ряды модели. Вернуть отчёт dict."""
    today = datetime.date.today()
    if int(year) != today.year:
        return {'checked': False, 'reason': 'не текущий год', 'refreshed': []}
    tid = db.territory_id(id_region, id_district)
    task = tasks_mod.load_task(cfg['task_id'])
    masks_df = db.get_masks_df()
    client = VegaClient(ukey=ukey)
    refreshed, stale = [], []
    for feat in task['features']:
        parm, _h = B._resolve_parm(feat, cfg['mask_culture'], id_region, masks_df)
        if parm is None:
            continue
        if is_stale(tid, parm, year, max_gap_days):
            stale.append(feat.get('name'))
            ok, st = refresh_series(client, id_region, id_district, parm, year)
            refreshed.append({'parm': feat.get('name'), 'status': st})
    return {'checked': True, 'stale': stale, 'refreshed': refreshed,
            'vega_ok': client.ok_count, 'vega_fail': client.fail_count}
