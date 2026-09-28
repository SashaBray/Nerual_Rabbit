"""Докачка недостающих временных рядов задания с Vega (строго через ukey) в БД.

Если ряд уже есть в БД — ничего не делает. В offline-режиме недостающие ряды
не качаются, а возвращаются как пропуск (для воспроизводимой сборки на кэше).
"""

import db


def ensure_series(client, id_region, id_district, parm, year, offline=True):
    """Гарантировать наличие ряда (parm, year) для территории в БД.

    Возвращает (ok: bool, status: str). При offline=True загрузка не выполняется.
    """
    tid = db.territory_id(id_region, id_district)
    if db.has_time_series(tid, parm, year):
        return True, 'cached'
    if offline:
        return False, 'missing'

    is_region = id_district is None or str(id_district).strip().lower() in ('', 'nan', 'none')
    if not is_region:
        # Вега перенумеровала районы: запрос идём по АКТУАЛЬНОМУ uid (старый — лишь ключ кэша)
        uid = db.current_vega_uid(tid)
        if uid is None:
            return False, 'no_uid'        # нет актуального uid района — пропускаем
        payload, ok = client.district(parm, year, uid)
        used_uid = uid
    else:
        payload, ok = client.region(parm, year, id_region)   # уровень региона (район None/nan)
        used_uid = id_region              # регион качается по id_region напрямую

    if ok and payload:
        try:
            block = payload['data']['1']
            xy = block.get('xy') or []
            labels = block.get('labels') or {}
        except (KeyError, TypeError):
            xy, labels = [], {}
        if xy:
            db.put_time_series(tid, parm, year, xy, labels, vega_uid=used_uid)
            return True, 'downloaded'
    return False, 'empty'
