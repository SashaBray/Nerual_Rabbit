"""Точечная чистка оверлея докачки по провенансу (столбец ``vega_uid``).

После правок сопоставления (``map_dist_uids.py``) у части районов меняется актуальный
vega_uid. Ряды, скачанные со СТАРОГО uid, в оверлее устаревают. Этот скрипт находит такие
ряды (записанный ``vega_uid`` != текущий ``db.current_vega_uid``) и удаляет их из
``time_series_downloaded.csv`` — тогда при следующей сборке ``--online`` перекачаются ТОЛЬКО
изменившиеся районы, без полной перекачки.

Что считается устаревшим (только РАЙОННЫЙ уровень; региональные ряды по id_region не трогаем):
  * записанный uid != текущий current_vega_uid(tid)  — район «переехал»;
  * текущего uid нет (район стал несопоставлен)       — ряд осиротел.
Ряды без провенанса (старый формат, uid неизвестен) для безопасности тоже помечаются.

Запуск:
    python code/purge_overlay.py            # сухой прогон: только показать, что удалится
    python code/purge_overlay.py --apply    # переписать оверлей без устаревших рядов
"""

import argparse
import os

import pandas as pd

import config
import db

_CHUNK = 500_000


def _stale_keys():
    """Множество (tid, parm, year), подлежащих удалению, + сводка по причинам."""
    uids = db.download_uids()                       # (tid, parm, year) -> vega_uid
    terr = db.load('territories')
    level = {str(t['territory_id']): t['level'] for _, t in terr.iterrows()}

    # все ряды оверлея (на случай рядов без провенанса)
    all_keys = set(db._downloads().keys())
    stale, reasons = set(), {'moved': 0, 'orphan': 0, 'no_provenance': 0}
    for key in all_keys:
        tid = key[0]
        if level.get(tid) != 'district':           # регион/нет в справочнике — не трогаем
            continue
        rec = uids.get(key)
        cur = db.current_vega_uid(tid)
        if rec is None:
            stale.add(key); reasons['no_provenance'] += 1
        elif cur is None:
            stale.add(key); reasons['orphan'] += 1
        elif int(cur) != int(rec):
            stale.add(key); reasons['moved'] += 1
    return stale, reasons


def purge(apply=False):
    path = os.path.join(config.DB_DIR, db.OVERLAY_TABLE + '.csv')
    if not os.path.exists(path):
        print('Оверлей не найден — чистить нечего.')
        return
    stale, reasons = _stale_keys()
    print(f'Рядов в оверлее: {len(db._downloads())}; помечено устаревших: {len(stale)} '
          f'[переехало: {reasons["moved"]}, осиротело: {reasons["orphan"]}, '
          f'без провенанса: {reasons["no_provenance"]}]')
    if not stale:
        print('Устаревших рядов нет.')
        return
    if not apply:
        for key in list(stale)[:20]:
            print('  удалится:', key)
        print('Сухой прогон. Для удаления запустите с --apply.')
        return

    tmp = path + '.tmp'
    kept_rows = dropped_rows = 0
    first = True
    with pd.read_csv(path, dtype={'territory_id': str, 'parm': str}, chunksize=_CHUNK) as reader:
        for chunk in reader:
            keys = list(zip(chunk['territory_id'].astype(str), chunk['parm'].astype(str),
                            chunk['year'].astype(int)))
            mask = pd.Series([k not in stale for k in keys], index=chunk.index)
            dropped_rows += int((~mask).sum())
            kept = chunk[mask]
            kept_rows += len(kept)
            kept.to_csv(tmp, mode='w' if first else 'a', header=first, index=False, encoding='utf-8')
            first = False
    db._downloads().invalidate()                    # индекс смещений устарел вместе с файлом
    os.replace(tmp, path)
    db.reload()                                     # сбросить кэш оверлея
    print(f'Готово: удалено рядов(строк) {dropped_rows}, осталось {kept_rows}. '
          f'Индекс оверлея пересоберётся при следующем чтении. '
          f'Перезапустите сборку с --online — изменившиеся районы перекачаются.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Чистка оверлея докачки по провенансу vega_uid.')
    ap.add_argument('--apply', action='store_true', help='переписать оверлей (без --apply — сухой прогон)')
    purge(apply=ap.parse_args().apply)
