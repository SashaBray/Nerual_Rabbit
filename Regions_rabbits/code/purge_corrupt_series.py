"""Удалить из базы time_series испорченные ряды (по отчёту scan_corrupt_series.py).

Причина порчи: легаси-загрузчик для ряда районов записал ряд ДАВЛЕНИЯ во все файлы
параметров (подтверждено: файлы mean_ndvi/mean_prec/… для района идентичны mean_p).
Поэтому чистим ЦЕЛИКОМ блок (территория, год), где найдена хоть одна порча — иначе
останутся параметры, где ~1000 формально «в пределах» (mean_p, temp_acc, prec_acc),
но на деле тоже мусор. После чистки эти блоки докачаются заново (build_dataset --online).

Запуск:
    python code/scan_corrupt_series.py        # сначала отчёт corrupt_series.csv
    python code/purge_corrupt_series.py        # сухой прогон
    python code/purge_corrupt_series.py --apply
"""

import argparse
import os

import numpy as np
import pandas as pd

import config
import db

_CHUNK = 1_000_000


def purge(apply=False):
    rep = os.path.join(config.REPORTS_DIR, 'corrupt_series.csv')
    if not os.path.exists(rep):
        print('Нет corrupt_series.csv — сначала запустите scan_corrupt_series.py')
        return
    bad = pd.read_csv(rep, encoding='utf-8-sig', dtype={'territory_id': str})
    affected = set(zip(bad['territory_id'].astype(str), bad['year'].astype(int)))
    print(f'Затронутых блоков (территория, год): {len(affected)} '
          f'(территорий: {bad.territory_id.nunique()})')

    series = db.load('series')
    series['territory_id'] = series['territory_id'].astype(str)
    series['year'] = series['year'].astype(int)
    mask = [ (t, y) in affected for t, y in zip(series['territory_id'], series['year']) ]
    drop_series = series[pd.Series(mask, index=series.index)]
    drop_ids = set(drop_series['series_id'].astype(np.int64))
    print(f'Рядов под удаление (все параметры этих блоков): {len(drop_ids)} '
          f'(в отчёте было {len(bad)} — расширили до всех параметров блока)')

    if not apply:
        ex = drop_series.merge(db.load('parms'), on='parm_id', how='left')
        print('примеры удаляемого:')
        print(ex[['territory_id', 'year', 'parm']].head(12).to_string(index=False))
        print('Сухой прогон. Для удаления — флаг --apply.')
        return

    # 1) переписать series без удаляемых
    keep_series = series[~series['series_id'].astype(np.int64).isin(drop_ids)]
    db.save('series', keep_series.reset_index(drop=True))

    # 2) стримингом переписать тяжёлую time_series без удаляемых series_id
    ts_path = db.path_of('time_series')
    tmp = ts_path + '.tmp'
    kept = dropped = 0
    first = True
    with pd.read_csv(ts_path, dtype={'series_id': np.int64}, chunksize=_CHUNK) as reader:
        for chunk in reader:
            keepmask = ~chunk['series_id'].isin(drop_ids)
            dropped += int((~keepmask).sum())
            out = chunk[keepmask]
            kept += len(out)
            out.to_csv(tmp, mode='w' if first else 'a', header=first, index=False, encoding='utf-8')
            first = False
    os.replace(tmp, ts_path)
    db.reload()
    print(f'Удалено рядов: {len(drop_ids)}; строк time_series: {dropped}, осталось {kept}.')
    print('Теперь перекачайте: пересборка build_dataset.py <task> --online докачает эти блоки заново.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Чистка испорченных рядов из time_series.')
    ap.add_argument('--apply', action='store_true', help='переписать БД (без флага — сухой прогон)')
    purge(apply=ap.parse_args().apply)
