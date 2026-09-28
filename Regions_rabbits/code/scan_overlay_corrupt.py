"""Скан оверлея скачанных рядов (time_series_downloaded.csv) на физическую невозможность.

scan_corrupt_series.py проверяет только БАЗОВУЮ time_series; оверлей (откуда build
читает в первую очередь) не проверялся. Здесь стримим оверлей и ищем ndvi-ряды со
значениями вне [-1.1, 2.0] и прочие параметры вне физичных границ.
"""
import os
import numpy as np
import pandas as pd
import config, db

BOUNDS = {
    'mean_ndvi_7dc_modis_int_agro': (-1.1, 2.0),
    'mean_ndvi_7dc_modis_int_ozim': (-1.1, 2.0),
    'mean_ndvi_7dc_modis_int_spring': (-1.1, 2.0),
    'mean_p': (500.0, 1100.0),
    'mean_prec': (-0.1, 300.0),
    'mean_prec_acc': (-0.1, 4000.0),
    'mean_rh': (-0.1, 101.0),
    'mean_snod': (-0.1, 10.0),
    'mean_snowc': (-0.1, 101.0),
    'mean_soilw10': (-0.1, 100.0),
    'mean_temp': (-70.0, 65.0),
    'mean_temp_acc': (-0.1, 7000.0),
    'mean_tmpgr10': (-70.0, 65.0),
}

path = os.path.join(config.DB_DIR, 'time_series_downloaded.csv')
print('scan', path, f'{os.path.getsize(path)/1e9:.2f} GB')

# агрегируем min/max/size по (tid,parm,year)
agg = {}
rows = 0
for chunk in pd.read_csv(path, dtype={'territory_id': str, 'parm': str},
                         usecols=['territory_id', 'parm', 'year', 'value'], chunksize=2_000_000):
    rows += len(chunk)
    g = chunk.groupby(['territory_id', 'parm', 'year'])['value'].agg(['min', 'max', 'size'])
    for k, r in g.iterrows():
        if k in agg:
            a = agg[k]
            agg[k] = (min(a[0], r['min']), max(a[1], r['max']), a[2] + r['size'])
        else:
            agg[k] = (r['min'], r['max'], r['size'])
    print(f'  ...{rows:,} строк, групп {len(agg):,}')

bad = []
for (tid, parm, year), (vmin, vmax, n) in agg.items():
    lo, hi = BOUNDS.get(parm, (-1e18, 1e18))
    if vmin < lo or vmax > hi:
        reasons = []
        if vmin < lo: reasons.append(f'min<{lo:g}')
        if vmax > hi: reasons.append(f'max>{hi:g}')
        bad.append((tid, parm, int(year), vmin, vmax, int(n), ';'.join(reasons)))

print(f'\nвсего групп оверлея: {len(agg):,}; ИСПОРЧЕННЫХ: {len(bad)}')
out = pd.DataFrame(bad, columns=['territory_id','parm','year','vmin','vmax','n_points','reason'])
if not out.empty:
    out['id_region'] = out['territory_id'].str.split('_').str[0]
    out = out.sort_values(['id_region','year','parm'])
    print('по параметрам:'); print(out['parm'].value_counts().to_string())
    print('по регионам (top15):'); print(out['id_region'].value_counts().head(15).to_string())
    print('по годам:'); print(out['year'].value_counts().sort_index().to_string())
    p = os.path.join(config.REPORTS_DIR, 'corrupt_overlay.csv')
    out.to_csv(p, index=False, encoding='utf-8-sig')
    print('отчёт:', p)
    print('\nпримеры:'); print(out.head(20).to_string(index=False))
