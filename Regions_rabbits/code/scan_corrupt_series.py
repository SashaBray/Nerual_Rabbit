"""Найти испорченные ряды в базе time_series по физическим границам параметров.

Брак (напр. Калининград 2021–2022): в несколько метеопараметров затекли значения
давления (~1019) и нули. Такие ряды выбиваются за физичные пределы своего параметра.

Пишет отчёт ``workspace/reports/corrupt_series.csv`` (territory_id, id_region, id_district,
parm, year, vmin, vmax, n_points, reason) и сводку по регионам/годам/параметрам.

Запуск:  python code/scan_corrupt_series.py
"""

import os

import numpy as np
import pandas as pd

import config
import db

# Физичные границы значений по параметру (широкие, чтобы не ловить ложное).
BOUNDS = {
    'mean_ndvi_7dc_modis_int_agro': (-1.1, 2.0),
    'mean_ndvi_7dc_modis_int_ozim': (-1.1, 2.0),
    'mean_ndvi_7dc_modis_int_spring': (-1.1, 2.0),
    'mean_p': (500.0, 1100.0),        # давление, гПа (0 -> брак)
    'mean_prec': (0.0, 300.0),        # осадки/сутки, мм
    'mean_prec_acc': (0.0, 4000.0),
    'mean_rh': (-0.1, 101.0),         # влажность, %
    'mean_snod': (-0.1, 10.0),        # высота снега, м
    'mean_snowc': (-0.1, 101.0),      # покрытие снегом, %
    'mean_soilw10': (-0.1, 100.0),
    'mean_temp': (-70.0, 65.0),
    'mean_temp_acc': (0.0, 7000.0),
    'mean_tmpgr10': (-70.0, 65.0),
}


def scan():
    config.ensure_dirs()
    series = db.load('series')
    parms = db.load('parms')
    pid2name = dict(zip(parms['parm_id'].astype(int), parms['parm'].astype(str)))

    # per-series min/max значения из тяжёлой time_series
    ts_path = db.path_of('time_series')
    print(f'Чтение {os.path.basename(ts_path)} …')
    ts = pd.read_csv(ts_path, usecols=['series_id', 'value'],
                     dtype={'series_id': np.int64, 'value': np.float64})
    agg = ts.groupby('series_id')['value'].agg(['min', 'max', 'size']).reset_index()
    print(f'  рядов с точками: {len(agg)}')

    s = series.copy()
    s['series_id'] = s['series_id'].astype(np.int64)
    s['parm_id'] = s['parm_id'].astype(int)
    m = s.merge(agg, on='series_id', how='inner')
    m['parm'] = m['parm_id'].map(pid2name)

    def reason(row):
        lo, hi = BOUNDS.get(row['parm'], (-1e18, 1e18))
        r = []
        if row['min'] < lo:
            r.append(f'min<{lo:g}')
        if row['max'] > hi:
            r.append(f'max>{hi:g}')
        return ';'.join(r)

    m['reason'] = m.apply(reason, axis=1)
    bad = m[m['reason'] != ''].copy()
    bad['id_region'] = bad['territory_id'].astype(str).str.split('_').str[0]
    bad['id_district'] = bad['territory_id'].astype(str).str.split('_').str[1]
    out = bad[['territory_id', 'id_region', 'id_district', 'parm', 'year',
               'min', 'max', 'size', 'reason']].rename(
        columns={'min': 'vmin', 'max': 'vmax', 'size': 'n_points'})
    out = out.sort_values(['id_region', 'year', 'parm']).reset_index(drop=True)

    path = os.path.join(config.REPORTS_DIR, 'corrupt_series.csv')
    out.to_csv(path, index=False, encoding='utf-8-sig')

    print(f'\nИспорченных рядов: {len(out)} (территорий: {out.territory_id.nunique()})')
    print('по регионам:'); print(out['id_region'].value_counts().head(15).to_string())
    print('по годам:'); print(out['year'].value_counts().sort_index().to_string())
    print('по параметрам:'); print(out['parm'].value_counts().to_string())
    print(f'\nОтчёт: {path}')
    return out


if __name__ == '__main__':
    scan()
