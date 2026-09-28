"""Диагностика битых строк в matrix.csv (одноразовый разбор)."""
import json, os, sys
import numpy as np
import pandas as pd
import config

task = sys.argv[1] if len(sys.argv) > 1 else 'soy_all'
ddir = config.dataset_dir(task)
meta = json.load(open(os.path.join(ddir, 'meta.json'), encoding='utf-8'))
F = int(meta['n_features'])
size = int(meta['time_rows_size'])
N = int(meta['duration_years'])
T = size * N
feats = meta['features']
print(f'task={task} F={F} size={size} N={N} T={T} width={F*T}')

mat = pd.read_csv(os.path.join(ddir, 'matrix.csv'), sep=' ', header=None,
                  dtype=np.float32, engine='c').dropna(axis=1, how='all').values
print('matrix shape', mat.shape)
X = mat.reshape(mat.shape[0], F, T)

rowmax = X.max(axis=2)
ceil = np.empty(F)
for f in range(F):
    nz = rowmax[:, f][rowmax[:, f] > 0]
    ceil[f] = 5.0 * (np.percentile(nz, 95) if nz.size else 0.0) + 1e-9
print('ceil per feature:')
for f in range(F):
    print(f'  {f:2d} {feats[f]:18s} ceil={ceil[f]:12.3f}  p95nz={np.percentile(rowmax[:,f][rowmax[:,f]>0],95):.3f}')

over = rowmax > ceil.reshape(1, F)
corrupt = over.any(axis=1)
idx = np.where(corrupt)[0]
print(f'\nCORRUPT rows: {corrupt.sum()} / {mat.shape[0]}')

plan = pd.read_csv(os.path.join(ddir, 'plan.csv'), encoding='utf-8-sig')
print('plan rows', len(plan))

# Какой признак вылетает в каждой битой строке
print('\nример битых строк (row, tid, year, признаки-нарушители, ndvi-блок макс):')
for i in idx[:20]:
    bad_feats = [feats[f] for f in range(F) if over[i, f]]
    pr = plan.iloc[i]
    ndvi_block = X[i, 0]  # признак 0 = ndvi
    print(f'  row={i} tid={pr["territory_id"]} year={pr["year"]} '
          f'ndvi_max={ndvi_block.max():.3f} нарушители={bad_feats}')

# Сводка: какие признаки чаще всего вылетают
print('\nчастота признаков-нарушителей:')
cnt = over[idx].sum(axis=0)
for f in range(F):
    if cnt[f]:
        print(f'  {feats[f]:18s} {int(cnt[f])}')

# по годам и территориям
plan_bad = plan.iloc[idx]
print('\nпо годам:')
print(plan_bad['year'].value_counts().sort_index().to_string())
print('\nуникальных территорий среди битых:', plan_bad['territory_id'].nunique())

# Сохранить список битых (tid, year)
plan_bad[['territory_id','id_region','id_district','year']].to_csv(
    os.path.join(ddir, 'corrupt_rows.csv'), index=False, encoding='utf-8-sig')
print('\nсохранено', os.path.join(ddir, 'corrupt_rows.csv'))

# Детально один пример: ndvi-блок по годам
i0 = idx[0]
pr = plan.iloc[i0]
print(f'\n=== детально row={i0} tid={pr["territory_id"]} year={pr["year"]} ===')
years = [int(pr['year']) - j for j in range(N)]  # newest_first
for f in range(F):
    block = X[i0, f]
    for j in range(N):
        seg = block[j*size:(j+1)*size]
        nz = seg[seg != 0]
        print(f'  {feats[f]:16s} year={years[j]} min={seg.min():9.3f} max={seg.max():9.3f} '
              f'mean_nz={nz.mean() if nz.size else 0:9.3f} nnz={nz.size}')
