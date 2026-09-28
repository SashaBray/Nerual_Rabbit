"""Быстрый скан битых строк по нескольким датасетам + выгрузка (tid, year)."""
import json, os, sys
import numpy as np
import pandas as pd
import config

tasks = sys.argv[1:] or ['soy_all','corn_all','spring_wheat_all','winter_wheat_all','rice_all','rye_spring_all']
for task in tasks:
    ddir = config.dataset_dir(task)
    mp = os.path.join(ddir,'meta.json')
    if not os.path.exists(mp):
        print(task,'нет meta'); continue
    meta = json.load(open(mp,encoding='utf-8'))
    F=int(meta['n_features']); size=int(meta['time_rows_size']); N=int(meta['duration_years']); T=size*N
    feats=meta['features']
    mat = pd.read_csv(os.path.join(ddir,'matrix.csv'),sep=' ',header=None,dtype=np.float32,engine='c').dropna(axis=1,how='all').values
    X = mat.reshape(mat.shape[0],F,T)
    rowmax=X.max(axis=2)
    ceil=np.empty(F)
    for f in range(F):
        nz=rowmax[:,f][rowmax[:,f]>0]
        ceil[f]=5.0*(np.percentile(nz,95) if nz.size else 0.0)+1e-9
    over=rowmax>ceil.reshape(1,F)
    corrupt=over.any(axis=1)
    idx=np.where(corrupt)[0]
    plan=pd.read_csv(os.path.join(ddir,'plan.csv'),encoding='utf-8-sig')
    viol=over[idx].sum(axis=0) if idx.size else np.zeros(F)
    vio_s=', '.join(f'{feats[f]}:{int(viol[f])}' for f in range(F) if viol[f])
    print(f'{task:20s} rows={mat.shape[0]:5d} corrupt={int(corrupt.sum()):3d} dropped_meta={meta.get("dropped_corrupt",0)}  [{vio_s}]')
    if idx.size:
        out=plan.iloc[idx][['territory_id','id_region','id_district','year']].copy()
        # какой признак макс вылетает
        out['top_viol']=[feats[int(np.argmax((rowmax[i]-ceil)/np.where(ceil>0,ceil,1)))] for i in idx]
        out['ndvi_max']=[float(X[i,0].max()) for i in idx]
        out.to_csv(os.path.join(ddir,'corrupt_rows.csv'),index=False,encoding='utf-8-sig')
        print('   примеры:', list(zip(out['territory_id'][:6],out['year'][:6],out['ndvi_max'][:6].round(2))))
