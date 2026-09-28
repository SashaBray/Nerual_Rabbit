import numpy as np, pandas as pd
import db, config

tid='62_1508'; parm='mean_temp'; year=2021
xy = db.get_time_series(tid, parm, year)
xy = np.array(xy, float)
print(f'{tid} {parm} {year}: точек={len(xy)} day[min,max]=[{xy[:,0].min()},{xy[:,0].max()}] val[min,max]=[{xy[:,1].min():.2f},{xy[:,1].max():.2f}]')
# дубликаты дней?
days=xy[:,0].astype(int)
u,c=np.unique(days,return_counts=True)
print('уник. дней:',len(u),'макс кратность дня:',c.max())
# где экстремум
bad=xy[xy[:,1]>65]
print('точек >65C:',len(bad),'примеры:',bad[:10].tolist())
# приведённый массив (как в build)
arr=db.get_time_series_array(tid,parm,year,365,'zeros')
print('get_time_series_array max=%.2f min=%.2f len=%d'%(arr.max(),arr.min(),len(arr)))

# в каких культурах есть урожайность 62_1508 в 2021?
yld=db.load('yields')
sub=yld[(yld['territory_id']==tid)]
print('\nурожайности 62_1508 по культурам/годам (2021):')
print(sub[sub['year']==2021][['culture','year','yield']].to_string(index=False))
print('\nвсе культуры этой территории:', sorted(sub['culture'].unique()))
