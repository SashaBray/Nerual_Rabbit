"""Финал архитектурного поиска: по сохранённым моделям — проверка топ-3 на ВАЛИДАЦИОННОЙ + график.

Берём конфигурации, у которых есть сохранённая модель (_models/*.pt), отбираем по ТЕСТОВОЙ,
валидационную раскрываем только для топ-3. Запуск:  python code/finalize_arch.py
"""
import os
import pandas as pd
import torch

import archsearch as A
import config
import explore_models as E
import train_nn as T

outdir = os.path.join(config.REPORTS_DIR, 'archsearch')
tmpdir = os.path.join(outdir, '_models')
cfgmap = {c['name']: c for c in A.CONFIGS}

D = A.prep()
res = pd.read_csv(os.path.join(outdir, f'archsearch_results{A.RND}.csv'))
res = res[res['config'].apply(lambda n: os.path.exists(os.path.join(tmpdir, f'{n}.pt')))].copy()
df = res.sort_values('test_MSE').reset_index(drop=True)
print('Готовые конфигурации (отбор по ТЕСТОВОЙ):')
print(df.to_string(index=False))

fin = []
for name in df['config'].head(3):
    m = A.ConvNet(D['in_ch'], D['in_len'], cfgmap[name])
    m.load_state_dict(torch.load(os.path.join(tmpdir, f'{name}.pt'), map_location='cpu'))
    m.eval()
    mval = E.metrics(D['yval'], T.predict(m, D['Xval'], 'cpu'))
    t = df[df['config'] == name].iloc[0]
    fin.append({'config': name, 'test_MSE': t['test_MSE'], 'test_R2': t['test_R2'],
                'val_MSE': round(mval['MSE'], 2), 'val_R2': round(mval['R2'], 3)})
fdf = pd.DataFrame(fin)
print('\nФИНАЛ на ВАЛИДАЦИОННОЙ (top-3 по тесту):')
print(fdf.to_string(index=False))

df.to_csv(os.path.join(outdir, f'archsearch_results{A.RND}.csv'), index=False, encoding='utf-8-sig')
fdf.to_csv(os.path.join(outdir, f'archsearch_final{A.RND}.csv'), index=False, encoding='utf-8-sig')
A._bar(df, fdf, outdir)
w = fdf.iloc[0]
print(f'\nЛучшая: {w["config"]} — ТЕСТ MSE {w["test_MSE"]}, ВАЛИДАЦИЯ MSE {w["val_MSE"]} '
      f'(эталон свёрточная v12: тест 26.7). Готово: {outdir}')
