"""Занести урожайности из таблицы БДПМО (выход ``bdpmo_archive_to_yields.py``) в БД приложения (``yields``).

Что делает:
  * фильтрует тип хозяйств (по умолчанию «Хозяйства всех категорий» — как в приложении);
  * оставляет только культуры, известные приложению (``bdpmo_culture`` из ``cultures.csv``);
  * сопоставляет (регион, район) -> ``territory_id`` по таблице ``territories`` (по нормализованным именам);
  * пакетно пишет в ``yields`` (``db.upsert_yields_bulk``), заменяя совпадающие (territory_id, culture, year);
  * перед записью делает бэкап ``yields``. ``--dry-run`` — только отчёт, без записи.

Запуск:
    python code/ingest_bdpmo_table.py <таблица.csv> [--category "Хозяйства всех категорий"] [--dry-run]
"""
import argparse
import re
import shutil
import sys

import pandas as pd

import config
import db

_PAREN = re.compile(r'\([^)]*\)')
_TYPE_WORDS = ('муниципальный район', 'муниципальный округ', 'городской округ с внутригородским делением',
               'городской округ', 'муниципальное образование', 'городское поселение', 'сельское поселение',
               'муниципальный', 'городской', 'район', 'округ', 'город')


def _norm(s):
    return ' '.join(str(s).strip().lower().replace('ё', 'е').split())


def _core(name):
    """Свести название района к «ядру» (прилагательному): убрать тип МО, скобки, «и город …».

    Примеры: «Александровский муниципальный район (до 2021 года)» -> «александровский»;
    «Алексеевский муниципальный район и город Алексеевка» -> «алексеевский»;
    «Муниципальный округ город Славгород» -> «славгород».
    """
    x = _norm(name)
    x = _PAREN.sub(' ', x)                                 # убрать (до 2021 года) и пр.
    x = re.sub(r'\bи город.*$', ' ', x)                   # «… и город N»
    for w in _TYPE_WORDS:
        x = x.replace(w, ' ')
    return ' '.join(x.split())


def _app_cultures():
    """Множество bdpmo_culture, известных приложению."""
    c = pd.read_csv(config.CULTURES_FILE)
    return set(str(x) for x in c['bdpmo_culture'].dropna())


def _territories():
    """Таблица (норм. регион, норм. район) -> (id_region, id_district)."""
    t = db.load('territories')[['region', 'district', 'id_region', 'id_district']].copy()
    t = t.dropna(subset=['region', 'district', 'id_region', 'id_district'])
    t = t[t['district'].astype(str).str.strip() != '']
    t['rk'] = t['region'].map(_norm); t['dk'] = t['district'].map(_core)
    return t.drop_duplicates(['rk', 'dk'])[['rk', 'dk', 'id_region', 'id_district']]


def ingest(table_csv, category='Хозяйства всех категорий', dry_run=False):
    df = pd.read_csv(table_csv, sep=';', dtype=str)
    df['year'] = df['year'].astype(int)
    df['yield'] = df['yield'].astype(float)
    total = len(df)

    if category:
        df = df[df['category'] == category]
    after_cat = len(df)

    app_cults = _app_cultures()
    df = df[df['culture'].isin(app_cults)]
    after_cult = len(df)

    t = _territories()
    df['rk'] = df['region_name'].map(_norm); df['dk'] = df['district'].map(_core)
    m = df.merge(t, on=['rk', 'dk'], how='left')
    matched = m[m['id_region'].notna()].copy()
    unmatched = m[m['id_region'].isna()]

    matched['territory_id'] = (matched['id_region'].astype(int).astype(str) + '_'
                               + matched['id_district'].astype(int).astype(str))
    recs = matched.rename(columns={'culture': 'culture'})[['territory_id', 'culture', 'year', 'yield']]

    print('=== сопоставление ===')
    print(f'  всего строк в таблице:            {total:,}')
    print(f'  после фильтра категории «{category}»: {after_cat:,}')
    print(f'  культура известна приложению:     {after_cult:,}')
    print(f'  район сопоставлен -> к записи:     {len(matched):,}')
    print(f'  район НЕ сопоставлен (отброшено):  {len(unmatched):,}')
    if len(unmatched):
        u = (unmatched[['region_name', 'district']].drop_duplicates()
             .groupby('region_name').size().sort_values(ascending=False))
        print('  несопоставленные районы по регионам (топ-8):')
        for reg, cnt in u.head(8).items():
            print(f'     {reg}: {cnt} районов')

    stats = {'total': total, 'after_cat': after_cat, 'after_cult': after_cult,
             'matched': len(matched), 'unmatched': len(unmatched), 'written': 0, 'backup': None}

    if dry_run:
        print('DRY-RUN: в БД ничего не записано.')
        return stats

    if len(recs):
        src = db.path_of('yields')
        shutil.copy(src, src + '.bak'); stats['backup'] = src + '.bak'; print('бэкап:', src + '.bak')
        stats['written'] = db.upsert_yields_bulk(recs.to_dict('records'))
        print(f'записано/обновлено в yields: {stats["written"]:,}')
    else:
        print('нечего записывать.')
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('table_csv', help='CSV из bdpmo_archive_to_yields.py')
    ap.add_argument('--category', default='Хозяйства всех категорий')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    ingest(a.table_csv, a.category, a.dry_run)
