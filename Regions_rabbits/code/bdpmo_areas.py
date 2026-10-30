"""Площади, сборы и удобрения из раздела 7 БДПМО -> одна длинная таблица по районам.

Зачем. В БД приложения из всего раздела 7 лежит один показатель — урожайность на УБРАННУЮ
площадь (``Y48007025``). Из-за этого недоступны две вещи, без которых зимний блок признаков
нечем проверить:

* **Списание.** Урожайность считается на убранную площадь, и в год сильной гибели она может
  оказаться выше обычной: плохие поля списали и в знаменатель не взяли. Убранная площадь
  восстанавливается как ``валовой сбор / урожайность``, доля списанного — как
  ``1 - убранная / посевная``.
* **Гибель от зимовки напрямую.** Росстат считает посевную площадь повторно весной
  (``Y47000006``, «весеннего учёта»). Для озимых разница между осенним посевом и весенним
  учётом и есть площадь, не дожившая до весны, — это метка, а не признак-прокси.

Заодно берутся удобрения (``Y48007022``): это фактор уровня «район × год», не метеорологический,
а именно взаимодействие «район × год» и составляет три четверти необъяснённой дисперсии.

Вход — каталог с файлами ``<КОД>.csv`` либо ``<КОД>.deflate``. Второй формат — это запись
zip-архива БДПМО, вынутая диапазонным запросом без скачивания всего архива (раздел 7 весит
2.1 ГБ, четыре нужных показателя — 64 МБ), поэтому она лежит сырым deflate-потоком и
распаковывается здесь на ходу: в развёрнутом виде это 3.3 ГБ, которые на диск не нужны.

    python code/bdpmo_areas.py <каталог> [--years 2014 2025] [--out crop_stats.csv]
"""
import argparse
import csv
import io
import os
import zlib

import pandas as pd

import config
import db
from ingest_bdpmo_table import _core, _norm, _territories

INDICATORS = {
    'Y48007018': 'посевная',
    'Y47000006': 'посевная_весна',
    'Y48007019': 'сбор',
    'Y48007022': 'удобрения',
    'Y48007025': 'урожайность',
}
DISTRICT_LEVEL = 'Муниципальное образование верхнего уровня'
ALL_CATEGORIES = 'Хозяйства всех категорий'
PLACEHOLDER = 999999999.0
OUT_COLS = ['territory_id', 'culture', 'year', 'indicator', 'value', 'unit']


def _rows(path, chunk=1 << 22):
    """Строки CSV из ``.csv`` или из сырого deflate-потока. -> итератор списков полей."""
    if path.endswith('.deflate'):
        def chunks():
            dec = zlib.decompressobj(-15)
            with open(path, 'rb') as f:
                while True:
                    blob = f.read(chunk)
                    if not blob:
                        break
                    out = dec.decompress(blob)
                    if out:
                        yield out
            tail = dec.flush()
            if tail:
                yield tail

        buf = ''
        for blob in chunks():
            buf += blob.decode('utf-8', 'replace')
            lines = buf.split('\n')
            buf = lines.pop()                              # последняя строка может быть обрезана
            for row in csv.reader(lines, delimiter=';'):
                if row:
                    yield row
        for row in csv.reader([buf], delimiter=';'):
            if row:
                yield row
    else:
        with io.open(path, encoding='utf-8-sig', newline='') as f:
            for row in csv.reader(f, delimiter=';'):
                if row:
                    yield row


def collect(src_dir, years=(2014, 2025), category=ALL_CATEGORIES):
    """Собрать показатели из каталога в длинную таблицу по территориям. -> (DataFrame, отчёт)."""
    terr = _territories()
    files = {}
    for n in sorted(os.listdir(src_dir)):
        code = n.split('.')[0].split('_')[0]
        if code in INDICATORS and n.endswith(('.csv', '.deflate')):
            files.setdefault(code, os.path.join(src_dir, n))
    if not files:
        raise FileNotFoundError(f'в {src_dir} нет файлов показателей {sorted(INDICATORS)}')

    parts, report = [], []
    for code, path in sorted(files.items()):
        seen = kept = 0
        acc = []
        it = _rows(path)
        hdr = next(it)
        ix = {c: i for i, c in enumerate(hdr)}
        need = ('region_name', 'mun_district', 'mun_level', 'year',
                'indicator_value', 'indicator_unit')
        miss = [c for c in need if c not in ix]
        if miss:
            raise KeyError(f'{code}: в файле нет колонок {miss}')
        # Разрезы есть не у всех показателей: удобрения идут одним рядом на район, без
        # разбивки по культуре и типу хозяйств. Тогда фильтр по категории не применяется.
        i_cult, i_cat = ix.get('kultur'), ix.get('kategor')
        last = max(ix[c] for c in need)
        for r in it:
            seen += 1
            if len(r) <= last or r[ix['mun_level']] != DISTRICT_LEVEL:
                continue
            if category and i_cat is not None and r[i_cat] != category:
                continue
            try:
                yr = int(r[ix['year']])
                val = float(r[ix['indicator_value']])
            except ValueError:
                continue
            if not years[0] <= yr <= years[1] or val < 0 or val >= PLACEHOLDER:
                continue
            acc.append((r[ix['region_name']], r[ix['mun_district']],
                        r[i_cult] if i_cult is not None else 'все культуры',
                        yr, val, r[ix['indicator_unit']]))
            kept += 1
        d = pd.DataFrame(acc, columns=['region_name', 'district', 'culture', 'year',
                                       'value', 'unit'])
        d['rk'] = d['region_name'].map(_norm)
        d['dk'] = d['district'].map(_core)
        m = d.merge(terr, on=['rk', 'dk'], how='left')
        ok = m[m['id_region'].notna()].copy()
        ok['territory_id'] = (ok['id_region'].astype(int).astype(str) + '_'
                              + ok['id_district'].astype(int).astype(str))
        ok['indicator'] = INDICATORS[code]
        parts.append(ok[OUT_COLS])
        report.append({'код': code, 'показатель': INDICATORS[code], 'строк_в_файле': seen,
                       'после_фильтров': kept, 'район_сопоставлен': len(ok),
                       'районов': ok['territory_id'].nunique(),
                       'культур': ok['culture'].nunique(),
                       'годы': f"{ok['year'].min()}–{ok['year'].max()}" if len(ok) else '—'})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(report)


def main():
    ap = argparse.ArgumentParser(description='Площади и сборы БДПМО по районам')
    ap.add_argument('src', help='каталог с <КОД>.csv или <КОД>.deflate')
    ap.add_argument('--years', nargs=2, type=int, default=[2014, 2025])
    ap.add_argument('--out', default='crop_stats.csv', help='имя файла в каталоге БД')
    ap.add_argument('--category', default=ALL_CATEGORIES,
                    help='тип хозяйств; пустая строка — все типы')
    a = ap.parse_args()

    tbl, rep = collect(a.src, tuple(a.years), a.category or None)
    print('=== что собрано')
    print(rep.to_string(index=False))

    out = os.path.join(os.path.dirname(db.path_of('yields')), a.out)
    tbl.to_csv(out, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'\nвсего строк {len(tbl):,}, записано: {out}')

    # покрытие: совпадают ли районы с теми, по которым у нас есть урожайности
    y = db.load('yields')
    ours = set(y['territory_id'].astype(str))
    got = set(tbl['territory_id'].astype(str))
    print(f'\nрайонов в наших урожайностях: {len(ours)}, в новых показателях: {len(got)}, '
          f'общих: {len(ours & got)}')
    piv = (tbl[tbl['culture'].str.contains('зимая|зимой', regex=True, na=False)]
           .pivot_table(index='year', columns='indicator', values='territory_id',
                        aggfunc='nunique'))
    if len(piv):
        print('\nрайонов по годам, озимая пшеница (сколько районов имеют показатель):')
        print(piv.to_string())


if __name__ == '__main__':
    main()
