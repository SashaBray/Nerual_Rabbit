"""Извлечь урожайность (на убранную площадь) из архива БДПМО (data_bdmo_*.zip) в одну реляционную таблицу.

Берётся показатель ``Y48007025`` «Урожайность сельскохозяйственных культур (в расчёте на убранную
площадь)» из раздела 7 «Сельское хозяйство». Уровень — муниципальные образования ВЕРХНЕГО уровня
(районы/округа). Пропуски и закрытые данные (значение 999999999) отбрасываются.

Так как урожайность зависит от культуры и типа хозяйств, таблица содержит и их
(иначе строки (регион, район, год) неоднозначны):
    region_id, region_name, district, oktmo, mun_type, category, culture, year, yield

Выход — CSV (utf-8, разделитель «;»). Запуск:
    python code/bdpmo_archive_to_yields.py <путь_к_архиву.zip> [выходной.csv]
"""
import csv
import io
import os
import sys
import zipfile

INDICATOR = 'Y48007025'
PLACEHOLDER = 999999999.0
DISTRICT_LEVEL = 'Муниципальное образование верхнего уровня'
OUT_COLS = ['region_id', 'region_name', 'district', 'oktmo', 'mun_type', 'category', 'culture', 'year', 'yield']


def _find_section7(z):
    """Имя вложенного zip раздела 7 внутри архива БДПМО."""
    for n in z.namelist():
        if n.startswith('data_section7_') and n.lower().endswith('.zip') and '__MACOSX' not in n:
            return n
    raise FileNotFoundError('в архиве не найден раздел 7 (data_section7_*.zip)')


def convert(archive_path, out_csv=None, progress=None):
    out_csv = out_csv or os.path.join(os.path.dirname(os.path.abspath(archive_path)), 'bdpmo_yields.csv')
    z = zipfile.ZipFile(archive_path)
    sec = _find_section7(z)
    print(f'раздел 7 (сельское хозяйство): {sec}')
    inner = zipfile.ZipFile(io.BytesIO(z.read(sec)))       # ~157 МБ в память
    csvname = next(n for n in inner.namelist() if n.lower().endswith('.csv'))

    kept = skipped_level = skipped_bad = seen = 0
    with inner.open(csvname) as f, open(out_csv, 'w', encoding='utf-8-sig', newline='') as o:
        rd = csv.reader(io.TextIOWrapper(f, encoding='utf-8'), delimiter=';')
        cols = next(rd); ix = {c: i for i, c in enumerate(cols)}
        w = csv.writer(o, delimiter=';'); w.writerow(OUT_COLS)
        for r in rd:
            seen += 1
            if progress and seen % 500000 == 0:
                progress(seen, kept)
            if r[ix['indicator_code']] != INDICATOR:
                continue
            if r[ix['mun_level']] != DISTRICT_LEVEL:       # только районный уровень
                skipped_level += 1; continue
            try:
                fv = float(r[ix['indicator_value']])
            except ValueError:
                skipped_bad += 1; continue
            if fv < 0 or fv >= PLACEHOLDER:                # пропуск/закрытые данные (999999999)
                skipped_bad += 1; continue
            w.writerow([r[ix['region_id']], r[ix['region_name']], r[ix['mun_district']], r[ix['oktmo']],
                        r[ix['mun_type']], r[ix['kategor']], r[ix['kultur']], r[ix['year']], fv])
            kept += 1

    print(f'строк урожайности записано: {kept}')
    print(f'отброшено — не районный уровень: {skipped_level}; пропуск/999999999: {skipped_bad}')
    print(f'таблица сохранена: {out_csv}')
    return out_csv, kept


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Использование: python code/bdpmo_archive_to_yields.py <архив.zip> [выход.csv]')
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
            progress=lambda seen, kept: print(f'  …просмотрено {seen:,}, урожайности {kept:,}'))
