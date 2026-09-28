"""Чтение первого листа .xlsx без обязательной зависимости openpyxl.

Сначала пробует ``pandas.read_excel`` (openpyxl), при отсутствии — встроенный
stdlib-парсер (zipfile + xml). Для устаревших ``.xls`` (BIFF) нужен ``xlrd`` —
он не покрывается фолбэком; см. :mod:`ingest_yields`.
"""

import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

_SS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def _col_letters(ref):
    return ''.join(ch for ch in ref if ch.isalpha())


def _read_xlsx_fallback(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()

    shared = []
    if 'xl/sharedStrings.xml' in names:
        ss = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in ss.findall(f'{_SS}si'):
            shared.append(''.join(t.text or '' for t in si.iter(f'{_SS}t')))

    sheet_files = sorted(n for n in names if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
    ws = ET.fromstring(z.read(sheet_files[0]))

    grid = []
    for r in ws.iter(f'{_SS}row'):
        cells = {}
        for c in r.findall(f'{_SS}c'):
            col = _col_letters(c.get('r', ''))
            t = c.get('t')
            text = None
            if t == 'inlineStr':
                is_ = c.find(f'{_SS}is')
                if is_ is not None:
                    text = ''.join(x.text or '' for x in is_.iter(f'{_SS}t'))
            else:
                v = c.find(f'{_SS}v')
                if v is not None:
                    text = shared[int(v.text)] if t == 's' else v.text
            cells[col] = text
        grid.append(cells)

    if not grid:
        return pd.DataFrame()

    header = grid[0]
    unnamed = 0
    col_names = {}
    for col, name in sorted(header.items()):
        if name is None or str(name).strip() == '':
            col_names[col] = f'Unnamed: {unnamed}'
            unnamed += 1
        else:
            col_names[col] = str(name)

    records = [{col_names.get(col, col): val for col, val in row.items()} for row in grid[1:]]
    return pd.DataFrame(records, columns=list(col_names.values()))


def read_xlsx(path):
    """Прочитать первый лист .xlsx в DataFrame (заголовок в первой строке)."""
    try:
        import openpyxl  # noqa: F401
        return pd.read_excel(path)
    except Exception:
        return _read_xlsx_fallback(path)
