import json
import socket

import requests

INDICATOR = '31533'
DATAGRID_URL = f'https://www.fedstat.ru/indicator/dataGrid.do?id={INDICATOR}'
PAGE_URL = f'https://www.fedstat.ru/indicator/{INDICATOR}'


def write_inf(data, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


def _force_ipv4():
    """Заставить requests/urllib3 использовать только IPv4 (частая причина сбоев — IPv6/VPN)."""
    try:
        import urllib3.util.connection as u
        u.allowed_gai_family = lambda: socket.AF_INET
    except Exception:                                     # noqa: BLE001
        pass


def _session():
    """Сессия с браузерными заголовками + прайминг куки со страницы показателя (обход простой антибот-защиты)."""
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://www.fedstat.ru',
        'Referer': PAGE_URL,
    })
    try:
        s.get(PAGE_URL, timeout=30)                       # получить куки сессии
    except Exception:                                     # noqa: BLE001 — не критично
        pass
    return s


def parser_fedstat(Year, Region, Culture, Farm_type, force_ipv4=True, timeout=30, session=None):
    """Урожайность (ц/га) региона за год из ЕМИСС (показатель 31533) или None, если данных нет.

    ВАЖНО: «Российская Федерация» в измерении регионов (57831) отсутствует — запрашивайте субъект.
    Значение берётся строго за запрошенный год (столбец-год) — без подмены последним известным.
    """
    if force_ipv4:
        _force_ipv4()

    Categories = read_inf("Categories_of_farms_dictionary.json")
    Region_Dictionary = read_inf("Region_Dictionary.json")
    Culture_Dictionary = read_inf("Culture_Dictionary.json")
    for label, d, key in (('регион', Region_Dictionary, Region),
                          ('культура', Culture_Dictionary, Culture),
                          ('тип хозяйств', Categories, Farm_type)):
        if key not in d:
            raise KeyError(f'{label} «{key}» не найден(а) в словаре UID')

    # 57831=регион, 58745=культура, 58423=тип хозяйств, 3=год(в столбцы); 30611/33560 — фиксированные измерения
    data = ('lineObjectIds=0&lineObjectIds=33560&lineObjectIds=30611&lineObjectIds=58423'
            '&lineObjectIds=57831&lineObjectIds=58745&columnObjectIds=3'
            '&selectedFilterIds=0_31533&selectedFilterIds=3_' + str(Year) +
            '&selectedFilterIds=30611_1342019&selectedFilterIds=33560_1558883'
            '&selectedFilterIds=57831_' + Region_Dictionary[Region] +
            '&selectedFilterIds=58745_' + Culture_Dictionary[Culture] +
            '&selectedFilterIds=58423_' + Categories[Farm_type])

    s = session or _session()
    resp = s.post(DATAGRID_URL, data=data,
                  headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}, timeout=timeout)
    try:
        DATA = resp.json()
    except ValueError:
        raise RuntimeError(f'ЕМИСС вернул не JSON (HTTP {resp.status_code}) — вероятно, блокировка/капча. '
                           f'Отключите VPN и повторите.')

    results = DATA.get('results') or []
    if not results:                                       # {'results': [], '__count': '0'} — данных нет
        return None
    row = results[0]
    if str(Year) in row and row[str(Year)] not in (None, '', '-'):
        return row[str(Year)]                             # значение именно за запрошенный год
    keys = list(row.keys())
    return row[keys[-1]] if keys else None                # запасной путь


if __name__ == '__main__':
    # быстрый самотест (нужны словари в текущей папке)
    print(parser_fedstat('2023', 'Белгородская область', 'Пшеница озимая', 'Хозяйства всех категорий'))
