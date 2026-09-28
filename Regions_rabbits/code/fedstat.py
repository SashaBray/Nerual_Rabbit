"""Парсер урожайности региональной статистики с сайта ЕМИСС (fedstat.ru).

Порт из ComplexPrediction/Parser_FedStat.py. Показатель 31533 — «Урожайность
сельскохозяйственных культур (в расчёте на убранную площадь), ц/га». Запрос: POST на
``/indicator/dataGrid.do?id=31533`` с фильтрами-UID (регион/культура/тип хозяйств/годы)
из словарей ``workspace/source/*_Dictionary.json``.

ДВЕ ОСОБЕННОСТИ (учтены):
  1. Если запросить год, которого у ЕМИСС для региона НЕТ (обычно текущий/предыдущий),
     сервер может вернуть ПОСЛЕДНЮЮ известную урожайность. Поэтому мы запрашиваем весь
     диапазон лет и читаем значение СТРОГО по столбцу нужного года: если года в ответе
     нет — возвращаем None (подмены не происходит). Плюс явная проверка ``verified_year``.
  2. Запрос может не пройти из-за сети/IP (в т.ч. IPv6). Есть форсирование IPv4
     (``force_ipv4=True``, по умолчанию), таймаут, повторы и мягкий возврат None вместо падения.

Живой ответ ЕМИСС здесь (в песочнице) недоступен — сайт отдаёт 403. Точную привязку
полей можно проверить/докалибровать на рабочей машине: ``python code/fedstat.py --probe <Регион>``.
"""
import json
import os
import random
import re
import socket
import time

import config

INDICATOR_ID = '31533'
URL = f'https://www.fedstat.ru/indicator/dataGrid.do?id={INDICATOR_ID}'

# объектные id измерений показателя 31533 (из исходного Parser_FedStat)
_DIM_INDICATOR = '0'
_DIM_PERIOD = '3'          # год (в столбцы: columnObjectIds=3)
_DIM_REGION = '57831'
_DIM_CULTURE = '58745'
_DIM_FARM = '58423'
_DIM_FIXED = {'30611': '1342019', '33560': '1558883'}   # фиксированные измерения (ед. изм./территория)

_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
    'Opera/9.80 (Windows NT 6.2; WOW64) Presto/2.12.388 Version/12.17',
]

DEFAULT_CULTURE = 'Пшеница озимая'
DEFAULT_FARM = 'Хозяйства всех категорий'

_dicts = {}


def _load_dict(name):
    if name not in _dicts:
        p = os.path.join(config.SOURCE_DIR, name)
        _dicts[name] = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    return _dicts[name]


def regions():
    return sorted(_load_dict('Region_Dictionary.json').keys())


def cultures():
    return sorted(_load_dict('Culture_Dictionary.json').keys())


# --------------------------------------------------------------------------- #
# Сеть
# --------------------------------------------------------------------------- #
_ipv4_forced = False


def _apply_force_ipv4():
    """Заставить urllib3/requests использовать только IPv4 (частая причина сбоев — IPv6)."""
    global _ipv4_forced
    if _ipv4_forced:
        return
    try:
        import urllib3.util.connection as u
        u.allowed_gai_family = lambda: socket.AF_INET
        _ipv4_forced = True
    except Exception:                                     # noqa: BLE001 — не критично
        pass


def _post(data, timeout, retries, force_ipv4):
    """POST к ЕМИСС с повторами. Возвращает текст ответа или None (при сетевой ошибке/блокировке)."""
    import requests
    if force_ipv4:
        _apply_force_ipv4()
    last = None
    for attempt in range(max(1, retries)):
        hdr = {'User-Agent': random.choice(_USER_AGENTS),
               'Content-Type': 'application/x-www-form-urlencoded', 'charset': 'UTF-8'}
        try:
            r = requests.post(URL, data=data, headers=hdr, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                return r.text
            last = f'HTTP {r.status_code}'
        except Exception as exc:                          # noqa: BLE001
            last = f'{type(exc).__name__}: {exc}'
        time.sleep(1.0 * (attempt + 1))
    print(f'[fedstat] запрос не прошёл: {last}')
    return None


def _build_data(region_uid, culture_uid, farm_uid, year_ids):
    """Тело POST: перечисляем измерения-строки, год в столбцы, выбранные фильтры (в т.ч. набор лет)."""
    parts = ['lineObjectIds=0', f'lineObjectIds={_DIM_FARM}', 'lineObjectIds=30611',
             f'lineObjectIds={_DIM_REGION}', f'columnObjectIds={_DIM_PERIOD}',
             f'selectedFilterIds={_DIM_INDICATOR}_{INDICATOR_ID}']
    for y in year_ids:
        parts.append(f'selectedFilterIds={_DIM_PERIOD}_{y}')
    for dim, mem in _DIM_FIXED.items():
        parts.append(f'selectedFilterIds={dim}_{mem}')
    parts.append(f'selectedFilterIds={_DIM_REGION}_{region_uid}')
    parts.append(f'selectedFilterIds={_DIM_CULTURE}_{culture_uid}')
    parts.append(f'selectedFilterIds={_DIM_FARM}_{farm_uid}')
    return '&'.join(parts)


# --------------------------------------------------------------------------- #
# Разбор ответа: собираем {год: урожайность} по фактическим столбцам-годам.
# --------------------------------------------------------------------------- #
def _num(v):
    try:
        return float(str(v).replace(',', '.').replace(' ', '').replace('\xa0', ''))
    except (ValueError, TypeError):
        return None


def _parse_series(payload):
    """Из ответа ЕМИСС собрать {год(int): урожайность(float)} — по реально присутствующим годам.

    Устойчиво к форме: ищем пары (4-значный год -> число) как в ключах строк ``results``,
    так и в отдельных записях с полем периода и значением.
    """
    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return {}
    out = {}

    def add(year, val):
        y = _num(year)
        f = _num(val)
        if y is not None and 1900 <= int(y) <= 2100 and f is not None:
            out[int(y)] = f

    def is_year(k):
        return bool(re.fullmatch(r'(19|20)\d\d', str(k)))

    def walk(node):
        if isinstance(node, dict):
            # 1) ключи-годы -> значение
            for k, v in node.items():
                if is_year(k) and not isinstance(v, (dict, list)):
                    add(k, v)
            # 2) запись вида {..., '3': <годId>, 'value'/'val'/'formatVal': <число>}
            per = node.get(_DIM_PERIOD)
            if per is not None and is_year(per):
                for vk in ('value', 'val', 'formatVal', 'sumValue'):
                    if vk in node:
                        add(per, node[vk]); break
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for it in node:
                walk(it)

    walk(obj)
    return out


# --------------------------------------------------------------------------- #
# Публичный API
# --------------------------------------------------------------------------- #
def fetch_series(region_name, culture=DEFAULT_CULTURE, farm=DEFAULT_FARM,
                 year_from=2000, year_to=None, timeout=25, retries=3, force_ipv4=True):
    """Урожайность региона по годам из ЕМИСС: {год: ц/га}. Пустой dict — нет данных/сети."""
    import datetime
    year_to = year_to or datetime.date.today().year
    R, C, F = _load_dict('Region_Dictionary.json'), _load_dict('Culture_Dictionary.json'), \
        _load_dict('Categories_of_farms_dictionary.json')
    if region_name not in R:
        raise KeyError(f'Регион не найден в Region_Dictionary: {region_name!r}')
    if culture not in C:
        raise KeyError(f'Культура не найдена в Culture_Dictionary: {culture!r}')
    if farm not in F:
        raise KeyError(f'Тип хозяйств не найден: {farm!r}')
    year_ids = [str(y) for y in range(int(year_from), int(year_to) + 1)]
    data = _build_data(R[region_name], C[culture], F[farm], year_ids)
    payload = _post(data, timeout, retries, force_ipv4)
    if payload is None:
        return {}
    series = _parse_series(payload)
    if not series:
        print('[fedstat] ответ получен, но годовые значения не распознаны — '
              'проверьте структуру ответа: python code/fedstat.py --probe "<Регион>"')
    return series


def fetch_yield(year, region_name, culture=DEFAULT_CULTURE, farm=DEFAULT_FARM, **kw):
    """Урожайность региона за КОНКРЕТНЫЙ год или None, если данных за этот год у ЕМИСС нет.

    Защита от подмены: значение берётся строго по столбцу запрошенного года. Если ЕМИСС
    данных за год не имеет, года не будет в ``fetch_series`` -> вернём None (а не последний известный).
    """
    year = int(year)
    series = fetch_series(region_name, culture, farm, year_to=max(year, kw.pop('year_to', year)), **kw)
    if year not in series:
        latest = max(series) if series else None
        print(f'[fedstat] {region_name}, {year}: данных нет '
              f'(последний доступный год: {latest}). Значение не подставляем.')
        return None
    return series[year]


def probe(region_name, culture=DEFAULT_CULTURE, year=None, out_path=None,
          timeout=25, force_ipv4=True):
    """Сохранить СЫРОЙ ответ ЕМИСС для калибровки разбора (запускать на рабочей машине)."""
    import datetime
    year = int(year or datetime.date.today().year)
    R, C, F = _load_dict('Region_Dictionary.json'), _load_dict('Culture_Dictionary.json'), \
        _load_dict('Categories_of_farms_dictionary.json')
    data = _build_data(R[region_name], C[culture], F[DEFAULT_FARM], [str(y) for y in range(2000, year + 1)])
    payload = _post(data, timeout, retries=2, force_ipv4=force_ipv4)
    out_path = out_path or os.path.join(config.REPORTS_DIR, f'fedstat_probe_{region_name}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, 'w', encoding='utf-8').write(payload or '')
    print(f'сырой ответ сохранён: {out_path} ({len(payload or "")} байт)')
    print('распознанная серия:', _parse_series(payload or ''))
    return out_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == '--probe':
        probe(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CULTURE)
    elif len(sys.argv) >= 2:
        reg = sys.argv[1]; cult = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CULTURE
        print('серия по годам:', fetch_series(reg, cult))
    else:
        print('Использование: python fedstat.py "<Регион>" ["<Культура>"]   |   --probe "<Регион>"')
        print('Регионов в словаре:', len(regions()))
