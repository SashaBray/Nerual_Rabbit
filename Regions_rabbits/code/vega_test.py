"""Быстрая проверка ukey: один запрос к Vega и печать сырого ответа.

Запуск:  python code/vega_test.py <ukey> [product] [year] [district_uid]
По умолчанию: product=mean_temp, year=2020, district_uid=120 (Белгородский район).

Если ответ — валидный JSON, ukey рабочий. Если приходит HTML с
«CGI SCRIPT ERROR / error in your SQL syntax» — ukey недействителен/просрочен.
"""

import sys

from vega import VegaClient


def main():
    if len(sys.argv) < 2:
        print('Использование: python code/vega_test.py <ukey> [product] [year] [district_uid]')
        sys.exit(1)
    ukey = sys.argv[1]
    product = sys.argv[2] if len(sys.argv) > 2 else 'mean_temp'
    year = sys.argv[3] if len(sys.argv) > 3 else '2020'
    uid = sys.argv[4] if len(sys.argv) > 4 else '120'

    client = VegaClient(ukey=ukey)
    print(f'Запрос: product={product}, year={year}, district_uid={uid}')
    result, ok = client.district(product, year, uid)

    if ok:
        block = (result.get('data', {}) or {}).get('1', {})
        xy = block.get('xy') or []
        print(f'УСПЕХ: ukey рабочий. Точек в ряду: {len(xy)}')
        if xy:
            print('  первые точки:', xy[:3])
    else:
        print('НЕУДАЧА. Сырой ответ Vega (начало):')
        print(' ', repr((client.last_text or '')[:400]))


if __name__ == '__main__':
    main()
