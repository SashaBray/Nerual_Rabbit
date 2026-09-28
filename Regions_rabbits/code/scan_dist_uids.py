"""Сканер актуальных районных uid Веги: определяет, под каким uid какой район.

Современная версия легаси-`main.py`: вместо логина/пароля — ukey; перебирает uid
района (``type=adm_dis``), читает из ``title`` ответа «Регион — Район» и пишет
актуальный справочник ``workspace/database/vega_districts.csv``
(``uid, region, district, has_data, status``).

Быстро (параллельные запросы), устойчиво (таймауты, ретраи, перехват всех ошибок)
и ВОЗОБНОВЛЯЕМО: повторный запуск пропускает уже просканированные uid. Прерывание
по Ctrl+C сохраняет прогресс.

Текущий районный uid Веги = (uid региона)*1000 + (номер района в регионе) — подтверждено
эмпирически. Поэтому по умолчанию сканируются только узкие полосы по каждому региону из
``regions.csv`` (mode=regions), а не весь диапазон подряд.

Запуск:
    python code/scan_dist_uids.py                       # полосы всех регионов, 16 потоков
    python code/scan_dist_uids.py --per-region 80
    python code/scan_dist_uids.py --mode range --start 34000 --end 34100   # одна полоса вручную
    python code/scan_dist_uids.py --ukey <ukey>
"""

import argparse
import csv
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config

_BASE = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl'
_UA = [
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
]
_local = threading.local()


def _session():
    s = getattr(_local, 's', None)
    if s is None:
        s = requests.Session()
        s.trust_env = False          # игнорировать прокси окружения
        _local.s = s
    return s


def _read_ukey():
    """ukey из ukeys.csv (первая непустая запись) либо config.read_ukey()."""
    path = os.path.join(config.CONFIG_DIR, 'ukeys.csv')
    if os.path.exists(path):
        try:
            import pandas as pd
            df = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
            for v in df['ukey']:
                v = str(v).strip()
                if v and v != 'your_ukey_here':
                    return v
        except Exception:               # noqa: BLE001
            pass
    return config.read_ukey()


def _probe(uid, ukey, product, year, retries=2):
    """Опросить один uid. Возвращает dict; ошибки не пробрасываются."""
    url = (_BASE + '?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&x2=366&query='
           + '[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + str(uid)
           + '%22,%22rows%22:{%22' + product + '%22:[' + year + ']}}]'
           + '&mode=basic&num_points=1&highcharts=1&a_week=51&a_year=' + year + '&label_year=' + year)
    headers = {'User-Agent': random.choice(_UA)}
    last = ''
    for attempt in range(retries + 1):
        try:
            r = _session().get(url, headers=headers, timeout=25)
            block = r.json()['data']['1']
            head = (block.get('title') or '').split('<br>')[0].strip()
            has_data = 1 if block.get('xy') else 0
            if ' - ' in head:
                region, district = [x.strip() for x in head.rsplit(' - ', 1)]
                if region and district:
                    return {'uid': uid, 'region': region, 'district': district,
                            'has_data': has_data, 'status': 'ok'}
            return {'uid': uid, 'region': '', 'district': '', 'has_data': has_data, 'status': 'empty'}
        except Exception as exc:                # noqa: BLE001
            last = str(exc)[:50]
    return {'uid': uid, 'region': '', 'district': '', 'has_data': 0, 'status': 'error:' + last}


def _load_done(path):
    done = set()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    done.add(int(row['uid']))
        except Exception:                       # noqa: BLE001
            pass
    return done


def _region_band_uids(per_region):
    """uid'ы по формуле region_uid*1000 + 1..per_region для всех регионов из regions.csv.

    Подтверждено эмпирически: текущий районный uid Веги = (uid региона)*1000 + (номер
    района в регионе). Поэтому сканируем только узкие полосы по каждому региону.
    """
    import pandas as pd
    path = os.path.join(config.DB_DIR, 'regions.csv')
    df = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
    region_uids = sorted({int(float(v)) for v in df['region_uid'] if str(v).strip().isdigit()})
    uids = []
    for ru in region_uids:
        uids.extend(ru * 1000 + d for d in range(1, per_region + 1))
    return uids


def main():
    ap = argparse.ArgumentParser(description='Сканер актуальных районных uid Веги.')
    ap.add_argument('--mode', choices=['regions', 'range'], default='regions',
                    help='regions: полосы region_uid*1000+1..N (по умолчанию); range: подряд --start..--end')
    ap.add_argument('--per-region', type=int, default=80, help='сколько uid сканировать в полосе региона')
    ap.add_argument('--start', type=int, default=0, help='для mode=range')
    ap.add_argument('--end', type=int, default=4000, help='для mode=range (не включительно)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--product', default='mean_ndvi_7dc_modis_int_ozim')
    ap.add_argument('--year', default='2023')
    ap.add_argument('--ukey', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    ukey = args.ukey or _read_ukey()
    if not ukey or ukey == 'your_ukey_here':
        print('Не задан ukey. Впишите его в workspace/config/ukeys.csv или передайте --ukey.')
        return
    out = args.out or os.path.join(config.DB_DIR, 'vega_districts.csv')

    done = _load_done(out)
    if args.mode == 'regions':
        all_uids = _region_band_uids(args.per_region)
        label = f'полосы регионов (region_uid*1000+1..{args.per_region})'
    else:
        all_uids = list(range(args.start, args.end))
        label = f'uid {args.start}..{args.end - 1}'
    todo = [u for u in all_uids if u not in done]
    print(f'Скан [{label}]: всего {len(all_uids)}, '
          f'уже сделано {len(set(all_uids) & done)}, осталось {len(todo)}, потоков {args.workers}')
    if not todo:
        print('Нечего сканировать.')
        return

    new_file = not os.path.exists(out) or os.path.getsize(out) == 0
    f = open(out, 'a', encoding='utf-8-sig', newline='')
    writer = csv.DictWriter(f, fieldnames=['uid', 'region', 'district', 'has_data', 'status'])
    if new_file:
        writer.writeheader()

    found, processed = 0, 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_probe, u, ukey, args.product, args.year) for u in todo]
            for fut in as_completed(futures):
                res = fut.result()
                writer.writerow(res)
                processed += 1
                if res['status'] == 'ok':
                    found += 1
                if processed % 200 == 0:
                    f.flush()
                    print(f'  обработано {processed}/{len(todo)}, найдено районов {found}')
    except KeyboardInterrupt:
        print('\nПрервано — сохраняю прогресс…')
    finally:
        f.flush()
        f.close()

    print(f'Готово: обработано {processed}, найдено районов с именем {found}. Файл: {out}')


if __name__ == '__main__':
    main()
