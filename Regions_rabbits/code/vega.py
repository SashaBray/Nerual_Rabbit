"""Клиент сервиса Vega-Science (sci-vega.ru) — строго через ukey.

Никаких логинов/паролей и сессий: запрос данных идёт GET-ом с параметром ``ukey``.
ukey берётся из ``workspace/config/ukey.txt`` (см. :mod:`config`). Шаблоны URL
портированы из основного проекта (``Nerual_Rabbit.VegaAPI``) с исправлением двух
багов: переданный ukey НЕ перезаписывается, и в region-URL не теряется ``&``.
"""

import json
import random

import requests

import config


_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
    'Opera/9.80 (Windows NT 6.2; WOW64) Presto/2.12.388 Version/12.17',
]

_BASE = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl'


class VegaClient:
    """Тонкий ukey-клиент: методы ``district`` и ``region`` возвращают (json, success)."""

    def __init__(self, ukey=None, a_week='51', ignore_proxy=True):
        self.ukey = ukey if ukey is not None else config.read_ukey()
        self.a_week = a_week
        # Vega — прямой адрес; по умолчанию игнорируем прокси из окружения
        # (иначе при выставленном socks://-прокси без PySocks все запросы падают).
        self._session = requests.Session()
        self._session.trust_env = not ignore_proxy
        self.ok_count = 0
        self.fail_count = 0
        self.net_fail_count = 0       # сетевые/прокси сбои (системные)
        self.bad_response_count = 0   # сервер вернул не-JSON для конкретного запроса (битый район/маска)
        self.last_error = None
        self.last_text = None

    def _headers(self):
        return {'User-Agent': random.choice(_USER_AGENTS)}

    def _get(self, url):
        text = None
        try:
            text = self._session.get(url, headers=self._headers(), timeout=60).text
            self.last_text = text
            result = json.loads(text)
            self.ok_count += 1
            return result, True
        except Exception as exc:                       # noqa: BLE001
            self.fail_count += 1
            self.last_error = exc
            if isinstance(exc, json.JSONDecodeError):
                self.bad_response_count += 1           # ответ сервера, а не сбой сети
            else:
                self.net_fail_count += 1
            if self.fail_count <= 3:                    # не засыпаем консоль одинаковыми строками
                print('Vega запрос не удался:', exc)
                if 'SOCKS' in str(exc):
                    print('  Подсказка: задан socks://-прокси, но нет PySocks. '
                          'Уберите HTTP_PROXY/HTTPS_PROXY/ALL_PROXY или установите: pip install pysocks')
                elif isinstance(exc, json.JSONDecodeError):
                    snippet = (text or '')[:200].replace('\n', ' ').strip()
                    print('  Ответ Vega — не JSON для этого запроса (нет данных/неверный район или маска).',
                          'Начало ответа:', repr(snippet))
                    if not self.ukey or self.ukey == 'your_ukey_here':
                        print('  ukey не задан! Впишите настоящий ключ в workspace/config/ukeys.csv')
            return None, False

    def district(self, product_type, year, district_uid):
        """Данные районного уровня (``adm_dis``)."""
        product_type, year, district_uid = str(product_type), str(year), str(district_uid)
        url = (_BASE + '?ukey=' + self.ukey
               + '&x_axis_type=time&w=0&h=0&x1=1&x2=366&query='
               + '[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + district_uid
               + '%22,%22rows%22:{%22' + product_type + '%22:[' + year + ']}}]'
               + '&mode=basic&num_points=1&highcharts=1&a_week=' + self.a_week
               + '&a_year=' + year + '&label_year=' + year)
        return self._get(url)

    def region(self, product_type, year, region_uid):
        """Данные регионального уровня (``adm_reg``, продукт с префиксом ``reg_``)."""
        product_type, year, region_uid = str(product_type), str(year), str(region_uid)
        url = (_BASE + '?ukey=' + self.ukey
               + '&x_axis_type=time&w=0&h=0&x1=1&x2=366&query='
               + '[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22' + region_uid
               + '%22,%22rows%22:{%22reg_' + product_type + '%22:[' + year + ']}}]'
               + '&mode=basic&num_points=1&highcharts=1&no_cache=34576&a_week=' + self.a_week
               + '&a_year=' + year + '&label_year=' + year)
        return self._get(url)
