import requests
#import fake_useragant
from bs4 import BeautifulSoup
import random
import json

def write_inf(data, file_name):
    data = json.dumps(data)
    data = json.loads(str(data))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
         return json.load(file)

def parser_vega_region(Login, Password, Reg_name, year, product_type):
    # Создание фальшивого User-Agent
    user = ['Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36',
            'Mozilla/5.0 (iPad; CPU OS 6_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/6.0 Mobile/10A5376e Safari/8536.25',
            'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
            'Opera/9.80 (Windows NT 6.2; WOW64) Presto/2.12.388 Version/12.17']  # Лист фальшивых имен
    index = random.randint(0, len(user) - 1)
    header = {'User-Agent': user[index], 'Content-Type': 'application/x-www-form-urlencoded', 'charset': 'UTF-8'}

    session = requests.Session()

        # Ссылка на сайт
    URL = 'http://sci-vega.ru/login/login.pl'

        # Логин и пароль
    # Login = os.environ.get('VEGA_LOGIN', '')
    # Password = os.environ.get('VEGA_PASSWORD', '')

        # Словарь для запроса
    data = {
        'first': Login,
        'second': Password,
        'mode': 'in',
        'from': '/'
    }

    # response = session.post(URL, data = data, headers=header)
    regs_uid_dict = read_inf("Regions_uid_dict.json")


    reg_uid = regs_uid_dict[Reg_name]

    a_week = '51'
    Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&x2=366&query=[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22'+ reg_uid +'%22,%22rows%22:{%22' + product_type + '%22:[' + year + ']}}]&mode=basic&num_points=1&highcharts=1&no_cache=25845&a_week=' + a_week + '&a_year='+ year +'&label_year='+ year

    Graph = requests.get(Save_graph_link).text
    Result = json.loads(Graph)


    return Result

def parser_vega_point(Login, Password, year, product_type, Eastern_longitude, Northern_latitude):

    # Создание фальшивого User-Agent
    user = ['Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36',
            'Mozilla/5.0 (iPad; CPU OS 6_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/6.0 Mobile/10A5376e Safari/8536.25',
            'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
            'Opera/9.80 (Windows NT 6.2; WOW64) Presto/2.12.388 Version/12.17']  # Лист фальшивых имен
    index = random.randint(0, len(user) - 1)
    header = {'User-Agent': user[index], 'Content-Type': 'application/x-www-form-urlencoded', 'charset': 'UTF-8'}

    # session = requests.Session()

        # Ссылка на сайт
    URL = 'http://sci-vega.ru/login/login.pl'

        # Логин и пароль
    # Login = os.environ.get('VEGA_LOGIN', '')
    # Password = os.environ.get('VEGA_PASSWORD', '')

        # Словарь для запроса
    data = {
        'first': Login,
        'second': Password,
        'mode': 'in',
        'from': '/'
    }

    response = session.post(URL, data = data, headers=header)
    regs_uid_dict = read_inf("Regions_uid_dict.json")

    # Координаты точки наблюдения
    # Eastern_longitude = '34.1591'
    # Northern_latitude = '45.589'
    # Reg_name = 'Нижегородская область'

        # Формирование полезной нагрузки запроса
    # year = '2022'
    a_week = '51'
    # a_week = '28'

    # product_type = 'ndvi_7dc_modis_int'
    # product_type = 'reg_mean_ndvi_7dc_modis_int_ozim'
    # reg_uid = regs_uid_dict[Reg_name]

        # URL запрос
    # Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&x2=366&query=[{"c":1,"type":"point","uid":"' + Eastern_longitude + ',' + Northern_latitude + '","rows":{"' + product_type + '":[' + year +']}}]&mode=basic&num_points=1&highcharts=1&no_cache=22835&a_week=' + a_week + '&a_year=' + year +'&label_year=' + year
    Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&x2=366&query=[{"c":1,"type":"point","uid":"83.01709,52.62451","rows":{"temp":[2023]}}]&mode=basic&num_points=1&highcharts=1&no_cache=97336&a_week=28&a_year=2023&label_year=2023'



    print('')
    print(Save_graph_link)


    Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?x_axis_type=time&w=0&h=0&x1=1&x2=366&query=[{"c":1,"type":"point","uid":"' + Eastern_longitude + ',' + Northern_latitude + '","rows":{"' + product_type + '":[' + year +']}}]&mode=basic&num_points=1&highcharts=1&no_cache=97336&a_week=' + a_week + '&a_year=' + year +'&label_year=' + year


    print('')
    print(Save_graph_link)
    print('')

    Graph = requests.get(Save_graph_link).text
    print(Graph)
    Result = json.loads(Graph)

    # print(Result["data"]["1"]["xy"])


    return Result














