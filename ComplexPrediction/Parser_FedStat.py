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

def parser_fedstat(Year, Region, Culture, Farm_type):
    # Создание фальшивого User-Agent
    user = ['Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36',
            'Mozilla/5.0 (iPad; CPU OS 6_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/6.0 Mobile/10A5376e Safari/8536.25',
            'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
            'Opera/9.80 (Windows NT 6.2; WOW64) Presto/2.12.388 Version/12.17'] #Лист фальшивых имен
    index = random.randint(0, len(user)-1)
    header = {'User-Agent': user[index], 'Content-Type': 'application/x-www-form-urlencoded', 'charset': 'UTF-8'}

    URL = 'https://www.fedstat.ru/indicator/dataGrid.do?id=31533'
    # h = {'Content-Type': 'application/x-www-form-urlencoded', 'charset': 'UTF-8'}

     # Dictionaries
    Categories_of_farms_dictionary = read_inf("Categories_of_farms_dictionary.json") #Загрузка словаря из json файла
    Region_Dictionary = read_inf("Region_Dictionary.json")
    Culture_Dictionary = read_inf("Culture_Dictionary.json")

    # print(Region, Region_Dictionary[Region])
    # print(Culture, Culture_Dictionary[Culture])
    # print(Farm_type, Categories_of_farms_dictionary[Farm_type])

    data = "lineObjectIds=0&lineObjectIds=33560&lineObjectIds=30611&lineObjectIds=58423&lineObjectIds=57831&lineObjectIds=58745&columnObjectIds=3&selectedFilterIds=0_31533&selectedFilterIds=3_" + Year + "&selectedFilterIds=30611_1342019&selectedFilterIds=33560_1558883&selectedFilterIds=57831_" + Region_Dictionary[Region] + "&selectedFilterIds=58745_"+ Culture_Dictionary[Culture] + "&selectedFilterIds=58423_" + Categories_of_farms_dictionary[Farm_type] + "&selectedFilterId"

    response = requests.post(URL, data=data, headers=header).text

    # print(response)

    DATA = json.loads(response)

    print(DATA)
    KEYS = list(DATA['results'][0].keys())
    # print(KEYS)
    index = KEYS[-1]
    # print(index)
    DATA = DATA['results'][0][index]

    return DATA



