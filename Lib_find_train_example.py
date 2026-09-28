import pandas as pd
import numpy as np
from sympy import collect
from torch.hub import download_url_to_file
from Library_working_with_time_series_v1 import TimeRow, Operations
import json
import requests


import numpy as np
import matplotlib.pyplot as plt


from datetime import datetime, date, time

import db                      # Файловая реляционная БД (CSV-таблицы) — кэш временных рядов

class VegaAPI():
    """Класс для запроса дынных на сайте Vega-Scince"""
    def __init__(self):
        pass

    def district(self, product_type, year, reg_uid, ukey):
        success = False
        try:  # Пробуем запросить данные
            product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)

            Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&x2=366&query=' \
                              +'[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + reg_uid + '%22,%22rows%22:{%22' \
                              + product_type + '%22:[' + year + ']}}]&mode=basic&num_points=1&highcharts=1&a_week=51&a_year=' \
                              + year + '&label_year=' + year

            print('Save_graph_link', Save_graph_link)
            Graph = requests.get(Save_graph_link).text
            Result = json.loads(Graph)
            success = True

        except:  # Ошибка скорее всего значит завершение сессии. Нужно зайти снова и повторить запрос
            try:
                product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)

                Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&x2=366&query=' \
                                  + '[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + reg_uid + '%22,%22rows%22:{%22' \
                                  + product_type + '%22:[' + year + ']}}]&mode=basic&num_points=1&highcharts=1&a_week=51&a_year=' \
                                  + year + '&label_year=' + year

                print('Save_graph_link', Save_graph_link)
                Graph = requests.get(Save_graph_link).text
                Result = json.loads(Graph)
                success = True

            except:
                print('No internet...')
                Result = None

        return Result, success

    def region(self, product_type, year, reg_uid, ukey):
        success = False
        a_week = '51'

        try:  # Пробуем запросить данные
            product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)
            print('product_type, year, reg_uid', product_type, year, reg_uid)
            a_week = '51'
            Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&' \
                              + 'x2=366&query=[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22' \
                              + reg_uid + '%22,%22rows%22:{%22reg_' + product_type + '%22:[' + year + ']}}]&mode' \
                              + '=basic&num_points=1&highcharts=1&no_cache=34576&a_week=' + a_week \
                              + '&a_year=' + year + '&label_year=' + year
            print('Save_graph_link', Save_graph_link)
            Graph = requests.get(Save_graph_link).text
            Result = json.loads(Graph)
            success = True

        except:  # Ошибка скорее всего значит завершение сессии. Нужно зайти снова и повторить запрос
            try:
                product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)
                print('product_type, year, reg_uid', product_type, year, reg_uid)
                a_week = '51'
                Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&' \
                                  + 'x2=366&query=[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22' \
                                  + reg_uid + '%22,%22rows%22:{%22reg_' + product_type + '%22:[' + year + ']}}]&mode' \
                                  + '=basic&num_points=1&highcharts=1&no_cache=34576&a_week=' + a_week \
                                  + '&a_year=' + year + '&label_year=' + year
                print('Save_graph_link', Save_graph_link)
                Graph = requests.get(Save_graph_link).text
                Result = json.loads(Graph)
                success = True
            except:
                print('No internet...')
                Result = None

        return Result, success




def specify_the_parameters(culture: str,
                           table_of_masks: pd.DataFrame,
                           id_region: str,
                           parms: list) -> list:
    '''In this function, specify the culture in this line of the task and set the desired ndvi mask,
    in accordance with the area where the culture grows'''
    culture = culture.lower()
    pd_file = table_of_masks
    parms_correct = []

    for i in range(len(parms)):
        parm_i = parms[i]
        if 'ndvi' in parm_i:
            res_ndvi = pd_file[(pd_file['id_region'].astype(str).str.contains(str(id_region)))]
            mask_i = res_ndvi[culture].values[0]
            if 'histor' in parm_i:
                mask_i = mask_i + '_historical'
        else:
            mask_i = parm_i
        parms_correct.append(mask_i)
    return parms_correct

def get_target_year_list(year, second_year, years_back):
    time_rows = []
    year = int(year)
    years = [year]

    if second_year:
        for i in range(int(years_back)):
            years.append(year - i - 1)

    years.reverse()
    print('years', years)

    return years


def make_time_row_name(year, id_dist, id_reg, parm):
    """Функция формирует имя для файла"""
    # mean_temp_69_nan_2022
    try:
        id_dist = int(id_dist)
    except:
        id_dist = 'nan'
    file_name = parm + "_" + str(id_reg) + "_" + str(id_dist) + "_" + str(year)
    print('file_name', file_name)

    return file_name


def write_inf(data, file_name):
    data = json.dumps(data)
    data = json.loads(str(data))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
         return json.load(file)


def is_it_district(id_dist):
    try:
        id_dist = int(id_dist)
        district = True
    except:
        district = False
    return district



def download_init_data(year, id_dist, id_reg, parm, vega_files_directory, average_last, ukey):
    """Гарантирует наличие нужных рядов в БД (скачивает с Vega при отсутствии)."""
    is_hist = 'hist' in parm
    parm = parm.replace('_historical', '').replace('_hist', '')

    tid = db.territory_id(id_reg, id_dist)
    district = is_it_district(id_dist)

    if is_hist:
        years = [year - i for i in range(average_last)]
    else:
        years = [year]

    for year_i in years:
        print('year_i', year_i)
        have = db.has_time_series(tid, parm, year_i)

        if is_hist:
            need = (not have) or (year_i >= datetime.now().year and ukey != '')
        else:
            need = (not have) or (year_i >= datetime.now().year)

        if not need:
            continue

        vega_parser = VegaAPI()
        if district:
            vega_json_file = vega_parser.district(parm, year_i, id_dist, ukey=ukey)[0]
        else:
            vega_json_file = vega_parser.region(parm, year_i, id_reg, ukey)[0]

        try:
            xy = vega_json_file['data']['1']['xy']
            labels = vega_json_file['data']['1'].get('labels', {})
            if xy:
                db.put_time_series(tid, parm, year_i, xy, labels)
        except (TypeError, KeyError):
            print('Полученный файл пуст! Он не будет записан в БД...')




def download_time_row(year, id_dist, id_reg, parm, vega_files_directory, frequency, average_last, ukey):
    tid = db.territory_id(id_reg, id_dist)
    time_rows_size = int(365 * frequency)
    parm_clean = parm.replace('_historical', '').replace('_hist', '')

    if 'hist' in parm:
        try:
            download_init_data(year, id_dist, id_reg, parm, vega_files_directory, average_last, ukey)
        except Exception:
            print('При загрузке рядов для расчета исторических рядов произошел неопределенный сбой!')

        time_data = db.mean_over_years(tid, parm_clean, year, average_last, time_rows_size)

    else:
        download_init_data(year, id_dist, id_reg, parm, vega_files_directory, 1, ukey=ukey)
        time_data = db.get_time_series_array(tid, parm_clean, year, time_rows_size)

        if time_data.shape[0] == 0:
            # Текущий год не найден — заменяем многолетним средним
            for i in range(average_last):
                download_init_data(year - i, id_dist, id_reg, parm_clean, vega_files_directory, 1, ukey=ukey)
            time_data = db.mean_over_years(tid, parm_clean, year, average_last, time_rows_size)

    return time_data


def download_and_dock_time_row(year,
                    id_dist,
                    id_reg,
                    parm,
                    second_year,
                    vega_files_directory,
                    average_last,
                    frequency,
                    years_back = 1,
                    ukey=''):
    """Функция запрашивает временной ряд, склеивает его, если треубется данные за два года ряда"""

    print('* average_last, frequency, ukey ', average_last, frequency, ukey)

    years = get_target_year_list(year, second_year, years_back)
    time_row = np.array([])

    for year in years:
        array_i = download_time_row(year, id_dist, id_reg, parm, vega_files_directory, frequency, average_last=average_last, ukey=ukey)
        time_row = np.concatenate((time_row, array_i), axis=0)

    return time_row

def get_one_time_row(year,
                    id_dist,
                    id_reg,
                    parm,
                    range_of_row,
                    vega_files_directory,
                    average_last,
                    frequency,
                    ukey):
    """Функция возвращает один временной ряд, длиной n лет"""

    if range_of_row[0] < 0: # Проверим, требуется ли загрузка данных предыдущего года
        second_year = True
    else:
        second_year = False

    print('* average_last, frequency, ukey ', average_last, frequency, ukey)

    time_row = download_and_dock_time_row(year,
                    id_dist,
                    id_reg,
                    parm,
                    second_year,
                    vega_files_directory,
                    average_last=average_last,
                    frequency=frequency,
                    ukey=ukey)

    return time_row


def collect_time_row_matrix(year: int,
                      id_dist: int,
                      id_reg: int,
                      parms: list, # Список продуктов веги
                      parms_of_statistic: list, # Параметры, отражающие статистику. Их программа игнорирует.
                      range_of_row: list,
                      vega_files_directory: str,
                      average_last: int,
                      frequency=1,
                      ukey=''
                            )-> np.array:

    print('average_last, ukey', average_last, ukey)

    matrix = np.array([])  # Инициализируем пустой массив
    for parm in parms:
        if parm in parms_of_statistic:
            pass
        else:
            new_time_row = get_one_time_row(year,
                                            id_dist,
                                            id_reg,
                                            parm,
                                            range_of_row,
                                            vega_files_directory,
                                            average_last=average_last,
                                            frequency=frequency,
                                            ukey=ukey
                                            )

            print('matrix_shape_i', matrix.shape, 'parm_i', parm)

            if matrix.shape[0] == 0:  # Набор нового тензора
                matrix = np.array([new_time_row])
            else:
                matrix = np.vstack([matrix, [new_time_row]])

    matrix[:, 0] = matrix[:, 1]
    matrix[:, 364] = matrix[:, 363]
    if range_of_row[0] < 0:
        matrix[:, -1] = matrix[:, -2]

    return matrix


def get_train_example(year: int,
                      id_dist: int,
                      id_reg: int,
                      parms: list, # Список продуктов веги
                      parms_of_statistic: list, # Параметры, отражающие статистику. Их программа игнорирует.
                      table_of_masks: pd.DataFrame,
                      range_of_row: list,
                      vega_files_directory: str,
                      culture: str,
                      average_last: int,
                      ukey: str)-> np.array:
    """Функция возвращает набор временных рядов в заданном разрешении.
    Первый фрагмент обучающей пары до нормировки"""

    # 1) Убедимся, что в списке параметров верно указаны элементы списка.
    # Нужно убедиться, что маски ndvi соответствуют маскам в справочнике table_of_masks

    parms_correct = specify_the_parameters(culture, table_of_masks, id_reg, parms)

    print('parms_correct', parms_correct)
    print('average_last, ukey', average_last, ukey)

    time_row_matrix = collect_time_row_matrix(year, id_dist ,id_reg, parms_correct,
                                                       parms_of_statistic, range_of_row,
                                                       vega_files_directory, average_last, ukey=ukey)

    # 2) Запросим все параметры по списку
    #

    return time_row_matrix


def main():

    list_of_parms = ['mean_ndvi',
                     'mean_ndvi_historical',
                     'mean_temp',
                     'mean_temp_historical',
                     'mean_temp_acc',
                     'mean_temp_acc_historical',
                     'mean_prod',
                     'trend'
                     ]

    # list_of_parms = [
    #                  'mean_ndvi_historical',
    #                  'mean_ndvi',
    #                  'mean_temp',
    #                  'mean_temp_historical',
    #                  'mean_temp_acc',
    #                  'mean_temp_acc_historical',
    #                  'mean_prec',
    #                  'mean_prec_acc',
    #                  'mean_prec_acc_historical',
    #                  'mean_rh',
    #                  'mean_rh_historical',
    #                  'mean_p',
    #                  'mean_snod',
    #                  'mean_snod_historical',
    #                  'mean_snowc',
    #                  'mean_snowc_historical',
    #                  'mean_htc_decade',
    #                  'mean_htc_decade_historical',
    #                  'mean_sdswr',
    #                  'mean_sdswr_historical',
    #                  'mean_sdlwr',
    #                  'mean_sdlwr_historical',
    #                  'mean_tmpgr10',
    #                  'mean_tmpgr10_historical',
    #                  'mean_soilw10',
    #                  'mean_soilw10_historical',
    #                  'mean_prod',
    #                  'trend'
    #                  ]

    parms_of_statistic = ['mean_prod',
                          'trend'
                          ]

    # range_of_row = [0, 160]
    range_of_row = [-1, 160]

    data_file_name = "Service_files\\Common_information\\Cultures_masks.xlsx".replace('\\', os.sep)
    pd_file = pd.read_excel(data_file_name, index_col='Unnamed: 0')

    directory = 'Temporary_information\\TimeRows\\'.replace('\\', os.sep)
    culture = 'пшеница озимая'

    average_last = 5

    #  region 58  district 977 2018

    get_example = get_train_example(2018,   # 2015
                                    10,   # 1575
                                    22, # 64
                                    list_of_parms,  # Список продуктов веги
                                    parms_of_statistic,  # Параметры, отражающие статистику. Их программа игнорирует.
                                    pd_file,    # Файл с масками ndvi
                                    range_of_row,   # диапазон дней
                                    directory,  # путь к папке, где хранятся файлы с временными рядами
                                    culture,    # имя культуры
                                    average_last,
                                    ukey)   # сколько нужно брать лет для расчета среднегодового ряда

    print('get_example', get_example)
    print('get_example', get_example.shape)

    for i in range(get_example.shape[0]):
        print(list_of_parms[i], get_example[i].shape, get_example[i])
        plt.plot(get_example[i])  # черные ромбы
        plt.show()


if __name__ == '__main__':
    # Make_arrays_of_normalized_NDVI()
    main()

