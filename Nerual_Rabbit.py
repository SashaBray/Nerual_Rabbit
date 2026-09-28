import os
from lib2to3.fixes.fix_tuple_params import find_params

import numpy as np
import pandas as pd
from openpyxl.styles.builtins import total
# from scipy.special import delta
from scipy.special import beta
from sympy.physics.units import years
from torch.utils.hipify.hipify_python import value
from tqdm import tqdm
import time
import alive_progress
from alive_progress import alive_bar
import copy

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statistics

import torch
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

import json
import requests

import random
import math

from Library_working_with_time_series_v1 import TimeRow, Operations

import pickle

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from typing import Tuple

import cProfile

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from Lib_find_train_example import get_train_example
from datetime import datetime, date, time

import db                      # Файловая реляционная БД (CSV-таблицы) — слой доступа к данным
import task_form               # Загрузка тонкой формы расчётного задания


class CalculationTask():
    """Класс является носителем информации о расчетном задании"""
    def __init__(self, task_file=None, years=None, prediction_years_iteration = 0, target=None, success=None, directory = 'Destination_files\\'.replace('\\', os.sep),
                 column_predict=None, column_models=None, models_list = None,
                 task_file_reserw=None,
                 culture=None, years_to_back=None, current_day=None,
                 current_year=None, total_historical_puding=False,
                 avegage_last=None, years_for_correct=1,
                 prediction_years_secret=None, correction=False, directory_time_rows=None, ukey='') -> None:

        self.task_file = task_file
        self.prediction_years = years
        self.prediction_years_iteration = prediction_years_iteration
        self.target = target
        self.success = success
        self.directory = directory
        self.directory_time_rows = directory_time_rows

        self.column_predict = column_predict
        self.column_models = column_models
        self.models_list = models_list

        self.task_file_reserw = task_file_reserw
        self.culture = culture

        day = datetime.now().day
        month = datetime.now().month
        year = datetime.now().year

        data_now = date(year, int(month), int(day))
        data_year_begining = date(year, 1, 1)
        delta_days = data_now - data_year_begining
        day_since_the_beginning_of_the_year = delta_days.days

        print('day_since_the_beginning_of_the_year', day_since_the_beginning_of_the_year)
        # print(hehehe)
        self.year_now = year

        self.years_to_back = years_to_back
        self.current_day = day_since_the_beginning_of_the_year
        self.current_year = datetime.now().year
        self.total_historical_puding = total_historical_puding
        self.avegage_last = avegage_last
        self.years_for_correct = years_for_correct
        self.prediction_years_secret = prediction_years_secret
        self.correction = correction
        self.ukey = ukey


class ModelWrapper():
    """Класс обертка для моделей"""
    def __init__(self, model_name,
                 parms_file = None,
                 timeline_file = None,
                 task=None,
                 masks=None):

        self.model_name = model_name
        self.parms_file = parms_file
        self.timeline_file = timeline_file
        self.task = task
        self.masks = masks

    def activate(self, directory=f'Service_files{os.sep}Models{os.sep}'.replace('\\', os.sep),
                 parm_file=f'{os.sep}parms.json', prossesing=f'{os.sep}prossesing.json'.replace('\\', os.sep),
                 masks_directory=f"Service_files{os.sep}Common_information{os.sep}".replace('\\', os.sep), masks_file='Cultures_masks.xlsx'):

        way = directory + self.model_name

        model_info = db.get_model(self.model_name)        # Метаданные модели из БД
        params = db.list_model_params(self.model_name)    # Упорядоченный список параметров + нормировка
        if model_info is None or not params:
            raise ValueError(f'Модель {self.model_name} не найдена в БД (Database/models.csv).')

        self.parms = [p['parm'] for p in params]
        self.normalization_parms = [p['normalization'] for p in params]
        self.first_day = int(model_info['time_border_first'])
        self.last_day = int(model_info['time_border_last'])

        # Совместимость с кодом, читающим model.parm_file_info[...]
        self.parm_file_info = {'parm_list': self.parms,
                               'time_borders': [self.first_day, self.last_day],
                               'normalization': self.normalization_parms}

        self.parms_scalars = [p['parm'] for p in params
                              if str(p.get('scalar')) not in ('', 'nan', 'None')] or None

        self.pd_masks = db.get_masks_df()                 # Маски культур из БД

        print('model.name', self.model_name)

        # try:
        if 'NN' in self.model_name: # self.model_name == 'NN_CN_v1_winter_wheat':
            model_file_name = '\\model_scripted.pt'.replace('\\', os.sep)
            model = torch.jit.load(way + model_file_name)
            model.train()

        if 'RF' in self.model_name: # self.model_name == 'RF_v1_winter_wheat':
            # way = 'Service_files\\Models\\'.replace('\\', os.sep) + self.model_name
            model_file_name = '\\my_model.pickle'.replace('\\', os.sep)
            model = pickle.load(open(way + model_file_name, "rb"))
            print(model)

        if 'LM' in self.model_name:
            model = Simple_linear_model()
            print('Выбрана простая линейная модель')

        if 'EXM' in self.model_name:
            model = Simple_exp_model()
            print('Выбрана простая экспоненциальная модель')

        self.model = model
        print('Активация модели успешно произведена!')
        pass

        if model == None:
            print('Ошибка! Модель не найдена! Проверьте корректность введенного в таблицу имени модели.')
        else:
            print('Модель успешно активирована!')


class Answer():
    """Класс ответа. Хранит вердикт и дополнительную информацию, которую нужно вывести в отчет"""
    def __init__(self, verdict: [float], additional_inf={}, inf_unconditional={}):
        self.verdict = verdict
        self.additional_inf = additional_inf
        self.inf_unconditional = inf_unconditional


class DataPrepearer():
    """Класс, принимающий на вход данные, полученные на сайте Vega и преобразует их в потребную для модели форму,
    производит отсечение нужного диапазона, нормализацию, усреднение, введение рядов с константой"""
    def __init__(self, model_name, parms_file=None, timeline_file=None, input_type=None):
        self.model_name = model_name
        self.parms_file = parms_file
        self.timeline_file = timeline_file
        self.input_type = input_type

    def activate(self):
        """Приведение экземпляра в готовность"""
        pass

    def transform_data(self):
        """Трансформация данных в нужный вид"""
        pass

class VegaAPI():
    """Класс для запроса дынных на сайте Vega-Scince"""
    def __init__(self):
        pass


    def connect(self):
        self.user = ['Mozilla/5.0 (Windows NT 6.3; WOW64; rv:36.0) Gecko/20100101 Firefox/36.0',
                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36',
                'Mozilla/5.0 (iPad; CPU OS 6_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/6.0 Mobile/10A5376e Safari/8536.25',
                'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
                'Opera/9.80 (Windows NT 6.2; WOW64) Presto/2.12.388 Version/12.17']  # Лист фальшивых имен
        index = random.randint(0, len(self.user) - 1)
        self.header = {'User-Agent': self.user[index], 'Content-Type': 'application/x-www-form-urlencoded', 'charset': 'UTF-8'}
        self.session = requests.Session()

            # Ссылка на сайт
        self.URL = 'http://sci-vega.ru/login/login.pl'

            # Логин и пароль
        # Login = self.password
        # Password = self.password

        # self.data = {'first': Login, 'second': Password, 'mode': 'in', 'from': '/'}
        # self.response = self.session.post(self.URL, data=self.data, headers=self.header)


    def district(self, product_type, year, reg_uid, ukey):

        ukey = os.environ.get('VEGA_UKEY', '')      # ключ Vega берётся из окружения, в коде его нет

        # print(product_type, year, reg_uid, ' Loading in progress...')  # Выводим на экран
        success = False
        try:  # Пробуем запросить данные
            product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)

            Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&x2=366&query=' \
                              +'[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + reg_uid + '%22,%22rows%22:{%22' \
                              + product_type + '%22:[' + year + ']}}]&mode=basic&num_points=1&highcharts=1&a_week=51&a_year=' \
                              + year + '&label_year=' + year

            print('Save_graph_link', Save_graph_link)
            # Graph = self.session.get(Save_graph_link).text
            Graph = requests.get(Save_graph_link).text
            # print('Graph', Graph)
            Result = json.loads(Graph)
            success = True

        except:  # Ошибка скорее всего значит завершение сессии. Нужно зайти снова и повторить запрос
            try:
                self.session = requests.Session()
                self.response = self.session.post(URL, data=data, headers=header)


                Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&x2=366&query=' \
                                  + '[{%22c%22:1,%22type%22:%22adm_dis%22,%22uid%22:%22' + reg_uid + '%22,%22rows%22:{%22' \
                                  + product_type + '%22:[' + year + ']}}]&mode=basic&num_points=1&highcharts=1&a_week=51&a_year=' \
                                  + year + '&label_year=' + year

                print('Save_graph_link', Save_graph_link)
                # Graph = self.session.get(Save_graph_link).text
                Graph = requests.get(Save_graph_link).text
                Result = json.loads(Graph)
                success = True

            except:
                print('No internet...')
                Result = None

        return Result, success

    def region(self, product_type, year, reg_uid, ukey):
        # print(product_type, year, reg_uid, ' Loading in progress...')  # Выводим на экран
        success = False
        a_week = '51'

        ukey = os.environ.get('VEGA_UKEY', '')      # ключ Vega берётся из окружения, в коде его нет

        try:  # Пробуем запросить данные

            product_type, year, reg_uid = str(product_type), str(year), str(reg_uid)

            print('product_type, year, reg_uid', product_type, year, reg_uid)

            a_week = '51'

            Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + 'x_axis_type=time&w=0&h=0&x1=1&' \
                              + 'x2=366&query=[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22' \
                              + reg_uid + '%22,%22rows%22:{%22reg_' + product_type + '%22:[' + year + ']}}]&mode' \
                              + '=basic&num_points=1&highcharts=1&no_cache=34576&a_week=' + a_week \
                              + '&a_year=' + year + '&label_year=' + year


            print('Save_graph_link', Save_graph_link)
            # print('type Save_graph_link', type(Save_graph_link))

            Graph = requests.get(Save_graph_link).text
            # print('Graph', Graph)
            Result = json.loads(Graph)
            success = True

        except:  # Ошибка скорее всего значит завершение сессии. Нужно зайти снова и повторить запрос
            try:
                self.session = requests.Session()
                self.response = self.session.post(URL, data=data, headers=header)
                Save_graph_link = 'http://sci-vega.ru/geosmis_charts_v2/plot.pl?ukey=' + ukey + '&x_axis_type=time&w=0&h=0&x1=1&'\
                                  + 'x2=366&query=[{%22c%22:1,%22type%22:%22adm_reg%22,%22uid%22:%22'\
                                  + reg_uid + '%22,%22rows%22:{%22reg_' + product_type + '%22:[' + year + ']}}]&mode'\
                                  + '=basic&num_points=1&highcharts=1&no_cache=34576&a_week=' + a_week\
                                  + '&a_year=' + year + '&label_year=' + year


                print('Save_graph_link', Save_graph_link)
                # Graph = self.session.get(Save_graph_link).text
                Graph = requests.get(Save_graph_link).text
                Result = json.loads(Graph)
                success = True

            except:
                print('No internet...')
                Result = None

        return Result, success


def print_metrics(y_true, y_predicted, x_train, answer, task, years_available, predicts_for_past) -> Answer:

    print('')
    print('true_for_metrics', y_true)
    print('predicted_for_metrics', y_predicted)
    print('')

    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    x_shape, y_shape = max(y_true.shape), min(y_true.shape)

    try:
        print(f"Mean squared error: {mean_squared_error(y_true, y_predicted):.3f}")
        print(
            "Root mean squared error: ",
            f"{mean_squared_error(y_true, y_predicted, squared=False):.3f}",
        )
        print(f"Mean absolute error: {mean_absolute_error(y_true, y_predicted):.3f}")
        print(f"R2 score: {r2_score(y_true, y_predicted):.3f}")

    except:
        print('Ошибка при расчете метрик!')

    try:
        answer.inf_unconditional.setdefault('R2 score', r2_score(y_true, y_predicted))      # Расчет основных метрик
        answer.inf_unconditional['R2 score'] = r2_score(y_true, y_predicted)
    except:
        print('Ошибка при расчете коэффициента детерминации!')
    try:
        answer.inf_unconditional.setdefault('Mean squared error', mean_squared_error(y_true, y_predicted))
        answer.inf_unconditional['Mean squared error'] = mean_squared_error(y_true, y_predicted)
    except:
        print('Ошибка при расчете СКО!')
    try:
        answer.inf_unconditional.setdefault('Root mean squared error', mean_squared_error(y_true, y_predicted, squared=False))
        answer.inf_unconditional['Root mean squared error'] = mean_squared_error(y_true, y_predicted, squared=False)
    except:
        print('Ошибка при расчете корня из СКО')

    try:
        answer.inf_unconditional.setdefault('Max absolute error', abs(y_true.reshape(-1, x_shape) - y_predicted.reshape(-1, x_shape)).max())
        answer.inf_unconditional['Max absolute error'] = abs(y_true.reshape(-1, x_shape) - y_predicted.reshape(-1, x_shape)).max()
    except:
        print('Ошибка при расчете максимальной ошибки!')

    try:
        answer.inf_unconditional.setdefault('Mean absolute error', mean_absolute_error(y_true, y_predicted))
        answer.inf_unconditional['Mean absolute error'] = mean_absolute_error(y_true, y_predicted)
    except:
        print('Ошибка при расчете MAE!')

    try:       # Расчет и запись коэффициента корреляции
        print('Расчет и запись коэффициента корреляции', y_true, y_predicted)
        correleatio_factor = statistics.correlation(y_true, y_predicted)
        answer.inf_unconditional.setdefault('Pearson correlation (real productove and predicted)', correleatio_factor)
        answer.inf_unconditional['Pearson correlation (real productove and predicted)'] = correleatio_factor
    except:
        print('Недостаточно точек для рассчета коэффициента кореляции!')
        answer.inf_unconditional.setdefault('Pearson correlation (real productove and predicted)', None)
        answer.inf_unconditional['Pearson correlation (real productove and predicted)'] = None

    print('answer -', answer.inf_unconditional)

    return answer


def print_settlement_information(answer, data, target, task):
    '''[maximums, prods, target_arg, years]'''

    answer.additional_inf = {}
    years = data[3]
    maximums = data[0]

    headline = task.prediction_years[task.prediction_years_iteration] + '_ndvi_max'
    value = target
    answer.additional_inf.setdefault(headline, value)

    for i in range(len(years)):
        year = years[i]
        headline = str(year) + '_ndvi_max'
        value = maximums[i]
        answer.additional_inf.setdefault(headline, value)

    return answer


def create_years_ndvi_line_graphic(data, task, row, way='Report_files\\Graphs\\'.replace('\\', os.sep), file_name='New_graphic', vector=True):
    '''Функция получает информацию и создает гарфик изменения ndvi и урожайности в разные годы'''

    YEARS = copy.deepcopy(data[3]) # Заводим массив годов
    # if not int(task.prediction_years[task.prediction_years_iteration]) in YEARS:
    #     YEARS.append(int(task.prediction_years[task.prediction_years_iteration]))  # Если в списке нет изучаемого года, он добавляется

    NDVI_YEARS = copy.deepcopy(data[0])    # Заводим переменную с макс знач ndvi

    # print('heheh', int(task.prediction_years[task.prediction_years_iteration]) in NDVI_YEARS)

    if not int(task.prediction_years[task.prediction_years_iteration]) in YEARS:
        NDVI_YEARS_interim = [data[2]]
        YEARS_interim = [int(task.prediction_years[task.prediction_years_iteration])]

        # NDVI_YEARS_interim.append(NDVI_YEARS)  # Если в списке нет изучаемого года, добавляем ndvi этого года
        # YEARS_interim.append(YEARS)

        NDVI_YEARS = NDVI_YEARS_interim + NDVI_YEARS
        YEARS = YEARS_interim + YEARS

    PROD = copy.deepcopy(data[1])  # График урожайности
    P_YEARS = copy.deepcopy(data[3])  # Годы под урожайность



    print('YEARS     ', YEARS, len(YEARS))
    print('NDVI_YEARS', NDVI_YEARS, len(NDVI_YEARS))

    print('P_YEARS', P_YEARS, len(P_YEARS))
    print('PROD   ', PROD, len(PROD))

    YEARS = [str(item) for item in YEARS]
    P_YEARS = [str(item) for item in P_YEARS]

    print('str YEARS     ', YEARS, len(YEARS))
    print('str_P_YEARS', P_YEARS, len(P_YEARS))

    YEARS.reverse()     # Разворачиваем все листы, чтобы соблюсти порядок
    P_YEARS.reverse()
    NDVI_YEARS.reverse()
    PROD.reverse()


    fig, ax1 = plt.subplots(figsize=(12, 6))  # График урожайности и пика NDVI
    ax2 = ax1.twinx()
    ax1.plot(YEARS, NDVI_YEARS, label=' NDVI ', color='blue')
    ax2.plot(P_YEARS, PROD, label=' Урожайность ', color='darkred')
    ax1.set_ylabel('NDVI', color='blue', size=11)
    ax2.set_ylabel('Урожайность (ц/га)', color='darkred', size=11)

    if str(row['district']) == 'nan':
        plt.title('NDVI/Урожайность по годам: ' + row['region'], size=14)
    else:
        plt.title('NDVI/Урожайность по годам: ' + row['region'] + ' - ' + row['district'], size=14)

    fig.legend(bbox_to_anchor=(0.65, 0.07), ncol=2, fontsize=11)
    plt.grid(True)

    if vector:
        pdf = PdfPages(way + file_name + '.pdf')
        pdf.savefig()
        pdf.close()

    else:
        plt.savefig(way + file_name + '.jpg')
        plt.close()
    pass


def get_regression_line_for_graph(ndvi, prod, regressor):

    prod = np.array(prod)
    ndvi = np.array(ndvi)

    ndvi_min = ndvi.min()
    ndvi_max = ndvi.max()

    n = (ndvi_max - ndvi_min) / 20
    n = 1 if n==0 else n

    points = np.arange(ndvi_min - 10 * n, ndvi_max + 10 * n, n)

    regression_points = []
    for i in range(len(points)):
        point = np.array(points[i]).reshape(1, -1)
        regression_point_i = np.exp(regressor.predict(point)).reshape(1)
        regression_points.append(regression_point_i)

    print('regression_points', regression_points)

    return points, regression_points


def create_years_ndvi_point_graphic(data, task, row, regressor, way, file_name, vector=True):
    '''Процедура получает информацию и создает гарфик изменения ndvi и урожайности в разные годы'''
    #   [maximums, prods, target_arg, years]

    YEARS = copy.deepcopy(data[3])  # Заводим массив годов # years
    NDVI_YEARS = copy.deepcopy(data[0])  # Заводим переменную с макс знач ndvi # maximums
    PROD = copy.deepcopy(data[1])  # График урожайности # prods
    P_YEARS = copy.deepcopy(data[3])  # Годы под урожайность # years

    print('YEARS     ', YEARS, len(YEARS))
    print('NDVI_YEARS', NDVI_YEARS, len(NDVI_YEARS))
    print('P_YEARS', P_YEARS, len(P_YEARS))
    print('PROD   ', PROD, len(PROD))
    print('str YEARS     ', YEARS, len(YEARS))
    print('str_P_YEARS', P_YEARS, len(P_YEARS))

    YEARS.reverse()  # Разворачиваем все листы, чтобы соблюсти порядок
    P_YEARS.reverse()
    NDVI_YEARS.reverse()
    PROD.reverse()

    x = NDVI_YEARS
    y = PROD

    points_arg, points_regression = get_regression_line_for_graph(NDVI_YEARS, PROD, regressor)

    plt.figure(figsize=(10, 6))  # Гарфик линейной регрессии
    plt.scatter(x, y, label=' NDVI ', color='blue')
    plt.plot(points_arg, points_regression, label=' Экспоненциальная регрессия ', color='darkred')
    plt.xlabel('NDVI max', size=11)
    plt.ylabel('Урожайность (ц/га) ', size=11)

    if str(row['district']) == 'nan':
        plt.title('Экспоненциальная: ' + row['region'] , size=14)
    else:
        plt.title('Экспоненциальная: ' + row['region'] + ' - ' + row['district'], size=14)
    plt.grid(True)

    if vector:
        pdf = PdfPages(way + file_name + '.pdf')
        pdf.savefig()
        pdf.close()

    else:
        plt.savefig(way + file_name + '.jpg')
        plt.close()


def make_graphs_to_illustrate_simple_linear_model(row, regressor, task, data):
    district = row['district']
    region = row['region']
    region_id = row['id_region']
    district_id = row['id_district']

    if str(district_id) == 'nan':
        district_id = '-'

    if str(district) == 'nan':
        district = '-'

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'line_graph' + '_' + 'SLM'

    way = 'Report_files\\Graphs\\PDF\\'.replace('\\', os.sep)
    create_years_ndvi_line_graphic(data, task, row, way=way, file_name=linear_graph_name, vector=True)

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'line_graph' + '_' + 'SLM'

    way = 'Report_files\\Graphs\\JPG\\'.replace('\\', os.sep)
    create_years_ndvi_line_graphic(data, task, row, way=way, file_name=linear_graph_name, vector=False)

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'point_graph' + '_' + 'SLM'
    way = 'Report_files\\Graphs\\PDF\\'.replace('\\', os.sep)
    create_years_ndvi_point_graphic(data, task, row, regressor, way=way, file_name=linear_graph_name, vector=True)

    linear_graph_name = region + '_' + str(region_id) + '_' + district + '_' + str(district_id) + '_' + str(
        task.prediction_years[task.prediction_years_iteration]) + \
                        '_' + 'point_graph' + '_' + 'SLM'
    way = 'Report_files\\Graphs\\JPG\\'.replace('\\', os.sep)
    create_years_ndvi_point_graphic(data, task, row, regressor, way=way, file_name=linear_graph_name, vector=False)


class Simple_linear_model():
    """Класс простой линейно-регрессионной модели"""

    def predict(self, data, task, row):
        print('predict is on', data)

        # if data[0].shape[0] >= 2:
        if len(data[0]) >= 2:
            # [maximums, prods, target_arg]

            # x_train_scaled = data['x_train']
            # y_train = data['y_train']
            # x_target_scaled = data['x_target_rows']

            x_train_scaled = np.array(data[0]).reshape(-1, 1)
            y_train = np.array(data[1]).reshape(-1, 1)
            x_target_scaled = np.array(data[2]).reshape(-1, 1)

            # print('')
            # print('x_train_scaled', x_train_scaled.shape)
            # print('y_train', y_train.shape)
            # print('x_target_scaled', x_target_scaled.shape)

            print('Изнач y', y_train)
            y_train_ln = np.log(y_train)
            print('Логарифм y', y_train_ln)

            regressor = LinearRegression()
            regressor.fit(x_train_scaled, y_train_ln)

            y_pred_s_ln = regressor.predict(x_target_scaled)
            y_pred_ln = regressor.predict(x_train_scaled)

            y_pred_s = np.exp(y_pred_s_ln)
            y_pred = np.exp(y_pred_ln)

            print('y_pred_s', y_pred_s)
            print('y_pred', y_pred)

            # print(heheh)

            print('argument', x_target_scaled)
            print('answer', y_pred_s)

            x, y = x_train_scaled, y_train

            answer = Answer(verdict=y_pred_s)

            years_available = gel_years_list_with_prods(task, row, all_points=True)
            predicts_for_past = y_pred

            answer = print_metrics(y_train, y_pred, x_train_scaled, answer, task, years_available, predicts_for_past)
            answer = print_settlement_information(answer, data, x_target_scaled, task)

            make_graphs_to_illustrate_simple_linear_model(row, regressor, task, data)

        else:
            print('Недостаточно точек для построения регрессий!')
            answer = Answer(verdict=None)

        return answer


class Simple_exp_model():
    """Класс простой линейно-регрессионной модели"""

    def predict(self, data):
        # [maximums, prods, target_arg]
        print('data[0]', data[0])
        print('data[1]', data[1])
        x_train_scaled = np.array(data[0]).reshape(-1, 1)
        y_train = np.array(data[1]).reshape(-1, 1)

        # x_train_scaled = np.log(x_train_scaled)
        y_train = np.log(y_train)

        print('y_train_log', y_train)
        print('x_train_scaled', x_train_scaled)

        x_test_scaled = np.array([2]).reshape(-1, 1)

        print('x_train_scaled.shape', x_train_scaled.shape)

        regressor = LinearRegression()
        regressor.fit(x_train_scaled, y_train)

        y_pred = regressor.predict(x_test_scaled)
        # y_pred = np.exp(y_pred)
        y_pred = 2.718281828**y_pred

        return y_pred


def write_inf(data, file_name):
    data = json.dumps(data)
    data = json.loads(str(data))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
         return json.load(file)

def scenario_1(task):
    directory = task.directory
    files = os.listdir(directory)

    while True:
        print('Укажите файл с расчетным заданием: ')
        print(files)
        print('или')
        print("['Назад']")

        phrase = input()

        if phrase.replace("'", '') in files:
            print('Имя файла записано!', phrase)
            task.task_file = phrase
            # task.write_task_file(phrase)
            print('Gaga', task.task_file)
            break
        elif phrase.replace(' ', '').replace("'", '') == 'Назад':
            break
        else:
            'Команда не распознана! \n'

    return task


def make_design(phrase: [str], task):       # Обработка входящих команд

    command_1 = '     Сделать прогноз по расчетному заданию '
    command_2 = '     Завершить работу'

    while True:

        if phrase.replace(' ', '') == command_1.replace(' ', ''):   # На ввод файла с заданием
            task = scenario_1(task)
            break

        elif phrase.replace(' ', '') == command_2.replace(' ', ''): # Завершение работы
            # task = scenario_2(task)
            break
        else:
            print('Команда не распознана...', '\n')
    return task


def сonduct_user_survey():
    availability_survey = True
    task = CalculationTask()
    command_2 = '     Завершить работу'

    while availability_survey:
        print('Введите команду:') # \n')
        print('     Сделать прогноз по расчетному заданию ')
        print('     Завершить работу')

        phrase = input()
        if phrase != command_2:
            task = make_design(phrase, task)
            break
        elif phrase == command_2:
            break
        else:
            print('Команда не распознана...', '\n')
    return task

def fast_user_survey(task_form_file='task_form_example.csv'):
    """Инициализация задания из тонкой формы (CSV/XLSX).

    Форма содержит только то, что нужно задать: ``id_region, id_district,
    culture, model, predict_year``. История урожайностей и временные ряды
    берутся из БД, результаты пишутся в таблицу ``predictions``.
    """
    task = CalculationTask(
                           task_file=task_form_file,
                           culture=None,
                           years_to_back=4, current_day=290, current_year=2024,
                           total_historical_puding=False, avegage_last=4,
                           years_for_correct=1, correction=False,
                           directory_time_rows='Temporary_information\\TimeRows\\'.replace('\\', os.sep))

    spec = task_form.load_task_form(task.directory + task_form_file)

    task.territory_ids = spec['territory_ids']
    task.culture_by_territory = spec['culture_by_territory']
    task.jobs = spec['jobs']

    pairs = spec['model_year_pairs']
    task.column_models = [model_name for model_name, _ in pairs]
    task.prediction_years = [str(year) for _, year in pairs]
    task.column_predict = [f'{model_name}_predict_{year}' for model_name, year in pairs]

    print('Загружено задание:', len(task.territory_ids), 'территорий,',
          len(pairs), 'моделей×лет')
    return task

def get_models_list(df_destination) -> list:
    """Получение списка моделей, упомянутых в файле"""
    column_names = df_destination.columns.tolist()
    column_predict = [] # Список заголовков под предсказание
    column_models = [] # Список моделей
    years = [] # Список годов
    print('Заголовки в файле задания', column_names)

    for name in column_names:
        if '_predict_' in str(name):
            column_predict.append(name)
    print('Заголовки под предсказание', column_predict)

    for name in column_predict:
        model_name = name.split('_predict_')[0] # Записываем текст до маякового слова '_predict_'
        year = name.split('_predict_')[-1] # Записываем текст после маякового слова '_predict_'
        column_models.append(model_name)
        years.append(year)

    print('Указанные модели', column_models)
    print('Указанные годы', years, '\n')

    return column_predict, column_models, years


def make_reserw_copy_destination(task, keyword = '_reserw'):
    """Создание резервной копии задания"""
    way = task.directory + task.task_file
    df_destination = pd.read_excel(way, index_col='Unnamed: 0')

    extension = '.xlsx'
    reserw_way = task.directory + task.task_file.replace(extension, '') + keyword + extension
    df_destination.to_excel(reserw_way)

    task.task_file_reserw = task.task_file.replace(extension, '') + keyword + extension       # Записываем в класс сведения о положении резервной копии

    return task


def destination_analysis(task):
    'Предварительный анализ файла с заданием. Выяснение года предсказания, получение списка моделей, упомянутых в файле'
    way = task.directory + task.task_file       # Путь файлв str
    df_destination = pd.read_excel(way, index_col='Unnamed: 0')     # Загрузка файла с заданием
    # print(df_destination)
    task.column_predict, task.column_models, task.prediction_years = get_models_list(df_destination)    # Вычитываем в файле указанные модели

    task = make_reserw_copy_destination(task)      # Создаем резервную копию на всякий случай

    # print('df_destination', df_destination)
    # print("df_destination['culture']", df_destination['culture'].values[0])
    # print(heheh)

    return task


def model_Loader(task):
    """Создаются и сохраняются в экземпляр класса задания списки экземпляров класса, служащего оберткой для моделей"""
    models_name_list = task.column_models
    models_list = []

    for model_name in models_name_list:
        model = ModelWrapper(model_name)
        models_list.append(model)

    task.models_list = models_list

    return task


def chek_file(parm, row, task):
    """Есть ли ряд в БД (замена проверки наличия .json-файла).

    Межгодовые (``historical``) ряды в БД не хранятся — считаются на лету,
    поэтому для них всегда ``False`` (как и раньше: файла ``interannual`` нет).
    """
    if 'historical' in parm:
        return False

    tid = db.territory_id(row['id_region'], row['id_district'])
    year = task.prediction_years[task.prediction_years_iteration]
    found = db.has_time_series(tid, str(parm), year)
    if not found:
        print('Ряд', parm, tid, year, 'в БД не найден!')
    return found


def read(parm, row, task, time_rows_size=365):
    """Прочитать ряд из БД и привести к нужному размеру."""
    tid = db.territory_id(row['id_region'], row['id_district'])

    if 'hist' in parm:
        year = task.prediction_years[task.prediction_years_iteration]
        print('Чтение межгодового ряда из БД...', parm, tid, year)
        return db.mean_over_years(tid, parm, year, task.years_to_back, time_rows_size)

    year = task.prediction_years[task.prediction_years_iteration]
    print('Чтение ряда из БД...', parm, tid, year)
    return db.get_time_series_array(tid, str(parm), year, time_rows_size)

def vega_product(parm: str, ndvi: bool, historical: bool, model, task, row) -> str:
    """Функция возвращает имя продукта веги"""

    pd_file = model.pd_masks
    culture = task.culture
    veg_prod = 'None'

    if str(row['id_region']) != 'nan':
        reg_uid = str(int(row['id_region']))
    else:
        reg_uid = str(row['id_region'])

    if ndvi:    # Загрузка маски ndvi из файла в зависимости от культуры
        res_ndvi = pd_file[(pd_file['id_region'].astype(str).str.contains(str(reg_uid)))]

        culture = row['culture'].lower()

        print('res_ndvi', res_ndvi)
        print('culture', culture)
        print('res_ndvi[culture]', res_ndvi[culture])

        mask = res_ndvi[culture].values[0]

        veg_prod = str(mask)
        print('Загружена маска', mask)

    if ndvi and historical:
        flag = 'n_n'
        splited_mask = veg_prod.split(flag)
        veg_prod = splited_mask[0] + 'n_interannual_n' + splited_mask[-1]

    if not ndvi and historical:
        # print('parm###', parm)
        veg_prod = parm.replace('_historical', '').replace('mean_', 'mean_interannual_')

    if not ndvi and not historical:
        veg_prod = parm

    print('Продукт веги -', veg_prod)
    return veg_prod


def write_time_row_json(json, row, task, parm, directory='Temporary_information\\TimeRows\\'.replace('\\', os.sep), hist=False):
    """Записать загруженный с Vega ряд в БД (замена записи .json-файла)."""
    tid = db.territory_id(row['id_region'], row['id_district'])
    parm_clean = str(parm).replace('_historical', '')
    year = task.prediction_years[task.prediction_years_iteration]
    file_name = parm_clean + '_' + tid + '_' + str(year)   # логическое имя (для логов/совместимости)

    try:
        block = json['data']['1']
        xy = block.get('xy') or []
        labels = block.get('labels') or {}
    except (TypeError, KeyError):
        xy, labels = [], {}

    if xy:
        db.put_time_series(tid, parm_clean, year, xy, labels)
        print('Ряд записан в БД:', file_name)
    else:
        print('Полученный ряд пуст — в БД не записан:', file_name)

    return file_name, directory


def find_in_vega(parm, row, vega_api, model, task, make_file=True, hist=False):

    product_type = parm
    year = task.prediction_years[task.prediction_years_iteration]
    masks_pd = model.pd_masks

    if str(row['id_district']) != 'nan':
        district_uid = str(int(row['id_district']))
    else:
        district_uid = str(row['id_district'])

    if str(row['id_region']) != 'nan':
        reg_uid = str(int(row['id_region']))
    else:
        reg_uid = str(row['id_region'])

    json, success = None, False

    veg_prod = vega_product(parm, ('ndvi' in parm), ('historical' in parm), model, task, row)

    print('district_uid', district_uid, 'reg_uid', reg_uid)

    if str(district_uid) == 'nan':
        json, success = vega_api.region(veg_prod, year, reg_uid, task.ukey)
        file_name, directory = write_time_row_json(json, row, task, parm, hist=hist)

    else:
        json, success = vega_api.district(veg_prod, year, district_uid, task.ukey)
        file_name, directory = write_time_row_json(json, row, task, parm, hist=hist)

    return json, veg_prod, success, file_name, directory


def empty_json(json):
    """Проверка пустоты загруженного файла"""
    xy = json["data"]["1"]["xy"]
    if xy == []:
        return True
    else:
        return False


def get_not_hist_row(parm, row, vega_api, model, task,          # Функция возвращает массив временного ряда нужной длины
                     directory='Temporary_information\\TimeRows\\'.replace('\\', os.sep),
                     time_rows_size=365, hist=False):

    json, veg_prod, success, file_name, directory = find_in_vega(parm, row, vega_api, model, task, hist=hist)    # Запрос в веге + запись ряда в БД

    tid = db.territory_id(row['id_region'], row['id_district'])
    year = task.prediction_years[task.prediction_years_iteration]
    time_data = db.get_time_series_array(tid, str(parm).replace('_historical', ''),  # Чтение из БД и приведение к размеру
                                         year, time_rows_size)

    return time_data, file_name, directory


def save_like_json(self, array, file_name, first_data = "2000-01-01 00:00:00",
                   last_data = "2000-12-31 23:59:59", directory = 'Dist_info_actual\\Average_long_term_data\\'.replace('\\', os.sep)):
        # file_name = directory + str(product) + "_" + str(dist_id) + "_" + str(year) + ".json"
        print('file_name_at_seving', file_name)
        file_name = directory + file_name
        target_array = []
        step = 365/(len(array)-1)
        for i in range(len(array)):
            time_i = round(array[i], 4)
            value_i = str(round(i * step, 2))
            target_array.append([value_i, time_i])
        json_file = {"data": {"1": {"xy": target_array,"labels": {target_array[0][0]: first_data, target_array[-1][0]: last_data}}}}
        write_inf(json_file, file_name)

        pass


def get_hist_row(parm, row, vega_api, model, task, directory='Temporary_information\\TimeRows\\'.replace('\\', os.sep),
                 time_rows_size=365):

    print('task.years_to_back', task.years_to_back)
    print('task.prediction_years', task.prediction_years)

    for i in range(task.years_to_back): # Загружаем несколько рядов за прошлые годы

        print(' int(task.prediction_years[0])',  int(task.prediction_years[task.prediction_years_iteration]))
        year_i = int(task.prediction_years[task.prediction_years_iteration]) - i
        task_fiction = copy.deepcopy(task)
        task_fiction.prediction_years[task.prediction_years_iteration] = year_i
        parm_i = parm.replace('_historical', '')
        print('year', task_fiction.prediction_years[task.prediction_years_iteration])

        time_data, file_name_last, directory = get_not_hist_row(parm_i, row, vega_api, model, task_fiction)

        print('New iteration', file_name_last)

    print('Iterations end')

    tid = db.territory_id(row['id_region'], row['id_district'])
    base_year = int(task.prediction_years[task.prediction_years_iteration])
    file_name = str(parm).replace('_historical', '') + '_' + tid + '_interannual'

    print('Расчёт межгодового среднего из БД:', file_name)
    time_data = db.mean_over_years(tid, parm, base_year, task.years_to_back, time_rows_size)

    return time_data, file_name, directory



def find_information(parm, model, row, vega_api, task):     # Если нужны исторические данные, их расчитывет отдельная функция

    print('_parm_', parm)
    succes = False

    if not 'historical' in parm:  # Функция возвращает json по запросу, но многолетние данные следует рассчитать самостоятельно
        time_data, file_name, directory = get_not_hist_row(parm, row, vega_api, model, task)

        if time_data.shape[0] == 0:
            print('Данные за текущий год не найдены. Они будут заменены на многолетние...')
            time_data, file_name, directory = get_hist_row(parm, row, vega_api, model, task)
        else:
            succes = True

    elif 'historical' in parm:
        print('историч')
        print('year', task.prediction_years, 'parm', parm)
        time_data, file_name, directory = get_hist_row(parm, row, vega_api, model, task)

    if str(time_data) == 'None':
        print('Данные не найдены...')
    else:
        succes = True

    return time_data


def get_new_line_timerow(parm, task, row, model, vega_api, df_new_matrix):

    print('parm', parm, 'year for predict', task.prediction_years)
    file_success = chek_file(parm, row, task)  # Если файл уже есть на компе, он будет загружен
    print('file_success', file_success)

    print('file_success and task.prediction_years_secret<task.year_now', file_success and task.prediction_years_secret<task.year_now)

    # print(hohoh)

    if file_success and task.prediction_years_secret<task.year_now:
        time_row_parm = read(parm, row, task)

    else:  # Если нет, информация будет найдена в интернете
        vega_api.connect()
        time_row_parm = find_information(parm, model, row, vega_api, task)

    print('time_row_parm.shape', time_row_parm.shape)
    print('parm', parm)
    # print('df_new_matrix', df_new_matrix)

    if time_row_parm.shape[0] == 0:  # Если матрица пустая,
        df_new_matrix, success = df_new_matrix, False
    else:
        if df_new_matrix.shape[0] == 0:
            df_new_matrix = np.array([time_row_parm])
        else:
            df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])

    return df_new_matrix


def find_params_of_statistic(parms):
    keywords = ['mean_prod',
                'prod',
                'mean_productive',
                'productive',
                'dispersion',
                'disp',
                'trend',
                'mean_trend',
                'mean_trend']
    list_params_of_statistic = []

    # print('hoho')

    for parm in parms:
        find = False
        for key in keywords:
            print(parm, key)
            if parm.lower() == key.lower():
                find = True

        if find:
            list_params_of_statistic.append(parm)

    print('list_params_of_statistic', list_params_of_statistic)

    return list_params_of_statistic


def get_raw_years_and_prods(row, task, all_years=False, including_current=False) -> tuple:
    '''Функция возвращает два списка. Список продуктивностей и соответствующие годы'''

    list_of_years = []
    list_of_prods = []
    heads = row.index

    print('heads', heads)
    counter = 0

    for head in heads:
        try:
            year = int(head)
            if int(year) <= int(task.prediction_years[task.prediction_years_iteration]) or including_current:
                if counter <= task.avegage_last or all_years:
                    year_i = int(year)
                    prod_i = float(str(row[year]).replace(',', '.'))
                    list_of_years.append(year_i)
                    list_of_prods.append(prod_i)
                    counter += 1
        except:
            pass

    print('list_of_years', list_of_years, 'list_of_prods', list_of_prods)

    return list_of_years, list_of_prods


def get_trend_parms(x_points: [list], y_points: [list]) -> tuple:
    '''Функция принимает массивы точек, строит линейную регрессию по методу наименьших квадратов. Возвращает параметры регрессии,
    такие как коэффициент наклона прямой и свдиг'''

    x_train = np.array(x_points).reshape(-1, 1)
    y_train = np.array(y_points).reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    coeff = regressor.coef_.reshape(-1, 1)[0][0]
    shift = regressor.intercept_.reshape(-1, 1)[0][0]

    return coeff, shift #, singular


def get_stat_parm_by_row(row, parm_of_statistic, average_last, predict_year):
    years_init = list(row.index)
    # print('years', years)
    years_init.sort()
    years_init.reverse()
    # print('years.sort()', years_init)
    years_in_attention = []
    prod_in_attentoin = []
    counter = 0

    for year in years_init:
        if counter < average_last+1:
            prod_i = row[year]
            year_i = year
            if (str(prod_i) != 'nan') and (str(year_i) != 'nan') and (int(year_i) < int(predict_year)):
                prod_in_attentoin.append(float(str(prod_i).replace(',', '.')))
                years_in_attention.append(int(year_i))
                counter += 1

    years_in_attention.reverse()
    prod_in_attentoin.reverse()

    print('predict_year', predict_year)
    print('years_in_attention', years_in_attention)
    print('prod_in_attentoin', prod_in_attentoin)

    if 'prod' in parm_of_statistic:
        mean_productive = np.array(prod_in_attentoin).mean()
        parm_of_statistic_value = float(mean_productive)
    elif 'disp' in parm_of_statistic:
        mean_productive = np.array(prod_in_attentoin).mean()
        list_of_productive_np = np.array(prod_in_attentoin)
        prod_disperssion = np.var(list_of_productive_np)
        prod_disperssion_norm = prod_disperssion / mean_productive
        try:
            parm_of_statistic_value = float(prod_disperssion_norm)
        except:
            parm_of_statistic_value = 0.43
    elif 'trend' in parm_of_statistic:

        trend_i, shift_i = get_trend_parms(years_in_attention, prod_in_attentoin)
        try:
            parm_of_statistic_value = float(trend_i)
        except:
            parm_of_statistic_value = 0

    return parm_of_statistic_value


def get_train_example_stat(df_new_matrix,
                           year,
                           row,
                           parms,
                           parms_of_statistic,
                           average_last):
    """Функция принимает на вход матрицу с временными рядами, строку из задания и
    добавляет в эту матрицу статистические параметры, которые указанные в списке"""

    matrix = np.array([])

    i, j = 0, 0
    for parm in parms:
        if parm in parms_of_statistic:
            value_of_parm_i = get_stat_parm_by_row(row, parm, average_last, year)
            print('** year', year, 'parm', parm, value_of_parm_i)

            new_line = value_of_parm_i * np.ones(df_new_matrix.shape[1])
            j += 1
            # print('*')
            # print('new_line', new_line)
        else:
            new_line = df_new_matrix[i]
            i += 1
            # print('**')

        if matrix.shape[0] == 0:  # Набор нового тензора
            matrix = np.array([new_line])
        else:
            matrix = np.vstack([matrix, [new_line]])
            # print('***')

    # print(hoho)
    # print('matrix', matrix.shape)

    return matrix


def inf_loader(model, row, vega_api, task) -> Tuple[np.array, bool]:     # Итерирование по параметрам и формирование таблицы входных данных.

    success = True
    # model.activate()
    parms = model.parms
    df_new_matrix = np.array([])
    productive = False

    print('Параметры для модели', parms)
    print('')

    if 'LM' in model.model_name or 'EXM' in model.model_name:       # Если модель линейная
        parm = parms[0]     # Загружается параметр для регрессии
        years, _ = gel_years_list_with_prods(task, row)
        print('years', years, 'prediction_year', task.prediction_years[task.prediction_years_iteration])

        for year in years:
            if year != task.prediction_years[task.prediction_years_iteration]:
                task_fiction = copy.deepcopy(task)  # Создаем фиктивное задание
                task_fiction.prediction_years[task.prediction_years_iteration] = year
                df_new_matrix = get_new_line_timerow(parm, task_fiction, row, model, vega_api, df_new_matrix)

        df_new_matrix = get_new_line_timerow(parm, task, row, model, vega_api, df_new_matrix)

    else:
        year = task.prediction_years[task.prediction_years_iteration]
        list_of_parms = parms
        parms_of_statistic = find_params_of_statistic(parms)
        pd_file = model.pd_masks
        range_of_row = model.parm_file_info["time_borders"]
        directory = task.directory_time_rows # путь к папке, где хранятся файлы с временными рядами
        culture = row['culture'].lower()
        average_last = task.avegage_last

        try:
            id_district = int(row['id_district'])
        except:
            id_district = 'nan'
            print('В задании не указан id района! Будет загружена информация регионального уровня.')
        id_region = int(row['id_region'])

        print('New library is on!')
        print('parms', parms)
        print('parms_of_statistic', parms_of_statistic)
        print('year', year)
        print('list_of_parms', list_of_parms)
        print('pd_file', pd_file)
        print('range_of_row', range_of_row)
        print('directory', directory)
        print('culture', culture)
        print('average_last', average_last)
        print('id_district')
        print('id_region')

        df_new_matrix = get_train_example(year,   # 2015
                                    id_district,   # 1575
                                    id_region, # 64
                                    list_of_parms,  # Список продуктов веги
                                    parms_of_statistic,  # Параметры, отражающие статистику. Их программа игнорирует.
                                    pd_file,    # Файл с масками ndvi
                                    range_of_row,   # диапазон дней
                                    directory,  # путь к папке, где хранятся файлы с временными рядами
                                    culture,    # имя культуры
                                    average_last,
                                    task.ukey)

        years = get_raw_years_list(row, task)
        print('years', years)
        row_years_only = row[years]
        print('row_years_only', row_years_only )

        df_new_matrix = get_train_example_stat(df_new_matrix,
                                               year,
                                               row_years_only,
                                               parms,
                                               parms_of_statistic,
                                               average_last
                                               )

        get_example = df_new_matrix

        print('parms', parms)
        print('parms_of_statistic', parms_of_statistic)

        print('df_new_matrix.shape', df_new_matrix.shape)

        # print(hoho)

        # for i in range(get_example.shape[0]):
        #     print(list_of_parms[i], get_example[i].shape, get_example[i])
        #     plt.plot(get_example[i])
        #     plt.show()


    return df_new_matrix, success


def normalize_matrix(matrix: [np.array], normalization_parms: [list]) -> [np.array]:       # Примитивная нормировка
    print('Длина массива и листа нормализации', matrix.shape, len(normalization_parms))
    df_new_matrix = np.array([])
    for i in range(matrix.shape[0]):
        line = matrix[i]
        norm_factor = normalization_parms[i]
        time_row_parm = line / norm_factor

        if df_new_matrix.shape[0] == 0:
            df_new_matrix = np.array([time_row_parm])
        else:
            df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])

    return df_new_matrix


def matrix_cut(matrix, first_day, last_day, model):
    '''Функция обрезает временные ряды, оставляя только заданный диапазон'''

    df_new_matrix = np.array([])
    for i in range(matrix.shape[0]):
        line = matrix[i]
        time_row_parm = line[first_day:last_day]
        if df_new_matrix.shape[0] == 0:
            df_new_matrix = np.array([time_row_parm])
        else:
            df_new_matrix = np.vstack([df_new_matrix, [time_row_parm]])
    matrix_norm_cut = df_new_matrix

    return matrix_norm_cut


def get_raw_years_list(row, task):
    years = []
    heads = row.index
    # print('heads', heads)

    for head in heads:
        try:
            year = int(head)
            if int(year) <= int(task.prediction_years[task.prediction_years_iteration]):
                years.append(int(year))
        except:
            pass

        # print('years_hehe', years)
    # print('years_list', years)

    return years


def get_raw_prod_list(row, years_list, task):
    prods = []

    years_in_attention = []
    years_list.sort()
    year_now = task.prediction_years_secret

    i = 0
    for j in range(len(years_list)):
        year_i = years_list[-j-1]
        prod_i = row[int(year_i)]

        print('int(year_now) > int(year_i)', year_now, year_i)

        if int(year_now) > int(year_i):
            if i < task.avegage_last:
                if str(prod_i) != 'nan':
                    if int(year_now) != int(year_i):
                        years_in_attention.append(year_i)
                        prod_i = float(str(prod_i).replace(',', '.'))
                        prods.append(prod_i)
                        i += 1


    if prods == []:
        i = 0
        for j in range(len(years_list)):
            year = years_list[-j - 1]
            prod_i = row[int(year)]
            if str(prod_i) != 'nan':
                if i <= task.avegage_last:
                    years_in_attention.append(year)
                    prod_i = float(str(prod_i).replace(',', '.'))
                    prods.append(prod_i)
                    i += 1

    print('')
    print('years_in_attention', years_in_attention)
    print('years_list', years_list)
    print('prod_i', prod_i)
    print('prods', prods)
    print('')

    return prods


def find_average_prod_in_row(row, task):
    """Функция принимает на вход строку из задания и возвращает среднюю продуктивность.
    Либо продуктивность дана, либо она будет подсчитана по приведенной статистикой"""

    avg_colm = row['Average productivity']
    print('avg_colm', avg_colm)

    if True: #str(avg_colm) == 'nan':
        years_list = get_raw_years_list(row, task)
        list_actual_prods = get_raw_prod_list(row, years_list, task)
        print('list_actual_prods', list_actual_prods)
        average_prod = np.array(list_actual_prods).mean()

    return average_prod


def NN_CN_v1_winter_wheat_prepeaer(matrix_norm_cut, model, task, mean_productive):
    """Функция принимает на вход массив переменных, продуктивность, нормирует ее
    и возвращает входные данные в формате, пригодном для загрузки в модель"""       # От данной функции лучше отказаться

    matrix_norm_cut.shape

    ones_line = np.ones((matrix_norm_cut.shape[1]))
    prod_line = ones_line * mean_productive             # Добавлена строка со средней продуктивностью
    # print('prod_line', prod_line)

    prod_idx = model.parms.index("mean_productive")
    print('prod_idx', prod_idx)

    prod_line_norm = prod_line / model.normalization_parms[prod_idx]
    # print('prod_line_norm', prod_line_norm)

    matrix_norm_cut = np.vstack([matrix_norm_cut, [prod_line_norm]])

    print('matrix_norm_cut', matrix_norm_cut.shape)

    return matrix_norm_cut


def roughen_up_to_a_month(line, points):
  counter = 0
  target_array = []
  month_array = []

  for i in range(line.shape[0]):
    # print(counter, (points), i, line.shape[0])
    # print('bool', counter < (points), i != line.shape[0])

    if counter <= (points):
      value_i = line[i]
      month_array.append(value_i)
      counter += 1

    if counter > points or i == line.shape[0] - 1:
      value_i = line[i]               # ???
      month_array.append(value_i)

      month_value = np.array(month_array).mean()
      target_array.append(month_value)

      month_value = np.array(month_array).max()
      target_array.append(month_value)

      month_value = np.array(month_array).min()
      target_array.append(month_value)

      month_value = sum(month_array)
      target_array.append(month_value)

      month_array = []
      counter = 0

  return target_array


def RF_v1_winter_wheat_prepeaer(matrix_mod_new_size, model, task, mean_productive):
    """Функция принимает на вход массив переменных, от функции, подготавливающей данные для нейросети
     и возвращает входные данные в формате, пригодном для загрузки в модель случайного леса"""

    mean_month = 30

    print('matrix_mod_new_size', matrix_mod_new_size.shape)
    points = math.ceil(matrix_mod_new_size.shape[1]/mean_month)
    print('points', points)

    list_of_parms = model.parms

    new_line = []
    for j in range(matrix_mod_new_size.shape[0]):  # Перебор по параметрам
        parm_j = list_of_parms[j]

        if parm_j != 'mean_productive':
            line = matrix_mod_new_size[j]

            # print('line', line.shape)
            points_array = roughen_up_to_a_month(line, mean_month)

            # print('points_array.shape', len(points_array))

            for point_i in points_array:
                new_line.append(point_i)

            additional_values = [np.array(line).max(), np.array(line).min(), sum(line)]

        else:
            new_line.append(matrix_mod_new_size[j][0])

    new_line = np.array(new_line)
    print('new_line', new_line.shape)

    return new_line


def SLM_v1_prepeaer(matrix_norm_cut, model, task, mean_productive, row):

    maximums = []
    for i in range(matrix_norm_cut.shape[0]-1):
        print('matrix_norm_cuti', matrix_norm_cut[i])
        maximums.append(matrix_norm_cut[i].max())

    print('matrix_norm_cut.shape', matrix_norm_cut.shape)
    # idx = matrix_norm_cut.shape[0]

    print('matrix_norm_cut', matrix_norm_cut)
    target_arg = matrix_norm_cut[-1].max()

    years, prods = gel_years_list_with_prods(task, row)
    print('years', years, 'prods', prods, 'target_arg', target_arg)

    print('[maximums, prods, target_arg]', [maximums, prods, target_arg])

    return [maximums, prods, target_arg, years]


def inf_processor(matrix, model, task, row):  # Предварительная подготовка временного ряда

    print('')
    current_day = task.current_day  # Текущий день
    parms_list = model.parms

    first_day = model.first_day
    last_day = model.last_day
    normalization_parms = model.normalization_parms

    if first_day<0 or last_day<0:
        first_day += 365
        last_day +=365

    print('model.model_name', model.model_name)

    if ('LM' in model.model_name) or ('EXM' in model.model_name):
        matrix_norm = matrix
        print('normalize_matrix_no')

    else:
        print('normalize_matrix_go')
        print('')
        print('matrix before norm shape', matrix.shape)
        print('matrix before norm', matrix)
        matrix_norm = normalize_matrix(matrix, normalization_parms)
        print('')
        print('matrix_norm shape', matrix_norm.shape)
        print('matrix_norm', matrix_norm)
        print('')

    print('matrix_norm', matrix_norm, '\n')

    matrix_norm_cut = matrix_cut(matrix_norm, first_day, last_day, model)
    print('matrix_norm_cut', matrix_norm_cut, '\n')

    mean_productive = find_average_prod_in_row(row, task)

    print('model.model_name', model.model_name)
    print('mean_productive', mean_productive)

    if 'NN' in model.model_name: # model.model_name == 'NN_CN_v1_winter_wheat':
        # np_array_data_processed = NN_CN_v1_winter_wheat_prepeaer(matrix_norm_cut, model, task, mean_productive)
        np_array_data_processed = matrix_norm_cut
        pass

    if 'RF' in model.model_name: # model.model_name == 'RF_v1_winter_wheat':
        # np_array_data_processed = NN_CN_v1_winter_wheat_prepeaer(matrix_norm_cut, model, task, mean_productive)
        np_array_data_processed = matrix_norm_cut
        print('np_array_data_processed.shape', np_array_data_processed.shape)
        np_array_data_processed = RF_v1_winter_wheat_prepeaer(np_array_data_processed, model, task, mean_productive)
        pass

    if 'LM' in model.model_name or 'EXM' in model.model_name:
        np_array_data_processed = SLM_v1_prepeaer(matrix_norm_cut, model, task, mean_productive, row)
        pass

    success = True
    return np_array_data_processed, success, mean_productive


def predictor(np_array_data_processed, model_class, task, row):        # Запуск модели, сохранение прогноза

    model_name = model_class.model_name
    model = model_class.model
    mode_enter_shape = len(model_class.parms)

    if 'NN' in model_name:   #model_name == 'NN_CN_v1_winter_wheat':

        # np.savetxt("Enter_matrix.csv", np_array_data_processed)

        get_example = np_array_data_processed
        # for i in range(get_example.shape[0]):
        #     # print(list_of_parms[i], get_example[i].shape, get_example[i])
        #     plt.plot(get_example[i])
        #     plt.show()

        predicts = []
        for i in range(100):     # Цикл, в котором нейросеть делает несколько предсказаний.
            # Каждый раз они немного отличаются из-за модуля Дропаута
            # np.savetxt("Enter_data.csv", np_array_data_processed)
            enter = np_array_data_processed
            enter = enter.reshape(1, mode_enter_shape, -1)
            enter = torch.from_numpy(enter)
            enter = enter.to(torch.float32)
            prediction = model(enter)
            predicts.append(float(prediction[0]))

        anwers = np.array(predicts)

        dispersion = np.var(anwers)
        verdict = anwers.mean()
        verdict = Answer(verdict = verdict, inf_unconditional={'dispersion': dispersion})

    if 'RF' in model_name: # model_name == 'RF_v1_winter_wheat':
        print('np_array_data_processed', np_array_data_processed.shape)
        verdict = model.predict(np_array_data_processed.reshape(1, -1))
        verdict = Answer(verdict=verdict)

    if 'LM' in model_name or 'EXM' in model_name:
        # print('np_array_data_processed_LM', np_array_data_processed)
        verdict = model.predict(np_array_data_processed, task, row)

    print('verdict', verdict)
    success = True

    return verdict, success


def generate_probable_future_matrix(np_array_data,
                                    prediction_year,
                                    day_now,
                                    year_now,
                                    total_padding,
                                    list_of_parms,
                                    parms_of_statistic,
                                    first_day,
                                    last_day,
                                    row,
                                    model, task):
    """Функция перебирает строчки матрицы и исправляет их, если там не хватает данных"""
    print('list_of_parms', list_of_parms)
    print('длина списка параметров', len(list_of_parms))
    print('длина матрицы', np_array_data.shape)

    for i in range(np_array_data.shape[0]):
        print('range_i', i)

    print('np_array_data[18]', np_array_data[18])
    print('list_of_parms[18]', list_of_parms[18])

    # print(hahah)

    for i in range(np_array_data.shape[0]):
        print('range_i', i)
        init_line = np_array_data[i]
        print('list_of_parms', list_of_parms)
        parm_name = list_of_parms[i]
        if not('hist' in parm_name) and not(parm_name in parms_of_statistic):
            year = prediction_year - 1
            try:
                id_district = int(row['id_district'])
            except:
                id_district = 'nan'
            id_region = int(row['id_region'])
            parms = [parm_name + '_historical']
            statistic = []
            pd_file =  model.pd_masks
            range_of_row = model.parm_file_info["time_borders"]
            directory = task.directory_time_rows
            culture = row['culture'].lower()
            average_last = task.avegage_last

            historical_line = get_train_example(year,   # 2015
                                    id_district,   # 1575
                                    id_region, # 64
                                    parms,  # Список продуктов веги
                                    statistic,  # Параметры, отражающие статистику. Их программа игнорирует.
                                    pd_file,    # Файл с масками ndvi
                                    range_of_row,   # диапазон дней
                                    directory,  # путь к папке, где хранятся файлы с временными рядами
                                    culture,    # имя культуры
                                    average_last,
                                    ukey=task.ukey
                                                )

            historical_line = historical_line[0]

            # if first_day < 0 or last_day < 0:
            #     first_day += 365
            #     last_day += 365
            #     day_now += 365  # Текущий день должен быть указан в задании с учетом знака
            #     # (Знак минус означает, что день соответсвует прошлому году)

            print('day_now', day_now)
            print('init_line.shape', init_line.shape)
            print('historical_line.shape', historical_line.shape)

            part_one = init_line[0:day_now]
            part_second = historical_line[day_now:historical_line.shape[0]]
            new_line = np.concatenate((part_one, part_second), axis=0)

            np_array_data[i] = new_line

            # print(hehehe)

    return np_array_data


def inf_processor_good_hope(np_array_data, model, task, row):
    """Функция принимает текущую дату (момент, когда делается прогноз) и в случае, если еще не все данные для
    расчетов известны, программа дозаполняет их историческими данными"""

    prediction_year = int(task.prediction_years[task.prediction_years_iteration])
    day_now = int(task.current_day)
    year_now = int(task.current_year)
    total_padding = task.total_historical_puding
    list_of_parms = model.parms
    parms_of_statistic = find_params_of_statistic(list_of_parms)
    first_day = int(model.first_day)
    last_day = int(model.last_day)




    if ('NN' in model.model_name)  or ('RF' in model.model_name):
        get_example = np_array_data
        # for i in range(get_example.shape[0]):
        #     # print(list_of_parms[i], get_example[i].shape, get_example[i])
        #     plt.plot(get_example[i])
        #     plt.show()

        if year_now < prediction_year:
            day_now_from_beggining = day_now
        else:
            day_now_from_beggining = day_now+365

        print('hope - функция работает')
        print('prediction_year', prediction_year)
        print('day_now', day_now)
        print('year_now', year_now)
        print('total_padding', total_padding)
        print('list_of_parms', list_of_parms)
        print('first_day', first_day)
        print('last_day', last_day)
        print('day_now_from_beggining', day_now_from_beggining)
        print('last_day - first_day', last_day - first_day)

        # print(hehehe)

        if prediction_year-task.year_now >= 1:
            delta_year_now = 2
        else:
            delta_year_now = 1



        """Условие начала исправления матрицы"""
        print('hope', total_padding or ((year_now <= prediction_year) and
                             (day_now_from_beggining < (last_day - first_day))))


        if total_padding or ((year_now <= prediction_year) and
                             (day_now_from_beggining < (last_day - first_day))):
            print('Начала процедура дозаполнения данных для заблаговременного прогноза!')

            np_array_data = generate_probable_future_matrix(np_array_data,
                                                            prediction_year,
                                                            day_now_from_beggining,
                                                            year_now-delta_year_now,
                                                            total_padding,
                                                            list_of_parms,
                                                            parms_of_statistic,
                                                            first_day,
                                                            last_day, row,
                                                            model, task)

        np_array_data_processed = np_array_data
        pass

    else:
        np_array_data_processed = np_array_data

    get_example = np_array_data
    # for i in range(get_example.shape[0]):
    #     # print(list_of_parms[i], get_example[i].shape, get_example[i])
    #     plt.plot(get_example[i])
    #     plt.show()

    # print(heh)

    success = True
    return np_array_data_processed, success


def get_one_verdict(model, row, vega_api, task) -> Tuple[Answer, bool]:
    '''Функция получает модель, строку из документа с заданием, загрузчик данных и задание, а возвращает готовый ответ'''

    success = True
    if success:  # Проверка успеха предыдущего действия (Данное место нужно переписать...)
        np_array_data, success = inf_loader(model, row, vega_api, task)  # Скачиваем информацию
    else:
        verdict = 'unexpected error'  # Если случится ошибка, система вернет комментарий ошибки вместо вердикта

    if success:
        print('Before Hope')
        np_array_data_processed, success = inf_processor_good_hope(np_array_data, model, task, row)
        """Данная функция должна дозаполнить отсутствующие временнные ряды средними историческими данными
         в том случае, если прогноз делается сильно заблаговременно"""
    else:
        verdict = 'inf loader error'

    if success:
        np_array_data_processed, success, average = inf_processor(np_array_data, model, task, row)  # Производим предварительную подготовку
    else:
        verdict = 'inf loader error'

    if success:
        # print('np_array_data_processed', np_array_data_processed, model.model_name)
        verdict, success = predictor(np_array_data_processed, model, task, row)  # Запуск модели, сохранение прогноза
    else:
        verdict = 'inf processor error'

    if success == False:
        verdict = 'No data'

    return verdict, success


def gel_years_list_with_prods(task, row, all_points=False):
    '''Функция возвращает n последних урожайностей и соответсвующих им лет, либо все возможные урожайнсти и годы, если all_points=True'''
    years_list = get_raw_years_list(row, task)
    years_list.sort(reverse=True)

    years_available = []
    prods_available = []

    print('row', row)
    print('task.avegage_last', task.avegage_last)
    print('all_points', all_points)

    if all_points:
        for i in range(len(years_list)):
            year_i = int(years_list[i])
            prod_i = float(str(row[int(year_i)]).replace(',', '.'))
            if str(prod_i) != 'nan':
                if year_i <= int(task.prediction_years[task.prediction_years_iteration]):
                    years_available.append(year_i)
                    prods_available.append(prod_i)
    else:
        counter = 0
        for i in range(len(years_list)):
            year_i = int(years_list[i])
            prod_i = float(str(row[int(year_i)]).replace(',', '.'))
            if counter <= task.avegage_last:
                if str(prod_i) != 'nan':
                    years_available.append(year_i)
                    prods_available.append(prod_i)
                    counter += 1

    if not all_points:
        years_available_short = []
        prods_available_short = []

        counter = 0
        for i in range(len(years_available)):

            if int(years_available[i]) < int(task.prediction_years[task.prediction_years_iteration]):
                if counter <= task.avegage_last:
                    years_available_short.append(years_available[i])
                    prods_available_short.append(prods_available[i])
                    counter += 1

        years_available = years_available_short
        prods_available = prods_available_short

    return years_available, prods_available


def get_correct_factor(prods_available, predicts_for_past):
    prods_available = np.array(prods_available)
    predicts_for_past = np.array(predicts_for_past)

    difference = prods_available - predicts_for_past

    correct_factor = difference.mean()
    return correct_factor


def exponental_filter(diff: np.array) -> np.array:
    print('diff', diff)
    return diff


def verdict_normalization(prods_available, predicts_for_past, verdict_bc):      # Происходит нормировка вердикта.
    predicts = np.append(np.array(predicts_for_past), np.array(verdict_bc))
    real = np.array(prods_available)

    print('predicts', predicts)
    print('predicts_for_past', predicts_for_past)
    print('real', real)

    diff = (real - np.array(predicts_for_past))

    diff_foltred = exponental_filter(diff)          # Функция не дописана.
    diff_foltred = diff_foltred.mean()

    print('diff_foltred', diff_foltred)
    preds_norm_shift = predicts + diff_foltred

    print('preds_norm_shift', preds_norm_shift)

    predicts_for_past_corrected = predicts + diff_foltred

    predicts_for_past_corrected = predicts_for_past_corrected[0:predicts_for_past_corrected.shape[0]-1]


    return preds_norm_shift[-1], predicts_for_past_corrected


def verdict_normalization_v2(prods_available, predicts_for_past, verdict_bc):  # Происходит нормировка вердикта.

    prods_available = np.array(prods_available).reshape(-1,1)
    predicts_for_past = np.array(predicts_for_past).reshape(-1,1)
    verdict_bc = np.array(verdict_bc).reshape(-1,1)

    # plt.plot(prods_available, predicts_for_past)  # черные ромбы
    # plt.show()

    regressor = LinearRegression()      # Инициализируем модель
    # regressor = RandomForestRegressor()
    regressor.fit(predicts_for_past, prods_available)    # Обучаем модель

    verdict_norm = float(regressor.predict(verdict_bc))
    predicts_for_past_corrected = regressor.predict(predicts_for_past)

    return verdict_norm, predicts_for_past_corrected


def predict_and_correction_real(model, row, vega_api, task):
    """Функция делает один прогноз. Функция может просто сделать прогноз, либо сделать прогноз для нескольких прошлых лет, для
    которых известны продуктивности, подсчитать систематическую ошибку и вычесть ее для прогноза на заданый год. Кроме того, длагодаря прогнозам на
    прошлые годы, программа оценивает ско и прочие метрики качества для данного района/региона"""

    years_available, prods_available = gel_years_list_with_prods(task, row)     # Получим список продуктивностей в соответствующие годы

    print('years_available_1', years_available)
    print('prods_available_1', prods_available)

    predicts_for_past = []

    if task.correction and (not('M' in model.model_name)):     # Для линейных моделей коррекция не предполагается

        task.prediction_years_secret = task.prediction_years[task.prediction_years_iteration]
        verdict_bc, succses = get_one_verdict(model, row, vega_api, task)
        average = find_average_prod_in_row(row, task)

        years_for_correct_avalible = min(task.years_for_correct, len(years_available))
        print('task.years_for_correct, len(years_available)', task.years_for_correct, len(years_available))
        print('year', task.prediction_years_secret)
        print('years_for_correct_avalible', years_for_correct_avalible)

        print('Год предсказания', task.prediction_years[task.prediction_years_iteration])

        # for i in range(task.years_for_correct):       # Делаем предсказания на N прошлых доступных лет
        for i in range(years_for_correct_avalible):  # Делаем предсказания на N прошлых доступных лет

            print('counter', i)
            year_i = years_available[i]                 # Получаем год
            task_fiction = copy.deepcopy(task)              # Создаем фиктивное задание
            task_fiction.prediction_years[task.prediction_years_iteration] = year_i
            task_fiction.prediction_years_secret = task.prediction_years[task.prediction_years_iteration]
            verdict_i, succses = get_one_verdict(model, row, vega_api, task_fiction)

            try:
                verdict_i.verdict.astype(float)
                print('astype(float) is done!')
            except:
                print('verdict_i_type', type(verdict_i.verdict))

            predicts_for_past.append(float(verdict_i.verdict))
            print('verdict_i ', verdict_i)


        prods_available = prods_available[0: len(predicts_for_past)]

        print('prods_available', prods_available)
        print('predicts_for_past', predicts_for_past)
        print('years_available', years_available[0: len(predicts_for_past)])

        if verdict_bc.verdict != 'inf processor error':
            verdict_corrected = copy.deepcopy(verdict_bc)
            if predicts_for_past != []:
                verdict_corrected.verdict, predicts_for_past_corrected = verdict_normalization_v2(prods_available, predicts_for_past, copy.deepcopy(verdict_bc.verdict))
                verdict_corrected = print_metrics(prods_available, predicts_for_past_corrected, None, verdict_corrected, task, years_available, predicts_for_past)
        else:
            verdict_corrected = Answer(verdict='')
            average = ''

        print('verdict_bc', verdict_bc.verdict)
        print('average', average)
        print('verdict_corrected', verdict_corrected.verdict)

    else:
        task.prediction_years_secret = task.prediction_years[task.prediction_years_iteration]
        verdict_bc, succses = get_one_verdict(model, row, vega_api, task)
        verdict_corrected, average = verdict_bc, find_average_prod_in_row(row, task)


    # print(hohoho)

    return verdict_corrected, average


def predict_array(row, task) -> Tuple[list, float]:   # Получаем строку и задание. Каждая модель из списка должна получить данные и отработать.
    """Функция принимает на вход задание, pd строку метаданных. Из задания принимает список оберток моделей.
    Те в свою очередь хранят список входных данных, которые нужно загрузить из интернета, и инструменты для
    предварительной обработки. Данные скачиваются, обрабатываются, пропускаются через модель и возвращается
    список прогнозов"""

    model_list = task.models_list
    column_models = task.column_models

    verdicts = []
    success = True

    vv = []

    vega_api = VegaAPI()   # Инициализируем экземпляр класса, с где будет храниться сеанс подключения к сайту для загрузки рассчетных данных

    for i in range(len(column_models)):         # Итерации по моделям
        years_list_in_row, _ = gel_years_list_with_prods(task, row, all_points=True)    # Получим список годов, для которых у нас имеются продуктивности в обрабатываемой строчке документа
        print('years_list_in_row', years_list_in_row)
        print('len(years_list_in_row)', len(years_list_in_row))
        print('column_models', column_models)
        task.prediction_years_iteration = i

        if len(years_list_in_row) != 0:
            print('Длины ', len(model_list), len(column_models))
            print(model_list, column_models)
            model, model_name = model_list[i], column_models[i]
            model.activate()    # Активируем модель (произойдут предварительные приготовления)

            model = specify_the_parameters(row, model)

            year = task.prediction_years[task.prediction_years_iteration]     # Загрузим год, для которого делается предсказание
            model_name_in_cap = model_name + '_predict_' + year
            print('model_name_in_cap', model_name_in_cap, 'years', year)

            if str(row['id_region']) == 'nan' and str(row['id_district']) == 'nan':     # Если строка полностью пустая, пропускаем ее
                verdicts, average = None, None    # Заполним целевые переменные None. Они будут возвращены в таком виде
                print('Пустрая строка в задании!')

            else:
                if str(row[model_name_in_cap]) == 'nan':     # Если в данной строке столбец с ответом все еще не заполнен, запускается расчет
                    try:
                        verdict, average = predict_and_correction_real(model, row, vega_api, task)  # Запускается расчет
                    except:
                        print('При попытке произвести рассчет произошла ошибка!')
                        verdict = Answer(verdict='')
                        average = ''
                else:
                    verdict = Answer(verdict=float(row[model_name_in_cap]))
                    print('Ведрикт уже записан!', verdict)
                    average = row['Average productivity']

        else:
            verdict = Answer(verdict='')
            average = ''
            year = task.prediction_years[task.prediction_years_iteration]

        print('* year', year, 'vardict', verdict.verdict)

        vv.append(verdict.verdict)

        verdicts.append(verdict)

    return verdicts, average, vv


def record_additional_information(df_destination, index, answer):

    unconditional_dict = answer.inf_unconditional
    dict_keys_list = list(unconditional_dict.keys())

    print('dict_keys_list', dict_keys_list)

    for i in range(len(dict_keys_list)):
        key_i = dict_keys_list[i]
        print('unconditional_dict[key]', key_i, unconditional_dict[key_i])

        print('index, str(key_i)', index, str(key_i))

        df_destination.loc[index, str(key_i)] = unconditional_dict[str(key_i)]

    additional_dict = answer.additional_inf
    dict_keys_list = list(additional_dict.keys())

    for i in range(len(dict_keys_list)):
        key = dict_keys_list[-i-1]
        df_destination.loc[index, str(key)] = additional_dict[key]


    return df_destination

def specify_the_parameters(row, model):
    '''In this function, specify the culture in this line of the task and set the desired ndvi mask,
    in accordance with the area where the culture grows'''
    print('df_destination', row)
    print("df_destination['culture']", row['culture'])

    culture = row['culture'].lower()
    id_region = row['id_region']

    print('culture', culture, 'id_region', id_region)

    pd_file = model.pd_masks
    parms = model.parms

    print('parms_in_model_before', parms)

    for i in range(len(parms)):
        parm_i = parms[i]
        if 'ndvi' in parm_i:
            res_ndvi = pd_file[(pd_file['id_region'].astype(str).str.contains(str(id_region)))]

            print('res_ndvi', res_ndvi)
            mask_i = res_ndvi[culture].values[0]

            if 'histor' in parm_i:
                mask_i = mask_i + '_historical'
            else:
                parms[i] = mask_i

    print('parms_in_model_after', parms)
    model.parms = parms

    # print(heheh)

    return model

def _verdict_to_float(value):
    """Привести вердикт модели (float / np.array / '' / None) к float либо None."""
    if value is None or isinstance(value, str):
        return None
    try:
        arr = np.asarray(value, dtype=float).flatten()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    result = float(arr[0])
    return None if np.isnan(result) else result


def build_row(tid, task):
    """Синтез строки задания из БД: метаданные территории + годовые урожайности.

    Воспроизводит форму прежней строки широкого Excel (с целочисленными
    ключами-годами и пустыми столбцами прогнозов), чтобы расчётные функции
    (``inf_loader``, ``gel_years_list_with_prods`` и др.) работали без изменений.
    """
    rec = db.get_territory(tid) or {}
    culture = task.culture_by_territory.get(tid, '')

    data = {
        'id_country': rec.get('id_country', ''),
        'region': rec.get('region', ''),
        'district': rec.get('district', ''),
        'culture': culture,
        'id_region': db._norm_id(rec.get('id_region')),
        'id_district': db._norm_id(rec.get('id_district')),
        'Average productivity': np.nan,
    }
    for year, value in db.get_yields(tid, culture).items():
        data[int(year)] = value
    for model_name, year in zip(task.column_models, task.prediction_years):
        data[f'{model_name}_predict_{year}'] = np.nan

    return pd.Series(data)


def download_and_prediction(task):  # Перебор территорий задания: расчёт моделей и запись прогнозов в БД
    print('Модели×годы:', task.column_predict)

    with alive_bar(len(task.territory_ids), force_tty=True) as bar:
        for tid in task.territory_ids:  # Итерации по территориям задания
            row = build_row(tid, task)
            culture = task.culture_by_territory.get(tid, '')

            row_predict, average, vv = predict_array(row, task)     # Расчёт всех моделей по синтезированной строке
            print('average', average)

            for i in range(len(row_predict)):
                answer = row_predict[i]
                model_name = task.column_models[i]
                predict_year = int(task.prediction_years[i])

                dispersion = None
                if getattr(answer, 'inf_unconditional', None):
                    dispersion = answer.inf_unconditional.get('dispersion')

                db.upsert_prediction(tid, culture, model_name, predict_year,
                                     value=_verdict_to_float(answer.verdict),
                                     dispersion=_verdict_to_float(dispersion),
                                     average_productivity=_verdict_to_float(average))

            print('vv', vv)
            bar()
    return True

def ukey_Loader(task,
                directory="Service_files\\Common_information\\".replace('\\', os.sep),
                file='Your_key.xlsx',
                key = None):
    'Откроем файл, запишем номер ukey в переменную экземпляра класса task'

    if key == None:
        file_name = directory + file
        file_key = pd.read_excel(file_name)
        # print('pd', file_key)
        ukey = str(file_key["you_key"].loc[0])
        print('ukey', ukey)
        task.ukey = ukey

    else:
        task.ukey = key
    # print(heheh)

    return task


def main():
    print('Добро пожаловать в Nerual Rabbit!', '\n')
    task = fast_user_survey()       # Загрузка задания из тонкой формы (территории, культура, модель, год)

    print('Получено задание: ', task.task_file, '\n')
    task = model_Loader(task)  # Загрузка выбранных моделей
    task = ukey_Loader(task)   # Ключ для запросов данных у онлайн-сервиса Vega
    succes = download_and_prediction(task)  # Расчёт прогнозов и запись в БД (predictions)

    import report             # Сборка отчёта из БД
    out_path = 'Report_files' + os.sep + 'report_' + os.path.splitext(task.task_file)[0] + '.xlsx'
    report.build_report(out_path,
                        territory_ids=task.territory_ids,
                        model_year_pairs=[(m, int(y)) for m, y in zip(task.column_models, task.prediction_years)],
                        culture_by_territory=task.culture_by_territory)

    print('Выполнение задания успешно завершено!', '\n')


if __name__ == '__main__':
    # cProfile.run('main()')
    main()


