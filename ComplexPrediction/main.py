import os
import json
from Parser_FedStat import parser_fedstat
from Parser_Vega import parser_vega_region
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statistics import mean
from DEFCOR import cor
import numpy as np
import math
import openpyxl
from openpyxl import load_workbook
import xlsxwriter
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import docx
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

# from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
# from docx.shared import Mm
# from docx import Document

from docx.shared import Inches, Pt



def write_inf(data, file_name):
    data = json.dumps(data)
    data = json.loads(str(data))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
         return json.load(file)

def NDVI_max(NDVI_list):
    # "NDVI_Processing_Options": {"Left_border": "0", "Right_border": "250", "Find": "Max", "comment": "# Max/Min/Mean"}

    x = []
    y = []
    Candidates = []

    INITIAL_DATA = read_inf('INITIAL_DATA.json')  # Загрузка датасета с указаниями к расчетам

    Left_border = int(INITIAL_DATA["NDVI_Processing_Options"]["Left_border"])
    Right_border = int(INITIAL_DATA["NDVI_Processing_Options"]["Right_border"])
    Find = str(INITIAL_DATA["NDVI_Processing_Options"]["Find"])

    print("Определение максимумов...", NDVI_list)

    for i in range(len(NDVI_list)):
        x.append(NDVI_list[i][0])
        y.append(NDVI_list[i][1])
        if (int(NDVI_list[i][0]) < Right_border) and (int(NDVI_list[i][0]) > Left_border):
            Candidates.append(NDVI_list[i][1])

    if Find == "Max":
        NDVI_point = max(Candidates)
    elif Find == "Min":
        NDVI_point = min(Candidates)
    elif Find == "Mean":
        NDVI_point = mean(Candidates)
    else:
        print("Ошибка! Некорректная формульровка инструкции по обработке графиков NDVI ")

    # print(NDVI_MAX)
    # plt.plot(x, y, c="r")     # Чтобы вывести миллион графиков при сборе информации
    # plt.show()
    return NDVI_point


def NDVI_max_for_dataset(DATASET, Reg_list, years, Analyzed_year): # В json файл будет добавлен максимальное значение NDVI в первой половине сезона
    for i in range(len(Reg_list)):
        Reg_name = Reg_list[i]
        try:
            for j in range(years+1):
                Year_j = str(int(Analyzed_year) - j)
                NDVI = DATASET[Reg_name][Year_j]["NDVI"]
                NDVI_m = NDVI_max(NDVI)
                DATASET[Reg_name][Year_j].setdefault('NDVI_max', NDVI_m)
        except BaseException:
            print('Нет данных для региона: '+ Reg_name + ', год: ' + Year_j)

    return DATASET


def Add_info_from_reference_fields(File_name): # File_name example: Пензенская область
    # File_name = "Пензенская область"

    INITIAL_DATA = read_inf('INITIAL_DATA.json')  # Загрузка датасета с указаниями к расчетам

    Line = int(INITIAL_DATA['Data_From_Fields_Format']['Line'])
    NDVI_column = int(INITIAL_DATA['Data_From_Fields_Format']['NDVI_column'])
    Productive_column = int(INITIAL_DATA['Data_From_Fields_Format']['Productive_column'])

    book = openpyxl.open( "Additional information\\" + File_name + ".xlsx", read_only = True)
    sheet = book.active

    NDVI_max_list = []
    Productive_list = []

    for row in range(Line, sheet.max_row+1):
        NDVI = sheet[row][NDVI_column].value
        Productive = sheet[row][Productive_column].value
        # print()
        # print(NDVI, Productive)
        # print(type(NDVI), type(Productive))
        # print(NDVI != None, Productive != None)
        # print("\(★ω★)/")

        if (NDVI != None) & (Productive != None):
            # print(NDVI, Productive)
            NDVI_max_list.append(NDVI)
            Productive_list.append(Productive)

    # print(NDVI_max_list)
    # print(Productive_list)


    class AddInfReport:
        ndvi = NDVI_max_list
        prod = Productive_list

    return AddInfReport


def regression_array(DATASET, Reg_list):            # Функция создает массивы для будущего построения регрессий
    Years_list = list(DATASET[Reg_list[0]].keys())      # Здесь стоит прописать более совершенную функцию определения временного диапазона исследования
    # print(DATASET)

    for i in range(len(Reg_list)):
        regression = []
        for j in range(len(Years_list)):
            x = []
            try:
                x.append(DATASET[Reg_list[i]][Years_list[j]]['NDVI_max'])
                x.append(DATASET[Reg_list[i]][Years_list[j]]['Productive'])
                x.append(Years_list[j])
                regression.append(x)
            except BaseException:
                print('Нет данных для региона: '+ Reg_list[i] + ', год: ' + Years_list[j])

        try:
            Add_inform = Add_info_from_reference_fields(Reg_list[i])

            for j in range(len(Add_inform.ndvi)):
                x = []
                a = Add_inform.ndvi[j]
                b = Add_inform.prod[j]
                x.append(a)
                x.append(b)
                regression.append(x)
            print('Найдена дополнительная информация! ' + Reg_list[i] + '. Данные добавлены в массивы для построения регрессии. ')
        except BaseException:
            print('Для региона - ' + Reg_list[i] + ' дополнительной информации не найдено...')

        # print("Выборка ", regression)

        DATASET[Reg_list[i]].setdefault('Regression_list', regression)
        DATASET[Reg_list[i]]['Regression_list'] = regression
        # print(DATASET[Reg_list[i]]['Regression_list'])                      # Что тут происходит....?

    return DATASET

def Alternative_input_EMISS(Year, Serch_Region): # File_name example: Пензенская область

    INITIAL_DATA = read_inf('INITIAL_DATA.json')  # Загрузка датасета с указаниями к расчетам

    Sheet = INITIAL_DATA['Additional_Information_File_Format']['Page_name']     # Параметры формата файла
    Line = int(INITIAL_DATA['Additional_Information_File_Format']['Line'])
    Region_name_column = int(INITIAL_DATA['Additional_Information_File_Format']['Region_name_column'])
    Prod_column = int(INITIAL_DATA['Additional_Information_File_Format']['Prod_column'])
    File_name = INITIAL_DATA['Additional_Information_File_Format']['File_name']

    book = openpyxl.open( "Additional information\\" + File_name + Year + ".xlsx", read_only = True)
    sheet = book[Sheet]  # 120_1(1104) # 120_1(1010400)
    for row in range(Line, sheet.max_row+1):
        Region = sheet[row][Region_name_column].value
        Productive = sheet[row][Prod_column].value
        if (Region == Serch_Region):
            Productive_search = Productive
            break
    class AddInfReport:
        reg = Region
        prod = Productive_search
    return AddInfReport

def collect_dataset(Reg_list, years, Analyzed_year, Culture, product_type, Farm_type, Login, Password, Vega, FS, AI, DATASET):
    # global Login, Password
    dataset = {}
    for i in range(len(Reg_list)): # Перебор по регионам
        Reg_name = Reg_list[i]  # Инициализация переменной региона
        for j in range(int(years)): # Перебор по годам
            year = str(int(Analyzed_year) - j)  # Инициализация переменной года

            if not(Reg_name in DATASET):            # Если ключ региона в БД отсутсвует,
                DATASET.setdefault(Reg_name, {})      # он создается.

            if not(year in DATASET[Reg_name]):            # Если ключ года для выбранного региона в БД отсутсвует,
                DATASET[Reg_name].setdefault(year, {})    # он создается.

            if not('NDVI' in DATASET[Reg_name][year]):        # Если в БД нет данных под ключем NDVI для запрашиваемого региона и года, производится попытка загрузить эти данные из интернета
                print("Поиск данных NDVI: " + Reg_name + ', ' + year + ' год в интернете...')
                try:
                    if Vega:
                        DATA_VEGA = parser_vega_region(Login, Password, Reg_name, year, product_type)
                        NDVI = DATA_VEGA["data"]["1"]["xy"]
                        DATASET[Reg_name][year].setdefault("NDVI", NDVI)
                        DATASET[Reg_name][year]["NDVI"] = NDVI

                except BaseException:
                    print('Не удалось загрузить данные из интернета: NDVI, '+ Reg_name + ', '+ year)   # Сообщение выводится, если при запросе данных из интернета случилась ошибка
            else:
                print("Данные NDVI: " + Reg_name + ', ' + year + ' найдены в БД программы! ')

            if not('Productive' in DATASET[Reg_name][year]):    # Если в БД нет данных под ключем Productive для запрашиваемого региона и года, производится попытка загрузить эти данные из интернета
                print("Поиск данных Productive: " + Reg_name + ', ' + year + ' год в дополнительных файлах...')
                try:
                    if FS:
                                                # print('Попытка найти данные в файлах дополнительной информации...')
                        DATA_FS = Alternative_input_EMISS(year, Reg_name)
                        # print(DATA_FS.prod)
                        DATA_FS = DATA_FS.prod
                        # Productive = float(DATA_FS.replace(',', '.'))
                        DATASET[Reg_name][year].setdefault("Productive", DATA_FS)
                        DATASET[Reg_name][year]["Productive"] = DATA_FS
                        print('Данные из файлы успешно записаны в БД! ')

                except BaseException:
                    print('Поиск данных в интернете: Productive, '+ Reg_name + ', '+ year)
                    try:
                        if FS:
                            print(year, Reg_name, Culture, Farm_type)
                            DATA_FS = parser_fedstat(year, Reg_name, Culture, Farm_type)
                            Productive = float(DATA_FS.replace(',', '.'))
                            DATASET[Reg_name][year].setdefault("Productive", Productive)
                            DATASET[Reg_name][year]["Productive"] = Productive


                    except:
                        print('Информация не найдена в файлах дополнительной информации и интернете: Productive, ' + Reg_name + ', ' + year)

            else:
                print("Данные Productive: " + Reg_name + ', ' + year + ' найдены в БД программы! ')



    return DATASET


def Find_mae_and_max(X,Y, func): # x - массив достоверных результатов модели, y - аргументы результатов
    if len(X) != len(Y):
        print("Массивы для проверки модели имеют разную длину")
    else:
        er = []
        er_std = []
        for i in range(len(X)):
            x = Y[i]
            er.append(abs(X[i] - eval(func)))
            er_std.append(X[i] - eval(func))
        # print(er)

        Delta_list_np = np.array(er_std)
        b = Delta_list_np
        df = pd.DataFrame(b)

        class erReport:
            mae = sum(er) / len(X)
            max = max(er)
            std = np.std( er_std )
            RETURN = {'count': df.describe()[0]['count'], 'mean': df.describe()[0]['mean'],
                      'std': df.describe()[0]['std'], 'min': df.describe()[0]['min'],
                      '25%': df.describe()[0]['25%'], '50%': df.describe()[0]['50%'],
                      '75%': df.describe()[0]['75%'], 'max': df.describe()[0]['max']}

    return erReport




def regression_function(DATASET, Reg_list):

    for i in range(len(Reg_list)):
        x = [] #NDVI
        y = [] #Productive
        z = [] #Year
        Regression = DATASET[Reg_list[i]]['Regression_list']
        for j in range(len(Regression)):    # Подготовка массивов для построения регрессии
            x.append(Regression[j][0])
            y.append(Regression[j][1])
            # z.append(Regression[j][2])

        print('Массивы для регрессии')
        print(x) #NDVI
        print(y) #Productive
        print() #Year

        print(' ', Reg_list[i])
        correlation_coefficient = cor(y, x)         # Вычисление коэфф кореляции между NDVI и Урожайностью
        print('Коэффициент корреляции: ', correlation_coefficient)

        print("Размер выборки: ",len(Regression))

        X = sm.add_constant(x)                  # Вычисление линейной регрессии
        model = sm.OLS(y, X).fit()
        function1 = str(model.params[0]) + ' + (' + str(model.params[1]) + ') * x'  # Линейная функция
        function1_for_report = str(round(model.params[0], 2)) + '+(' + str(round(model.params[1], 2)) + ')*x'  # Линейная функция


        print('Полином 1-й степени: ', function1)
        # print(model.summary())          # Можешь попробовать прикрепить этот отчет к документу целеком
        print()
        print('R-squared: ', model.rsquared)
        mae = Find_mae_and_max(y, x, function1)
        print('MAE: ', mae.mae)
        print('Max error: ', mae.max)
        print()

        DATASET[Reg_list[i]].setdefault('Cor', correlation_coefficient)
        DATASET[Reg_list[i]].setdefault('Linear_approximation', {})
        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('Regression_Function', function1)  # Запись полученных данных в БД
        DATASET[Reg_list[i]]['Linear_approximation']['Regression_Function'] = function1

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('Regression_function_for_report', function1_for_report)  # Запись полученных данных в БД
        DATASET[Reg_list[i]]['Linear_approximation']['Regression_function_for_report'] = function1_for_report

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('R_squared', model.rsquared)
        DATASET[Reg_list[i]]['Linear_approximation']['R_squared'] = model.rsquared
        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('MAE', mae.mae)
        DATASET[Reg_list[i]]['Linear_approximation']['MAE'] = mae.mae
        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('Max_error', mae.max)
        DATASET[Reg_list[i]]['Linear_approximation']['Max_error'] = mae.max

        # class erReport:

        #     RETURN = {'count': df.describe()[0]['count'], 'mean': df.describe()[0]['mean'],
        #               'std': df.describe()[0]['std'], 'min': df.describe()[0]['min'],
        #               '25%': df.describe()[0]['25%'], '50%': df.describe()[0]['50%'],
        #               '75%': df.describe()[0]['75%'], 'max': df.describe()[0]['max']}

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('STD', mae.std)
        DATASET[Reg_list[i]]['Linear_approximation']['STD'] = mae.max

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('min', mae.RETURN['min'])
        DATASET[Reg_list[i]]['Linear_approximation']['min'] = mae.RETURN['min']

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('25%', mae.RETURN['25%'])
        DATASET[Reg_list[i]]['Linear_approximation']['25%'] = mae.RETURN['25%']

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('50%', mae.RETURN['50%'])
        DATASET[Reg_list[i]]['Linear_approximation']['50%'] = mae.RETURN['50%']

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('75%', mae.RETURN['75%'])
        DATASET[Reg_list[i]]['Linear_approximation']['75%'] = mae.RETURN['75%']

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('mim', mae.RETURN['max'])
        DATASET[Reg_list[i]]['Linear_approximation']['max'] = mae.RETURN['max']

        DATASET[Reg_list[i]]['Linear_approximation'].setdefault('IQR', (mae.RETURN['75%'] - mae.RETURN['25%']))
        DATASET[Reg_list[i]]['Linear_approximation']['IQR'] = (mae.RETURN['75%'] - mae.RETURN['25%'])


        Yln = np.log(y)     # Степенная аппроксимация
        # print(Yln)
        model = sm.OLS(Yln, X).fit()

        a = model.params[0]
        b = model.params[1]

        a = math.exp(a)
        b = math.exp(b)

        print(' NDVI = ', x)
        print(' Prod = ', y)
        print()

        function2 = str(a) + ' * (' + str(b) + ') ** x '    # Степенная функция
        function2_for_report = str(round(a, 2)) + '*(' + str(round(b, 2)) +')^x'    # Степенная функция

        print('Степенная функция: ', function2)
        R_sqr = model.rsquared  # Коэффициент детерминации
        print()
        print('R-squared: ', R_sqr)
        mae = Find_mae_and_max(y, x, function2)
        print('MAE: ', mae.mae)
        print('Max error: ', mae.max)
        print()

        DATASET[Reg_list[i]].setdefault('Power_approximation', {})
        DATASET[Reg_list[i]]['Power_approximation'].setdefault('Regression_function_for_report', function2_for_report)  # Запись полученных данных в БД
        DATASET[Reg_list[i]]['Power_approximation']['Regression_function_for_report'] = function2_for_report

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('Regression_Function', function2)  # Запись полученных данных в БД
        DATASET[Reg_list[i]]['Power_approximation']['Regression_Function'] = function2

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('R_squared', model.rsquared)
        DATASET[Reg_list[i]]['Power_approximation']['R_squared'] = model.rsquared
        DATASET[Reg_list[i]]['Power_approximation'].setdefault('MAE', mae.mae)
        DATASET[Reg_list[i]]['Power_approximation']['MAE'] = mae.mae
        DATASET[Reg_list[i]]['Power_approximation'].setdefault('Max_error', mae.max)
        DATASET[Reg_list[i]]['Power_approximation']['Max_error'] = mae.max

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('STD', mae.std)
        DATASET[Reg_list[i]]['Power_approximation']['STD'] = mae.max

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('min', mae.RETURN['min'])
        DATASET[Reg_list[i]]['Power_approximation']['min'] = mae.RETURN['min']

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('25%', mae.RETURN['25%'])
        DATASET[Reg_list[i]]['Power_approximation']['25%'] = mae.RETURN['25%']

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('50%', mae.RETURN['50%'])
        DATASET[Reg_list[i]]['Power_approximation']['50%'] = mae.RETURN['50%']

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('75%', mae.RETURN['75%'])
        DATASET[Reg_list[i]]['Power_approximation']['75%'] = mae.RETURN['75%']

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('min', mae.RETURN['max'])
        DATASET[Reg_list[i]]['Power_approximation']['max'] = mae.RETURN['max']

        DATASET[Reg_list[i]]['Power_approximation'].setdefault('IQR', (mae.RETURN['75%'] - mae.RETURN['25%']) )
        DATASET[Reg_list[i]]['Power_approximation']['IQR'] = (mae.RETURN['75%'] - mae.RETURN['25%'])

        # DATASET[Reg_list[i]].setdefault('Regression_model', function)

        print('Параметры аппроксимации записаны!')

    return DATASET

def prediction(DATASET, DATASET_ANSWER, Reg_list, Analyzed_year):   # Расчет продуктивности по регрессии
    for i in range(len(Reg_list)):

        Function = DATASET[Reg_list[i]]['Linear_approximation']['Regression_Function']
        x = DATASET_ANSWER[Reg_list[i]][Analyzed_year]['NDVI_max']
        Pred = eval(Function)
        DATASET_ANSWER[Reg_list[i]][Analyzed_year].setdefault('Estimated productivity according to the linear model', Pred)
        DATASET_ANSWER[Reg_list[i]][Analyzed_year]['Estimated productivity according to the linear model'] = Pred

        Function = DATASET[Reg_list[i]]['Power_approximation']['Regression_Function']
        Pred = eval(Function)
        DATASET_ANSWER[Reg_list[i]][Analyzed_year].setdefault('Estimated productivity according to the power model', Pred)
        DATASET_ANSWER[Reg_list[i]][Analyzed_year]['Estimated productivity according to the power model'] = Pred
    return DATASET_ANSWER


def answer_print(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years):
    for i in range(len(Reg_list)):
        Reg_name = Reg_list[i]
        NDVI_m = DATASET_ANSWER[Reg_list[i]][Analyzed_year]['NDVI_max']
        Regression_func1 = DATASET[Reg_list[i]]['Linear_approximation']['Regression_Function']
        Regression_func2 = DATASET[Reg_list[i]]['Power_approximation']['Regression_Function']
        Calculation1 = DATASET_ANSWER[Reg_list[i]][Analyzed_year]['Estimated productivity according to the linear model']
        Calculation2 = DATASET_ANSWER[Reg_list[i]][Analyzed_year]['Estimated productivity according to the power model']

        fact = DATASET_ANSWER[Reg_list[i]][Analyzed_year]["Productive"]

        print('Region: ', Reg_name)
        print('Analyzed_year: ', Analyzed_year)
        print('Max NDVI: ', NDVI_m)
        print('Power model: ', Regression_func1)
        print('Power line: ', Regression_func2)
        sample = len(DATASET[Reg_name]['Regression_list'])
        print('Sample for calculations: ', sample)
        print('Estimated Regression Productivity (Line model): ', Calculation1)
        print('Estimated Regression Productivity (Power model): ', Calculation2)
        # print('Fact: ', fact)
        # print('')

def Regression_list_to_two_arrays(Regression_list):
    a = []  # NDVI
    b = []  # Prod
    c = []  # Year
    aia = []
    aib = []
    for i in range(len(Regression_list)):
        if len(Regression_list[i]) > 2:
            a.append(Regression_list[i][0])
            b.append(Regression_list[i][1])
            c.append(Regression_list[i][2])
        else:
            aia.append(Regression_list[i][0])
            aib.append(Regression_list[i][1])

    a.reverse()
    b.reverse()
    c.reverse()

    # print(a)
    # print(b)
    # print(c)

    class Answer:
        Year_NDVI = a
        Year_Prod = b
        AI_NDVI = aia
        AI_Prod = aib
        Years = c
    return Answer


def Create_a_excel_report(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years, Last_year, Average_productivity, Culture):

    workbook = xlsxwriter.Workbook('Excel report\\Report_' + Analyzed_year +'_'+ Culture + '.xlsx')
    worksheet = workbook.add_worksheet()
    workbook.close()

    INPUT = []
    Years_in_dataset = []
    for i in range(len(Reg_list)):  # Происходит оценка временных границ исследования (Самый старый и молодой год)
        Regression_array = DATASET[Reg_list[i]]['Regression_list']
        for j in range(len(Regression_array)):
            if len(Regression_array[j]) == 3:
                Years_in_dataset.append(Regression_array[j][2])

    year_min = min(Years_in_dataset)
    year_max = max(Years_in_dataset)
    # print(year_min, year_max)

    fn = 'Excel report\\Report_' + Analyzed_year +'_'+ Culture + '.xlsx' # Открываем пустой файл
    wb = load_workbook(fn)
    ws = wb['Sheet1']   # На первой странице

        # Прописываем заголовок
    ws.append(['Урожайность сельскохозяйственных культур (в расчете на убранную площадь) (значение показателя за год), центнеров с гектара, Хозяйства всех категорий'])

    ws.append([]) # Пустая строка после заголовка

    LINE = ['Регион ', 'Показатель ']
    for i in range(int(year_min), int(year_max)+1): # Формируем строку годов (шапка таблицы)
        LINE.append(i)

    LINE_ANSWER = [Analyzed_year, '', 'Корреляция', '', 'Модель аппроксимации ', 'R^2', 'STD','MAE', 'IQR', 'Minimum', '25%', '50%', '75%', 'Maximum', 'Прогноз']
    LINE = LINE + LINE_ANSWER
    # print(LINE)
    ws.append(LINE) # Запись шапки


    for i in range(len(Reg_list)):
        LINE1 = []
        LINE2 = []

        LINE1.append(Reg_list[i])
        LINE2.append('')

        LINE1.append('NDVI')
        LINE2.append('Урожайность')

        for j in range(int(year_min), int(year_max) + 1):
            try: # Если информация по годам имеется в БД, она записывается
                ndvi_j = DATASET[Reg_list[i]][str(j)]['NDVI_max']
                LINE1.append(ndvi_j)
            except BaseException:      # Иначе, ячейка оставляется пустой
                LINE1.append('')

            try:
                prod_j = DATASET[Reg_list[i]][str(j)]['Productive']
                LINE2.append(prod_j)
            except BaseException:\
                LINE2.append('')

        LINE1.append(DATASET_ANSWER[Reg_list[i]][Analyzed_year]['NDVI_max'])
        LINE2.append('')

        LINE1.append(' ')
        LINE2.append(' ')

        LINE1.append(DATASET[Reg_list[i]]['Cor'])
        LINE2.append(' ')

        LINE1.append('')
        LINE2.append('')


        #  'Модель аппроксимации ', 'R^2', 'STD','MAE', 'IQR', 'Minimum', '25%', '50%', '75%', 'Maximum', 'Прогноз'

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['Regression_Function'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['Regression_Function'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['R_squared'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['R_squared'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['STD'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['STD'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['MAE'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['MAE'])

        #  'Модель аппроксимации ', 'R^2', 'STD','MAE', 'IQR', 'Minimum', '25%', '50%', '75%', 'Maximum', 'Прогноз'

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['IQR'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['IQR'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['min'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['min'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['25%'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['25%'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['50%'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['50%'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['75%'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['75%'])

        LINE1.append(DATASET[Reg_list[i]]['Linear_approximation']['max'])
        LINE2.append(DATASET[Reg_list[i]]['Power_approximation']['max'])


        LINE1.append(DATASET_ANSWER[Reg_list[i]][Analyzed_year]['Estimated productivity according to the linear model'])
        LINE2.append(DATASET_ANSWER[Reg_list[i]][Analyzed_year]['Estimated productivity according to the power model'])

        # print(LINE1)
        # print(LINE2)
        ws.append(LINE1)
        ws.append(LINE2)        # Записываем основную таблицу



    ws.append([])
    ws.append([])
    ws.append([])

    ws.append(['Массивы дополнительных данных'])
    for i in range(len(Reg_list)):

        Regression_list = DATASET[Reg_list[i]]['Regression_list']
        Regression_list = Regression_list_to_two_arrays(Regression_list)

        if len(Regression_list.AI_NDVI) > 0:
            LINE1 = []
            LINE2 = []

            LINE1.append(Reg_list[i])
            LINE2.append('')

            LINE1.append('NDVI')
            LINE2.append('Урожайность')

            LINE1 = LINE1 + Regression_list.AI_NDVI
            LINE2 = LINE2 + Regression_list.AI_Prod

            # print(LINE1)
            # print(LINE2)

            ws.append(LINE1)
            ws.append(LINE2)


    wb.save(fn)     # Сохраняем и закрываем документ
    wb.close()


def Make_array_by_formula(xmax, xmin, Function):
    Y = []
    n = (xmax-xmin)/50
    X = np.arange(xmin - 3*n, xmax + 3*n, n)
    for i in range(len(X)):
        x = X[i]
        Pred = eval(Function)
        Y.append(Pred)
    class Answer:
        x = X
        y = Y
    return Answer



def Create_a_word_report(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years, Last_year, Average_productivity, Culture):

    doc = docx.Document()   # Создаем документ

    # _______1
    section = doc.sections[-1]
    section.top_margin = Inches(0.8)  # Верхний отступ
    section.bottom_margin = Inches(0.8)  # Нижний отступ
    section.left_margin = Inches(1.2)  # Отступ слева
    section.right_margin = Inches(0.6)  # Отступ справа
    # _______2
    # paragraph_format = doc.styles['Normal'].paragraph_format
    # paragraph_format.line_spacing = Pt(12)  # межстрочный интервал
    # _______3
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'  # Стиль шрифта
    font.size = Pt(12)  # Размер шрифта

    doc.add_heading(Culture, 0)     # Пишем культуру в заголовок

    for i in range( len(Reg_list) ):

        Region = Reg_list[i]

        doc.add_heading(Region, 1)  # Заголовок раздела
        doc.add_paragraph('')


        if len(DATASET[Reg_list[i]]['Regression_list']) > years:    # Если данных регрессии больше, чем лет в статистике, в отчете упоминается о полевых измерениях
            doc.add_paragraph('Выборка для регрессии: ' + str(len(
                DATASET[Reg_list[i]]['Regression_list'])) + '. Данные с ' + Analyzed_year + ' по ' + str(
                round((int(Analyzed_year) - int(years)))) + ' год, включая данные полевых измерений.  ')
        else:
            doc.add_paragraph('Выборка для регрессии: ' + str(len(
                DATASET[Reg_list[i]]['Regression_list'])) + '. Данные с ' + Analyzed_year + ' по ' + str(
                round((int(Analyzed_year) - int(years)))) + ' год. ')

        doc.add_paragraph('Коэффициент корреляции урожая ('+ Culture + ') и максимального значения вегетационного индекса NDVI: '+ str( round(DATASET[Reg_list[i]]['Cor'], 3) ) )
        doc.add_paragraph('Максимальное значение NDVI, измеренное в '+ str(Analyzed_year) + ' году: ' + str(round(DATASET_ANSWER[Reg_list[i]][Analyzed_year]["NDVI_max"], 3))  )


        p = doc.add_paragraph()
        run = p.add_run()
        run.add_picture('Graphs\\JPG\\' + 'NDVI и Урожайность по годам - ' + Region + '.jpg',
                        width=docx.shared.Cm(15))  # Вставка изображения
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        paragraph = doc.add_paragraph(
            'Рис. ' + 'Сопоставление урожайности и максимальных значений вегетационного индекса. ')
        p_fmt = paragraph.paragraph_format
        p_fmt.alignment
        p_fmt.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        # doc.add_paragraph('')

        doc.add_page_break()


        p = doc.add_paragraph()
        run = p.add_run()
        run.add_picture('Graphs\\JPG\\' + 'NDVI - ' + Region + '.jpg', width=docx.shared.Cm(15))  # Вставка изображения
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        paragraph = doc.add_paragraph('Рис. ' + 'График вегетационного индекса NDVI измеренный в '+ Analyzed_year + ' году. ')
        p_fmt = paragraph.paragraph_format
        p_fmt.alignment
        p_fmt.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        # doc.add_paragraph('')



        p = doc.add_paragraph()
        run = p.add_run()
        run.add_picture('Graphs\\JPG\\' + 'Линейная регрессия - ' + Region + '.jpg',
                        width=docx.shared.Cm(15))  # Вставка изображения
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        paragraph = doc.add_paragraph('Рис. ' + 'Линейная модель аппроксимации зависимости средней урожайности в регионе от max_NDVI.')
        p_fmt = paragraph.paragraph_format
        p_fmt.alignment
        p_fmt.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        # doc.add_paragraph('')

        p = doc.add_paragraph()
        run = p.add_run()
        run.add_picture('Graphs\\JPG\\' + 'Экспоненциальная регрессия - ' + Region + '.jpg',
                        width=docx.shared.Cm(15))  # Вставка изображения
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        paragraph = doc.add_paragraph('Рис. ' + 'Экспоненциальная модель аппроксимации зависимости средней урожайности в регионе от max_NDVI.')
        p_fmt = paragraph.paragraph_format
        p_fmt.alignment
        p_fmt.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        # doc.add_paragraph('')


        doc.add_paragraph('Таблица – Параметры аппроксимации. ')

        table = doc.add_table(rows=3, cols=7)
        table.style = 'Table Grid'

        cell = table.cell(0, 1)
        cell.text = str('f(x)')
        cell = table.cell(0, 2)
        cell.text = str('R^2')
        cell = table.cell(0, 3)
        cell.text = str('MAE')
        cell = table.cell(0, 4)
        cell.text = str('STD')
        cell = table.cell(0, 5)
        cell.text = str('IQR')
        cell = table.cell(0, 6)
        cell.text = str('Ожидаемая урожайность в 2023 году (ц/га)')

        # 'f(x)' 'R^2' 'Correlation' 'MAE' 'STD' 'IQR' 'Ожидаемая урожайность в 2023 году (ц/га)'

        cell = table.cell(1, 0)
        cell.text = str('Линейная модель')
        cell = table.cell(1, 1)
        cell.text = str(DATASET[Reg_list[i]]['Linear_approximation']['Regression_function_for_report'])
        cell = table.cell(1, 2)
        cell.text = str(round(DATASET[Reg_list[i]]['Linear_approximation']['R_squared'], 3))

        cell = table.cell(1, 3)
        cell.text = str(round(DATASET[Reg_list[i]]['Linear_approximation']['MAE'], 3))
        cell = table.cell(1, 4)
        cell.text = str(round(DATASET[Reg_list[i]]['Linear_approximation']['STD'], 3))
        cell = table.cell(1, 5)
        cell.text = str(round(DATASET[Reg_list[i]]['Linear_approximation']['IQR'], 3))
        cell = table.cell(1, 6)
        cell.text = str(
            round(DATASET_ANSWER[Reg_list[i]][Analyzed_year]["Estimated productivity according to the linear model"],
                  3))

        cell = table.cell(2, 0)
        cell.text = str('Эксп. модель ')
        cell = table.cell(2, 1)
        cell.text = str(DATASET[Reg_list[i]]['Power_approximation']['Regression_function_for_report'])
        cell = table.cell(2, 2)
        cell.text = str(round(DATASET[Reg_list[i]]['Power_approximation']['R_squared'], 3))

        cell = table.cell(2, 3)
        cell.text = str(round(DATASET[Reg_list[i]]['Power_approximation']['MAE'], 3))
        cell = table.cell(2, 4)
        cell.text = str(round(DATASET[Reg_list[i]]['Power_approximation']['STD'], 3))
        cell = table.cell(2, 5)
        cell.text = str(round(DATASET[Reg_list[i]]['Power_approximation']['IQR'], 3))
        cell = table.cell(2, 6)
        cell.text = str(
            round(DATASET_ANSWER[Reg_list[i]][Analyzed_year]["Estimated productivity according to the power model"],
                  3))

        ###########################
        Regression_graph = Regression_list_to_two_arrays(DATASET[Reg_list[i]]['Regression_list'])
        Average_productivity = mean(Regression_graph.Year_Prod)

        doc.add_paragraph('')
        doc.add_paragraph('Средняя урожайность за прошлые годы: ' + str(round( Average_productivity, 3)))
        doc.add_paragraph('')

        doc.add_section()  # Разрыв раздела


    doc.save('Word report\\' + Culture + '_' + Analyzed_year + '.doc')
    print('Отчет Word сохранен! ')
    return

def massiv_to_two_arrays(ARRAY):
    Array1 = []
    Array2 = []
    for i in range(len(ARRAY)):
        day = str(round(int(ARRAY[i][0])/7))
        number = ARRAY[i][1]
        Array1.append(day)
        Array2.append(number)
    # print()
    # print(Array1)
    # print()
    # print(Array2)
    # print()
    Answer = []
    Answer.append(Array1)
    Answer.append(Array2)

    return Answer

def Create_a_report(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years, Last_year, Culture):

    for i in range( len(Reg_list) ):

        Regression_graph = Regression_list_to_two_arrays(DATASET[Reg_list[i]]['Regression_list'])

        YEARS = list(Regression_graph.Years)
        YEARS.append(Analyzed_year) # Годы по которым имеется NDVI
        NDVI_YEARS = Regression_graph.Year_NDVI
        NDVI_YEARS.append(DATASET_ANSWER[Reg_list[i]][Analyzed_year]["NDVI_max"]) # График ndvi
        PROD = list(Regression_graph.Year_Prod) # График урожайности
        P_YEARS = Regression_graph.Years    # Годы под урожайность

        fig, ax1 = plt.subplots(figsize=(12, 6))        # График урожайности и пика NDVI
        ax2 = ax1.twinx()
        ax1.plot(YEARS, NDVI_YEARS, label=' NDVI ', color = 'blue')
        ax2.plot(P_YEARS, PROD, label=' Урожайность ', color='darkred')
        ax1.set_ylabel('NDVI', color = 'blue', size = 11)
        ax2.set_ylabel('Урожайность (ц/га)', color='darkred', size = 11)
        plt.title('NDVI/Урожайность по годам - ' + Reg_list[i], size = 14)
        fig.legend(bbox_to_anchor=(0.65, 0.07), ncol=2, fontsize=11)
        plt.grid(True)
        pdf = PdfPages('Graphs\\PDF\\NDVI и Урожайность по годам - ' + Reg_list[i] + '.pdf')
        pdf.savefig()
        plt.savefig('Graphs\\JPG\\NDVI и Урожайность по годам - ' + Reg_list[i] + '.jpg')
        plt.close()
        pdf.close()

        x = []  # NDVI
        y = []  # Productive
        z = []  # Year


        Regression = DATASET[Reg_list[i]]['Regression_list']
        for j in range(len(Regression)):  # Подготовка массивов для построения регрессии
            x.append(Regression[j][0])
            y.append(Regression[j][1])


        Line_array = Make_array_by_formula(max(x), min(x), DATASET[Reg_list[i]]['Linear_approximation']['Regression_Function'])


        plt.figure(figsize=(10, 6)) # Гарфик линейной регрессии
        plt.scatter(x, y, label=' NDVI ', color='blue')
        plt.plot(Line_array.x, Line_array.y, label=' Линейная регрессия ', color='darkred')

        plt.xlabel('NDVI max', size = 11)
        plt.ylabel('Урожайность (ц/га) ', size = 11)
        plt.title('Линейная регрессия - ' + Reg_list[i], size = 14)
        plt.grid(True)
        # plt.show()
        pdf = PdfPages('Graphs\\PDF\\Линейная регрессия - ' + Reg_list[i] + '.pdf')
        pdf.savefig()
        plt.savefig('Graphs\\JPG\\Линейная регрессия - ' + Reg_list[i] + '.jpg')
        plt.close()
        pdf.close()


        Line_array = Make_array_by_formula(max(x), min(x),
                                           DATASET[Reg_list[i]]['Power_approximation']['Regression_Function'])

        plt.figure(figsize=(10, 6)) # Гарфик экспоненциальной регрессии
        plt.scatter(x, y, label=' NDVI ', color='blue')
        plt.plot(Line_array.x, Line_array.y, label=' Экспоненциальная регрессия ', color='darkred')

        plt.xlabel('NDVI max', size = 11)
        plt.ylabel('Урожайность (ц/га) ', size = 11)
        plt.title('Экспоненциальная регрессия - ' + Reg_list[i], size = 14) # Заголовок
        plt.grid(True)
        # plt.show()
        pdf = PdfPages('Graphs\\PDF\\Экспоненциальная регрессия - ' + Reg_list[i] + '.pdf')
        pdf.savefig()
        plt.savefig('Graphs\\JPG\\Экспоненциальная регрессия - ' + Reg_list[i] + '.jpg')
        plt.close()
        pdf.close()

        INITIAL_DATA = read_inf('INITIAL_DATA.json')  # Загрузка датасета с указаниями к расчетам

        # print(INITIAL_DATA)

        Login = os.environ.get('VEGA_LOGIN', '')         # учётные данные Vega — из окружения
        Login = INITIAL_DATA['Login']
        Password = INITIAL_DATA['Password']
        Culture = INITIAL_DATA['Culture']
        Reg_list = INITIAL_DATA['Reg_list']
        Analyzed_year = INITIAL_DATA['Analyzed_year']
        Last_year = INITIAL_DATA['Last_year']
        years = INITIAL_DATA['Years']


        NDVI_list = DATASET_ANSWER[Reg_list[i]][Analyzed_year]["NDVI"]
        DATA_VEGA = parser_vega_region(Login, Password, Reg_list[i], Analyzed_year,
                                       "reg_mean_interannual_ndvi_7dc_v2_modis_int_ozim")  # Средний многолетний график NDVI
        NDVI_list_manyears = DATA_VEGA['data']['1']['xy']
        NDVI_list_manyears = massiv_to_two_arrays(NDVI_list_manyears)
        NDVI_list = massiv_to_two_arrays(NDVI_list)

        matplotlib.rcParams.update({'font.size': 7})  # Гарфик NDVI
        plt.figure(figsize=(11, 6))
        plt.ylim(0, 1)
        plt.xlim(0, 52)
        plt.plot(NDVI_list[0], NDVI_list[1], 'g', linewidth=3, label=' NDVI ')
        plt.plot(NDVI_list_manyears[0], NDVI_list_manyears[1], 'g--', label=' Многолетняя норма NDVI ', )

        plt.legend(fontsize=11)

        plt.xlabel('Номер недели в году', color='blue', size=11)
        # plt.ylabel('NDVI', color='darkred', size = 14)
        plt.title(' NDVI - ' + Reg_list[i] + ' ' + Analyzed_year, size=14)
        plt.grid(True)
        # plt.show()
        pdf = PdfPages('Graphs\\PDF\\NDVI - ' + Reg_list[i] + ' ' + Analyzed_year + '.pdf')
        pdf.savefig()
        plt.savefig('Graphs\\JPG\\NDVI - ' + Reg_list[i] + '.jpg')
        plt.close()
        pdf.close()




        Average_productivity = mean(Regression_graph.Year_Prod)
        Average_NDVI = mean(Regression_graph.Year_NDVI)

        print('')
        print('Регион: ' + Reg_list[i])
        print('Выборка для регрессии: ', len(DATASET[Reg_list[i]]['Regression_list']))
        print('Коэффициент корреляции: ', DATASET[Reg_list[i]]['Cor'])
        print('Максимальный NDVI в ' + str(Analyzed_year) + ' году: ', DATASET_ANSWER[Reg_list[i]][Analyzed_year]["NDVI_max"])
        print()
        print('Линейная функция регрессии: ', DATASET[Reg_list[i]]['Linear_approximation']['Regression_Function'])
        print('R^2: ', DATASET[Reg_list[i]]['Linear_approximation']['R_squared'], ' Средняя ошибка (MAE): ', DATASET[Reg_list[i]]['Linear_approximation']['MAE'], ' Макс. ошибка: ', DATASET[Reg_list[i]]['Linear_approximation']['Max_error'])
        print()
        print('Степенная функция регрессии: ', DATASET[Reg_list[i]]['Power_approximation']['Regression_Function'])
        print('R^2: ', DATASET[Reg_list[i]]['Power_approximation']['R_squared'], ' Средняя ошибка (MAE): ',
              DATASET[Reg_list[i]]['Power_approximation']['MAE'], ' Макс. ошибка: ',
              DATASET[Reg_list[i]]['Power_approximation']['Max_error'])
        print()
        print('Ожидаемая урожайность по линейной моделе в ' + str(Analyzed_year) + ' году: ',
              DATASET_ANSWER[Reg_list[i]][Analyzed_year]["Estimated productivity according to the linear model"])
        print('Ожидаемая урожайность по степенной моделе в ' + str(Analyzed_year) + ' году: ',
              DATASET_ANSWER[Reg_list[i]][Analyzed_year]["Estimated productivity according to the power model"])
        print()
        print('Средняя урожайность за прошлые годы: ', Average_productivity)

        print('')

    Create_a_excel_report(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years, Last_year, Average_productivity, Culture)
    Create_a_word_report(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years, Last_year, Average_productivity, Culture)









def main():



    Login = os.environ.get('VEGA_LOGIN', '')         # учётные данные Vega — из окружения
    Password = os.environ.get('VEGA_PASSWORD', '')   # пароль в коде не хранится
    # product_type = 'ndvi_7dc_modis_int'
    product_type = 'reg_mean_ndvi_7dc_modis_int_ozim'   # Кодировка среднего по региону индекса NDVI в Vega Scince
    Culture = "Пшеница озимая"      # Анализируемая культура
    Farm_type = "Хозяйства всех категорий"      # Тип хозяйства (Информация требуется для сайта ЕМИСС)
    Reg_list = ['Пензенская область', 'Новгородская область', 'Ставропольский край', 'Республика Крым']
    Analyzed_year = '2020'  # Анализируемый год
    Last_year = '2019'      # Год, с которого начнется сбор информации
    years = 20              # На сколько лет в прошлое нужно найти информации

    INITIAL_DATA = read_inf('INITIAL_DATA.json')  # Загрузка датасета с указаниями к расчетам

    print(INITIAL_DATA)

    Login = INITIAL_DATA['Login']
    Password = INITIAL_DATA['Password']
    product_type = INITIAL_DATA['product_type']
    Culture = INITIAL_DATA['Culture']
    Farm_type = INITIAL_DATA['Farm_type']
    Reg_list = INITIAL_DATA['Reg_list']
    Analyzed_year = INITIAL_DATA['Analyzed_year']
    Last_year = INITIAL_DATA['Last_year']
    years = INITIAL_DATA['Years']

    Sheet = '120_1(1010400)'
    # global Sheet

    # Application_1 = read_inf('Categories_of_farms_dictionary.json')  # Загрузка датасета с указаниями к расчетам
    #
    # print("Application_1: ",Application_1)
    # print()
    #
    # Application_2 = read_inf('Culture_Dictionary.json')  # Загрузка датасета с указаниями к расчетам
    #
    # print("Application_2: ", Application_2)
    # print()
    #
    # Application_3 = read_inf('Region_Dictionary.json')  # Загрузка датасета с указаниями к расчетам
    #
    # print("Application_3: ", Application_3)
    # print()
    #
    # Application_4 = read_inf('Regions_uid_dict.json')  # Загрузка датасета с указаниями к расчетам
    #
    # print("Application_4: ", Application_4)
    # print()



    while True:

        print(' Введите команду: Обновить БД! / Запустить расчет! / Сформировать отчет! / Показать DATASET! / Завершить работу!') # Ввод команды
        # print('     Обновить БД!')
        # print('     Запустить расчет!')
        # print('     Сформировать отчет!')
        action = input()

        if action.upper() == 'ОБНОВИТЬ БД!':
            DATASET = read_inf('DATASET.json')  # Загрузка датасета
            # print(DATASET)
            Vega = True
            FS = True
            AI = True
            DATASET = collect_dataset(Reg_list, years, Last_year, Culture, product_type, Farm_type, Login, Password, Vega, FS, AI, DATASET) # Сбор датасета
            # DATASET = Add_info_from_reference_fields(DATASET, Reg_list, years, Last_year)


            write_inf(DATASET, 'DATASET.json') # Запись датасета
            print('База данных обновлена!')

        elif action.upper() == 'ЗАПУСТИТЬ РАСЧЕТ!':
            DATASET = read_inf('DATASET.json')  # Загрузка датасета
            DATASET = NDVI_max_for_dataset(DATASET, Reg_list, years, Last_year)   # Определение максимального значения NDVI в начале сезона
            write_inf(DATASET, 'DATASET.json')  # Запись датасета
            print('Максимумы определены')

            DATASET = regression_array(DATASET, Reg_list) # Подготовка массивов для вычисления регрессии        # Где-то тут возникает артефакт "Нет данных для региона: Белгородская область, год: 1st_order_polynomial"
            write_inf(DATASET, 'DATASET.json')  # Запись датасета
            print('Массивы подготовлены')

            DATASET = regression_function(DATASET, Reg_list)    # Вычисление регрессий
            write_inf(DATASET, 'DATASET.json')  # Запись датасета
            print('Регрессии построены')

            Vega = True
            FS = True
            AI = False
            DATASET_ANSWER = read_inf('DATASET_ANSWER.json')  # Загрузка датасета
            DATASET_ANSWER = collect_dataset(Reg_list, 1, Analyzed_year, Culture, product_type, Farm_type, Login, Password, Vega, FS, AI, DATASET_ANSWER) # Сбор датасета
            DATASET_ANSWER = NDVI_max_for_dataset(DATASET_ANSWER, Reg_list, 1, Analyzed_year)  # Определение максимального значения NDVI в конце сезона

            write_inf(DATASET_ANSWER, 'DATASET_ANSWER.json')  # Запись датасета
            DATASET_ANSWER = read_inf('DATASET_ANSWER.json')  # Загрузка датасета


            DATASET_ANSWER = prediction(DATASET, DATASET_ANSWER, Reg_list, Analyzed_year) # Расчет ожидаемой продуктивности
            write_inf(DATASET_ANSWER, 'DATASET_ANSWER.json')  # Запись датасета

            # print( 'DATASET = ', DATASET)
            print('')
            # print( 'DATASET_ANSWER = ', DATASET_ANSWER)
            print('')
            answer_print(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years)
            # print(DATASET)
            # DATASET_ANSWER = prediction(DATASET, DATASET_ANSWER, Reg_list)
            # print(DATASET)`
            print('Расчеты завершены!')

        elif action.upper() == 'ПОКАЗАТЬ DATASET!':
            DATASET = read_inf('DATASET.json')  # Загрузка датасета
            print(DATASET)

        elif action.upper() == 'СФОРМИРОВАТЬ ОТЧЕТ!':
            DATASET = read_inf('DATASET.json')  # Загрузка датасета
            DATASET_ANSWER = read_inf('DATASET_ANSWER.json')  # Загрузка датасета
            Create_a_report(DATASET_ANSWER, DATASET, Reg_list, Analyzed_year, years, Last_year, Culture)
            print("Отчет сформирован! Графики сохранены в формате .PDF ")

        elif action.upper() == 'ЗАВЕРШИТЬ РАБОТУ!':
            print('Завершение работы...')
            break

        else:
            print(' Команда не распознана! ')

# Пусть падает красиво +
# Проверка наличия данных в БД +
# Сохранение БД в полном объеме +
# Степенная аппроксимация +
# Оценка точности +
# Дополнительный ввод информации +
# Коэфф корреляции +
# Сделать так, чтобы программа использовала список годов, а не просто их количество в глубину
# Формирование отчета в word  +
# Формирование отчета в excel +
# Файлы ввода информации +
# Название отчета должно содержать инофрмацию о параметрах расчета


if __name__ == '__main__':
    main()