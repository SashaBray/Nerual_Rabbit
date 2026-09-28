import pandas as pd
import json
import datetime
import numpy as np
import os

from datetime import datetime
import matplotlib.pyplot as plt
import pylab

import cProfile


def write_inf(data, file_name):
    data = json.dumps(data)
    data = json.loads(str(data))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
         return json.load(file)


class District:
    """Описаине района"""

    def __init__(self, id_district, list_of_products, culture, directory):
        """Свойства района"""
        self.id_district = int(id_district)
        self.list_of_products = list_of_products
        self.culture = culture
        self.directory = directory
        self.name = None
        self.region_name = None

    def find_init_inf(self):
        """Поиск дополнительной информации в общем файле"""
        # print('Search for information...')
        df = pd.read_pickle(self.directory[1] + self.directory[2])
        index = df.index[(df['id_district'] == self.id_district) & (df['culture'] == self.culture)].tolist()[0]
        self.name = df.iloc[index]['district']
        self.region_name = df.iloc[index]['region']
        self.id_district = df.iloc[index]['id_district']
        self.prod = df.drop(['id_district', 'district', 'region', 'id_region', 'culture'], axis=1).iloc[[index]]  # Датафрейм с урожайностью в разные годы

    def find_list_of_satellite_data(self):

        print(self.prod)
        column_names = self.prod.columns.tolist()
        print('Заголовки ', column_names)

        # list_of_products

        for year in column_names:
            prod_value = self.prod.iloc[0][year]
            print(prod_value)

        pass



class Operations:

    def __init__(self):
        """Инициализация экземпляра класса""" # Набор функций для работы с временными рядами
        # self

    def plus_by_files(self, file_1, file_2, size, padding = 'zeros'):

        array_1, _ = TimeRow(file_1).get_array_by_size(size, padding)
        array_2, _ = TimeRow(file_2).get_array_by_size(size, padding)

        target_array = []
        for i in range(len(array_1)):
            value_i = array_1[i] + array_2[i]
            target_array.append(value_i)

        return target_array

    def minus_by_files(self, file_1, file_2, size, padding = 'zeros'):

        array_1, _ = TimeRow(file_1).get_array_by_size(size, padding)
        array_2, _ = TimeRow(file_2).get_array_by_size(size, padding)

        target_array = []
        for i in range(len(array_1)):
            value_i = array_1[i] - array_2[i]
            target_array.append(value_i)

        return target_array

    def mean_by_files(self, file_1, file_2, size, padding='zeros'):

        array_1, _ = TimeRow(file_1).get_array_by_size(size, padding)
        array_2, _ = TimeRow(file_2).get_array_by_size(size, padding)

        target_array = []
        for i in range(len(array_1)):
            value_i = (array_1[i] + array_2[i])
            target_array.append(value_i)

        return target_array

    def mean_by_files(self, file_1, file_2, size, padding='zeros'):

        array_1, _ = TimeRow(file_1).get_array_by_size(size, padding)
        array_2, _ = TimeRow(file_2).get_array_by_size(size, padding)

        target_array = []
        for i in range(len(array_1)):
            value_i = (array_1[i] + array_2[i])
            target_array.append(value_i)

        return target_array

    def binary_array_by_files_by_threshold(self, file_1, size, threshold, padding='zeros', value_if_true = 1, value_if_fals = 0):

        array_1, _ = TimeRow(file_1).get_array_by_size(size, padding)

        target_array = []
        for i in range(len(array_1)):
            value_i = array_1[i]
            if value_i > threshold:
                target_array.append(value_if_true)
            else:
                target_array.append(value_if_fals)

        return target_array

    def mean_in_district(self, product: str,
                         dist_id: int,
                         file_name: str,
                         size: int,
                         directory: str,
                         padding='zeros', save=True, year=None, average_last=None) -> list:
        {os.sep}
        files = os.listdir(directory)
        file_name = file_name

        if file_name in files:
            TimeRow_1 = TimeRow(directory  + file_name.replace('.json', ''))
            target_array, time_arr = TimeRow_1.get_array_by_size(size)

        else:
            files = os.listdir(directory)
            array_of_same_prod_files = []

            print('file_name', file_name)
            file_name_last_splited = file_name.split('_')
            print('file_name_last_splited', file_name_last_splited)
            mask = file_name.replace(file_name_last_splited[-1], '')
            mask = mask.replace(' ', '')

            for file_name_i in files:
                # print(mask, file_name_i)
                if mask in file_name_i and not 'historical' in file_name_i:
                    array_of_same_prod_files.append(file_name_i)

            print('Список файлов для расчетов:', '\n')
            print(array_of_same_prod_files)

            if year != None and average_last != None:
                array_for_open = []
                for i in range(average_last):
                    name_i = mask + str(int(year) - i) + '.json'
                    print('name_i', name_i)
                    if name_i in array_of_same_prod_files:
                        array_for_open.append(name_i)

                array_of_same_prod_files = array_for_open

            intermediate_array = np.zeros(size)
            time_arr = np.arange(0, 365, 365/size)

            arrays_list = []
            target_array = []

            for file_i in array_of_same_prod_files:  # Скрадываем все графики
                array_i, time_arr = TimeRow(file_name=directory + file_i.replace('.json', '')).get_array_by_size(size,
                                                                                                                 padding)
                # intermediate_array = intermediate_array + array_i
                if array_i.shape[0] != np.array([]).shape[0]:
                    arrays_list.append(array_i)

            print('arrays_list_len', len(arrays_list))

            for i in range(size):
                sum_of_values = 0
                not_null = 0
                for array_i_i in arrays_list:
                    # print('array_i_i', array_i_i)

                    sum_of_values = sum_of_values + array_i_i[i]
                    if array_i_i[i] != 0:
                        not_null = not_null + 1

                if not_null == 0:
                    not_null = 1
                target_value = sum_of_values / not_null
                target_array.append(target_value)
            target_array = np.array(target_array)

            if save:
                if '.json' not in file_name:
                    file_name = file_name + '.json'

                # print('file_name_i 111',file_name_i)
                self.save_like_json(target_array, product.replace('historical', ''), dist_id, 'mean', directory=directory, file_name=file_name)

        return target_array, time_arr

    def save_like_json(self, array, product, dist_id, year, first_data = "2000-01-01 00:00:00",
                       last_data = "2000-12-31 23:59:59",
                       directory = f'Dist_info_actual{os.sep}Average_long_term_data{os.sep}', file_name=''):

        file_name = directory + file_name
        print('file_name_at_seving', file_name)

        target_array = []

        step = 365/(len(array)-1)

        for i in range(len(array)):
            time_i = round(array[i], 4)
            value_i = str(round(i * step, 2))
            target_array.append([value_i, time_i])

        json_file = {"data": {"1": {"xy": target_array,"labels": {target_array[0][0]: first_data, target_array[-1][0]: last_data}}}}

        write_inf(json_file, file_name)

        pass




class TimeRow:

    def __init__(self, file_name=None, data=None):
        """Инициализация экземпляра класса.

        ``file_name`` — путь к .json (без расширения) для чтения с диска,
        либо ``data`` — уже готовый словарь того же формата (источник из БД).
        """
        self.file_name = file_name
        self.status = True
        self.open(data=data)

    def open(self, data=None):
        """Открываем источник и определяем год, первую и последнюю дату"""
        if data is None:
            filejson = read_inf(self.file_name + '.json')
        else:
            filejson = data

        # print('filejson', filejson)

        self.data_array = filejson['data']['1']['xy']
        self.calendar_dict = filejson['data']['1']['labels']


        if self.data_array != []:   # Если в файле нет никаких данных, переменой статуса будет присвоено значение False
            self.status = True
        else:
            self.status = False

        if self.status:
            data_array = filejson['data']['1']['xy']
            calendar_dict = filejson['data']['1']['labels']

            first_data = calendar_dict[data_array[0][0]]
            last_data = calendar_dict[data_array[-1][0]]

            # print('Time_borders: ', first_data, last_data)

            if len(first_data.split(' ')) >= 2:         # Разные варианты работы для разных типов данных. Где ест время и где нет.
                year = int(first_data.split('-')[0])
                data_ij_f = first_data.split(' ')
                data_j_f = data_ij_f[0].split('-')
                time_i_f = data_ij_f[1].split(':')

                # print(int(data_j_f[0]), int(data_j_f[1]), int(data_j_f[2]), int(time_i_f[0]), int(time_i_f[1]), int(time_i_f[2]))

                first_data_data = datetime(int(data_j_f[0]), int(data_j_f[1]), int(data_j_f[2]), int(time_i_f[0]), int(time_i_f[1]), int(time_i_f[2]))

                data_ij_l = last_data.split(' ')
                data_j_l = data_ij_l[0].split('-')
                time_i_l = data_ij_l[1].split(':')

                last_data_data = datetime(int(data_j_l[0]), int(data_j_l[1]), int(data_j_l[2]), int(time_i_l[0]), int(time_i_l[1]), int(time_i_l[2]))


            else:
                year = int(first_data.split('-')[0])
                data_j_f = first_data.split('-')
                first_data_data = datetime(int(data_j_f[0]), int(data_j_f[1]), int(data_j_f[2]))
                data_j_f = last_data.split('-')
                last_data_data = datetime(int(data_j_f[0]), int(data_j_f[1]), int(data_j_f[2]))

            # print('first_data_data - ', first_data_data)
            # print('last_data_data - ', last_data_data)

            self.first_data_data = first_data_data
            self.last_data_data = last_data_data
            self.year = year
            self.zero_data = datetime(year, 1, 1, 0, 0, 0)
            self.minus_zero_data = datetime(year, 12, 31, 23, 59, 59)

            if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:  # Проверка года на високосность
                days_in_this_year = float(366)
            else:
                days_in_this_year = float(365)

            self.days_in_this_year = days_in_this_year


        # print((last_data_data - first_data_data)/3)

    def get_original_shape(self):
        """Функция возвращает размер исходного массива"""
        original_shape = len(self.data_array)
        return original_shape

    def get_init_data(self):
        """Функция возвращает исходные данные"""
        return self.data_array, self.calendar_dict



    def function(self, target_date, padding = 'zeros', parm=0):        # target date - номер дня в формате float
        """Функция возвращает значение графика в произвольное время, используя линейную аппроксимацию между двумя ближайшими имеющимися датами"""
        last_data = self.last_data_data
        first_data = self.first_data_data
        year = self.year
        zero_data = self.zero_data
        minus_zero_data = self.minus_zero_data
        data_array = self.data_array
        target_date = float(target_date)
        target_date = target_date + parm

        days_in_this_year = self.days_in_this_year

        # print(data_array)

        if padding == 'closest_value':
            new_data_array = [[-days_in_this_year*2, data_array[0][1]], [data_array[0][0], data_array[0][1]]]
            new_data_array = new_data_array + data_array
            new_data_array = new_data_array + [[data_array[-1][0], data_array[-1][1]], [days_in_this_year*2, data_array[-1][1]]]
            data_array = new_data_array

        if padding == 'zeros':
            new_data_array = [[-days_in_this_year * 2, 0], [data_array[0][0], 0]]
            new_data_array = new_data_array + data_array
            new_data_array = new_data_array + [[data_array[-1][0], 0], [days_in_this_year * 2, 0]]
            data_array = new_data_array

        new_data_array_np = np.array(new_data_array)
        new_data_array_np = np.float32(new_data_array_np)

        time_data = new_data_array_np[:, 0]
        value_data = new_data_array_np[:, 1]

        function = np.interp(target_date, time_data, value_data)

        return function


    def get_array_by_size(self, my_size,
                          padding='zeros', # 'zeros',
                          parm=0):

        if self.status == True:

            if my_size > len(self.data_array): # Если требуемый размер больше, чем исходный, нужно лишь сделать интерполяцию.

                init_size = len(self.data_array)
                target_array = []
                step = self.days_in_this_year/my_size
                time_array = []

                #for i in range(my_size):
                #    step_i = i * step
                #    target_array.append(self.function(step_i, padding, parm=parm))
                #    time_array.append(step_i)


                self_data = np.array(self.data_array).astype(float)

                x = np.linspace(0,self.days_in_this_year,my_size) #np.arange(0,my_size, step)
                xp = self_data[:, 0]
                fp = self_data[:,1]

                xp = np.concatenate((np.array([xp[0]]), xp))
                xp = np.concatenate((xp, np.array([xp[-1]])))

                fp = np.concatenate((np.array([0]), fp))
                fp = np.concatenate((fp, np.array([0])))

                target_array = np.interp(x,xp,fp)
                time_array = x

            else:                              # Если требуемый размер меньше, чем исходный
                first_size = my_size

                while first_size < len(self.data_array):
                    first_size = first_size * 2
                first_size = first_size * 2
                # print(first_size)
                first_target_array, time_array = self.get_array_by_size(first_size, padding, parm)

                target_array = []
                step = int(first_size/my_size)

                for i in range(my_size):
                    mean_i = sum(first_target_array[i * step : i * step + step]) / step
                    target_array.append(mean_i)

            target_array = np.array(target_array)
            time_array = np.array(time_array)

        else:
            target_array, time_array = np.array([]), np.array([])

        return target_array, time_array

    def get_init_arrays(self):

        if self.status == True:

            data_array = []
            data_time = []
            for i in range(len(self.data_array)):
                data_array.append(self.data_array[i][1])
                data_time.append(self.data_array[i][0])

            data_array = np.array(data_array)
            data_time = np.array(data_time)

        else:
            data_array, data_time = np.array([]), np.array([])

        return data_array, data_time



def main():

    product_type_list = ['mean_ndvi_7dc_modis_int_agro', 'mean_prec_acc',
                         'mean_temp_acc', 'mean_ndvi_7dc_modis_int_ozim', 'mean_ndvi_7dc_modis_int_spring',
                         'mean_ndvi_7dc_modis_int_agro', 'mean_temp', 'mean_rh', 'mean_p', 'mean_prec', 'mean_prec_acc',
                         'mean_temp_acc', 'mean_tmpgr10', 'mean_soilw10', 'mean_snod', 'mean_snowc']

    Directory_of_files_with_information = "Dist_info_actual"
    General_information_file_directory = f"File_with_all_information{os.sep}"
    General_information_file_name = "Final_result_df_v2.pkl"

    direct_info = [Directory_of_files_with_information, General_information_file_directory,
                   General_information_file_name]

    file_names = ['mean_temp_60_1006_2022', 'mean_prec_acc_60_1006_2022', 'mean_rh_60_1006_interannual', 'mean_p_59_1005_2020']
    culture = 'Пшеница озимая'

    way = f'Service_files{os.sep}Temporary_information{os.sep}TimeRows{os.sep}'

    op = Operations()

    array_mean, _ = op.mean_in_district(product = 'mean_temp', dist_id = 1006, file_name = 'mean_temp_60_1006_2022_interannual',
    size = 365, directory = way, padding = 'zeros')

    ndvi_array, _ = TimeRow(way + file_names[0]).get_array_by_size(my_size=365, padding='zeros')
    init_ndvi, init_time = TimeRow(way + file_names[0]).get_init_arrays()


if __name__ == '__main__':
    cProfile.run('main()')

    # main()