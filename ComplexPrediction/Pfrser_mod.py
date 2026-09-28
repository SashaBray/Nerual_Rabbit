import os
from Parser_Vega import parser_vega_region
import json
import matplotlib
import matplotlib.pyplot as plt


def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


INITIAL_DATA = read_inf('INITIAL_DATA.json')  # Загрузка датасета с указаниями к расчетам

print(INITIAL_DATA)

Login = os.environ.get('VEGA_LOGIN', '')          # учётные данные Vega — из окружения
Login = INITIAL_DATA['Login']
Password = INITIAL_DATA['Password']
product_type = INITIAL_DATA['product_type']
Culture = INITIAL_DATA['Culture']
Farm_type = INITIAL_DATA['Farm_type']
Reg_list = INITIAL_DATA['Reg_list']
Analyzed_year = INITIAL_DATA['Analyzed_year']
Last_year = INITIAL_DATA['Last_year']
years = INITIAL_DATA['Years']

Reg_name = Reg_list[2]
print(Reg_name)
year = Analyzed_year


DATA_VEGA = parser_vega_region(Login, Password, Reg_name, year, product_type) # Средний график NDVI
NDVI_list = DATA_VEGA['data']['1']['xy']

DATA_VEGA = parser_vega_region(Login, Password, Reg_name, year, "reg_mean_interannual_ndvi_7dc_v2_modis_int_ozim")  # Средний многолетний график NDVI
NDVI_list_manyears = DATA_VEGA['data']['1']['xy']

def massiv_to_two_arrays(ARRAY):
    Array1 = []
    Array2 = []
    for i in range(len(ARRAY)):
        day = str(round(int(ARRAY[i][0])/7))
        number = ARRAY[i][1]
        Array1.append(day)
        Array2.append(number)
    print()
    print(Array1)
    print()
    print(Array2)
    print()
    Answer = []
    Answer.append(Array1)
    Answer.append(Array2)

    return Answer


NDVI_list = massiv_to_two_arrays(NDVI_list)
NDVI_list_manyears = massiv_to_two_arrays(NDVI_list_manyears)

# plt.plot(NDVI_list_manyears[0], NDVI_list_manyears[1], 'g--')
# plt.plot(NDVI_list[0], NDVI_list[1],'g', linewidth= 3)
#
# plt.show()




matplotlib.rcParams.update({'font.size': 7})    # Гарфик NDVI
plt.figure(figsize=(11, 6))
# plt.plot(x, y, label=' NDVI ', color='green')

plt.plot(NDVI_list_manyears[0], NDVI_list_manyears[1], 'g--', label=' Многолетняя норма NDVI ')
plt.plot(NDVI_list[0], NDVI_list[1],'g', linewidth= 3, label=' NDVI ')

plt.legend(fontsize=11)
plt.ylim(0, 1)
plt.xlim(0, 52)
plt.xlabel('Номер недели в году', color='blue', size = 11)
# plt.ylabel('NDVI', color='darkred', size = 14)
plt.title(' NDVI - ' + Reg_list[1] + ' ' + Analyzed_year, size = 14)
plt.grid(True)
plt.show()