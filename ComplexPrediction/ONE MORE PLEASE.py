import json
from Parser_Vega import parser_vega_point


def write_inf(data, file_name):
    data = json.dumps(data)
    data = json.loads(str(data))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def read_inf(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
         return json.load(file)


INITIAL_DATA = read_inf('INITIAL_DATA.json')  # Загрузка датасета с указаниями к расчетам
Login = INITIAL_DATA['Login']
Password = INITIAL_DATA['Password']
coord = ['53.520', '83.835']


DICTIONARY_PARAM_VEGA = {
    'Температура': 'temp',
    'Относительная влажность': 'rh',
    'Атмосферное давление': 'p',
    'Кол - во осадков': 'prec'
    }


parm = DICTIONARY_PARAM_VEGA['Температура']

# parm = 'temp'
print(str(coord[0]), str(coord[1]))

print('')
print( Login, ' - ', type(Login))
print( Password, ' - ', type(Password))
print( '2022', ' - ', type('2022'))
print( parm, ' - ', type(parm))
print( coord[1], ' - ', type(coord[1]))
print( coord[0], ' - ', type(coord[0]))
print('')


# data_vega = parser_vega_point(Login, Password, '2022', parm, coord[1], coord[0])



YEAR_LIST = [2023]

year_i = str(YEAR_LIST[0])
data_vega = parser_vega_point(Login, Password, year_i, parm, coord[1], coord[0])



print(data_vega)