"""Тест парсера урожайности ЕМИСС (fedstat, показатель 31533).

Правки: используем ПРАВИЛЬНЫЕ полные словари UID из проекта (а не перепутанные хардкод-ID),
запрашиваем реальный субъект (в измерении регионов 57831 нет «Российской Федерации»),
верную культуру и «Хозяйства всех категорий» (UID 1750642).

Если ЕМИСС отвечает пусто — значит нет данных за эти параметры/год (не блок).
Если сайт блокирует — обычно из-за VPN: отключите VPN и повторите.
"""
import os
import shutil

import pandas as pd

from Parser_FedStat import parser_fedstat

# полные корректные словари лежат в проекте (скопированы из ComplexPrediction)
_SRC_CANDIDATES = [
    os.path.join('Regions_rabbits', 'workspace', 'source'),
    'ComplexPrediction',
]
_DICTS = ('Region_Dictionary.json', 'Culture_Dictionary.json', 'Categories_of_farms_dictionary.json')


def ensure_dictionaries():
    """Положить рядом со скриптом ПРАВИЛЬНЫЕ словари UID (с верными кодами регионов/культур/категорий)."""
    src = next((d for d in _SRC_CANDIDATES if os.path.isdir(d) and
                all(os.path.exists(os.path.join(d, n)) for n in _DICTS)), None)
    if src is None:
        raise FileNotFoundError('Не найдены словари UID (Region/Culture/Categories) в проекте. '
                                'Ожидались в Regions_rabbits/workspace/source или ComplexPrediction.')
    print(f'Использую корректные словари UID из: {src}')
    for n in _DICTS:
        shutil.copy(os.path.join(src, n), n)


def main():
    ensure_dictionaries()

    year = '2023'
    region = 'Белгородская область'          # реальный субъект (RF в измерении 57831 отсутствует!)
    culture = 'Пшеница озимая'               # верный UID из словаря культур
    farm_type = 'Хозяйства всех категорий'   # UID 1750642

    print(f'\nЗапрос за {year}: {region} -> {culture} -> {farm_type}')
    try:
        val = parser_fedstat(Year=year, Region=region, Culture=culture, Farm_type=farm_type)
    except KeyError as e:
        print(f'[ОШИБКА] {e}. Проверьте точное написание из словарей.'); return
    except RuntimeError as e:
        print(f'[БЛОК] {e}'); return
    except Exception as e:                                 # noqa: BLE001
        print(f'[ОШИБКА] Непредвиденная ошибка: {e}'); return

    if val is None:
        print('[ПУСТО] ЕМИСС вернул 0 записей за эти параметры/год. '
              'Попробуйте другой субъект/культуру/год (или данные ещё не опубликованы).')
        return

    print(f'[УСПЕХ] Урожайность: {val} ц/га')
    pd.DataFrame([{'Год': year, 'Регион': region, 'Культура': culture,
                   'Категория': farm_type, 'Урожайность (ц/га)': val}]).to_excel('yield_output.xlsx', index=False)
    print("Сохранено в 'yield_output.xlsx'")


if __name__ == '__main__':
    main()
