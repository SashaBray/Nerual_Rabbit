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


def main():

    Region_Dictionary = read_inf("Regions_uid_dict.json")

    print(Region_Dictionary)

    # print(Region_Dictionary["Кемеровская область - Кузбасс"])

    print('')
    print("Республика Татарстан (Татарстан)")
    print('Кемеровская область – Кузбасс')

    Region_Dictionary.setdefault('Республика Татарстан (Татарстан)', '16')

    print(Region_Dictionary)

    write_inf(Region_Dictionary, "Regions_uid_dict.json")


if __name__ == '__main__':
    main()




