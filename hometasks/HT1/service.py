from pydantic import BaseModel
import pandas as pd
from typing import List
import pickle
import re
import ssl
from fastapi import FastAPI, UploadFile
from io import BytesIO
ssl._create_default_https_context = ssl._create_unverified_context
pd.options.mode.chained_assignment = None

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]


# очищаю 'mileage', 'engine', 'max_power' от str и иставляю только числа
def delete_str_in_column(data, column: str):
    '''
    Функция получает на вход набор данных в формете pd.DataFrame, колонку датафрейма в формате str
    В dict_error попадают все значения, которые не удалось обработать 
    Возвращает: 
    * pd.Series с данными приведенным к типу float без единиц измерения 
    * словавь с наблюдениями которые не удалось обработать 
    * количество таких наблюдений
    '''
    column_data = data.loc[:, column].copy()
    dict_error = dict()
    for i in range(len(data)):
        try:
            column_data[i] = float(data[column][i].split()[0])
        except:
            column_data[i] = None
            dict_error[str(i)] = data[column][i]
    return column_data, dict_error, len(dict_error)


# применяю delete_str_in_column к колонке
def norm_data(data, list_column):
    """
    Функция принимает набор данных в pd.DataFrame и список колонок над которыми применять функцию для изменения данных
    Применяет преобразования на месте
    """
    for column_name in list_column:
        data[column_name], _, _ = delete_str_in_column(data, column_name)


def torque_clear_data(data):
    """
    Функция получает на вход DataFrame
    При вызове функции создает в нем две колонки из колонки torque для удобства дальнейшей обработки 
    Возвращает список с индексами наблюдений, которые не получилось обработать 
    """

    list_column = ['mileage', 'engine', 'max_power']
    norm_data(data, list_column)

    data['torque_clear'] = data['torque'].copy()
    data['max_torque_rpm'] = data['torque'].copy()
    list_error_index = list()
    # цикл бежит по элементам колонки 
    for i in range(len(data['torque'])):
       # пробую обработать элемент колонки 
       try:
           value = data['torque'][i] # одно наблюдение в колонке
           # если крутящий момент выражен в кг/метр
           if 'kgm' in value.lower():
               kgm = re.sub("[^0-9.,]", " ", value).replace(',', '').split() # оставляю в строке только числа, точки и запятые, заменяю запятые для корректного разбиения и применяю split
               rpm = int(kgm[-1]) # выбираю из списка только число оборотов и привожу его к int
               kgm = int(float(kgm[0]) * 9.80665) # крутящий момент перевожу из кг/метр в Н/метр
               data['max_torque_rpm'][i] = rpm # количество оборотов в одну колонку
               data['torque_clear'][i] = kgm # крутящий момент в другую колонку
           # если крутящий момент выражен в Н/метр
           elif 'nm' in value.lower():
               nm = re.sub("[^0-9.,]", " ", value).replace(',', '').split() 
               rpm = int(nm[-1])
               nm = int(float(nm[0]))
               data['max_torque_rpm'][i] = rpm
               data['torque_clear'][i] = nm
            # если указаний измерения крутящего момента нет
           else:
               torque = re.sub("[^0-9.,]", " ", value).replace(',', '').split() 
               rpm = int(torque[-1])
               torque = int(float(torque[0]))
               data['max_torque_rpm'][i] = rpm
               data['torque_clear'][i] = torque    
       # если возникает ошибка, то индекс в список индексов наблюдений, где возникла ошибка
       except:
           list_error_index.append(i) 
    return list_error_index


def transtorm(data, scaller, OH_encoder):
    torque_clear_data(data)
    data.drop(['torque', 'name', 'selling_price'], axis= 1, inplace= True)
    object_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    OH_cols_data = pd.DataFrame(OH_encoder.transform(data[object_cols]))
    # # One-hot encoding removed index; put it back
    OH_cols_data.index = data.index
    # Getting column names
    OH_cols_data.columns = OH_encoder.get_feature_names_out()
    # Remove categorical columns (will replace with one-hot encoding)
    num_data = data.drop(object_cols, axis=1)
    # Add one-hot encoded columns to numerical features
    OH_data = pd.concat([num_data, OH_cols_data], axis=1)
    # стандартизирую признаки
    test_scaller = pd.DataFrame(scaller.transform(OH_data))

    return test_scaller


# загружаю веса модели для предказания
loaded_model = pickle.load(open('model_weight.pickle', 'rb'))
encoder = pickle.load(open('onehotencoder_weight.pickle', 'rb'))
std_scaller = pickle.load(open('scaller_weight.pickle', 'rb'))

@app.post("/predict_item")
def predict_item(name: str, year: int, selling_price: int , km_driven: int, fuel: str, 
                 seller_type: str, transmission: str, owner: str, mileage: str, 
                 engine: str, max_power: str, torque: str, seats: float) -> float:
    
    item = Item(name= name, year= year, selling_price= selling_price, km_driven= km_driven, fuel= fuel, 
                seller_type= seller_type, transmission= transmission, owner= owner, mileage= mileage, engine= engine, 
                max_power= max_power, torque= torque, seats= seats)
    
    item = pd.DataFrame([dict(item)])

    item = transtorm(item, std_scaller, encoder)
    return loaded_model.predict(item)[0]

@app.post('/predict_items')
def predict_items(file: UploadFile) -> List[float]:
    content = file.file.read()
    buffer = BytesIO(content)
    df = pd.read_csv(buffer, index_col=False)
    buffer.close() 
    file.close()
    df = transtorm(df, std_scaller, encoder)
    return list(loaded_model.predict(df))
