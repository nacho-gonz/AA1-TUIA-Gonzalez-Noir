import joblib
import pandas as pd
import numpy as np
from utils.imputador_cat import imputador_por_semana

num_vars = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine','WindGustSpeed',
                'WindSpeed9am','WindSpeed3pm', 'Humidity9am', 'Humidity3pm','Pressure9am', 
                'Pressure3pm', 'Temp9am', 'Temp3pm','Cloud3pm','Cloud9am']

num_vars_date = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine','WindGustSpeed',
                'WindSpeed9am','WindSpeed3pm', 'Humidity9am', 'Humidity3pm','Pressure9am', 
                'Pressure3pm', 'Temp9am', 'Temp3pm','Cloud3pm','Cloud9am', 'cos_week', 'sin_week']

categ_vars = ['Location','week_year','WindGustDir', 'WindDir9am', 'WindDir3pm','RainToday']

def modif(dataset):
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['week_year'] = dataset['Date'].dt.strftime('%U')

    dataset['Cloud9am'].replace(to_replace={9:8}, inplace=True)
    dataset.index = dataset['Date']
    dataset.drop('Date', axis=1, inplace=True)

    puntos_8 = {'NNE': 'NE', 'ENE': 'E', 'ESE': 'SE', 'SSE':'S', 'SSW': 'SW', 'WSW': 'W', 'WNW': 'NW', 'NNW': 'N'}

    dataset['WindGustDir'] = dataset['WindGustDir'].replace(puntos_8)
    dataset['WindDir9am'] = dataset['WindDir9am'].replace(puntos_8)
    dataset['WindDir3pm'] = dataset['WindDir3pm'].replace(puntos_8)
    return dataset


def positivisar(df):
    """
    Esta función recibe un dataframe escalado con MinMax para ajustar posibles valores fuera del rango [0,1]
    """
    bins = [0,0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375,1]
    bins_n = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    columnas = df.columns
    for col in columnas:
        if col in ['RainToday', 'Location','WindGustDir', 'WindDir9am', 'WindDir3pm','cos_week','sin_week']:
            continue
        df[col][df[col] < 0] = 0
        df[col][df[col] > 1] = 1
        if col == 'Cloud9am' or col == 'Cloud3pm':
            df[col] = pd.cut(df[col], bins=bins, labels=bins_n, include_lowest=True)

    return df

def codificar_fecha(dataset):
    """
    Codifica la variable fecha en seno y coseno
    """
    dataset['week_year'] = dataset['week_year'].astype(int)
    dataset['sin_week'] = np.sin(2 * np.pi * dataset['week_year'] / 52)
    dataset['cos_week'] = np.cos(2 * np.pi * dataset['week_year'] / 52)
    dataset.drop('week_year', axis=1, inplace=True)

    return dataset


def codificar_variables(dataset):
    """
    Codifica y reemplaza date, WindGustDir, WindDir9am, WindDir3pm, Raintoday, Location
    """

    puntos_cardinales = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W','NW']

    angulos = np.arange(0, 360, 45)
    df_angulos_cardinal = pd.DataFrame({'puntos': puntos_cardinales, 'angulos': angulos})
    
    dataset_ang = dataset.merge(df_angulos_cardinal, left_on='WindGustDir', right_on=['puntos'], how='left')
    dataset_ang = dataset.merge(df_angulos_cardinal, left_on='WindDir9am', right_on=['puntos'], how='left')
    dataset_ang = dataset.merge(df_angulos_cardinal, left_on='WindDir3pm', right_on=['puntos'], how='left')

    dataset_ang['sin_WindGustDir'] = round(np.sin(2 * np.pi * dataset_ang['angulos'] / 360), ndigits=4)
    dataset_ang['cos_WindGustDir'] = round(np.cos(2 * np.pi * dataset_ang['angulos'] / 360), ndigits=4)

    dataset_ang['sin_WindDir9am'] = round(np.sin(2 * np.pi * dataset_ang['angulos'] / 360), ndigits=4)
    dataset_ang['cos_WindDir9am'] = round(np.cos(2 * np.pi * dataset_ang['angulos'] / 360), ndigits=4)

    dataset_ang['sin_WindDir3pm'] = round(np.sin(2 * np.pi * dataset_ang['angulos'] / 360), ndigits=4)
    dataset_ang['cos_WindDir3pm'] = round(np.cos(2 * np.pi * dataset_ang['angulos'] / 360), ndigits=4)

        
    dataset_ang.drop('WindGustDir', axis=1, inplace=True)
    dataset_ang.drop('WindDir9am', axis=1, inplace=True)
    dataset_ang.drop('WindDir3pm', axis=1, inplace=True)

    dataset_ang.drop('angulos', axis=1, inplace=True)
    dataset_ang.drop('puntos', axis=1, inplace=True)
    
    dataset_codificado = pd.get_dummies(dataset_ang, columns=['Location'], drop_first=True)
    dataset_codificado['RainToday_Yes'] = dataset_codificado['RainToday_Yes'].astype(int)
    return dataset_codificado



def post_imputado(dataset_num, dataset_cat):
    dataset_cat.drop('week_year', axis=1,inplace=True)
    dataset_num.reset_index(drop=True,inplace=True)
    dataset_cat.reset_index(drop=True,inplace=True)
    dataset_merge = pd.merge(dataset_num, dataset_cat,left_index=True, right_index=True, how='left')
    dataset_merge = pd.get_dummies(dataset_merge,columns=['RainToday'], drop_first=True)
    dataset_merge = codificar_variables(dataset_merge)
    
    return dataset_merge



escalador = joblib.load('/app/escalador.pkl')
imputador_categorico = joblib.load('/app/imputador_categorico.pkl')
imp_mean = joblib.load('/app/imputador_numerico.pkl')
nn = joblib.load('/app/red_neuronal.pkl')

def aplicar_todo(dataset):
    dataset = modif(dataset)
    dataset[num_vars] = escalador.transform(dataset[num_vars])
    dataset_cat = imputador_categorico.transform(dataset[categ_vars])
    dataset = codificar_fecha(dataset)
    dataset_imp = imp_mean.transform(dataset[num_vars_date])
    dataset_imp = positivisar(dataset_imp)
    dataset_tot = post_imputado(dataset_imp, dataset_cat)

    return dataset_tot


input_file = '/app/files/input.csv'
output_file = '/app/files/output.csv'


input = pd.read_csv(input_file)

input_modificado = aplicar_todo(input)

prediccion = nn.predict(input_modificado)

prediccion = (prediccion > 0.49)*1

prediccion_df = pd.DataFrame(prediccion)

prediccion_df.to_csv(output_file, index=False)

