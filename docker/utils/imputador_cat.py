import pandas as pd


class imputador_por_semana:

    def __init__(self, group_col: str, columnas_categoricas: list[str]):

        self.group_col = group_col
        self.columnas_categoricas = columnas_categoricas
        self.weekly_modes = {}


    def fit(self, df):

        # Iteramos sobre todas las columnas del dataset de fiteo, y guardamos la moda de cada columna por semana.
        for col in self.columnas_categoricas:
            self.weekly_modes[col] = df.groupby(self.group_col)[col].agg(lambda x: x.mode()[0]).to_dict()

        return self

    def transform(self, df):

        df_imputed = df.copy()
        # Una vez entrenado el modelo, tenemos un diccionario de clave: variable, valor: lista de modas por semana
        for col in self.columnas_categoricas:
            weekly_modes = self.weekly_modes[col]
            df_imputed[col] = df_imputed.apply(
                lambda row: weekly_modes[row[self.group_col]] if pd.isnull(row[col]) else row[col],
                axis=1
            )

        return df_imputed


    def fit_transform(self, df):
        
        self.fit(df)
        return self.transform(df)