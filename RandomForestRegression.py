# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:07:51 2023

@author: Martinez, Nicolas Agustin
"""

# Importando las bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np


# Cargar el conjunto de datos Boston
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]



# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
regr.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = regr.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
