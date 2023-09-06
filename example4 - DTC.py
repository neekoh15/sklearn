# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:38:12 2023

@author: 20373637824
"""
#importo el modelo de ML
#import el separador de datos
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
import pandas as pd

#obtengo los datos:
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
data = pd.read_csv(url, header=None)

#filtro la data
data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)

#normalizar las variables
def normalize(col, x):
    try:
        return pd.to_numeric(x)
    except ValueError:
        return list(col.unique()).index(x)

for col in data.columns:
    data[col] = data[col].apply(lambda x: normalize(data[col], x))


y = data[len(data.columns)-1].to_numpy()
data.drop(data.columns[-1], axis=1, inplace=True)

X = data.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred_DTC = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_DTC)

print(f'accuracy: {accuracy*100:2}%')

