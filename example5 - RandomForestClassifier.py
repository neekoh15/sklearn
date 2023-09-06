# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:06:54 2023

@author: 20373637824
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')  # El archivo utiliza ';' como separador

# Convertimos la calificación en una clasificación binaria
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)  # 1: alta calidad, 0: baja calidad

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_predict))
print('classification report: ', classification_report(y_test, y_predict))

feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
