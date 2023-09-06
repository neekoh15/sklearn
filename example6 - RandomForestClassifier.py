# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:41:44 2023

@author: 20373637824
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#cargar los datos del modelo
iris = load_iris()

#previsualizar los datos
#print(iris.data[:1000])
X = iris.data
y = iris.target

#cargar el modelo
model = RandomForestClassifier(n_estimators=100, max_depth=5)

#dividir los datos de entrenamiento y los de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#entrenar el modelo
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(f'accuracy: {accuracy_score(y_test, y_pred)*100:2}%')
print(f'report: {classification_report(y_test, y_pred)}')
print(f'confusion matrix: \n{confusion_matrix(y_test, y_pred)}')
