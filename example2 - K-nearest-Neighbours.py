# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:54:09 2023

@author: Nicolas Martinez

Ejercicios de clasificacion
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data[:,:3]
y = iris.target


#separo datos de entrenamiento, y datos de testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#cargo el escalador
scaler = StandardScaler()

X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train2, y_train)

dtc = DecisionTreeClassifier()
dtc.fit(X_train2, y_train)

y_pred_DTC = dtc.predict(X_test2)
y_pred_KNN = knn.predict(X_test2)

print(y_pred_DTC)
print(y_pred_KNN)
accuracy = accuracy_score(y_test, y_pred_DTC)
print(f"Accuracy: {accuracy*100:.2f}%")


accuracy = accuracy_score(y_test, y_pred_KNN)
print(f"Accuracy: {accuracy*100:.2f}%")


colors = ['blue', 'red', 'green']
marker = ['s', '^', 'o']
plt.figure(dpi=300, figsize=(3,3))
plt.axes(projection='3d')

for i,x in enumerate(X):
    plt.scatter(*x, color=colors[y[i]], marker=marker[y[i]], alpha=0.5)
plt.grid(False)

