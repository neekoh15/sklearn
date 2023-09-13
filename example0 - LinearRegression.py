# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 07:53:47 2023
Basic Regression Example
@author: Nicolas Martinez
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#datos de entrada: [2D ARRAY]
X = np.array([[5],[6],[7],[8]])

#datos de salida: [1D ARRAY]
Y = np.array([60, 70, 75, 85])

#setear el tipo de modelo
model = LinearRegression()
#entrenar el modelo
model.fit(X,Y)


plt.figure(dpi=300)
plt.scatter(X,Y, label='data', marker='x', color='red')
plt.plot([x for x in range(10)], [model.predict([[x]])[0] for x in range(10)], label='Regression')
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title('Basic implementation of linear regression')
plt.legend()
plt.grid()
plt.show()




