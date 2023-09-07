# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:29:52 2023

@author: Nicolas Martinez
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris = load_iris()

flores = iris.data
clase = iris.target

class NN:
    def __init__(self, model_params, data, clase):
        n_neighbors, algorithm = model_params
        self.model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
        self.data = data
        self.clase = clase
        
        self.__format_data()
        
        self.__train_model()
        
    def __format_data(self):
        #skip formatting       
        self.__normalized_data = self.data 
        
    def __train_model(self):
        
        self.trained_model = self.model.fit(self.__normalized_data)
        
    def prediction(self, flower):
        
        distances, indexes =self.trained_model.kneighbors([flower])
        
        return pd.DataFrame( {
                'Flowers': indexes[0][1:],
                'Distances': distances[0][1:],
                'Class': [self.clase[i] for i in indexes[0][1:]]
                })
    
    
model = NN([5, 'brute'], flores, clase)

print(model.prediction(flores[10]))

"""
Output:
    
    Flowers  Distances  Class
 0      144   0.244949      2
 1      120   0.264575      2
 2      143   0.346410      2
 3      112   0.346410      2
 
"""

        
        
        
        
    