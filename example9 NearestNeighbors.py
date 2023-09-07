# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:57:49 2023

@author: 20373637824
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

breast_cancer = load_breast_cancer()

#print(breast_cancer)

bc = pd.DataFrame(data= breast_cancer.data, columns=breast_cancer.feature_names)

#print(bc.head())

class NN:
    def __init__(self, data):
        
        self.model = NearestNeighbors(n_neighbors=5, algorithm='brute')
        self.data = data
        
        self.__train_model()
        
    def __format_data(self):
        pass
    
    def __train_model(self):
        
        self.trained_model = self.model.fit(self.data.values)
        
    def get_nearest(self, cancer):
        
        dd, indx = self.trained_model.kneighbors([cancer])

        return pd.DataFrame({
            
            'index': indx[0][1:],
            'distance': dd[0][1:]
            })


model = NN(bc)

print(model.get_nearest(bc.iloc[0]))

"""
Output:
    
    index    distance
 0    337  186.617630
 1    254  194.568813
 2     56  204.171305
 3     70  209.537125

"""
        
        

