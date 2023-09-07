# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:17:04 2023

@author: Nicolas Martinez
"""

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class NN:
    def __init__(self, model, path_data):
        
        self.model = model
        try:
            print('cargando los datos..')
            self.data = pd.read_csv(path_data, encoding='utf-8' )
        except Exception as e:
            print(e)
            raise ValueError()
            
        
    def data_formatting(self):
        print('formateando data..')
        #cargar los datos y dar formato a los datos:
        self.data = self.data.drop(self.data.columns[0], axis=1)
        
        #normalizar la data:
        self.normalized_data = StandardScaler().fit_transform(self.data)
        
    def train_model(self):
        print('entrenando el modelo..')
        self.trained_model = self.model.fit(self.normalized_data)
        
    def get_neareast_neighbours(self, song):
        song = [self.data.iloc[song]]
        print('buscando los vecinos mas cercanos..')
        distances, indexes = self.trained_model.kneighbors(song)
        
        nn = pd.DataFrame({
            'Songs': indexes[0][1:],
            'Distances': distances[0][1:]
            })
        
        return nn
        
path = '../YearPredictionMSD.csv'       

model = NN(NearestNeighbors(n_neighbors=5, algorithm='brute'), path)

model.data_formatting()
model.train_model()

while True:
    print('Ingresa el indice de una cancion para encontrar las mas parecidas entre si..')
    index = input('Cancion numero >> ')
    try:
        int(index)
    except ValueError as e:
        print(f'{e}.. \n')
    neighbors = model.get_neareast_neighbours(int(index))
    print(neighbors, '\n')