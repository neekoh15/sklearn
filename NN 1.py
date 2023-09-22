from sklearn.datasets import load_wine
from sklearn.neighbors import NearestNeighbors
import pandas as pd

data = load_wine()
wines = data.data
classes = data.target

class NN:
    def __init__(self, wines, classes) -> None:

        self.model = NearestNeighbors(n_neighbors=5, algorithm='brute')
        self.wines = wines
        self.classes = classes

        self.__train_model()

    def __train_model(self):
        self.model = self.model.fit(self.wines)

    def get_nearests(self, wine):
        distances, indexes = self.model.kneighbors([wine])

        dataset = pd.DataFrame({
            'index': indexes[0][1:],
            'similitud': (1-1/distances[0][1:])*100,
            'type': [self.classes[i] for i in indexes[0][1:]],
        })

        return dataset
    
model = NN(wines, classes)
print(model.get_nearests(wines[0]))

