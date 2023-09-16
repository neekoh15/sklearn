"""
ML basic algorithms implemented atm:
"""
#Dummy data for model testing only
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Utilities (data splitting, scaler for NN model and metrics)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class KNN:
    
    def __init__(self, data):
        
        self.model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        self.X = data.data
        self.y = data.target
        
        self.__split_data()
        self.__preprocesing()        
        self.__train_model()
    
    def __split_data(self):    
        #split data 
        self.X_train, self.X_true, self.y_train, self.y_true = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        
    def __preprocessing(self):
        #preprocesing data
        
        scaler = StandardScaler()
        self.X_train = scaler.fit(self.X_train)
        self.X_true = scaler.transform(self.X_true)
    
    def __train_model(self):
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, value):
        
        return self.model.predict(value)

    def evaluate_model(self):    
    
        #predicciones:
        y_pred = self.model.predict(self.X_true)
        
        return {
            'accuracy': accuracy_score(self.y_true, y_pred),
            'report': classification_report(self.y_true, y_pred)}

class NN:
    
    def __init__(self, data):
        
        self.model = NearestNeighbors(n_neighbors=5, algorithm='brute')
        self.X = data.data
        
        self.__preprocesing()
        
        self.__train_model()
        
    
    def __preprocesing(self):
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = scaler.transform(self.X)
    
    def __train_model(self):
        
        self.model.fit(self.X)
        
    def get_nn(self, value):
        
        distances, indexes = self.model.kneighbors(value)
        
        return {
            'indexes': indexes[0][1:],
            'distances': distances[0][1:]
            }
        
class DTC:
    
    def __init__(self, data):
        
        self.model = DecisionTreeClassifier()
        
        self.X = data.data
        self.y = data.target
        
        self.__preprocessing()
        self.__split_data()
        self.__train_model()
        
    def __preprocessing(self):
        pass
        
    def __split_data(self):
        
        self.X_train, self.X_test, self.y_train, self.y_true = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        
    def __train_model(self):
        
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, value):
        
        return self.model.predict([value])
    
    def evaluate_model(self):
        
        y_pred = self.model.predict(self.X_test)
        
        return {
            'accuracy': accuracy_score(self.y_true, y_pred),
            'report': classification_report(self.y_true, y_pred)
            }
    
class RFC:
    
    def __init__(self, data):
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=1)
        
        self.X = data.data
        self.y = data.target
        
        self.__split_data()
        self.__preprocessing()
        self.__train_model()
        
    def __split_data(self):
        
        self.X_train, self.X_test, self.y_train, self.y_true = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        
    def __preprocessing(self):
        pass
    
    def __train_model(self):
        
        self.model.fit(self.X_train, self.y_train)

    def predict(self, value):
        
        return self.model.predict([value])
    
    def evaluate_model(self):
        
        y_pred = self.model.predict(self.X_test)
        
        return {
            'accuracy': accuracy_score(self.y_true, y_pred),
            'report': classification_report(self.y_true, y_pred)
            'confusion matrix': confusion_matrix(self.y_true, y_pred)
            }

class LR:

    def __init__(self, data):
        
        self.model = LinearRegression()
        
        self.X = data.data
        self.y = data.target
        
        self.__train_model()
        
    def __train_model(self):
        
        self.model.fit(self.X, self.y)
        
    def predict(self, value):
        return self.model.predict([value])
    

#data a utilizar
dummy_data = load_iris()

#Modelos que he aprendido a usar hasta el momento:
model0 = LinearRegression()
model1 = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
model2 = NearestNeighbors(n_neighbors=3, algorithm='brute')
model3 = DecisionTreeClassifier(random_state=1)
model4 = RandomForestClassifier(n_estimators=100, random_state=1)
