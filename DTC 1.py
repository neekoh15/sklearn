from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

data = load_wine()

classes = data.target
wines = data.data


wines_train, wines_test, classes_train, classes_test = train_test_split(wines, classes, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(wines_train, classes_train)

classes_predicted = model.predict(wines_test)

accuracy = accuracy_score(classes_test, classes_predicted)*100
report = classification_report(classes_test, classes_predicted)
matrix = confusion_matrix(classes_test, classes_predicted)

print(f'accuracy: {accuracy:.2f}%')
print(report)

print(matrix)