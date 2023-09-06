# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:14:31 2023

@author: 20373637824
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=names)

outCome = data['Outcome'].to_numpy()
data = data.drop('Outcome', axis=1).to_numpy()

Data_train, Data_test, outCome_train, outCome_test = train_test_split(data, outCome, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()

model.fit(Data_train, outCome_train)

data_predicted = model.predict(Data_test)

print(f'Accuracy: {accuracy_score(outCome_test, data_predicted)*100:.2f}%')

precision = precision_score(outCome_test, data_predicted)
recall = recall_score(outCome_test, data_predicted)

print("Precision:", precision)
print("Recall:", recall)

# For ROC curve
fpr, tpr, thresholds = roc_curve(outCome_test, data_predicted)
roc_auc = auc(fpr, tpr)

plt.figure(dpi=300)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
