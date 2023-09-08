# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:50:07 2023

@author: Martinez, Nicolas Agustin
"""


from sklearn.datasets import load_digits

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = load_digits()

X = data.data
y = data.target

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)

X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('accuracy: ', accuracy_score(y_true, y_pred))
print('report: \n', classification_report(y_true, y_pred))

