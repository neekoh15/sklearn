"""
@author: Nicolas Martinez

Exercise:
Given a dataset that relates engine power to fuel efficiency, use linear regression to predict the fuel efficiency for an engine with 150 HP.
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
df = pd.read_csv('../datasets/AutoMPG/auto-mpg.data', sep=r'\s+', header=None, names=names)

# Replace missing values and drop rows with missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Create the linear regression model
model = LinearRegression()

# Extract the features (horsepower) and target (fuel consumption)
horsepower = np.array([[float(x)] for x in df['horsepower']])
fuel_consumption = np.array([float(x) for x in df['mpg']], dtype=float)

# Fit the model
model.fit(horsepower, fuel_consumption)

# Create a range of horsepower values for predictions
horsepower_range = np.array([[x] for x in range(45, 200)])

# Predict fuel consumption for the range of horsepower values
predictions = model.predict(horsepower_range)

# Plot the training data, predictions curve, and target point
plt.figure(dpi=300)
plt.scatter(horsepower, fuel_consumption, label='Training Data', marker='x', c='r', alpha=0.5)
plt.plot(horsepower_range, predictions, label='Predictions Curve', c='black', ls='dashed')
plt.scatter([150], model.predict([[150]]), label='Target (150 HP)', marker='s')
plt.legend()
plt.xlabel('Horsepower')
plt.ylabel('Fuel Consumption (mpg)')
plt.title('Linear Regression for Fuel Efficiency Prediction')
plt.show()
