# Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_excel('data.xlsx')

# Getting the inputs and output
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Creating the Training Set and the Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Building the model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)


# Inference

# Making the predictions of the data points in the test set
y_pred = model.predict(X_test)

# Making the prediction of a single data point with AT = 15, V = 40, AP = 1000, RH = 75
model.predict([[15,40,1000,75]])


# Evaluating the model

# R-Squared
r2 = r2_score(y_test, y_pred)

# Adjusted R-Squared
k = X_test.shape[1]
n = X_test.shape[0]
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)