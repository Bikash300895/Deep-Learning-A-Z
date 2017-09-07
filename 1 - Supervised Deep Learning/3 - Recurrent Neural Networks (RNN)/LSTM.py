import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1: Data preprocessing

# import the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

# Feature Scalling
from sklearn.preprocessing import MinMaxScaler  # Normalize the dataset (x - Xmin)/(Xmax - Xmin)
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshape
X_train = np.reshape(X_train, (1257, 1, 1))