import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

# Feature Scalling
from sklearn.preprocessing import MinMaxScaler  # Normalize the dataset (x - Xmin)/(Xmax - Xmin)
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

