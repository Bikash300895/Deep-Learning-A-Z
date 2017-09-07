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

# Part 2: Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initializing the RNN
regressor = Sequential()
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

# adding the output layer
regressor.add(Dense(units=1))

# Compile the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, batch_size=32, epochs=200)


# Part 3 : Making the prediction
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

# Getting the predicted price
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))

predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualize the results

