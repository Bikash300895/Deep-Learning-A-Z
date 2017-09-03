""" Step 1: Data preprocessing"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# lebel encoder for country
label_encoder_1 = LabelEncoder()
X[:, 1] = label_encoder_1.fit_transform(X[:, 1])


# label encoder for gender
label_encoder_2 = LabelEncoder()
X[:, 2] = label_encoder_2.fit_transform(X[:, 2])

# one hot encoder
one_hot = OneHotEncoder(categorical_features= [1])
X = one_hot.fit_transform(X).toarray()

# Removing one variable of one hot encoder to avoid dummy variable trup
X = X[:, 1:]


# Splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_trian, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



""" Step 2: Meking the deep learning model"""
import keras
from keras.models import Sequential
from keras.layers import Dense

# Lets start building the neural network
# 1. defining ther neural network itself
classifier = Sequential()

# 2. Add the first hidden layer
# as a rule of thumb we will take the dimension as (input_dim + output_dim)/2
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))

# Add second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


# 3. Compile the neural network
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Finally fit data train data to train the model
classifier.fit(X_train, y_trian, batch_size=10, epochs=100)

"""Step 3: Evaluating the model"""
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




"""" Step 4: Tuning the ANN """
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
     classifier = Sequential()
     classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))
     classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
     classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
     classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
     return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {
            'batch_size':[25, 32],
            'epochs': [100, 500],
            'optimizer' : ['adam', 'rmsprop']
        }
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_trian)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_









