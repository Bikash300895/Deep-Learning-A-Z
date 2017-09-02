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