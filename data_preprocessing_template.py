# Data Preprocessing

# importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")

# iloc can take the indexes of the column we want to extract
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# taking care of missing data
# Imputer class will help us take care of missing data
# we also need an object for the class
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the dataset into the TRaining set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# we don't need to fit test set, because we have already fitted train test
# and on that note,x_train and x_test will be scaled on same basis
X_test = sc_X.transform(X_test)


















