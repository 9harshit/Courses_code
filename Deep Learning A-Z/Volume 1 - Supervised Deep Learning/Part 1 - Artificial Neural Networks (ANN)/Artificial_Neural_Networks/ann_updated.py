# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = pd.DataFrame(dataset.iloc[:, 3:13].values)
y = pd.DataFrame(dataset.iloc[:, 13].values)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[1] = labelencoder_X_1.fit_transform(X[1])
labelencoder_X_2 = LabelEncoder()
X[2] = labelencoder_X_2.fit_transform(X[2])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense 

# Initalizing layers of ANN
classifier = Sequential()

# Adding input and hidden layer
classifier.add(Dense(output_dim = 6, init = "uniform" , activation = 'relu' , input_dim = 11))

# Adding second hiddent layer
classifier.add(Dense(output_dim = 6, init = "uniform" , activation = 'relu' ))

# Adding Output layer
classifier.add(Dense(output_dim = 1, init = "uniform" , activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the Tr
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)