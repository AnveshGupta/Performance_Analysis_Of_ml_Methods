import numpy as np
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')
x= data.iloc[:,3:13].values
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le1 = LabelEncoder()
x[:,1] = le1.fit_transform(x[:,1])
le2 = LabelEncoder()
x[:,2]= le2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain,ytest = train_test_split(x,y,test_size = 0.2, random_state =0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform' , input_dim = 11))

classifier.add(Dense(units = 10, activation = 'relu', kernel_initializer = 'uniform' ))

classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform' ))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(xtrain , ytrain ,batch_size = 1 , epochs =10)

ypred = classifier.predict(xtest)
ypred = (ypred>.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest , ypred)