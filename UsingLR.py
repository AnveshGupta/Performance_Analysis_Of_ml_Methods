import numpy as np
import matplotlib.pyplot as plt
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

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)


ypred = classifier.predict(xtest)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)