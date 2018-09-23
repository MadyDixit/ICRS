import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:12].values
y = dataset.iloc[:, 12].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

import matplotlib.pyplot 
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model)
rfe = rfe.fit(X, y)
print(rfe.support_)
print(rfe.ranking_)


X = X[:,[0,1,3,4,5,6,7,8]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 8))

classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)



'''x = ['agra','alwar','Male',42,1,0,1,0]
if((ord(x[0][0])>=97 and ord(x[0][0])<=107) or(ord(x[0][0])>=65 and ord(x[0][0])<=75)):
    x[0]=0
    x[1]=0
else:
    x[0]=1
    x[1]=0

if(x[2][0]=='M' or x[2][0]=='m'):
    x[2]=1
else:
    x[2]=0
print(x)
from numpy import array
z = [q,0,0,42,2,1,1,1]
z = [z]
a = array(z).reshape(8,1)
'''
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
y_pred = classifier.predict(X_test)

'''if(y_pred>=.60 and y_pred<=.80):
    print("Congratulation! you got 5% discount on meal")
elif(y_pred>=.80 and y_pred<=.90):
    print("Congratulation! you got 7% discount on meal")
else:
    print("Thanks for Booking => Enjoy your journey and Stay tuned")
'''


'''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
y=dict()
m = []
z=0
xa = list(dataset.iloc[:, 6])
for i in range(1000):
    if xa[i] > 45:
        y[xa[i]]=y_pred[z]
    z=z+1
    











    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

print(y)
import matplotlib.pyplot as plt
plt.scatter(y.keys(),y.values())
