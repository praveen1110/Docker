import pandas as pd
import sklearn
import warnings
import os
os.getcwd()
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score
df = pd.read_csv(r'/Users/praveen/Desktop/BankNote_Authentication.csv')

df.columns
X=df.drop('class',axis=1)
Y=df[['class']]
# Y=np.array(Y).reshape(-1,1)
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2)
model = RandomForestClassifier().fit(X_train,Y_train)
y_pred=model.predict(X_test)
accuracy_score(Y_test,y_pred)
confusion_matrix(Y_test,y_pred)
rfc=RandomForestClassifier().fit(X,Y)
# joblib.dump(model,'rfc.sav')
X_test.to_csv('X_test.csv')
Y_train.head()
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(rfc, pickle_out)