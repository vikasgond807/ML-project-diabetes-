import pandas as pd 
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Reading dataset from csv file 
dataset=pd.read_csv('diabetes.csv')
print(len(dataset))
#print(dataset.head())

# Labeled Data 
zero_not_accepted=['preg','glucose','plasma','pres','skin','test','mass','age']

# Replacing zeros with Mean values 
for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0,np.nan)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.nan,mean)
#print(dataset.head())

#spliting the dataset
X=dataset.iloc[:,0:8] # : all rows and 0:8 all columns except the 8th one i.e y
y=dataset.iloc[:,8]# the last column .i.e the output
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# Feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)


#using KNeighborsClassifier for building the model
clf=KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
clf.fit(X_train,y_train)


# Predict the test results 
y_pred=clf.predict(X_test)


# Evaluate the model
print(confusion_matrix(y_test,y_pred))
print("f1_score",f1_score(y_test,y_pred))
print("Accuracy score",accuracy_score(y_test,y_pred))
print("Model accuracy",clf.score(X_test,y_test))
