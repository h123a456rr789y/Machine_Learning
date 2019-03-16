import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import graphviz
import pydot
import os
from statistics import mode
#--read data--
data = pd.read_csv('../processed_data/iris.csv',engine='python',header=None)
#--read data fin--

#--split data into feature/answer (X/Y)--
X = data.drop(data.columns[4],axis=1)
Y = data[data.columns[4]]
#--split data into feature/answer (X/Y) fin--

#--resubstitution validation--
SUM_CONF=[[0,0,0],[0,0,0],[0,0,0]]
result_array=np.empty((6,150,))
result_array=result_array.astype(int)
ct=0
#--random forest C(4,2)--
for i in range(0,4):
    for j in range(i+1,4):
        X_train = X.drop(data.columns[i],axis=1)
        X_train = X_train.drop(data.columns[j],axis=1)
        classify = tree.DecisionTreeClassifier()
        classify.fit(X_train,Y)
        Y_pred=classify.predict(X_train)
        #store result into array
        result_array[ct]=np.asarray(Y_pred)
        ct=ct+1
#--random forest C(4,2) fin--
        
#--collect data (vote)--
result_matrix=(np.asmatrix(result_array)).transpose()
trained_output=np.zeros(150)
for i in range (0,150):
    temp=np.bincount(np.asarray(result_matrix[i]).flatten())
    trained_output[i]=np.argmax(temp)
for i in range (0,150):
    j=int(trained_output[i])-1	#use the type(1,2,3) as column/row type
    k=int(Y.values[i])-1
    SUM_CONF[j][k]=SUM_CONF[j][k]+1
#--collect data (vote) fin--
    
#--Count TP FP FN--
TP=SUM_CONF[0][0]+SUM_CONF[1][1]+SUM_CONF[2][2]
FP=SUM_CONF[0][1]+SUM_CONF[0][2]+SUM_CONF[1][2]
FN=SUM_CONF[1][0]+SUM_CONF[2][0]+SUM_CONF[2][1]
#--Count TP FP FN fin--
    
#--print result--
print("$$RV+RF$$")
print("CM:")
print (np.asmatrix(SUM_CONF))
print("PS:")
print(TP/(TP+FP))
print("RC:")
print(TP/(TP+FN))
print("AC:")
print(TP/(TP+FN+FP))
#--pirnt result fin--
#--resubstitution validation fin--

#--K-fold cross validation--
split=10
kf=model_selection.KFold(n_splits=split)
ans_ct=int(150/split)
SUM_CONF=[[0,0,0],[0,0,0],[0,0,0]]
for train,test in kf.split(X):
    result_array=np.empty((6,ans_ct,))
    result_array=result_array.astype(int)
    ct=0
    for i in range (0,4):
        for j in range(i+1,4):
            X_split=X.drop(data.columns[i],axis=1)
            X_split=X_split.drop(data.columns[j],axis=1)
            X_split_train,X_split_test,Y_train,Y_test=X_split.values[train],X_split.values[test],Y.values[train],Y.values[test]
            classify = tree.DecisionTreeClassifier()
            classify.fit(X_split_train,Y_train)
            Y_pred=classify.predict(X_split_test)
            result_array[ct]=np.asarray(Y_pred)
            ct=ct+1
            #SUM_K_FOLD+=accuracy_score(Y_test,Y_pred)
    result_matrix=(np.asmatrix(result_array)).transpose()
    trained_output=np.zeros(ans_ct)
    for i in range (0,ans_ct):
        temp=np.bincount(np.asarray(result_matrix[i]).flatten())
        trained_output[i]=np.argmax(temp)
    for i in range (0,ans_ct):
        j=int(trained_output[i])-1	#use the type(1,2,3) as column/row type
        k=int(Y_test[i])-1
        SUM_CONF[j][k]=SUM_CONF[j][k]+1
#--Count TP FP FN--
TP=SUM_CONF[0][0]+SUM_CONF[1][1]+SUM_CONF[2][2]
FP=SUM_CONF[0][1]+SUM_CONF[0][2]+SUM_CONF[1][2]
FN=SUM_CONF[1][0]+SUM_CONF[2][0]+SUM_CONF[2][1]
#--Count TP FP FN fin--
    
#--print result--
print("$$KF+RF$$")
print("CM:")
print (np.asmatrix(SUM_CONF))
print("PS:")
print(TP/(TP+FP))
print("RC:")
print(TP/(TP+FN))
print("AC:")
print(TP/(TP+FN+FP))
#--pirnt result fin--
#--K-fold cross validation fin--
input()