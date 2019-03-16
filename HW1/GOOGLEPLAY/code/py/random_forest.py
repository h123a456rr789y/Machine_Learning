import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

#--read data--
data = pd.read_csv('../../processed_data/processed_googleplaystore_merged_final.csv',engine='python')
#--read data fin--

#--split data into feature/answer (X/Y)--
X = data.drop(data.columns[0],axis=1)
X = X.drop(data.columns[1],axis=1)
Y = data[data.columns[1]]
#--split data into feature/answer (X/Y) fin--
#--resubstitution validation--
SUM_CONF=np.zeros((8,8), dtype=np.int).tolist()
result_array=[]
#--random forest C(4,2)--
for i in range(2,7):
    for j in range(i+1,7):
        X_train = X.drop(data.columns[i],axis=1)
        X_train = X_train.drop(data.columns[j],axis=1)
        classify = tree.DecisionTreeClassifier()
        classify.fit(X_train,Y)
        Y_pred=classify.predict(X_train)
        #store result into array
        result_array.append(np.asarray(Y_pred))
#--random forest C(4,2) fin--
#--collect data (vote)--
result_matrix=(np.asmatrix(result_array)).transpose()
length=len(result_matrix)
trained_output=np.zeros(length)
for i in range (0,length):
    temp=np.bincount(np.asarray(result_matrix[i]).flatten())
    trained_output[i]=np.argmax(temp)
for i in range (0,length):
    j=int(trained_output[i])-1	#use the type(1,2,3) as column/row type
    k=int(Y.values[i])-1
    SUM_CONF[j][k]=SUM_CONF[j][k]+1
#--collect data (vote) fin--
#--Count TP FP FN--
TP=0
FP=0
FN=0
for i in range(0,8):
    for j in range(0,8):
        if(i==j):
            TP+=SUM_CONF[i][j]
        elif(i<j):
            FP+=SUM_CONF[i][j]
        elif(i>j):
            FN+=SUM_CONF[i][j]
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
SUM_CONF=np.zeros((8,8), dtype=np.int).tolist()
for train,test in kf.split(X):
    result_array=[]
    for i in range (2,7):
        for j in range(i+1,7):
            X_split=X.drop(data.columns[i],axis=1)
            X_split=X_split.drop(data.columns[j],axis=1)
            X_split_train,X_split_test,Y_train,Y_test=X_split.values[train],X_split.values[test],Y.values[train],Y.values[test]
            classify = tree.DecisionTreeClassifier()
            classify.fit(X_split_train,Y_train)
            Y_pred=classify.predict(X_split_test)
            result_array.append(np.asarray(Y_pred))
    result_matrix=(np.asmatrix(result_array)).transpose()
    length=len(result_matrix)
    trained_output=np.zeros(length)
    for i in range (0,length):
        temp=np.bincount(np.asarray(result_matrix[i]).flatten())
        trained_output[i]=np.argmax(temp)
    for i in range (0,length):
        j=int(trained_output[i])-1	#use the type(1,2,3) as column/row type
        k=int(Y_test[i])-1
        SUM_CONF[j][k]=SUM_CONF[j][k]+1
#--Count TP FP FN--
TP=0
FP=0
FN=0
for i in range(0,8):
    for j in range(0,8):
        if(i==j):
            TP+=SUM_CONF[i][j]
        elif(i<j):
            FP+=SUM_CONF[i][j]
        elif(i>j):
            FN+=SUM_CONF[i][j]
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
