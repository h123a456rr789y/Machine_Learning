import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score,recall_score
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
#--build tree
classify=tree.DecisionTreeClassifier()
classify.fit(X,Y)
Y_pred=classify.predict(X)
#--build tree fin--
#--print result--
print("$$RV+DT$$")
print("CM:")
print(confusion_matrix(Y,Y_pred))
print("PS:")
print(precision_score(Y,Y_pred,average='micro'))
print("RC:")
print(recall_score(Y,Y_pred,average='micro'))
print("AC:")
print(accuracy_score(Y,Y_pred))
#--print result fin--
#--resubstitution validation fin--
#--K-fold validation
split=10
kf=model_selection.KFold(n_splits=split)
SUM_CONF=[[0,0,0],[0,0,0],[0,0,0]]
for train,test in kf.split(X):
    X_train,X_test,Y_train,Y_test=X.values[train],X.values[test],Y.values[train],Y.values[test]
    classify=tree.DecisionTreeClassifier()
    classify.fit(X_train,Y_train)
    Y_pred=classify.predict(X_test)
    SUM_CONF+=confusion_matrix(Y_test,Y_pred)
#--K-fold validation fin--
#--Count TP FP FN--
    TP=SUM_CONF[0][0]+SUM_CONF[1][1]+SUM_CONF[2][2]
    FP=SUM_CONF[0][1]+SUM_CONF[0][2]+SUM_CONF[1][2]
    FN=SUM_CONF[1][0]+SUM_CONF[2][0]+SUM_CONF[2][1]
#--Count TP FP FN fin-- 
#--print result--
print("$$KF+DT$$")
print("CM:")
print(SUM_CONF)
print("PS:")
print(TP/(TP+FP))
print("RC:")
print(TP/(TP+FN))
print("AC:")
print(TP/(TP+FN+FP))
#--pirnt result fin--
input()
