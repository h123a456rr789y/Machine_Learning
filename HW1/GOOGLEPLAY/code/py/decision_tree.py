import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score,recall_score
#from statistics import mode
#--read data--
data = pd.read_csv('../../processed_data/processed_googleplaystore_merged_final.csv',engine='python')
#--read data fin--
#--split data into feature/answer (X/Y)--
X = data.drop(data.columns[0],axis=1)
X = X.drop(data.columns[1],axis=1)
Y = data[data.columns[1]]
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
SUM_CONF=np.zeros((8,8), dtype=np.int).tolist()
for train,test in kf.split(X):
    result_array=[]
    X_train,X_test,Y_train,Y_test=X.values[train],X.values[test],Y.values[train],Y.values[test]
    classify=tree.DecisionTreeClassifier()
    classify.fit(X_train,Y_train)
    Y_pred=classify.predict(X_test)
    result_array=np.asarray(Y_pred)
    for i in range(len(result_array)):
        j=int(result_array[i])-1
        k=int(Y_test[i])-1
        SUM_CONF[j][k]+=1
#--K-fold validation fin--
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
print("$$KF+DT$$")
print("CM:")
print(np.asmatrix(SUM_CONF))
print("PS:")
print(TP/(TP+FP))
print("RC:")
print(TP/(TP+FN))
print("AC:")
print(TP/(TP+FN+FP))
#--pirnt result fin--

input()
