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
data = pd.read_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/IRIS/processed_data/iris.csv',engine='python',header=None)
#--read data fin--
'''
X = data.drop(data.columns[4],axis=1)
Y = data[data.columns[4]]

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.20)

classify = tree.DecisionTreeClassifier()
classify.fit(X_train,Y_train)

Y_pred=classify.predict(X_test)
#print(np.array(list(Y_test.to_csv(index=False)),dtype=int))
print(np.array(list(str(Y_test.to_csv(index=False)).replace('\n','')),dtype=int))
print(Y_pred)
'''
'''
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
'''
'''
dot_data=tree.export_graphviz(classify,out_file=None)
(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png('mytree.png')
'''
#--split data into feature/answer (X/Y) and train and test--
X = data.drop(data.columns[4],axis=1)
Y = data[data.columns[4]]
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.20)
#--split data into feature/answer (X/Y) and train and test fin--
#--Confusion matrix--
SUM_CONF=[[0,0,0],[0,0,0],[0,0,0]]
result_array=np.empty((6,30,))
result_array=result_array.astype(int)
ct=0
for i in range(0,4):
    for j in range(i+1,4):
        X_train_split = X_train.drop(data.columns[i],axis=1)
        X_train_split = X_train_split.drop(data.columns[j],axis=1)
        X_test_split = X_test.drop(data.columns[i],axis=1)
        X_test_split = X_test_split.drop(data.columns[j],axis=1)
        classify = tree.DecisionTreeClassifier()
        classify.fit(X_train_split,Y_train)
        Y_pred=classify.predict(X_test_split)
        #store result into array
        result_array[ct]=np.asarray(Y_pred)
        ct=ct+1
        #SUM_CONF+=confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix:")
#print(SUM_CONF)
result_matrix=(np.asmatrix(result_array)).transpose()
#print(result_matrix)
trained_output=np.zeros(30)
for i in range (0,30):
    temp=np.bincount(np.asarray(result_matrix[i]).flatten())
    trained_output[i]=np.argmax(temp)
for i in range (0,30):
    j=int(trained_output[i])-1	#use the type(1,2,3) as column/row type
    k=int(Y_test.values[i])-1
    SUM_CONF[j][k]=SUM_CONF[j][k]+1
print (np.asmatrix(SUM_CONF))
#--Confusion matrix fin--
#--Resubstitution Validation--
result_array=np.empty((6,150,))
result_array=result_array.astype(int)
ct=0
SUM_RV_AC=0
for i in range(0,4):
    for j in range(i+1,4):
        X_train_split = X.drop(data.columns[i],axis=1)
        X_train_split = X_train_split.drop(data.columns[j],axis=1)
        classify = tree.DecisionTreeClassifier()
        classify.fit(X_train_split,Y)
        Y_pred=classify.predict(X_train_split)
        result_array[ct]=np.asarray(Y_pred)
        ct=ct+1
        #SUM_RV_AC+=accuracy_score(Y,Y_pred)
        '''
        dot_data=tree.export_graphviz(classify,out_file=None)
        (graph,) = pydot.graph_from_dot_data(dot_data)
        graph.write_png('mytree'+str(i)+str(j)+'.png')
        '''
        #print(SUM_RV_AC)
        #print(np.array(list(str(Y.to_csv(index=False)).replace('\n','')),dtype=int))
        #print(Y_pred)
print("Resubstitution Validation:")
#print((SUM_RV_AC/6)*100,"%")
result_matrix=(np.asmatrix(result_array)).transpose()
trained_output=np.zeros(150)
for i in range (0,150):
    temp=np.bincount(np.asarray(result_matrix[i]).flatten())
    trained_output[i]=np.argmax(temp)
right=0
for i in range (0,150):
    if int(trained_output[i])==int(Y.values[i]):
        right=right+1
print((right/150)*100,"%")
#--Resubstitution Validation fin--
#--K-fold cross validation--
#SUM_K_FOLD=0
split=10
kf=model_selection.KFold(n_splits=split)
right=0
ans_ct=int(150/split)
for train,test in kf.split(X):
    '''
    train_X=train.drop(data.columnns[4],axis=1)
    train_y=train[data.column[4]]
    test_X=test.drop(data.columns[4],axis=1)
    test_Y=test[data.column[4]]
    '''
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
            #print(Y_pred)
            result_array[ct]=np.asarray(Y_pred)
            ct=ct+1
            #SUM_K_FOLD+=accuracy_score(Y_test,Y_pred)
    result_matrix=(np.asmatrix(result_array)).transpose()
    trained_output=np.zeros(ans_ct)
    for i in range (0,ans_ct):
        temp=np.bincount(np.asarray(result_matrix[i]).flatten())
        trained_output[i]=np.argmax(temp)
    for i in range (0,ans_ct):
        if int(trained_output[i])==int(Y_test[i]):
            right=right+1
print("K-fold cross validation:")
#print((SUM_K_FOLD/(split*6))*100,"%")
print((right/150)*100,"%")
#--K-fold cross validation fin--
input()
