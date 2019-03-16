import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score,recall_score
#from statistics import mode
#--read data--
data = pd.read_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/test_data/processed_googleplaystore_merged_final.csv',engine='python')
#--read data fin--
#--split data into feature/answer (X/Y)--
X_origin = data.drop(data.columns[0],axis=1)
X_origin = X_origin.drop(data.columns[1],axis=1)
Y = data[data.columns[1]]
#--split data into feature/answer (X/Y) fin--
MAX_AC=float(0)
for a in range(2,11):
    for b in range(a+1,11):
        for c in range(b+1,11):
            for d in range(c+1,11):
                for e in range(d+1,11):
                    X = X_origin.drop(data.columns[a],axis=1)
                    X = X_origin.drop(data.columns[b],axis=1)
                    X = X_origin.drop(data.columns[c],axis=1)
                    X = X_origin.drop(data.columns[d],axis=1)
                    X = X_origin.drop(data.columns[e],axis=1)
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
                    #print("$$KF+DT$$")
                    #print("CM:")
                    #print(np.asmatrix(SUM_CONF))
                    #print("PS:")
                    #print(TP/(TP+FP))
                    #print("RC:")
                    #print(TP/(TP+FN))
                    #print("AC:")
                    if (MAX_AC<(TP/(TP+FN+FP))):
                        print(a,b,c,d,e)
                        print(MAX_AC)
                        MAX_AC=(TP/(TP+FN+FP))
                    #--pirnt result fin--
                    #print(ct)
                    #--K-fold validation

print("fin")
