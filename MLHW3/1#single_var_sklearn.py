
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("Concrete_Data.csv",engine = 'python');
columnNames = list(df.head(0))
length=len(df.index)
split_num=int(length*0.2)
df_feature=df.iloc[:,0]
df_target=df.iloc[:,8]
df_feature_train=df_feature[:-split_num].values.reshape(-1,1)
df_feature_test=df_feature[-split_num:].values.reshape(-1,1)
df_target_train=df_target[:-split_num].values.reshape(-1,1)
df_target_test=df_target[-split_num:].values.reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(df_feature_train,df_target_train)

predict_ans=regr.predict(df_feature_test)

print("weight:\n",regr.coef_)
print('bias: \n', regr.intercept_)
#print("MSE:%.2f"%mean_squared_error(df_target_test,predict_ans))
print("R2 SCORE:%.2f"%r2_score(df_target_test,predict_ans))

plt.scatter(df_feature_test, df_target_test,  color='black')
plt.plot(df_feature_test, predict_ans, color='blue', linewidth=3)

plt.title("Y="+str(regr.coef_[0][0])+"X+"+str(regr.intercept_[0]))
plt.xlabel(columnNames[0])
plt.ylabel(columnNames[8])
plt.axis('on')
plt.savefig('Q1/sklearn_1,9.png')
#plt.show()
input()
