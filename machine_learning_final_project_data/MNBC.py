#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import sklearn
import scipy
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# In[9]:


df_X = pd.read_csv('./feature_select/Feature_select_3/BEST_FEATURES/BEST5.csv', header=None, index_col=False)

df_Y = pd.read_csv('./feature_select/Feature_select_3/TARGET/TARGET.csv',header=None, index_col=False)

df_X = df_X.drop(df_X.index[0])


# In[10]:


# print(type(df_X))
# print(df_X.shape)
# print(type(df_Y))
# print(df_Y.shape)


# In[11]:


X = df_X.values
y = df_Y.values
# print(X.shape)
# print(y.shape)


# In[12]:


total=0
for x in range(0, 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# print(type(X_train))
# print(type(X_test))
# print(type(y_train))
# print(type(y_test))
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

    clf = MultinomialNB()
    y_pred =clf.fit(X_train, y_train.ravel()).predict(X_train)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))

    s = clf.score(X_test, y_test)
    total= s+total
    final=total/100


# In[13]:


print("The final socre:{}".format(final))


# In[ ]:




