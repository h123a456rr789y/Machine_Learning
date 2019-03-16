#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import scipy
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

# self defined function
import def_function as foo


# In[22]:


df = pd.read_csv('INSECURITY.csv')


# In[23]:


print(df.shape)


# In[24]:


list(df)


# In[25]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[26]:


df = df.drop(['State'], axis=1)


# In[27]:


df = df.drop(['County'], axis=1)


# In[28]:


df = df.drop(['CH_FOODINSEC_12_15'], axis=1)


# In[29]:


df = df.drop(['CH_VLFOODSEC_12_15'], axis=1)


# In[30]:


print(df.shape)


# In[31]:


list(df)


# In[32]:


'''
FIPS : int
FOODINSEC_10_12 : 
FOODINSEC_13_15 : 
VLFOODSEC_10_12 : 
VLFOODSEC_13_15 : 
FOODINSEC_CHILD_01_07 : 
FOODINSEC_CHILD_03_11 : 
'''


# In[33]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[34]:


for i in range(7):
    foolist[i] = df[df.columns[i]].tolist()


# In[35]:


for i in [1, 2, 3, 4, 5, 6]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[36]:


se1 = pd.Series(foolist[1])
se2 = pd.Series(foolist[2])
se3 = pd.Series(foolist[3])
se4 = pd.Series(foolist[4])
se5 = pd.Series(foolist[5])
se6 = pd.Series(foolist[6])


# In[37]:


df['FOODINSEC_10_12'] = se1.values
df['FOODINSEC_13_15'] = se2.values
df['VLFOODSEC_10_12'] = se3.values
df['VLFOODSEC_13_15'] = se4.values
df['FOODINSEC_CHILD_01_07'] = se5.values
df['FOODINSEC_CHILD_03_11'] = se6.values


# In[38]:


df.to_csv('PROCESSED_INSECURITY.csv', index=False)


# In[ ]:




