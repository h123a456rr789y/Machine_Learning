#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


df = pd.read_csv('RESTAURANTS.csv')


# In[6]:


print(df.shape)


# In[7]:


list(df)


# In[8]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[9]:


df = df.drop(['State'], axis=1)


# In[10]:


df = df.drop(['County'], axis=1)


# In[11]:


df = df.drop(['FFR09'], axis=1)


# In[12]:


df = df.drop(['FFR14'], axis=1)


# In[13]:


df = df.drop(['PCH_FFR_09_14'], axis=1)


# In[14]:


df = df.drop(['PCH_FFRPTH_09_14'], axis=1)


# In[15]:


df = df.drop(['FSR09'], axis=1)


# In[16]:


df = df.drop(['FSR14'], axis=1)


# In[17]:


df = df.drop(['PCH_FSR_09_14'], axis=1)


# In[18]:


df = df.drop(['FSRPTH09'], axis=1)


# In[19]:


df = df.drop(['FSRPTH14'], axis=1)


# In[20]:


df = df.drop(['PCH_FSRPTH_09_14'], axis=1)


# In[21]:


print(df.shape)


# In[22]:


list(df)


# In[23]:


'''
FIPS : int
FFRPTH09 : float
FFRPTH14 : float
PC_FFRSALES07 : float
PC_FFRSALES12 : float
PC_FSRSALES07 : float
PC_FSRSALES12 : float
'''


# In[24]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[25]:


for i in range(7):
    foolist[i] = df[df.columns[i]].tolist()


# In[27]:


for i in [1, 2, 3, 4, 5, 6]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[28]:


se1 = pd.Series(foolist[1])
se2 = pd.Series(foolist[2])
se3 = pd.Series(foolist[3])
se4 = pd.Series(foolist[4])
se5 = pd.Series(foolist[5])
se6 = pd.Series(foolist[6])


# In[29]:


df['FFRPTH09'] = se1.values
df['FFRPTH14'] = se2.values
df['PC_FFRSALES07'] = se3.values
df['PC_FFRSALES12'] = se4.values
df['PC_FSRSALES07'] = se5.values
df['PC_FSRSALES12'] = se6.values


# In[30]:


df.to_csv('PROCESSED_RESTAURANTS.csv', index=False)


# In[ ]:




