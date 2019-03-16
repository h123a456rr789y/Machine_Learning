#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv('HEALTH.csv')


# In[3]:


print(df.shape)


# In[4]:


list(df)


# In[5]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[6]:


df = df.drop(['State'], axis=1)


# In[7]:


df = df.drop(['County'], axis=1)


# In[8]:


df = df.drop(['RECFAC09'], axis=1)


# In[9]:


df = df.drop(['RECFAC14'], axis=1)


# In[10]:


df = df.drop(['PCH_RECFAC_09_14'], axis=1)


# In[11]:


df = df.drop(['PCH_RECFACPTH_09_14'], axis=1)


# In[12]:


print(df.shape)


# In[13]:


list(df)


# In[14]:


'''
FIPS : int
PCT_DIABETES_ADULTS08 : float
PCT_DIABETES_ADULTS13 : float
PCT_OBESE_ADULTS08 : float
PCT_OBESE_ADULTS13 : float
PCT_HSPA15 : float
RECFACPTH09 : float
RECFACPTH14 : float
'''


# In[15]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[16]:


for i in range(8):
    foolist[i] = df[df.columns[i]].tolist()


# In[17]:


for i in [1, 2, 3, 4, 5, 6, 7]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[18]:


se1 = pd.Series(foolist[1])
se2 = pd.Series(foolist[2])
se3 = pd.Series(foolist[3])
se4 = pd.Series(foolist[4])
se5 = pd.Series(foolist[5])
se6 = pd.Series(foolist[6])
se7 = pd.Series(foolist[7])


# In[19]:


df['PCT_DIABETES_ADULTS08'] = se1.values
df['PCT_DIABETES_ADULTS13'] = se2.values
df['PCT_OBESE_ADULTS08'] = se3.values
df['PCT_OBESE_ADULTS13'] = se4.values
df['PCT_HSPA15'] = se5.values
df['RECFACPTH09'] = se6.values
df['RECFACPTH14 '] = se7.values


# In[20]:


df.to_csv('PROCESSED_HEALTH.csv', index=False)


# In[ ]:





# In[ ]:




