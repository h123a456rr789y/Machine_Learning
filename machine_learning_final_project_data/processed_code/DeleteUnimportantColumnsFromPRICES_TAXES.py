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


df = pd.read_csv('PRICES_TAXES.csv')


# In[3]:


# THE WHOLE DATA IS POTENTIALLY USEFUL, NO NEED TO DELETE ANY COLUMNS


# In[4]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[5]:


df = df.drop(['State'], axis=1)


# In[6]:


df = df.drop(['County'], axis=1)


# In[7]:


print(df.shape)


# In[8]:


list(df)


# In[9]:


'''
FIPS : int
MILK_PRICE10 : float
SODA_PRICE10 : float
MILK_SODA_PRICE10 : float
SODATAX_STORES14 : float
SODATAX_VENDM14 : float
CHIPSTAX_STORES14 : float
CHIPSTAX_VENDM14 : float
FOOD_TAX14 : float
'''


# In[10]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[11]:


for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    foolist[i] = df[df.columns[i]].tolist()


# In[14]:


for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[15]:


se1 = pd.Series(foolist[1])
se2 = pd.Series(foolist[2])
se3 = pd.Series(foolist[3])
se4 = pd.Series(foolist[4])
se5 = pd.Series(foolist[5])
se6 = pd.Series(foolist[6])
se7 = pd.Series(foolist[7])
se8 = pd.Series(foolist[8])


# In[16]:


df['MILK_PRICE10'] = se1.values
df['SODA_PRICE10'] = se2.values
df['MILK_SODA_PRICE10'] = se3.values
df['SODATAX_STORES14'] = se4.values
df['SODATAX_VENDM14'] = se5.values
df['CHIPSTAX_STORES14'] = se6.values
df['CHIPSTAX_VENDM14'] = se7.values
df['FOOD_TAX14'] = se8.values


# In[17]:


df.to_csv('PROCESSED_PRICES_TAXES.csv', index=False)


# In[ ]:




