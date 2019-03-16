#!/usr/bin/env python
# coding: utf-8

# In[111]:


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


# In[112]:


df = pd.read_csv('SOCIOECONOMIC.csv')


# In[113]:


print(df.shape)


# In[114]:


list(df)


# In[115]:


# THE WHOLE DATA IS POTENTIALLY USEFUL, NO NEED TO DELETE ANY COLUMNS


# In[116]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[117]:


df = df.drop(['State'], axis=1)


# In[118]:


df = df.drop(['County'], axis=1)


# In[119]:


print(df.shape)


# In[120]:


list(df)


# In[121]:


'''
FIPS : int
PCT_NHWHITE10 : float
PCT_NHBLACK10 : float
PCT_HISP10 : float
PCT_NHASIAN10 : float
PCT_NHNA10 : float
PCT_NHPI10 : float
PCT_65OLDER10 : float
PCT_18YOUNGER10 : float
MEDHHINC15 : float
POVRATE15 : float
PERPOV10 : bool(0 or 1)
CHILDPOVRATE15 : float
PERCHLDPOV10 : bool(0 or 1)
METRO13 : bool(0 or 1)
POPLOSS10 : bool(0 or 1)
'''


# In[122]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[123]:


for i in range(16):
    foolist[i] = df[df.columns[i]].tolist()


# In[126]:


for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[137]:


for i in [11, 13, 14, 15]:
    foo.bool_check(foolist[i])
    #foo.bool_filled(foolist[i])


# In[138]:


se1 = pd.Series(foolist[1])
se2 = pd.Series(foolist[2])
se3 = pd.Series(foolist[3])
se4 = pd.Series(foolist[4])
se5 = pd.Series(foolist[5])
se6 = pd.Series(foolist[6])
se7 = pd.Series(foolist[7])
se8 = pd.Series(foolist[8])
se9 = pd.Series(foolist[9])
se10 = pd.Series(foolist[10])
se11 = pd.Series(foolist[11])
se12 = pd.Series(foolist[12])
se13 = pd.Series(foolist[13])
se14 = pd.Series(foolist[14])
se15 = pd.Series(foolist[15])


# In[139]:


df['PCT_NHWHITE10'] = se1.values
df['PCT_NHBLACK10'] = se2.values
df['PCT_HISP10'] = se3.values
df['PCT_NHASIAN10'] = se4.values
df['PCT_NHNA10'] = se5.values
df['PCT_NHPI10'] = se6.values
df['PCT_65OLDER10'] = se7.values
df['PCT_18YOUNGER10'] = se8.values
df['MEDHHINC15'] = se9.values
df['POVRATE15'] = se10.values
df['PERPOV10'] = se11.values
df['CHILDPOVRATE15'] = se12.values
df['PERCHLDPOV10'] = se13.values
df['METRO13'] = se14.values
df['POPLOSS10'] = se15.values


# In[141]:


df.to_csv('PROCESSED_SOCIOECONOMIC.csv', index=False)


# In[ ]:




