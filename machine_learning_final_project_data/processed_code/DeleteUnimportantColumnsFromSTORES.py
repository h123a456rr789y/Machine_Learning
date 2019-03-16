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


df = pd.read_csv('STORES.csv')


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


df = df.drop(['GROC09'], axis=1)


# In[9]:


df = df.drop(['GROC14'], axis=1)


# In[10]:


df = df.drop(['PCH_GROC_09_14'], axis=1)


# In[11]:


df = df.drop(['PCH_GROCPTH_09_14'], axis=1)


# In[12]:


df = df.drop(['SUPERC09'], axis=1)


# In[13]:


df = df.drop(['SUPERC14'], axis=1)


# In[14]:


df = df.drop(['PCH_SUPERC_09_14'], axis=1)


# In[15]:


df = df.drop(['PCH_SUPERCPTH_09_14'], axis=1)


# In[16]:


df = df.drop(['CONVS09'], axis=1)


# In[17]:


df = df.drop(['CONVS14'], axis=1)


# In[18]:


df = df.drop(['PCH_CONVS_09_14'], axis=1)


# In[19]:


df = df.drop(['PCH_CONVSPTH_09_14'], axis=1)


# In[20]:


df = df.drop(['SPECS09'], axis=1)


# In[21]:


df = df.drop(['SPECS14'], axis=1)


# In[22]:


df = df.drop(['PCH_SPECS_09_14'], axis=1)


# In[23]:


df = df.drop(['PCH_SPECSPTH_09_14'], axis=1)


# In[24]:


df = df.drop(['SNAPS12'], axis=1)


# In[25]:


df = df.drop(['SNAPS16'], axis=1)


# In[26]:


df = df.drop(['PCH_SNAPS_12_16'], axis=1)


# In[27]:


df = df.drop(['PCH_SNAPSPTH_12_16'], axis=1)


# In[28]:


df = df.drop(['WICS08'], axis=1)


# In[29]:


df = df.drop(['WICS12'], axis=1)


# In[30]:


df = df.drop(['PCH_WICS_08_12'], axis=1)


# In[31]:


df = df.drop(['PCH_WICSPTH_08_12'], axis=1)


# In[32]:


print(df.shape)


# In[33]:


list(df)


# In[34]:


'''
FIPS : int
GROCPTH09 : float
GROCPTH14 : float
SUPERCPTH09 : float
SUPERCPTH14 : float
CONVSPTH09 : float
CONVSPTH14 : float
SPECSPTH09 : float
SPECSPTH14 : float
SNAPSPTH12 : float
SNAPSPTH16 : float
WICSPTH08 : float
WICSPTH12 : float
'''


# In[35]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[36]:


for i in range(13):
    foolist[i] = df[df.columns[i]].tolist()


# In[38]:


for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[39]:


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


# In[40]:


df['GROCPTH09'] = se1.values
df['GROCPTH14'] = se2.values
df['SUPERCPTH09'] = se3.values
df['SUPERCPTH14'] = se4.values
df['CONVSPTH09'] = se5.values
df['CONVSPTH14'] = se6.values
df['SPECSPTH09'] = se7.values
df['SPECSPTH14'] = se8.values
df['SNAPSPTH12'] = se9.values
df['SNAPSPTH16'] = se10.values
df['WICSPTH08'] = se11.values
df['WICSPTH12'] = se12.values


# In[41]:


df.to_csv('PROCESSED_STORES.csv', index=False)


# In[ ]:




