#!/usr/bin/env python
# coding: utf-8

# In[39]:


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


# In[40]:


df = pd.read_csv('ACCESS.csv')


# In[41]:


print(df.shape)


# In[42]:


#list(df)


# In[43]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[44]:


df = df.drop(['State'], axis=1)


# In[45]:


df = df.drop(['County'], axis=1)


# In[46]:


df = df.drop(['LACCESS_POP10'], axis=1)


# In[47]:


df = df.drop(['LACCESS_POP15'], axis=1)


# In[48]:


df = df.drop(['PCH_LACCESS_POP_10_15'], axis=1)


# In[49]:


df = df.drop(['LACCESS_LOWI10'], axis=1)


# In[50]:


df = df.drop(['LACCESS_LOWI15'], axis=1)


# In[51]:


df = df.drop(['PCH_LACCESS_LOWI_10_15'], axis=1)


# In[52]:


df = df.drop(['LACCESS_HHNV10'], axis=1)


# In[53]:


df = df.drop(['LACCESS_HHNV15'], axis=1)


# In[54]:


df = df.drop(['PCH_LACCESS_HHNV_10_15'], axis=1)


# In[55]:


df = df.drop(['LACCESS_SNAP15'], axis=1)


# In[56]:


df = df.drop(['LACCESS_CHILD10'], axis=1)


# In[57]:


df = df.drop(['LACCESS_CHILD15'], axis=1)


# In[58]:


df = df.drop(['LACCESS_CHILD_10_15'], axis=1)


# In[59]:


df = df.drop(['LACCESS_SENIORS10'], axis=1)


# In[60]:


df = df.drop(['LACCESS_SENIORS15'], axis=1)


# In[61]:


df = df.drop(['PCH_LACCESS_SENIORS_10_15'], axis=1)


# In[62]:


df = df.drop(['LACCESS_WHITE15'], axis=1)


# In[63]:


df = df.drop(['LACCESS_BLACK15'], axis=1)


# In[64]:


df = df.drop(['LACCESS_HISP15'], axis=1)


# In[65]:


df = df.drop(['LACCESS_NHASIAN15'], axis=1)


# In[66]:


df = df.drop(['LACCESS_NHNA15'], axis=1)


# In[67]:


df = df.drop(['LACCESS_NHPI15'], axis=1)


# In[68]:


df = df.drop(['LACCESS_MULTIR15'], axis=1)


# In[69]:


print(df.shape)


# In[70]:


list(df)


# In[71]:


'''
FIPS : int
PCT_LACCESS_POP10 : float
PCT_LACCESS_POP15 : float
PCT_LACCESS_LOWI10 : float
PCT_LACCESS_LOWI15 : float
PCT_LACCESS_HHNV10 : float
PCT_LACCESS_HHNV15 : float
PCT_LACCESS_SNAP15 : float 
PCT_LACCESS_CHILD10 : float
PCT_LACCESS_CHILD15 : float
PCT_LACCESS_SENIORS10 : float
PCT_LACCESS_SENIORS15 : float
PCT_LACCESS_WHITE15 : float
PCT_LACCESS_BLACK15 : float
PCT_LACCESS_HISP15 : float
PCT_LACCESS_NHASIAN15 : float
PCT_LACCESS_NHNA15 : float
PCT_LACCESS_NHPI15 : float
PCT_LACCESS_MULTIR15 : float
'''


# In[72]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[73]:


for i in range(19):
    foolist[i] = df[df.columns[i]].tolist()


# In[74]:


for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
    # foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[75]:


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
se16 = pd.Series(foolist[16])
se17 = pd.Series(foolist[17])
se18 = pd.Series(foolist[18])


# In[76]:


df['PCT_LACCESS_POP10'] = se1.values
df['PCT_LACCESS_POP15'] = se2.values
df['PCT_LACCESS_LOWI10'] = se3.values
df['PCT_LACCESS_LOWI15'] = se4.values
df['PCT_LACCESS_HHNV10'] = se5.values
df['PCT_LACCESS_HHNV15'] = se6.values
df['PCT_LACCESS_SNAP15 '] = se7.values
df['PCT_LACCESS_CHILD10'] = se8.values
df['PCT_LACCESS_CHILD15'] = se9.values
df['PCT_LACCESS_SENIORS10'] = se10.values
df['PCT_LACCESS_SENIORS15'] = se11.values
df['PCT_LACCESS_WHITE15'] = se12.values
df['PCT_LACCESS_BLACK15'] = se13.values
df['PCT_LACCESS_HISP15'] = se14.values
df['PCT_LACCESS_NHASIAN15'] = se15.values
df['PCT_LACCESS_NHNA15'] = se16.values
df['PCT_LACCESS_NHPI15'] = se17.values
df['PCT_LACCESS_MULTIR15'] = se18.values


# In[77]:


df.to_csv('PROCESSED_ACCESS.csv', index=False)


# In[ ]:




