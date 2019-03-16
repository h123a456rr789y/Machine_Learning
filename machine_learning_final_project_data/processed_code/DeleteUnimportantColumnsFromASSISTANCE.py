#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


df = pd.read_csv('ASSISTANCE.csv')


# In[4]:


print(df.shape)


# In[5]:


list(df)


# In[6]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[7]:


df = df.drop(['State'], axis=1)


# In[8]:


df = df.drop(['County'], axis=1)


# In[9]:


df = df.drop(['PCH_SNAP_12_16'], axis=1)


# In[10]:


df = df.drop(['PCH_REDEMP_SNAPS_12_16'], axis=1)


# In[11]:


df = df.drop(['PCH_PC_SNAPBEN_10_15'], axis=1)


# In[12]:


df = df.drop(['PCH_PC_WIC_REDEMP_08_12'], axis=1)


# In[13]:


df = df.drop(['PCH_REDEMP_WICS_08_12'], axis=1)


# In[14]:


df = df.drop(['PCH_WIC_09_15'], axis=1)


# In[15]:


df = df.drop(['PCH_CACFP_09_15'], axis=1)


# In[16]:


df = df.drop(['PCH_NSLP_09_15'], axis=1)


# In[17]:


df = df.drop(['PCH_SBP_09_15'], axis=1)


# In[18]:


df = df.drop(['PCH_SFSP_09_15'], axis=1)


# In[19]:


print(df.shape)


# In[20]:


list(df)


# In[21]:


'''
FIPS : 
REDEMP_SNAPS12 : float
REDEMP_SNAPS16 : float
PCT_SNAP12 : float
PCT_SNAP16 : float
PC_SNAPBEN10 : float
PC_SNAPBEN15 : float
SNAP_PART_RATE08 : int
SNAP_PART_RATE13 : float
SNAP_OAPP09 : boolean(three kinds : 0, 0.5, 1)
SNAP_OAPP16 : boolean(three kinds : 0, 0.5, 1)
SNAP_CAP09 : boolean(two kinds : 0, 1)
SNAP_CAP16 : boolean(two kinds : 0, 1)
SNAP_BBCE09 : boolean(two kinds : 0, 1)
SNAP_BBCE16 : boolean(two kinds : 0, 1)
SNAP_REPORTSIMPLE09 : boolean(two kinds : 0, 1)
SNAP_REPORTSIMPLE16 : boolean(two kinds : 0, 1)
PCT_NSLP09 : float
PCT_NSLP15 : float
PCT_FREE_LUNCH09 : float
PCT_FREE_LUNCH14 : float
PCT_REDUCED_LUNCH09 : float
PCT_REDUCED_LUNCH14 : float
PCT_SBP09 : float
PCT_SBP15 : float
PCT_SFSP09 : float
PCT_SFSP15 : float
PC_WIC_REDEMP08 : float
PC_WIC_REDEMP12 : float
REDEMP_WICS08 : float
REDEMP_WICS12 : float
PCT_WIC09 : float
PCT_WIC15 : float
PCT_CACFP09 : float
PCT_CACFP15 : float
'''


# In[22]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[23]:


for i in range(35):
    foolist[i] = df[df.columns[i]].tolist()


# In[24]:


# the data type of 7th column(SNAP_PART_RATE08) is int
for i in [1, 2, 3, 4, 5, 6, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[25]:


# the data type of 9th and 10th columns(SNAP_OAPP09, SNAP_OAPP16) is not completely boolean(it has three kinds)
for i in [11, 12, 13, 14, 15, 16]:
    #foo.bool_check(foolist[i])
    foo.bool_filled(foolist[i])


# In[26]:


# for i in [9, 10]:
#     for ind, j in enumerate(foolist[i]):
#         if (j != 0) & (j != 1) & (j != 0.5):
#             print(ind, j)


# In[27]:


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
se19 = pd.Series(foolist[19])
se20 = pd.Series(foolist[20])
se21 = pd.Series(foolist[21])
se22 = pd.Series(foolist[22])
se23 = pd.Series(foolist[23])
se24 = pd.Series(foolist[24])
se25 = pd.Series(foolist[25])
se26 = pd.Series(foolist[26])
se27 = pd.Series(foolist[27])
se28 = pd.Series(foolist[28])
se29 = pd.Series(foolist[29])
se30 = pd.Series(foolist[30])
se31 = pd.Series(foolist[31])
se32 = pd.Series(foolist[32])
se33 = pd.Series(foolist[33])
se34 = pd.Series(foolist[34])


# In[28]:


df['REDEMP_SNAPS12'] = se1.values
df['REDEMP_SNAPS16'] = se2.values
df['PCT_SNAP12'] = se3.values
df['PCT_SNAP16'] = se4.values
df['PC_SNAPBEN10'] = se5.values
df['PC_SNAPBEN15'] = se6.values
df['SNAP_PART_RATE08'] = se7.values
df['SNAP_PART_RATE13'] = se8.values
df['SNAP_OAPP09'] = se9.values
df['SNAP_OAPP16'] = se10.values
df['SNAP_CAP09'] = se11.values
df['SNAP_CAP16'] = se12.values
df['SNAP_BBCE09'] = se13.values
df['SNAP_BBCE16'] = se14.values
df['SNAP_REPORTSIMPLE09'] = se15.values
df['SNAP_REPORTSIMPLE16'] = se16.values
df['PCT_NSLP09'] = se17.values
df['PCT_NSLP15'] = se18.values
df['PCT_FREE_LUNCH09'] = se19.values
df['PCT_FREE_LUNCH14'] = se20.values
df['PCT_REDUCED_LUNCH09'] = se21.values
df['PCT_REDUCED_LUNCH14'] = se22.values
df['PCT_SBP09'] = se23.values
df['PCT_SBP15'] = se24.values
df['PCT_SFSP09'] = se25.values
df['PCT_SFSP15'] = se26.values
df['PC_WIC_REDEMP08'] = se27.values
df['PC_WIC_REDEMP12'] = se28.values
df['REDEMP_WICS08'] = se29.values
df['REDEMP_WICS12'] = se30.values
df['PCT_WIC09'] = se31.values
df['PCT_WIC15'] = se32.values
df['PCT_CACFP09'] = se33.values
df['PCT_CACFP15'] = se34.values


# In[29]:


df.to_csv('PROCESSED_ASSISTANCE.csv', index=False)


# In[ ]:




