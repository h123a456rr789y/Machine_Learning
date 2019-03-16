#!/usr/bin/env python
# coding: utf-8

# In[20]:


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


# In[21]:


def float_check(tmplist):
    for ind, i in enumerate(tmplist):
        if type(i) != float:
            print(ind, i)


# In[22]:


def bool_check(tmplist):
    for ind, i in enumerate(tmplist):
        if (i != 0) & (i != 1):
            print(ind, i)


# In[23]:


def float_nan_filled(tmplist):
    
    num = 0
    sum = 0
    
    # calculate the average to fill the NaN
    for i in tmplist:
        if np.isnan(i):
            pass
        else:
            num = num + 1
            sum = sum + i
    aver = sum/num

    # fill the NaN with average
    for ind, i in enumerate(tmplist):
        if np.isnan(i):
            tmplist[ind] = aver
        elif type(i) != float:
            tmplist[ind] = aver
        else:
            pass


# In[24]:


def bool_filled(tmplist):
    
    num_of_one = 0
    num_of_zero = 0
    
    # calculate the mode of list
    for i in tmplist:
        if i == 0:
            num_of_zero = num_of_zero + 1
        elif i == 1:
            num_of_one = num_of_one + 1
        else:
            pass
    
    if num_of_zero > num_of_one:
        mode0 = 0
    else:
        mode0 = 1
    
    for ind, i in enumerate(tmplist):
        if (i != 0) & (i != 1):
            tmplist[ind] = mode0


# In[ ]:




