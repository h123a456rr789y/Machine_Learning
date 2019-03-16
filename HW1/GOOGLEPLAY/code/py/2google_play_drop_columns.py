#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import statistics as stats
import re
df_googleplaystore = pd.read_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/processed_data/processed_googleplaystore.csv',engine='python') # read the whole data

# In[29]:


# df_googleplaystore = df_googleplaystore.drop(['App'], axis=1)


# In[30]:


df_googleplaystore = df_googleplaystore.drop(['Category'], axis=1)


# In[31]:


df_googleplaystore = df_googleplaystore.drop(['Size'], axis=1)


# In[32]:


df_googleplaystore = df_googleplaystore.drop(['Type'], axis=1)


# In[33]:


df_googleplaystore = df_googleplaystore.drop(['Content Rating'], axis=1)


# In[34]:


df_googleplaystore = df_googleplaystore.drop(['Android Ver'], axis=1)


# In[35]:


df_googleplaystore.to_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/processed_data/processed_googleplaystore_dropped.csv', sep=',', index=0)


# In[ ]:




