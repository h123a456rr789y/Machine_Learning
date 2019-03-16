#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import statistics as stats
import re
df_review = pd.read_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/origin_data/googleplaystore_user_reviews.csv',engine='python') # read the whole data
df_googleplaystore = pd.read_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/processed_data/processed_googleplaystore_dropped.csv',engine='python') # read the whole data


# In[37]:


listnan = []
for e, i in enumerate(df_review.iloc[:, 2]):
    if type(i) is float:
        listnan.append(e)

df_review = df_review.drop(listnan)

skind = set()
for i in df_review.iloc[:, 0]:
    skind.add(i)


# In[38]:


for e, i in enumerate(df_review.iloc[:, 2]):
    if i == "Neutral":
        df_review.iloc[e, 2] = 0
    elif i == "Negative":
        df_review.iloc[e, 2] = -1
    else:
        df_review.iloc[e, 2] = 1


# In[39]:


dtimes = dict()
for i in skind:
    dtimes.update({i: 0})

#for key, value in d.items():
#    print(key, value)


# In[40]:


for i in df_review.iloc[:, 0]:
    dtimes[i] += 1


# In[41]:


dsum = dict()
for i in skind:
    dsum.update({i: 0})

for e, i in enumerate(df_review.iloc[:, 0]):
    dsum[i] += df_review.iloc[e, 2]

#for key, value in dsum.items():
#    print(key, ":", value)


# In[42]:


for key, value in dtimes.items():
    dsum[key] /= value


# In[45]:


listnan=[]
for e, i in enumerate(df_googleplaystore.iloc[:, 0]):
    if not i in skind:
        listnan.append(e)

df_googleplaystore = df_googleplaystore.drop(listnan)


# In[47]:


dtmp = dict()
for i in skind:
    dtmp.update({i: 0})

for i in df_googleplaystore.iloc[:, 0]:
    dtmp[i] += 1


# In[49]:


append_array=[]
for i in df_googleplaystore.iloc[:,0]:
    append_array.append(dsum[i])
df_googleplaystore.insert(6, 'review_rating', append_array)


# In[50]:

for e,i in enumerate(df_googleplaystore.iloc[:,1]):
    if i <= 1.5:
        df_googleplaystore.iloc[e,1]='1'
    elif i<= 2:
        df_googleplaystore.iloc[e,1]='2'
    elif i<= 2.5:
        df_googleplaystore.iloc[e,1]='3'
    elif i<= 3:
        df_googleplaystore.iloc[e,1]='4'
    elif i<= 3.5:
        df_googleplaystore.iloc[e,1]='5'
    elif i<= 4:
        df_googleplaystore.iloc[e,1]='6'
    elif i<= 4.5:
        df_googleplaystore.iloc[e,1]='7'
    elif i<= 5:
        df_googleplaystore.iloc[e,1]='8'


df_googleplaystore.to_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/processed_data/processed_googleplaystore_merged_final.csv', sep=',',index=0)


# In[ ]:




