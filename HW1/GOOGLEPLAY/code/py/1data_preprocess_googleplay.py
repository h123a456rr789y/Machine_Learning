#!/usr/bin/env python
# coding: utf-8

# In[10]:


# IMPORT THE NEEDED MODULE

import numpy as np
import pandas as pd
import statistics as stats
import re


# In[11]:


# IMPORT ORIGIN DATA

df_googleplaystore = pd.read_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/origin_data/googleplaystore.csv', engine = 'python')# read the whole data


# In[12]:


# DROP SOME NO USE COLUMN IN THE FIRST

# The column "Current Ver" is too hard to handle, so I drop it.
df_googleplaystore = df_googleplaystore.drop(['Current Ver'], axis=1)
# The column "Genres" is too similar to the column "Catorory", so I drop it.
df_googleplaystore = df_googleplaystore.drop(['Genres'], axis=1)


# In[13]:


# HANDLE THE COLUMN "Catogory"

listnan = [] # The index of wrong or missing data would be append here, and will be drop 
for e, i in enumerate(df_googleplaystore.iloc[:, 1]): # replace the catogorical data with number 1~33
    if i == 'ENTERTAINMENT':
        df_googleplaystore.iloc[e, 1] = 'A'
    elif i == "BOOKS_AND_REFERENCE":
        df_googleplaystore.iloc[e, 1] = 'B'
    elif i == "EDUCATION":
        df_googleplaystore.iloc[e, 1] = 'C'
    elif i == "FAMILY":
        df_googleplaystore.iloc[e, 1] = 'D'
    elif i == "HEALTH_AND_FITNESS":
        df_googleplaystore.iloc[e, 1] = 'E'
    elif i == "GAME":
        df_googleplaystore.iloc[e, 1] = 'F'
    elif i == "TRAVEL_AND_LOCAL":
        df_googleplaystore.iloc[e, 1] = 'G'
    elif i == "PRODUCTIVITY":
        df_googleplaystore.iloc[e, 1] = 'H'
    elif i == "SHOPPING":
        df_googleplaystore.iloc[e, 1] = 'I'
    elif i == "BEAUTY":
        df_googleplaystore.iloc[e, 1] = 'J'
    elif i == "SPORTS":
        df_googleplaystore.iloc[e, 1] = 'K'
    elif i == "WEATHER":
        df_googleplaystore.iloc[e, 1] = 'L'
    elif i == "COMMUNICATION":
        df_googleplaystore.iloc[e, 1] = 'M'
    elif i == "PARENTING":
        df_googleplaystore.iloc[e, 1] = 'N'
    elif i == "HOUSE_AND_HOME":
        df_googleplaystore.iloc[e, 1] = 'O'
    elif i == "ART_AND_DESIGN":
        df_googleplaystore.iloc[e, 1] = 'P'
    elif i == "LIBRARIES_AND_DEMO":
        df_googleplaystore.iloc[e, 1] = 'Q'
    elif i == "MEDICAL":
        df_googleplaystore.iloc[e, 1] = 'R'
    elif i == "PHOTOGRAPHY":
        df_googleplaystore.iloc[e, 1] = 'S'
    elif i == "FINANCE":
        df_googleplaystore.iloc[e, 1] = 'T'
    elif i == "NEWS_AND_MAGAZINES":
        df_googleplaystore.iloc[e, 1] = 'U'
    elif i == "BUSINESS":
        df_googleplaystore.iloc[e, 1] = 'V'
    elif i == "PERSONALIZATION":
        df_googleplaystore.iloc[e, 1] = 'W'
    elif i == "DATING":
        df_googleplaystore.iloc[e, 1] = 'X'
    elif i == "TOOLS":
        df_googleplaystore.iloc[e, 1] = 'Y'
    elif i == "FOOD_AND_DRINK":
        df_googleplaystore.iloc[e, 1] = 'Z'
    elif i == "SOCIAL":
        df_googleplaystore.iloc[e, 1] = 'a'
    elif i == "AUTO_AND_VEHICLES":
        df_googleplaystore.iloc[e, 1] = 'b'
    elif i == "EVENTS":
        df_googleplaystore.iloc[e, 1] = 'c'
    elif i == "LIFESTYLE":
        df_googleplaystore.iloc[e, 1] = 'd'
    elif i == "VIDEO_PLAYERS":
        df_googleplaystore.iloc[e, 1] = 'e'
    elif i == "MAPS_AND_NAVIGATION":
        df_googleplaystore.iloc[e, 1] = 'f'
    elif i == "COMICS":
        df_googleplaystore.iloc[e, 1] = 'g'
    else: # if it is not corresponding to the 33 class
        listnan.append(e)

for i in listnan:
    df_googleplaystore = df_googleplaystore.drop(i) # delete the wrong or missing data


# In[15]:


# HANDLE THE COLUMN "Rate"

SUM = 0 # the temporary variable to calculate the mean of the column "Rate"

# MEAN is a temporary variable of the mean of the column "Rate"

#-----------calculate the mean of the normal data---------------
for i in df_googleplaystore.iloc[:, 2]:
    if (i <= 5) & (i >= 0):
        SUM += i
MEAN = SUM / df_googleplaystore.shape[0]
MEAN = format(MEAN, '.6f') # round up to six digits after decimal point
#-----------calculate the mean of the normal data---------finish

for e, i in enumerate(df_googleplaystore.iloc[:, 2]): # replace the abnormal or missing data with MEAN
    if not ((i <= 5) & (i >= 0)):
        df_googleplaystore.iloc[e, 2] = MEAN


# In[16]:


# HANDLE THE COLUMN "Reviews"

for e, i in enumerate(df_googleplaystore.iloc[:, 3]):
    df_googleplaystore.iloc[e, 3] = int(i)
for i in df_googleplaystore.iloc[:, 3]: # still need to check if there is abnornal or missing data
    if not ((i >= 0) & (i <= 100000000)):
        print(i)


# In[17]:


# HANDLE THE COLUMN "Size"

listtmp=[] # The number of normal data to calculate MEAN and STANDARD DEVIATION
listnan=[] # The index of wrong or missing data would be append here, and will be replaced with MEAN
for e, i in enumerate(df_googleplaystore.iloc[:, 4]):
    i = i.replace(",", "") # remove all the ','
    if re.search("\+$", i): # if there is a '+' at the end, remove it
        tmp = i.replace("+", "") 
        tmp = float(tmp)
        tmp = int(tmp)
        df_googleplaystore.iloc[e, 4] = tmp
        listtmp.append(tmp)
    elif re.search("M$", i): # if there is a 'M' at the end, remove it and multiply the number with 1000000
        tmp = i.replace("M", "")
        tmp = float(tmp)
        tmp = tmp*1000000
        tmp = int(tmp)
        df_googleplaystore.iloc[e, 4] = tmp
        listtmp.append(tmp)
    elif re.search("k$", i): # if there is a 'k' at the end, remove it and multiply the number with 1000
        tmp = i.replace("k", "")
        tmp = float(tmp)
        tmp = tmp*1000
        tmp = int(tmp)
        df_googleplaystore.iloc[e, 4] = tmp
        listtmp.append(tmp)
    else: # if it is not corresponding to the above R.E., prepare to replace it with the MEAN
        listnan.append(e)

# calculate the MEAN of data without outliers
listtmp = np.array(listtmp)
MEAN = np.mean(listtmp, axis=0)
SD = np.std(listtmp, axis=0)
listtmp = [x for x in listtmp if (x > MEAN - 2*SD)]
listtmp = [x for x in listtmp if (x < MEAN + 2*SD)]
MEAN = np.mean(listtmp, axis=0)
MEAN = int(MEAN)
# calculate the mean of data without outliers --finish

for i in listnan:
    df_googleplaystore.iloc[i, 4] = MEAN


# In[18]:


# HANDLE THE COLUMN "Installs"

for e, i in enumerate(df_googleplaystore.iloc[:, 5]):
    tmp = re.sub('[,+]', '', i) # remove ',' and '+'
    df_googleplaystore.iloc[e, 5] = tmp


# In[21]:


# HANDLE THE COLUMN "Type"

listnan=[] # The index of wrong or missing data would be append here, and will be replace with MODE
listtmp=[] # find the MODE
for e, i in enumerate(df_googleplaystore.iloc[:, 6]):
    if i == "Free":
        df_googleplaystore.iloc[e, 6] = 0
        listtmp.append(i)
    elif i == "Paid":
        df_googleplaystore.iloc[e, 6] = 1
        listtmp.append(i)
    else:
        listnan.append(e)

tmp = 0
if stats.mode(listtmp) == "Free":
    tmp = 0
else:
    tmp = 1

for i in listnan:
    df_googleplaystore.iloc[i, 6] = tmp


# In[22]:


# HANDLE THE COLUMN "Price"

for e, i in enumerate(df_googleplaystore.iloc[:, 7]):
    tmp = i.replace("$", "") # remove '$'
    tmp = float(tmp)
    tmp = int(tmp * 100)
    df_googleplaystore.iloc[e, 7] = tmp


# In[23]:


# HANDLE THE COLUMN "Content Rating"

# There are five type of "Content Rating" : 'Mature 17+', 'Unrated', 'Everyone', 'Everyone 10+', 'Adults only 18+', 'Teen'
for e, i in enumerate(df_googleplaystore.iloc[:, 8]):
    if i == "Unrated" :
        df_googleplaystore.iloc[e, 8] = 6
    elif i == "Everyone" :
        df_googleplaystore.iloc[e, 8] = 6
    elif i == "Everyone 10+" :
        df_googleplaystore.iloc[e, 8] = 10
    elif i == "Teen" :
        df_googleplaystore.iloc[e, 8] = 12
    elif i == "Mature 17+":
        df_googleplaystore.iloc[e, 8] = 17
    elif i == "Adults only 18+" :
        df_googleplaystore.iloc[e, 8] = 18


# In[24]:


# HANDLE THE COLUMN "Content Rating"

# replace each instance with the day difference with "January 1, 2010"
# no missing or wrong data
YEAR_DIFF = 0
MONTH_DIFF = 0
DAY_DIFF = 0
for e, i in enumerate(df_googleplaystore.iloc[:, 9]):
    tmp = i.split(' ', ) # split the string with whitespace
    tmp[1] = tmp[1].replace(",", "")
    if tmp[0] == "January":
        MONTH_DIFF = 0 * 30.5
    elif tmp[0] == "February":
        MONTH_DIFF = 1 * 30.5
    elif tmp[0] == "March":
        MONTH_DIFF = 2 * 30.5
    elif tmp[0] == "April":
        MONTH_DIFF = 3 * 30.5
    elif tmp[0] == "May":
        MONTH_DIFF = 4 * 30.5
    elif tmp[0] == "June":
        MONTH_DIFF = 5 * 30.5
    elif tmp[0] == "July":
        MONTH_DIFF = 6 * 30.5
    elif tmp[0] == "August":
        MONTH_DIFF = 7 * 30.5
    elif tmp[0] == "September":
        MONTH_DIFF = 8 * 30.5
    elif tmp[0] == "October":
        MONTH_DIFF = 9 * 30.5
    elif tmp[0] == "November":
        MONTH_DIFF = 10 * 30.5
    elif tmp[0] == "December":
        MONTH_DIFF = 11 * 30.5
    DAY_DIFF = int(tmp[1])-1
    YEAR_DIFF = (int(tmp[2]) - 2010)*365.25
    
    df_googleplaystore.iloc[e, 9] = DAY_DIFF + MONTH_DIFF + YEAR_DIFF


# In[25]:


# HANDLE THE COLUMN "Android Ver"

listnan = []
MEAN = 0
SUM = 0
NUMBER = 0
for e, i in enumerate(df_googleplaystore.iloc[:, 10]):
    if type(i) is str:
        tmp = i.split(' ', )
        if (tmp[0][0].isdigit()) & (tmp[0][1] == '.') & (tmp[0][2].isdigit()):
            tmp1 = tmp[0][0] + tmp[0][1] + tmp[0][2]
            NUMBER = NUMBER + 1
            tmp2 = float(tmp1)
            tmp2 = format(tmp2, '.3f')
            tmp2 = float(tmp2)
            SUM += tmp2
            df_googleplaystore.iloc[e, 10] = float(tmp1) 
        else:
            listnan.append(e)
    elif type(i) is float:
        if np.isnan(i):
            listnan.append(e)
        else:
            SUM += i
            NUMBER = NUMBER + 1

MEAN = float(SUM/NUMBER)

for i in listnan:
    df_googleplaystore.iloc[i, 10] = MEAN


# In[26]:


df_googleplaystore.to_csv('G:/我的雲端硬碟/大三上/機器學習/MLHW1/GOOGLEPLAY/processed_data\processed_googleplaystore.csv', sep = ',', index = 0)

