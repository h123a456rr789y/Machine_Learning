#!/usr/bin/env python
# coding: utf-8

# In[92]:


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


# In[93]:


df = pd.read_csv('LOCAL.csv')


# In[94]:


print(df.shape)


# In[95]:


list(df)


# In[96]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[97]:


df = df.drop(['State'], axis=1)


# In[98]:


df = df.drop(['County'], axis=1)


# In[99]:


df = df.drop(['DIRSALES_FARMS07'], axis=1)


# In[100]:


df = df.drop(['DIRSALES_FARMS12'], axis=1)


# In[101]:


df = df.drop(['PCH_DIRSALES_FARMS_07_12'], axis=1)


# In[102]:


df = df.drop(['DIRSALES07'], axis=1)


# In[103]:


df = df.drop(['DIRSALES12'], axis=1)


# In[104]:


df = df.drop(['PCH_DIRSALES_07_12'], axis=1)


# In[105]:


df = df.drop(['PCH_PC_DIRSALES_07_12'], axis=1)


# In[106]:


df = df.drop(['FMRKT09'], axis=1)


# In[107]:


df = df.drop(['FMRKT16'], axis=1)


# In[108]:


df = df.drop(['PCH_FMRKT_09_16'], axis=1)


# In[109]:


df = df.drop(['PCH_FMRKTPTH_09_16'], axis=1)


# In[110]:


df = df.drop(['FMRKT_SNAP16'], axis=1)


# In[111]:


df = df.drop(['FMRKT_WIC16'], axis=1)


# In[112]:


df = df.drop(['FMRKT_WICCASH16'], axis=1)


# In[113]:


df = df.drop(['FMRKT_SFMNP16'], axis=1)


# In[114]:


df = df.drop(['FMRKT_CREDIT16'], axis=1)


# In[115]:


df = df.drop(['FMRKT_FRVEG16'], axis=1)


# In[116]:


df = df.drop(['FMRKT_ANMLPROD16'], axis=1)


# In[117]:


df = df.drop(['FMRKT_BAKED16'], axis=1)


# In[118]:


df = df.drop(['FMRKT_OTHERFOOD16'], axis=1)


# In[119]:


df = df.drop(['VEG_FARMS07'], axis=1)


# In[120]:


df = df.drop(['VEG_FARMS12'], axis=1)


# In[121]:


df = df.drop(['PCH_VEG_FARMS_07_12'], axis=1)


# In[122]:


df = df.drop(['VEG_ACRES07'], axis=1)


# In[123]:


df = df.drop(['VEG_ACRES12'], axis=1)


# In[124]:


df = df.drop(['PCH_VEG_ACRES_07_12'], axis=1)


# In[125]:


df = df.drop(['PCH_VEG_ACRESPTH_07_12'], axis=1)


# In[126]:


df = df.drop(['FRESHVEG_FARMS07'], axis=1)


# In[127]:


df = df.drop(['FRESHVEG_FARMS12'], axis=1)


# In[128]:


df = df.drop(['PCH_FRESHVEG_FARMS_07_12'], axis=1)


# In[129]:


df = df.drop(['FRESHVEG_ACRES07'], axis=1)


# In[130]:


df = df.drop(['FRESHVEG_ACRES12'], axis=1)


# In[131]:


df = df.drop(['PCH_FRESHVEG_ACRES_07_12'], axis=1)


# In[132]:


df = df.drop(['PCH_FRESHVEG_ACRESPTH_07_12'], axis=1)


# In[133]:


df = df.drop(['ORCHARD_FARMS07'], axis=1)


# In[134]:


df = df.drop(['ORCHARD_FARMS12'], axis=1)


# In[135]:


df = df.drop(['PCH_ORCHARD_FARMS_07_12'], axis=1)


# In[136]:


df = df.drop(['ORCHARD_ACRES07'], axis=1)


# In[137]:


df = df.drop(['ORCHARD_ACRES12'], axis=1)


# In[138]:


df = df.drop(['PCH_ORCHARD_ACRES_07_12'], axis=1)


# In[139]:


df = df.drop(['PCH_ORCHARD_ACRESPTH_07_12'], axis=1)


# In[140]:


df = df.drop(['BERRY_FARMS07'], axis=1)


# In[141]:


df = df.drop(['BERRY_FARMS12'], axis=1)


# In[142]:


df = df.drop(['PCH_BERRY_FARMS_07_12'], axis=1)


# In[143]:


df = df.drop(['BERRY_ACRES07'], axis=1)


# In[144]:


df = df.drop(['BERRY_ACRES12'], axis=1)


# In[145]:


df = df.drop(['PCH_BERRY_ACRES_07_12'], axis=1)


# In[146]:


df = df.drop(['PCH_BERRY_ACRESPTH_07_12'], axis=1)


# In[147]:


df = df.drop(['SLHOUSE07'], axis=1)


# In[148]:


df = df.drop(['SLHOUSE12'], axis=1)


# In[149]:


df = df.drop(['PCH_SLHOUSE_07_12'], axis=1)


# In[150]:


df = df.drop(['GHVEG_FARMS07'], axis=1)


# In[151]:


df = df.drop(['GHVEG_FARMS12'], axis=1)


# In[152]:


df = df.drop(['PCH_GHVEG_FARMS_07_12'], axis=1)


# In[153]:


df = df.drop(['GHVEG_SQFT07'], axis=1)


# In[154]:


df = df.drop(['GHVEG_SQFT12'], axis=1)


# In[155]:


df = df.drop(['PCH_GHVEG_SQFT_07_12'], axis=1)


# In[156]:


df = df.drop(['PCH_GHVEG_SQFTPTH_07_12'], axis=1)


# In[157]:


df = df.drop(['FOODHUB16'], axis=1)


# In[158]:


df = df.drop(['CSA07'], axis=1)


# In[159]:


df = df.drop(['CSA12'], axis=1)


# In[160]:


df = df.drop(['PCH_CSA_07_12'], axis=1)


# In[161]:


df = df.drop(['AGRITRSM_OPS07'], axis=1)


# In[162]:


df = df.drop(['AGRITRSM_OPS12'], axis=1)


# In[163]:


df = df.drop(['PCH_AGRITRSM_OPS_07_12'], axis=1)


# In[164]:


df = df.drop(['AGRITRSM_RCT07'], axis=1)


# In[165]:


df = df.drop(['AGRITRSM_RCT12'], axis=1)


# In[166]:


df = df.drop(['PCH_AGRITRSM_RCT_07_12'], axis=1)


# In[167]:


print(df.shape)


# In[168]:


list(df)


# In[169]:


'''
FIPS : int
PCT_LOCLFARM07 : float
PCT_LOCLFARM12 : float
PCT_LOCLSALE07 : float
PCT_LOCLSALE12 : float
PC_DIRSALES07 : float
PC_DIRSALES12 : float
FMRKTPTH09 : float
FMRKTPTH16 : float
PCT_FMRKT_SNAP16 : float
PCT_FMRKT_WIC16 : float
PCT_FMRKT_WICCASH16 : float
PCT_FMRKT_SFMNP16 : float
PCT_FMRKT_CREDIT16 : float
PCT_FMRKT_FRVEG16 : float
PCT_FMRKT_ANMLPROD16 : float
PCT_FMRKT_BAKED16 : float
PCT_FMRKT_OTHERFOOD16 : float
VEG_ACRESPTH07 : float
VEG_ACRESPTH12 : float
FRESHVEG_ACRESPTH07 : float
FRESHVEG_ACRESPTH12 : float
ORCHARD_ACRESPTH07 : float
ORCHARD_ACRESPTH12 : float
BERRY_ACRESPTH07 : float
BERRY_ACRESPTH12 : float
GHVEG_SQFTPTH07 : float
GHVEG_SQFTPTH12 : float
FARM_TO_SCHOOL09 : boolean
FARM_TO_SCHOOL13 : boolean
'''


# In[170]:


foolist = list()
for i in range(100):
    barlist = list()
    foolist.append(barlist)


# In[171]:


for i in range(30):
    foolist[i] = df[df.columns[i]].tolist()


# In[172]:


for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
    #foo.float_check(foolist[i])
    foo.float_nan_filled(foolist[i])


# In[173]:


for i in [28, 29]:
    #foo.bool_check(foolist[i])
    foo.bool_filled(foolist[i])


# In[174]:


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


# In[175]:


df['PCT_LOCLFARM07'] = se1.values
df['PCT_LOCLFARM12'] = se2.values
df['PCT_LOCLSALE07'] = se3.values
df['PCT_LOCLSALE12'] = se4.values
df['PC_DIRSALES07'] = se5.values
df['PC_DIRSALES12'] = se6.values
df['FMRKTPTH09'] = se7.values
df['FMRKTPTH16'] = se8.values
df['PCT_FMRKT_SNAP16'] = se9.values
df['PCT_FMRKT_WIC16'] = se10.values
df['PCT_FMRKT_WICCASH16'] = se11.values
df['PCT_FMRKT_SFMNP16'] = se12.values
df['PCT_FMRKT_CREDIT16'] = se13.values
df['PCT_FMRKT_FRVEG16'] = se14.values
df['PCT_FMRKT_ANMLPROD16'] = se15.values
df['PCT_FMRKT_BAKED16'] = se16.values
df['PCT_FMRKT_OTHERFOOD16'] = se17.values
df['VEG_ACRESPTH07'] = se18.values
df['VEG_ACRESPTH12'] = se19.values
df['FRESHVEG_ACRESPTH07'] = se20.values
df['FRESHVEG_ACRESPTH12'] = se21.values
df['ORCHARD_ACRESPTH07'] = se22.values
df['ORCHARD_ACRESPTH12'] = se23.values
df['BERRY_ACRESPTH07'] = se24.values
df['BERRY_ACRESPTH12'] = se25.values
df['GHVEG_SQFTPTH07'] = se26.values
df['GHVEG_SQFTPTH12'] = se27.values
df['FARM_TO_SCHOOL09'] = se28.values
df['FARM_TO_SCHOOL13'] = se29.values


# In[176]:


df.to_csv('PROCESSED_LOCAL.csv', index=False)


# In[ ]:




