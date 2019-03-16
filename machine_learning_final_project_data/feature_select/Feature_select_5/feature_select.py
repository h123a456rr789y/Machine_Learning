import pandas as pd
import numpy as np
import glob
a=pd.read_csv("PROCESSED_ACCESS.csv")
b=pd.read_csv("PROCESSED_ASSISTANCE.csv")
c=pd.read_csv("PROCESSED_HEALTH_CLASSIFIED.csv")
d=pd.read_csv("PROCESSED_INSECURITY.csv")
e=pd.read_csv("PROCESSED_LOCAL.csv")
f=pd.read_csv("PROCESSED_PRICES_TAXES.csv")
g=pd.read_csv("PROCESSED_RESTAURANTS.csv")
h=pd.read_csv("PROCESSED_SOCIOECONOMIC.csv")
i=pd.read_csv("PROCESSED_STORES.csv")
merged = a.merge(b, on='FIPS')
merged = merged.merge(c, on='FIPS')
merged = merged.merge(d, on='FIPS')
merged = merged.merge(e, on='FIPS')
merged = merged.merge(f, on='FIPS')
merged = merged.merge(g, on='FIPS')
merged = merged.merge(h, on='FIPS')
merged = merged.merge(i, on='FIPS')
merged.to_csv('ALL_FEATURE.csv',index=False)

from sklearn.feature_selection import SelectKBest,chi2
y=merged['PCT_DIABETES_ADULTS13_CLASS']
X=merged.drop('PCT_DIABETES_ADULTS13_CLASS',axis=1)
X=X.drop('FIPS',axis=1)
X=X.drop('Unnamed: 0',axis=1)
X=X.drop('PCT_DIABETES_ADULTS08',axis=1)
X=X.drop('PCT_DIABETES_ADULTS13',axis=1)
X['PCT_LACCESS_SNAP15'].fillna(X['PCT_LACCESS_SNAP15'].mean(), inplace = True)
print(type(X))

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)


print("best 5 feature after standerd")
selector = SelectKBest(chi2, k=5).fit(X_scaled, y)
cols = selector.get_support(indices=True)
out_5=merged['FIPS']
for i in cols:
    name=X_scaled.columns.tolist()[i]
    print(name)
    temp=X_scaled[name]
    out_5=pd.concat([out_5,temp],axis=1)
out_5=out_5.drop('FIPS',axis=1)
out_5.to_csv('BEST_FEATURES/BEST5.csv',index=False)
print('--------------------------')
'''
print("best 5 feature before standerd")
selector = SelectKBest(chi2, k=5).fit(X, y)
cols = selector.get_support(indices=True)
for i in cols:
    print(X.columns.tolist()[i])
print('---------------------------')
'''
print("best 10 feature after standerd")
selector = SelectKBest(chi2, k=10).fit(X_scaled, y)
cols = selector.get_support(indices=True)
out_10=merged['FIPS']
for i in cols:
    name=X_scaled.columns.tolist()[i]
    print(name)
    temp=X_scaled[name]
    out_10=pd.concat([out_10,temp],axis=1)
out_10=out_10.drop('FIPS',axis=1)
out_10.to_csv('BEST_FEATURES/BEST10.csv',index=False)
print('--------------------------')

print("best 15 feature after standerd")
selector = SelectKBest(chi2, k=15).fit(X_scaled, y)
cols = selector.get_support(indices=True)
out_15=merged['FIPS']
for i in cols:
    name=X_scaled.columns.tolist()[i]
    print(name)
    temp=X_scaled[name]
    out_15=pd.concat([out_15,temp],axis=1)
out_15=out_15.drop('FIPS',axis=1)
out_15.to_csv('BEST_FEATURES/BEST15.csv',index=False)
print('--------------------------')

print("20 feature selected humanly")
cols=['GROCPTH14']
cols.append('SUPERCPTH14')
cols.append('CONVSPTH14')
cols.append('SPECSPTH14')
cols.append('SNAPSPTH12')
cols.append('WICSPTH12')
cols.append('FFRPTH14')
#cols.append('FSRPTH14')
cols.append('PCT_SNAP12')
cols.append('PCT_NSLP15')
cols.append('PCT_SBP15')
cols.append('PCT_WIC15')
cols.append('VLFOODSEC_10_12')
cols.append('MILK_PRICE10')
cols.append('SODA_PRICE10')
cols.append('VEG_ACRESPTH12')
cols.append('BERRY_ACRESPTH12')
cols.append('GHVEG_SQFTPTH12')
cols.append('POVRATE15')
#cols.append('PCT_OBESE_ADULTS13')

out_20=merged['FIPS']
for i in cols:
    print(i)
    temp=X_scaled[i]
    out_20=pd.concat([out_20,temp],axis=1)
out_20=out_20.drop('FIPS',axis=1)
out_20.to_csv('BEST_FEATURES/HUMAN20.csv',index=False)
print('--------------------------')

print("human 5 feature selected from 20")
selector = SelectKBest(chi2, k=5).fit(out_20, y)
cols = selector.get_support(indices=True)
out_5_h=merged['FIPS']
for i in cols:
    name=out_20.columns.tolist()[i]
    print(name)
    temp=out_20[name]
    out_5_h=pd.concat([out_5_h,temp],axis=1)
out_5_h=out_5_h.drop('FIPS',axis=1)
out_5_h.to_csv('BEST_FEATURES/HUMAN5.csv',index=False)
print('--------------------------')

y.to_csv('TARGET/TARGET.csv',index=False)

input()


