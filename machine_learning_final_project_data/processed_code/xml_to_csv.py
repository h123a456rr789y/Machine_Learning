import pandas as pd
import numpy as np

newXls = pd.ExcelFile('DataDownload.xls')

df1 = df1 = pd.read_excel(newXls, 'Read_Me')
df1.to_csv('Read_Me.csv')

df2 = pd.read_excel(newXls, 'Variable List')
df2.to_csv('Variable List.csv')

df3 = pd.read_excel(newXls, 'Supplemental Data - County')
df3.to_csv('Supplemental Data - County.csv')

df4 = pd.read_excel(newXls, 'Supplemental Data - State')
df4.to_csv('Supplemental Data - State.csv')

df5 = pd.read_excel(newXls, 'ACCESS')
df5.to_csv('ACCESS.csv')

df6 = pd.read_excel(newXls, 'STORES')
df6.to_csv('STORES.csv')

df7 = pd.read_excel(newXls, 'RESTAURANTS')
df7.to_csv('RESTAURANTS.csv')

df8 = pd.read_excel(newXls, 'ASSISTANCE')
df8.to_csv('ASSISTANCE.csv')

df9 = pd.read_excel(newXls, 'INSECURITY')
df9.to_csv('INSECURITY.csv')

df10 = pd.read_excel(newXls, 'PRICES_TAXES')
df10.to_csv('PRICES_TAXES.csv')

df11 = pd.read_excel(newXls, 'LOCAL')
df11.to_csv('LOCAL.csv')

df12 = pd.read_excel(newXls, 'HEALTH')
df12.to_csv('HEALTH.csv')

df13 = pd.read_excel(newXls, 'SOCIOECONOMIC')
df13.to_csv('SOCIOECONOMIC.csv')