import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("Concrete_Data.csv",engine = 'python');
columnNames = list(df1.head(0))
for i in range(8):
	plt.plot(df1.iloc[:,8],df1.iloc[:,i],'ro',ms=0.5)
	plt.xlabel(columnNames[8])
	plt.ylabel(columnNames[i])
	plt.savefig('plotted_graph/testplot'+str((i+1))+',9.png')
	plt.cla()
	plt.clf()

