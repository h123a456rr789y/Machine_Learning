
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with open(r'../origin_data/iris.data',newline='') as csvfile:
    rows=csv.reader(csvfile,quotechar='"')
    for row in rows:
        if row:
            xs = float(row[0])  #pl
            ys = float(row[1])  #pw
            zs = float(row[2])  #sl
            if row[4] == "Iris-setosa":
                ax.scatter(xs,ys,zs,c='r',marker='o')
            elif row[4] == "Iris-versicolor":
                ax.scatter(xs,ys,zs,c='g',marker='^')
            else:
                ax.scatter(xs,ys,zs,c='b',marker='x')  
ax.set_xlabel('PL')
ax.set_ylabel('PW')
ax.set_zlabel('SL')

plt.show()

