import csv
import random
#replace the flower name with number
with open('../processed_data/iris.csv','w',newline='') as o_f:
    writer = csv.writer(o_f)
    with open('../origin_data/iris.data',newline='') as i_f:
        rows=csv.reader(i_f,quotechar='"')
        for row in rows:
            if row:
                if row[4] == 'Iris-setosa':
                    row[4]='1'
                elif row[4] == 'Iris-versicolor':
                    row[4]='2'
                else:
                    row[4]='3'
                print(row)
                writer.writerow(row)
#randomize the order of samples(for k-fold) 
file = open('../processed_data/iris.csv','r')
lines = [line for line in file.readlines() if line.strip()]
file.close()
random.shuffle(lines)
file = open('../processed_data/iris.csv','w',newline='')
file.writelines(lines)
file.close()
