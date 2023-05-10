import numpy as np
import matplotlib.pyplot as plt
import csv
data = np.genfromtxt('feature_acc_100_3000.csv',delimiter=',')#画IFS过程
print(data.shape)
x = data[:,0]
y = data[:,1]
plt.figure()
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.plot(x,y)
plt.savefig('myrfe.jpg',dpi=600)
plt.show()