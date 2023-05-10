from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
import matplotlib.pyplot as plt
# load label
#data = np.load(r'C:\Users\Amber\Desktop\feature\cdkmerifs.npy')
data = np.load('cdkmerifs.npy')#画IFS过程
#print(data)
dataorder = np.load('cdkmerifsoder.npy')
print(f'best dimension is {int(dataorder[0][0])} and the acc is {dataorder[0][1]}')
kmer = np.load('../data/8merbino.npy')
kmer_sel = kmer[:, :int(dataorder[0][0])-1]
np.save('8merifs.npy', kmer_sel)
x = data[:,0]
y = data[:,1]
plt.figure()
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.plot(x,y)
plt.savefig('myifs.jpg',dpi=600)
plt.show()