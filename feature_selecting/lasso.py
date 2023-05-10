import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

kmer = np.load(r'8merifs.npy')
print(f'kmer shape is {kmer.shape}')
#DACC = np.genfromtxt(r'C:\Users\Amber\Desktop\feature\DACC.txt',delimiter=' ')
DACC = np.load(r'../data/DACC_10.npy')
print(f'DACC shape is {DACC.shape}')
CTD = np.genfromtxt(r'../data/CTD.txt',delimiter=' ')
print(f'CTD shape is {CTD.shape}')
#label = np.genfromtxt(r'C:\Users\Amber\Desktop\feature\label.txt',delimiter=' ')
Mathfea = np.load('../data/Mathfea.npy')
label = np.load(r'../data/label.npy')
label = label.astype('int64')
data = np.hstack((kmer,DACC,CTD,Mathfea))
data = MinMaxScaler().fit_transform(data)
print(data.shape)
pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv=5,
                      scoring='neg_mean_squared_error',
                      verbose=3)
search.fit(data,label)
print(search.best_params_)
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print(importance)