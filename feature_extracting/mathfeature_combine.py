import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler

def standardization(data):
    mu = np.mean(data, axis=0)
    print(mu)
    sigma = np.std(data, axis=0)
    print(sigma)
    return (data - mu) / sigma

np.seterr(divide='ignore',invalid='ignore')
file_name = 'Shannon'
raw_data = pd.read_csv(f'../data/{file_name}.csv',header=None)
data = np.array(raw_data)
data_stand = MinMaxScaler().fit_transform(data)
data_stand = np.nan_to_num(data_stand)
np.save(f'data/{file_name}',data_stand)

fourier = np.load('../data/Fourier.npy')
graph = np.load('../data/Graph.npy')
shannon = np.load('../data/Shannon.npy')
mathfea = np.hstack((fourier,graph,shannon))
print(mathfea.shape)
np.save('../data/Mathfea',mathfea)
