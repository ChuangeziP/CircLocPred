import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.io import loadmat
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from lightgbm import LGBMClassifier
import time
#rfe选择的过程
kmer = np.load(r'8merifs.npy')
print(f'kmer shape is {kmer.shape}')
DACC = np.load(r'../data/DACC_10.npy')
print(f'DACC shape is {DACC.shape}')
CTD = np.genfromtxt(r'../data/CTD.txt',delimiter=' ')
print(f'CTD shape is {CTD.shape}')
Mathfea = np.load('../data/Mathfea.npy')
print(f'Mathfea shape is {Mathfea.shape}')
label = np.load(r'../data/label.npy')
data = np.hstack((kmer,DACC,CTD,Mathfea))
print(f'data shape is {data.shape}')
data = MinMaxScaler().fit_transform(data)

best_dim, best_score = 0, 0.0

model = LinearSVC(max_iter=10000)

#feature_num = [i for i in range(3000, 5001, 50)]
feature_score = []
rfe_support = []
feature_num = [1100]
for i in feature_num:
    t1 = time.perf_counter()
    rfe = RFE(model, n_features_to_select=i, step=10)
    features = rfe.fit_transform(data, label)
    score = cross_val_score(model, features, label).mean()
    feature_score.append(score)
    if score > best_score:
        best_dim, best_fea, best_score = i, features, score
        rfe_support = rfe.get_support()
    t2 = time.perf_counter()
    print(f'feature dimension {i} score = {score:.4f}, time cost is {t2-t1:.3f}s')
print(f'best feature dimension is {best_dim}, the score is {best_score:.4f}')
np.save(f'../data/f_rfe_{best_dim}_lsvc', arr=best_fea)
tuples = zip(feature_num,feature_score)
matrix = [[x, y] for x, y in tuples]
np.savetxt('feature_acc.csv', matrix,delimiter=',')
np.savetxt('feature_ranking.csv', rfe.ranking_, delimiter=',')
np.savetxt('rfe_support.csv', rfe_support,delimiter=',')