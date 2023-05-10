import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import time
from collections import Counter
from imblearn.over_sampling import SMOTE

data = np.load('data/f_rfe_1450_lsvc.npy')
label = np.load('data/label.npy')
data = MinMaxScaler().fit_transform(data)
label = label.astype('int64')
print(f'data shape is {data.shape}, label shape is {label.shape}')
#data, label = SMOTE(sampling_strategy={0: 100, 3: 100, 4: 100, 5: 100, 6: 100}, random_state=30).fit_resample(data, label)
#data, label = NearMiss(sampling_strategy={1: 150, 2: 150, 7: 150}).fit_resample(data, label)
sample_count = Counter(label)
n_classes = 8
sample_weights = {i: len(label)/sample_count[i]/8 for i in range(n_classes)}

lr = LogisticRegression(max_iter=10000, class_weight=sample_weights)
ri = RidgeClassifier(class_weight=sample_weights)
lgb = LGBMClassifier(class_weight=sample_weights)
lsvc = LinearSVC(class_weight=sample_weights, max_iter=10000)
mlp = MLPClassifier(max_iter=400)
rf = RandomForestClassifier(class_weight=sample_weights)
svc = SVC(class_weight=sample_weights)
xgb = XGBClassifier()
dt = DecisionTreeClassifier(class_weight=sample_weights)

model = svc

#param_grid = {'n_estimators': [k for k in range(50, 200,5)]}
param_grid = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'C':[pow(2, k) for k in range(-5, 15)],
               'gamma':[pow(2, k) for k in range(-15, 5)]}]
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
t1 = time.perf_counter()
grid_search.fit(data, label)
t2 = time.perf_counter()
print(grid_search.best_params_)
print(grid_search.best_score_)
print(f'time cost is {t2-t1:.3f}')
