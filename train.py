import time
from collections import Counter

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from draw import draw_roc
from evalutefunc import metric
from shap_visualization import shap_visualization

# Features and label loading
data = np.load('data/f_rfe_1100_lsvc.npy')
label = np.load('data/label.npy')
data = MinMaxScaler().fit_transform(data)
label = label.astype('int64')
#data, label = SMOTE(sampling_strategy={0: 100, 3: 100, 4: 100, 5: 100, 6: 100}, random_state=30).fit_resample(data, label)
#data, label = NearMiss(sampling_strategy={1: 150, 2: 150, 7: 150}).fit_resample(data, label)
print(f'data shape is {data.shape}, label shape is {label.shape}')
n_classes = 8
sample_count = Counter(label)
y_pred_list = []
y_true_list = []
y_pred_prob_list =[]
onehot = OneHotEncoder()
sample_weights = {i: len(label)/sample_count[i]/8 for i in range(n_classes)}
#print(f'Sample weights = {sample_weights}')
class_names = ['Chromatin', 'Cytoplasm', 'Exosome', 'Insoluble cyto', 'Membrane',
                   'Nucleolus', 'Nucleoplasm', 'Nucleus']

ri = RidgeClassifier(alpha=0.5, class_weight=sample_weights)

'''lr = LogisticRegression(C=pow(2,16), solver='lbfgs', max_iter=10000, class_weight=sample_weights)
lgb = LGBMClassifier(class_weight=sample_weights, n_estimators=75)
svc = SVC(C=64, gamma=pow(2, -5), kernel='sigmoid', class_weight=sample_weights,decision_function_shape='ovr')
rf = RandomForestClassifier(n_estimators=125, class_weight=sample_weights)
lsvc = LinearSVC(C=8, max_iter=10000, class_weight=sample_weights)
mlp = MLPClassifier(alpha=pow(2, -12), max_iter=400)
xbg = XGBClassifier(n_estimators=85)
dt = DecisionTreeClassifier(class_weight=sample_weights)'''

# Model
model = ri

#cv = LeaveOneOut().split(data, label)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=30).split(data, label)
t1 = time.perf_counter()
flag = 0
for idx, (train_index, test_index) in enumerate(cv):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    model.fit(x_train, y_train)
    '''if flag == 0:
        shap_visualization(model,x_train,x_test,class_names)
        flag = 1'''
    y_pred = model.predict(x_test)
    calibrated_model = CalibratedClassifierCV(model, cv="prefit").fit(x_test, y_test)
    y_pred_prob = calibrated_model.predict_proba(x_test)
    y_pred_list.append(y_pred)
    y_true_list.append(y_test)
    y_pred_prob_list.append(list(y_pred_prob.reshape(-1)))
    #print(f'第{idx}折准确率：{accuracy_score(y_test,y_pred)}')
t2 = time.perf_counter()
y_pred_list = np.array(y_pred_list).reshape(-1, 1)
y_true_list = np.array(y_true_list).reshape(-1, 1)
y_pred_prob_list = np.array(y_pred_prob_list).reshape(-1, n_classes)
auc = roc_auc_score(onehot.fit_transform(y_true_list).toarray(),
                    y_pred_prob_list,
                    multi_class='ovr')
y_pred_list, y_true_list = list(y_pred_list), list(y_true_list)
#print(f'pred_list shape is {np.shape(pred_list)}, y_true_list shape is {np.shape(y_true_list)}')
confusion_matrix = confusion_matrix(y_true=y_true_list, y_pred=y_pred_list)
print(confusion_matrix)
#draw_conf_matrix(y_true_list,y_pred_list,class_names)
draw_roc(y_true_list, y_pred_prob_list, n_classes,class_names)
Sn, Sp, Mcc = metric(confusion_matrix)
F1 = f1_score(y_true=y_true_list, y_pred=y_pred_list,average='weighted')
np.set_printoptions(precision=4)
print(f'Acc={accuracy_score(y_true=y_true_list, y_pred=y_pred_list):.4f}')
print(f'Sn ={Sn}')
print(f'Sp ={Sp}')
print(f'Mcc={Mcc}')
print(f'F1 ={F1:.4f}')
#print(f'AUC={auc:.4f}')
print(f'time cost is {t2-t1:.3f}s')