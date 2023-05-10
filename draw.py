import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

def draw_roc(y_ture,y_pred,n_classes,class_names):
    # 将标签二值化
    y_ture = label_binarize(y_ture, classes=[i for i in range(n_classes)])
    #y_pred = label_binarize(y_pred, classes=[i for i in range(n_classes)])
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_ture[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_ture.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 1
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average(area = {roc_auc["micro"]:0.4f})',
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average(area = {roc_auc["macro"]:0.4f})',
             color='navy', linestyle=':', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','blue','green','yellow','purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'{class_names[i]} (area = {roc_auc[i]:0.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc of each class and overall')
    plt.legend(fontsize=7, loc="lower right")
    plt.savefig('myroc.jpg', dpi=600)
    plt.show()

def draw_conf_matrix(y_true,y_pred,class_names):
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred,
                                                   display_labels=class_names, colorbar=False,
                                                   cmap=plt.cm.Blues, xticks_rotation=75)
    plt.tight_layout()
    plt.savefig('confusion_matrix.jpg', dpi=300)
    plt.show()