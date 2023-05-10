import shap
from matplotlib import pyplot as plt
import numpy as np
import csv

def shap_visualization(model,x_train,x_test,class_names):
    # feature_names = []
    with open(f'feature_selecting/selected_feature_name.txt', "r") as file:
        feature_names = file.readlines()
    feature_names = [line.strip() for line in feature_names]
    explainer = shap.LinearExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, class_names=class_names,
                      feature_names=feature_names, show=False)
    plt.subplots_adjust(left=0.3, right=0.8, top=0.9, bottom=0.1)
    plt.savefig('shap.jpg', dpi=300)
    plt.show()
    # 对第三维进行求和
    shap_values = np.array(shap_values)
    mean = np.abs(shap_values).mean(1)
    sum = mean.sum(0)
    top20=list(sum.argsort()[-20:][::-1])
    for i in top20:
        print(feature_names[i])
    mean = np.transpose(mean)
    new_array = mean[top20,:]
    with open('top20.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_array)