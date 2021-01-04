import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import json


path = '/media/guillaume/Data/data/features/resnet50'
K = 10
auc_scores = {}

for class_folder in os.listdir(path):

    if os.path.isdir(os.path.join(path, class_folder)):

        features_path = os.path.join(path, class_folder)

        train = pickle.load(open(os.path.join(features_path, 'output_train.p'), 'rb'))
        test = pickle.load(open(os.path.join(features_path, 'output_test.p'), 'rb'))

        features_train = train['features']
        features_test = test['features']

        names_test = test['names']

        features_train = np.mean(features_train, axis=(1, 2))
        features_test = np.mean(features_test, axis=(1, 2))

        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(features_train)
        distances, indices = nbrs.kneighbors(features_test)

        scores = np.mean(distances, axis=1)

        labels = np.array([float("good" not in x.decode()) for x in names_test])

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        plt.savefig(f"/home/guillaume/anomaly_detection/results/detection_roc_curves/fig_{class_folder}.png")
        plt.close()

        auc_scores[class_folder] = roc_auc

json.dump(auc_scores, open("/home/guillaume/anomaly_detection/results/detection_roc_curves/roc_scores.json", "w"))


mean_res = np.mean(np.array(list(auc_scores.values())))
print(mean_res)

