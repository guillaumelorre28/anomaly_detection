import os
import json
import pickle
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn import metrics
import imageio
from skimage.transform import resize

eps = 0.01
n_features = 100
path = '/media/guillaume/Data/data/multiscale_features/resnet50/bottle'
gt_path = '/media/guillaume/Data/data/mvtec_anomaly_detection/bottle/ground_truth'

train = pickle.load(open(os.path.join(path, 'output_train.p'), 'rb'))
test = pickle.load(open(os.path.join(path, 'output_test.p'), 'rb'))

features_train = train['features']
features_test = test['features']


N = features_train.shape[0]
H = features_train.shape[1]
W = features_train.shape[2]
C = features_train.shape[3]

N_t = features_test.shape[0]

indices = np.arange(C)
np.random.shuffle(indices)
indices = indices[:n_features]
features_train = np.take(features_train, indices, axis=3)
features_test = np.take(features_test, indices, axis=3)
C = n_features

names_test = test['names']

mean = np.mean(features_train, axis=0)
# features_norm = features_train - mean
#
# cov = np.sum(np.reshape(features_norm, [N, H, W, C, 1]) * np.reshape(features_norm, [N, H, W, 1, C]), axis=0) / (N-1)

cov = [[np.cov(features_train[:, i, j].T) for i in range(H)] for j in range(W)]
cov = np.array(cov) + eps * np.reshape(np.eye(C), (1, 1, C, C))

cov_inv = [[np.linalg.inv(cov[i, j]) for i in range(H)] for j in range(W)]
cov_inv = np.array(cov_inv)

scores = [[[spatial.distance.mahalanobis(features_test[n, j, i], mean[j, i], cov_inv[j, i])
               for i in range(H)] for j in range(W)] for n in range(N_t)]
scores = np.array(scores)

print(scores.shape)

scores_image = np.max(scores, axis=(1, 2))

labels = np.array([float("good" not in x.decode()) for x in names_test])

fpr, tpr, thresholds = metrics.roc_curve(labels, scores_image)
roc_auc = metrics.auc(fpr, tpr)
print(f"ROC AUC: {roc_auc}")
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()

plt.show()

print(names_test)

gt_maps = []

for name in names_test:

    name = name.decode().split('/')
    if name[0] == "good":
        gt = np.zeros([56, 56], dtype=np.int32)
    else:
        gt_name = os.path.join(name[0], f"{name[1][:3]}_mask.png")
        gt = imageio.imread(os.path.join(gt_path, gt_name))
        gt = gt / 255
        gt = resize(gt, (256, 256))
        crop_pad = int((256-224)/2)
        gt = gt[crop_pad:crop_pad + 224, crop_pad: crop_pad + 224]
        gt = resize(gt, (56, 56))
        gt =gt.astype(np.int32)

    gt_maps.append(gt)

gt_maps = np.array(gt_maps).astype(np.float32)

gt_maps_flat = np.reshape(gt_maps, (N_t*56*56))
scores_flat = np.reshape(scores, (N_t*56*56))

gt_maps = np.array(gt_maps).astype(np.float32)

fpr, tpr, thresholds = metrics.roc_curve(gt_maps_flat, scores_flat)
roc_auc = metrics.auc(fpr, tpr)
print(f"ROC AUC: {roc_auc}")
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()

plt.show()

scores = (scores - np.min(scores))/np.ptp(scores)

visu_map = np.concatenate([scores, gt_maps], axis=2)

for score_map in visu_map:

    plt.imshow(score_map)
    plt.show()
