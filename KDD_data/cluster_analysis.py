# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:47:15 2021

@author: Theresa
"""
# sklearn >= 0.24
from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import completeness_score, homogeneity_score, mutual_info_score, fowlkes_mallows_score

from matplotlib import cm
cmap = cm.get_cmap('gist_rainbow')

def getData():
    kdd = fetch_kddcup99(as_frame = True)
    print("fetched")
    
    for c in kdd.data.columns:
        if kdd.data[c].dtype == type(object):
            le = preprocessing.LabelEncoder()
            kdd.data.loc[:,c] = le.fit_transform(kdd.data[c])
    labels = le.fit_transform(kdd.target)
    return kdd, labels

def plot_2d(data, dim1, dim2, features):
    plt.scatter(data[:,dim1], data[:,dim2], s=20)
    x = features[dim1]
    plt.ylabel(x)
    y = features[dim2]
    plt.xlabel(y)
    plt.show()
    return

kdd, labels = getData()
print('features:', kdd.feature_names)

classes, n_class_sizes = np.unique(kdd.target, return_counts = True)
n_classes = len(classes)

for t in zip(classes, n_class_sizes):
    print(f'class: {t[0].astype(str)}, size: {t[1]}')

# kmeans
#n_clusters = 5
#kn = KMeans(n_clusters = n_clusters)
#kn.fit(kdd.data)

#cluster = kn.predict(kdd.data)
#centers = kn.cluster_centers_
#cluster_id, cluster_sizes = np.unique(kn.labels_, return_counts=True)

# k means for different cluster sizes with metric evaluation
scores = [[], [], [], []]

for k in range(1,23):
    kn = KMeans(n_clusters = k)
    y_pred = kn.fit_predict(kdd.data)
    #y_pred = kn.predict(kdd.data)
    cs = completeness_score(labels, y_pred)
    hom = homogeneity_score(labels, y_pred)
    mi = mutual_info_score(labels, y_pred)
    fms = fowlkes_mallows_score(labels, y_pred)
    print(f'{k} clusters: cs={cs}, hom={hom}, mi={mi}, fms={fms}')
    scores[0] += [cs]
    scores[1] += [hom]
    scores[2] += [mi]
    scores[3] += [fms]

metrics = ['completeness', 'homogeneity', 'Mutual Info', 'Fowlkes Mallows']
plt.figure(figsize=(9,6))
for i in range(4):
    plt.plot(np.arange(1,len(scores[0])), scores[i], label = metrics[i])
plt.xlabel('Cluster size', fontsize=15)
plt.ylabel('Metric result', fontsize=15)
plt.legend(fontsize=15)
plt.show()

'''
features = kdd.feature_names
dim1 = 4    # 4 good
dim2 = 8 # 8,7,6,0 bad
f, ax = plt.subplots(nrows=1, ncols = 2, sharey=True, figsize = (12,9))

rgba_values = cmap(np.arange(n_clusters) / n_clusters)
c = rgba_values[kn.labels_] # color for each point, points with same label have the same color
ax[0].scatter(kdd.data[features[dim1]], kdd.data[features[dim2]], color=c, s=5)
ax[0].set_xlabel(features[dim1])
ax[0].set_title("clusters")
ax[0].set_ylabel(features[dim2])

# true labels
rgba_values = cmap(np.arange(n_classes) / n_classes)
c = rgba_values[labels] # color for each point, points with same label have the same color
ax[1].scatter(kdd.data[features[dim1]], kdd.data[features[dim2]], color=c, s=5)
ax[1].set_xlabel(features[dim1])
ax[1].set_ylabel(features[dim2])
ax[1].set_title("true classes")

plt.show()
#dim1 = 0
#dim2 = 1
#plot_2d(centers, dim1, dim2, features)
#plot_2d(centers, 1, 2, features)
'''