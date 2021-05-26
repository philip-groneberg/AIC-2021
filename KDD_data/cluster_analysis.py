# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:47:15 2021

@author: Theresa
"""
# sklearn >= 0.24
from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import completeness_score, homogeneity_score, mutual_info_score, fowlkes_mallows_score, normalized_mutual_info_score, calinski_harabasz_score, adjusted_mutual_info_score, davies_bouldin_score, silhouette_score

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
print(len(kdd.feature_names), 'features:', kdd.feature_names)

classes, n_class_sizes = np.unique(kdd.target, return_counts = True)
n_classes = len(classes)

for t in zip(classes, n_class_sizes):
    print(f'class: {t[0]}, size: {t[1]}')


# k means for different cluster sizes with metric evaluation
metrics = ['Completeness', 'Homogeneity', 'Adjusted Mutual Information', 'Fowlkes Mallows']

scores = [[], [], [], []]
rounds = 20
print(scores)
for k in range(2, 24):
    cs = 0
    hom = 0
    mi = 0
    fms = 0
    for j in range(rounds):    
        kn = KMeans(n_clusters = k)
        y_pred = kn.fit_predict(kdd.data)
        cs += completeness_score(labels, y_pred)
        hom += homogeneity_score(labels, y_pred)
        mi += adjusted_mutual_info_score(labels, y_pred)
        fms += fowlkes_mallows_score(labels, y_pred)
        
    cs = cs/rounds
    hom= hom/rounds
    mi = mi/rounds
    fms = fms/rounds
    print(f'{k} clusters: cs={cs}, hom={hom}, mi={mi}, fms={fms}')
    scores[0] += [cs]
    scores[1] += [hom]
    scores[2] += [mi]
    scores[3] += [fms]
    cluster_id, cluster_sizes = np.unique(kn.labels_, return_counts=True)
    print(cluster_sizes)

plt.figure(figsize=(9,6))
for i in range(len(metrics)):
    plt.plot(np.arange(2,24), scores[i], label = metrics[i])
plt.xlabel('Cluster size', fontsize=15)
plt.ylabel('Metric result', fontsize=15)
plt.legend(fontsize=15)
plt.show()
