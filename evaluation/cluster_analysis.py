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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import completeness_score, homogeneity_score, mutual_info_score, fowlkes_mallows_score, normalized_mutual_info_score, calinski_harabasz_score, adjusted_mutual_info_score, davies_bouldin_score, silhouette_score

def getData():
    kdd = fetch_kddcup99(as_frame = True)
    print("fetched")
    integers = list(np.arange(4,23+1)) + [31,32]
    floats = list(np.arange(24,30+1)) + list(np.arange(33,41)) + [0]
    
    # change all floats and int from 'object' type to correct type
    for i in integers:
        c = kdd.data.columns[i]
        kdd.data[c] = kdd.data[c].astype(int)
    for i in floats:
        c = kdd.data.columns[i]
        kdd.data[c] = kdd.data[c].astype(float) 
    # use hot encoding (also called dummy encoding). puts eg for each protocol (tcp, udp, icmp) one column which is 0 or 1 -> 3 new columns
    oce = preprocessing.OneHotEncoder(sparse=False)
    dummy_encoded = oce.fit_transform(kdd.data.iloc[:,1:4].values)
    categories = list(oce.categories_[0]) + list(oce.categories_[1]) + list(oce.categories_[2])
    df_new = pd.DataFrame(dummy_encoded, columns = categories)
    
    #del kdd.data['protocol_type', 'service', 'flag']
    df_dropped = kdd.data.drop(['protocol_type', 'service', 'flag'], axis=1)
    kdd.data = df_dropped.join(df_new)
    # categorical 1,2,3
    # int 4 .. 23, 31, 32
    # float 0, 24 .. 30, 33..41
    '''
    for c in kdd.data.columns:
        if kdd.data[c].dtype == type(object):
            le = preprocessing.LabelEncoder()
            kdd.data.loc[:,c] = le.fit_transform(kdd.data[c])
    '''
    le = preprocessing.LabelEncoder()
    kdd.target = le.fit_transform(kdd.target)
    
    return kdd

def plot_2d(data, dim1, dim2, features):
    plt.scatter(data[:,dim1], data[:,dim2], s=20)
    x = features[dim1]
    plt.ylabel(x)
    y = features[dim2]
    plt.xlabel(y)
    plt.show()
    return

def find_k(data, labels):
    # k means for different cluster sizes with metric evaluation
    metrics = ['Completeness', 'Homogeneity', 'Adjusted Mutual Information', 'Fowlkes Mallows']
    
    scores = [[], [], [], []]
    rounds = 10
    print(scores)
    r = np.arange(50,101)
    for k in r:
        cs = 0
        hom = 0
        mi = 0
        fms = 0
        for j in range(rounds):    
            kn = KMeans(n_clusters = k)
            y_pred = kn.fit_predict(data)
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
        plt.plot(r, scores[i], label = metrics[i])
    plt.xlabel('Cluster amount', fontsize=15)
    plt.ylabel('Metric result', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    return scores

# load preprocessed data
kdd = getData()

data = kdd.data
labels = kdd.target
print(len(data.columns), ' features')


classes, n_class_sizes = np.unique(kdd.target, return_counts = True)
n_classes = len(classes)
for t in zip(classes, n_class_sizes):
    print(f'class: {t[0]}, size: {t[1]}')


#TODO
# train test split

# train for k=15?
k = 15
kn = KMeans(n_clusters = k)
#kn.fit(train_data)

# map cluster labels to class label according to which class has most training points in the cluster

# predict test data

# classify test point according to closest cluster label (or other methods..)


