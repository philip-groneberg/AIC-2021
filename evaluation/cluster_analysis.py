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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import completeness_score, homogeneity_score, mutual_info_score, fowlkes_mallows_score, normalized_mutual_info_score, calinski_harabasz_score, adjusted_mutual_info_score, davies_bouldin_score, silhouette_score,precision_score, recall_score, accuracy_score
import time
range_k = [15, 40, 110]#np.arange(2,120, 10)
    

def getData():
    kdd = fetch_kddcup99(as_frame = True)
    print("fetched")
    integers = list(np.arange(4,23+1)) + [31,32]
    double = list(np.arange(24,30+1)) + list(np.arange(33,41)) + [0]
    
    # change all floats and int from 'object' type to correct type
    for i in integers:
        c = kdd.data.columns[i]
        kdd.data[c] = kdd.data[c].astype(np.double)
        #print(min(kdd.data[c]), max(kdd.data[c]))
        if max(kdd.data[c].values) != min(kdd.data[c].values):
            kdd.data[c] = (kdd.data[c]-min(kdd.data[c].values))/ (max(kdd.data[c].values)-min(kdd.data[c].values) )
    for i in double:
        c = kdd.data.columns[i]
        kdd.data[c] = kdd.data[c].astype(np.double) 
        kdd.data[c] = (kdd.data[c]-min(kdd.data[c].values)) / (max(kdd.data[c].values)-min(kdd.data[c].values) )
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
    labels = np.zeros(len(kdd.target))
    labels[kdd.target == 11] = 0 # negative = normal
    labels[kdd.target != 11] = 1 # positive = anomaly
    kdd.target = labels
    return kdd

def plot_2d(data, dim1, dim2, features):
    plt.scatter(data[:,dim1], data[:,dim2], s=20)
    x = features[dim1]
    plt.ylabel(x)
    y = features[dim2]
    plt.xlabel(y)
    plt.show()
    return

def get_radien(kn, train_data, y_pred):
    # [0,k-1] list of cluster labels
    labels = np.unique(kn.labels_)
    # entry for each test point assigned to the corresponding cluster: max distance to cluster centroid of all training points in that cluster, 
    radien = np.zeros(len(y_pred))
    for label in labels:
        # get all data points with that label (points that are in this cluster)
        cluster_points = train_data[kn.labels_ == label] 
        # get distance of points to each cluster centroid
        dist = kn.transform(cluster_points)
        # filter for cluster in which the data points lie and take maximum distance
        max_radius = np.max(dist[:,label])
        radien[y_pred==label] = max_radius
    return radien 

def get_dist_to_centroid(kn, data, y_pred):
    dist = kn.transform(data)
    y_dist = np.zeros(len(dist))
    for i in range(len(dist)):
        y_dist[i] = dist[i, y_pred[i]]
    #y_dist = dist[:,y_pred]
    return y_dist
        

def find_k_validation(train_data, validation_data, validation_labels):
    # k means for different cluster sizes with metric evaluation
    scores = [[], [], []]
    rounds = 1
    print(scores)
    for k in range_k:
        p = 0
        r = 0
        a = 0
        for j in range(rounds):    
            kn = KMeans(n_clusters = k)
            # cluster normal data -> only normal data in cluster after training
            kn.fit(train_data)
            # predict labels for validation set
            y_pred = kn.predict(validation_data)
            # radius of its cluster for each validation point
            y_radius = get_radien(kn, train_data, y_pred)
            # distance to cluster centroid for each validation point
            y_dist = get_dist_to_centroid(kn, validation_data, y_pred)
            # if distance bigger than radius -> difference smaller 0 -> label=1 -> anomaly
            is_anomaly = (y_radius - y_dist) < 0
            p += precision_score(validation_labels, is_anomaly)
            r += recall_score(validation_labels, is_anomaly)
            a += accuracy_score(validation_labels, is_anomaly)
            
        print(f'{k} clusters: p={p/rounds}, a={a/rounds}, r={r/rounds}')
        scores[0] += [p/rounds]
        scores[1] += [r/rounds]
        scores[2] += [a/rounds]
        #cluster_id, cluster_sizes = np.unique(kn.labels_, return_counts=True)
        print(scores)
    return scores

"""
splits data into three subsets. 
train_size in [0,1] how many of the normal data points are training samples
the rest of the normal data points is split in two equal parts for validation and testing

val_size in [0,1] how many of the anomaly samples are validation samples
rest is test samples
"""
def train_val_test_split(data, labels,  train_size, val_size):
    normal_data = data[labels == 0]
    normal_labels = labels[labels == 0]
    print(normal_data.shape, normal_labels.shape)
    
    # use train_size (0.75) of normal data for training, half of the rest (0.25) for validation and test each
    data_train, data_rest, labels_train, labels_rest = train_test_split(normal_data, 
                                                                normal_labels, test_size=1-train_size, random_state=42)
    data_normal_validation, data_normal_test, labels_normal_validation, labels_normal_test = train_test_split(data_rest, 
                                                                  labels_rest, test_size=0.5, random_state=42)
    
    data_rest = 0
    labels_rest = 0
    
    anomaly_data = data[labels == 1]
    anomaly_labels = labels[labels==1]
    
    # use val_size (0.25) of anomaly data for validation and rest for test
    data_anomaly_validation, data_anomaly_test, labels_anomaly_validation, labels_anomaly_test = train_test_split(anomaly_data,
                                                                  anomaly_labels, test_size=1-val_size, random_state=42)
    print(len(labels_anomaly_validation), len(labels_anomaly_test))
    
    # concatenate normal and anomormal data for test and validation
    data_validation = np.append(data_normal_validation, data_anomaly_validation, axis = 0)
    labels_validation = np.append(labels_normal_validation, labels_anomaly_validation)
    data_test = np.append(data_normal_test, data_anomaly_test, axis = 0)
    labels_test = np.append(labels_normal_test, labels_anomaly_test)
    return data_train, labels_train, data_validation, labels_validation, data_test, labels_test


# calculate execution time
start_time = time.time()
# load preprocessed data
kdd = getData()

data = kdd.data
labels = kdd.target
print(len(data.columns), ' features')

classes, n_class_sizes = np.unique(labels, return_counts = True)
n_classes = len(classes)
print("---------------Total--------------")
for t in zip(classes, n_class_sizes):
    print(f'class: {t[0]}, size: {t[1]}')

# split into test and train data set
data_train, labels_train, data_validation, labels_validation, data_test, labels_test = train_val_test_split(data, labels, train_size=0.75, val_size=0.25)


train_classes, train_n_class_sizes = np.unique(labels_train, return_counts = True)
print("---------------Train--------------")
for t in zip(train_classes, train_n_class_sizes):
    print(f'class: {t[0]},	size: {t[1]}')

validation_classes, validation_n_class_sizes = np.unique(labels_validation, return_counts = True)
print("---------------Validation---------")
for t in zip(validation_classes, validation_n_class_sizes):
    print(f'class: {t[0]},	size: {t[1]}')

test_classes, test_n_class_sizes = np.unique(labels_test, return_counts = True)
print("---------------Test--------------")
for t in zip(test_classes, test_n_class_sizes):
    print(f'class: {t[0]},	size: {t[1]}')

print("--- Calculation time split: %s seconds ---" % (time.time() - start_time))

# TRAINING AND VALIDATION
#scores_validation = find_k_validation(data_train, data_validation, labels_validation)

# PLOT METRICS FOR EACH K
metrics = ['Precision', 'Recall', 'Accuracy']
plt.figure(figsize=(9,6))
for i in range(len(metrics)):
    plt.plot(range_k, scores_validation[i], label = metrics[i])
plt.xlabel('Cluster amount', fontsize=15)
plt.ylabel('Metric result', fontsize=15)
plt.legend(fontsize=15)
plt.savefig("validation.pdf", pad_inches=0.05, bbox_inches='tight')
    
# TESTING WITH SOME K
test_dict = {"metrics": [], "k": [], "values": [], "round": []}
ks = [15, 40,110]
for k in ks:
    p=0
    r=0
    a=0
    rounds = 10
    for i in range(rounds):
        kn = KMeans(n_clusters = k)
        kn.fit(data_train)
        
        # test data
        y_pred = kn.predict(data_test)
        # radius of its cluster for each test point
        y_radius = get_radien(kn, data_train, y_pred)
        # distance to cluster centroid for each validation point
        y_dist = get_dist_to_centroid(kn, data_test, y_pred)
        # if distance bigger than radius -> difference smaller 0 -> label=1 -> anomaly
        is_anomaly = (y_radius - y_dist) < 0
        temp = precision_score(labels_test, is_anomaly)
        test_dict["values"].append(temp)
        p += temp
        temp = recall_score(labels_test, is_anomaly)
        test_dict["values"].append(temp)
        r += temp
        temp = accuracy_score(labels_test, is_anomaly)
        test_dict["values"].append(temp)
        a += temp
        test_dict["k"].append(k)
        test_dict["k"].append(k) 
        test_dict["k"].append(k) 
        test_dict["round"].append(i)
        test_dict["round"].append(i)
        test_dict["round"].append(i)
        test_dict["metrics"].append("precision")
        test_dict["metrics"].append("recall")
        test_dict["metrics"].append("accuracy")
    
    print("---------------Result--------------")
    print(f'k: {k}, accuracy: {a/rounds}, precision: {p/rounds}, recall: {r/rounds}')         

test_frame = pd.DataFrame(test_dict, columns = ["k", "metrics", "values"])
sns.set_theme(style="whitegrid")
#ax = sns.swarmplot(data=test_frame, x="metrics", y="values", hue="round", col="k") #, legend_out=True
ax = sns.swarmplot(data=test_frame, x="metrics", y="values", hue="k")

"""
# predict test data
result = kn.predict(data_test)
accuracy_score = accuracy_score(labels_test, result)
accuracy_score_formated = format(accuracy_score, '.7f')
accuracy_unique, accuracy_counts = np.unique(result, return_counts=True)

print(f'Accuracy:{accuracy_score_formated}')
for t in zip(accuracy_unique, accuracy_counts):
	print(f'class: {t[0]},	count: {t[1]}')

score_results = kn.score(data_test)
print("---------------Score--------------")
print(score_results)

# classify test point according to closest cluster label (or other methods..)

print("--- Calculation time total: %s seconds ---" % (time.time() - start_time))
"""