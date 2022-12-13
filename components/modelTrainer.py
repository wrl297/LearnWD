import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from learnwd.components.featureExtractor import disturbanceVector

def predictInvalid(km_center,data_array):
    min_distance = 1000
    cluster_index = 0
    i = 0
    for center in km_center:

        tmp_distance = euclidean_distances([center],[data_array])[0][0]
        if tmp_distance < min_distance:
            min_distance = tmp_distance
            cluster_index = i
        i += 1
    return cluster_index

def updateCenter(km_center,cluster_index,bias_array,items_num,insert_num=1):
    km_center[cluster_index] = (km_center[cluster_index] * items_num + bias_array) / (items_num+insert_num)
    return km_center

def insertInvalid(address_table, address, value, km_center, items_per_cluster):
    data_array = disturbanceVector(data_value=value)
    cluster_index = predictInvalid(km_center, data_array)
    insert_index = np.searchsorted(address_table[:, 0], cluster_index)
    insert_items = np.c_[cluster_index,address]
    address_table = np.insert(address_table, insert_index, insert_items[0], axis=0)
    items_num = items_per_cluster[cluster_index]
    km_center = updateCenter(km_center,cluster_index,data_array,items_num)
    return address_table, km_center, items_per_cluster

def trainModel(old_data_value,K):
    data_array = []
    for data_value in old_data_value:
        data_array.append(disturbanceVector(data_value))
    data_array = np.array(data_array)
    km_model = KMeans(n_clusters=K, max_iter=100, init='k-means++')
    km_model.fit(data_array)
    return km_model

def modelTag(km_model):
    data_label = km_model.labels_
    model_tag = []
    for tag in data_label:
        model_tag.append(tag)
    return model_tag

def modelOut(data_table,K):
    data_table = np.array(data_table)
    old_data_value = data_table[:,1]
    address = data_table[:,0]
    for i in range(len(address)):
        address[i] = int(address[i],base=16)
    address = address.astype('int')

    km_model = trainModel(old_data_value, K)
    model_tag = modelTag(km_model)
    model_tag = np.array(model_tag)
    address_table = np.c_[model_tag, address]
    address_table = address_table[np.argsort(model_tag)]
    items_per_cluster = np.zeros(K).astype('int')
    for i in range(K):
        items_per_cluster[i] = np.sum(address_table[:, 0] == i)
    return km_model, address_table, items_per_cluster

