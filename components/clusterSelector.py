import numpy as np
from learnwd.components.featureExtractor import aggressorExtractor

def deleteInvalidItem(tag,address):
    return np.delete(tag,np.where(tag[:,-1]==address),axis=0)

def findFinalCluster(data_value,center):

    data_array = aggressorExtractor(data_value)
    min_distance = 1000
    index = 0
    cluster_num = index

    for tmp in center:
        if tmp[0] != 10000:
            distance = np.sum(data_array * tmp)
            if distance < min_distance:
                min_distance = distance
                cluster_num = index
        index += 1
    return cluster_num

def writeValue(value, km_center, address_table, items_per_cluster):
    for i in range(len(items_per_cluster)):
        if items_per_cluster[i] <= 0:
            km_center[i][0] = 10000
    cluster_index = findFinalCluster(value, km_center)
    write_address_index = np.searchsorted(address_table[:, 0], cluster_index)
    write_address_index = write_address_index if write_address_index < len(address_table[:, 0]) else 0
    address = address_table[:, -1][write_address_index]
    address_table = deleteInvalidItem(address_table, address)
    items_per_cluster[cluster_index] -= 1
    return address_table, address, items_per_cluster