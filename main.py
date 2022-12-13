
from learnwd.components.modelTrainer import modelOut
from learnwd.components.clusterSelector import writeValue
from learnwd.toolKit.expOption import writeDisturbance

import numpy as np

def file_load(file_name):
    data = []
    with open(file_name,'r') as f:
        while True:
            line = f.readline().strip('\n')
            if not line:
                break
            data.append(line.split(','))
    return data

def randomWD(old_data_file,new_data_file):
    old_data = file_load(old_data_file)
    new_data = file_load(new_data_file)

    disturbance_number = 0
    for i in range(len(new_data)):
        disturbance_number += writeDisturbance(new_data[i][0],old_data[i][1])
    return disturbance_number / len(new_data)

def clusterWD(old_data_file,write_quene_file):
    old_data = file_load(old_data_file)
    old_data = np.array(old_data)

    write_quene = file_load(write_quene_file)

    disturbance_number = 0
    for i in range(len(write_quene)):

        write_address = write_quene[i][0]
        overwritten_data = old_data[:,1][np.where(old_data[:,0]==write_address)[0]][0]


        disturbance_number += writeDisturbance(write_quene[i][1], overwritten_data)
    return disturbance_number / len(write_quene)

if __name__ == '__main__':

    #load the stale data and train the stale data
    old_data_table = file_load('./data/sample_old.txt')
    km_model, address_table, items_per_cluster = modelOut(old_data_table,K=16)

    #load the new data
    new_data = file_load('./data/sample_new.txt')
    km_center = km_model.cluster_centers_


    #select cluster for each new value
    file_write_quene = open('./data/sample_write.txt','w')
    for i in range(len(new_data)):
        address_table, write_address, items_per_cluster = writeValue(new_data[i][0],km_center,address_table,items_per_cluster)
        file_write_quene.write(str(hex(write_address))+','+new_data[i][0]+'\n')
    file_write_quene.close()

    #verify write disturbance errors

    wd_r = randomWD('./data/sample_old.txt', './data/sample_new.txt')
    wd_c = clusterWD('./data/sample_old.txt', './data/sample_write.txt')
    print("Random select errors",wd_r,"LearnWD errors",wd_c,"reduction ratio:",1-wd_c/wd_r)
    

