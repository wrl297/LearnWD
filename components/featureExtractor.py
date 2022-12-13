import numpy as np


from learnwd.toolKit.minHash import byteToArray



# disturbance vector value di = vi·(2−vi−1 −vi+1)
def disturbanceVector(stale_value):
    data_array = byteToArray(stale_value,hm_flag=False)

    feature_array = np.zeros(len(data_array) - 2)
    feature_array = feature_array - data_array[:-2] - data_array[2:]
    disturbance_vector = (feature_array + 2) * data_array[1:-1]
    return disturbance_vector

#aggressor vector value ai = 1 if vi = 0 and (vi-1 = 0 or vi+1 = 0), ai = 0 in others.
def aggressorExtractor(new_value):
    data_array = byteToArray(new_value,hm_flag=False)
    aggressive = data_array[1:-1] + (data_array[:-2] * data_array[2:])
    aggressor_vector = np.where(np.array(aggressive) >= 1, 0, 1)
    return aggressor_vector


def patternProbability(mlc_value,pattern_value):
    if mlc_value == '00':
        return 0
    else:
        if pattern_value == '00':
            return 0.246
        elif pattern_value == '01':
            return 0.312
        elif pattern_value == '11':
            return 0.552
        else:
            return 0


def pickMLCWD(data_value):
    data = bin(int('1'+data_value,base=16))[3:]
    feature_length = len(data)//2
    feature_array = np.zeros(feature_length)

    feature_array[0] = patternProbability(data[:2],data[2:4])
    feature_array[feature_length-1] = patternProbability(data[-2:],data[-4:-2])
    for i in range(feature_length-2):
        feature_array[i+1] = patternProbability(data[(i+1)*2:(i+2)*2],data[i*2:(i+1)*2]) + patternProbability(data[(i+1)*2:(i+2)*2],data[(i+2)*2:(i+3)*2])
    return feature_array

def patternAgg(mlc_value,close_mlc):
    if mlc_value == '00':
        if close_mlc == '10':
            return 0
        else:
            return 1
    else:
        return 0

def pickMLCAGG(data_value):
    data = bin(int('1'+data_value,base=16))[3:]
    feature_length = len(data) // 2
    feature_array = np.zeros(feature_length)
    feature_array[0] = patternAgg(data[:2],data[2:4])
    feature_array[feature_length - 1] = patternAgg(data[-2:],data[-4:-2])
    for i in range(feature_length - 2):
        feature_array[i+1] = patternAgg(data[(i+1)*2:(i+2)*2],data[i*2:(i+1)*2]) or patternAgg(data[(i+1)*2:(i+2)*2],data[(i+2)*2:(i+3)*2])
    return feature_array
