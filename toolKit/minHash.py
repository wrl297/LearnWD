import numpy as np

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

def byteToArray(data_value,hm_flag = True):
    #Hex -> Bin
    if hm_flag:
        bin_data = bin(int('1' + data_value + 'A', base=16))[3:-4]
    else:
        bin_data = bin(int('1' + data_value + 'A', base=16))[2:-3]
    #trans to numpy array
    data_array = np.array(cut(bin_data, 1)).astype(int)
    return data_array

def pickNaiveFeature(data_value):
    return byteToArray(data_value)

def shuffleFunc(N,M,length=512):
    range_hash = int(2**M)
    hash_func = []
    for i in range(N):
        #permutaion:生成随机序列
        hash_func.append(np.random.permutation(length)[:range_hash])
    return np.array(hash_func)


#value:待计算哈希值的数据
#N：运用N个哈希函数
#M：哈希函数占M bit的标志位
def minHashTag(hash_value,hash_func):
    N = len(hash_func)
    M = len(hash_func[0])
    all_hash = np.zeros((len(hash_value),N)) + M-1
    for v in range(len(hash_value)):
        value = hash_value[v]
        data_array = pickNaiveFeature(value)

        for i in range(N):
            shuffle_array = data_array[hash_func[i]]
            for j in range(len(shuffle_array)):
                if shuffle_array[j] == 1:#1:
                    all_hash[v][i] = j
                    break
    return all_hash

def minHashTagPart(hash_value,hash_func):
    N = len(hash_func)
    M = len(hash_func[0])
    all_hash = np.zeros((len(hash_value),N)) + M-1
    for v in range(len(hash_value)):
        value = hash_value[v]
        data_array = pickNaiveFeature(value)

        for i in range(N):
            shuffle_array = data_array[i*(512//N):(i+1)*(512//N)][hash_func[i]]
            for j in range(len(shuffle_array)):
                if shuffle_array[j] == 1:#1:
                    all_hash[v][i] = j
                    break
    return all_hash


