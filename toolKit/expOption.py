import math
import numpy as np

def count_bin_num(num):
    return bin(num)[2:].count('1')

def readLatency():
    read_latency = 100
    return read_latency

def isflipOperation(new,old,base_x):
    new = int(new,base=base_x)
    old = int(old,base=base_x)
    flip_bit = new ^ old
    set_flag = True if new&flip_bit else False
    reset_flag = True if (~new)&flip_bit else False
    return set_flag,reset_flag


def writeLatency(new,old,cacheline_size=512,base_x=16):
    reset_latency = 100
    set_latency = 200
    parallel_bit = 128
    write_latency = 0

    single_c = int(math.log(base_x,2))
    for i in range(cacheline_size//parallel_bit):
        new_part = new[i*(parallel_bit//single_c):(i+1)*(parallel_bit//single_c)]
        old_part = old[i*(parallel_bit//single_c):(i+1)*(parallel_bit//single_c)]
        set_flag,reset_flag = isflipOperation(new_part,old_part,base_x)
        write_latency += set_latency if set_flag else 0
        write_latency += reset_latency if reset_flag else 0
    return write_latency + readLatency()

def readEnergy():
    read_energy = 1.075
    return read_energy

def numFlipOperation(new,old,base_x=16):
    new = int(new, base=base_x)
    old = int(old, base=base_x)
    flip_bit = new ^ old
    set_num = count_bin_num(new&flip_bit)
    reset_num = count_bin_num((~new)&flip_bit)
    return set_num, reset_num

def writeEnergy(new,old,base_x=16):
    reset_energy = 0.013733
    set_energy = 0.0268
    fixed_energy = 4.1
    read_energy = 1.075
    set_num, reset_num = numFlipOperation(new,old,base_x)
    write_energy = fixed_energy + read_energy + set_num * set_energy + reset_num * reset_energy
    return write_energy

def writeEndurance(new,old,base_x=16):
    reset_cost = 2
    set_cost = 1
    set_num,reset_num = numFlipOperation(new,old,base_x)
    write_flip_cost = set_num * set_cost + reset_num * reset_cost
    return write_flip_cost


def failedCell(victim,error_rate=0.099):
    victim_num = np.sum(victim)
    fail_cell = np.zeros(victim_num)
    #fail_locate = np.random.choice(np.arange(victim_num),round(victim_num*error_rate),replace=False)
    fail_locate = np.where(np.random.rand(victim_num) < error_rate)
    fail_cell[fail_locate] = 1
    victim[np.where(victim == 1)] = fail_cell
    return victim

def shiftArr(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

#def writeDisturbanceFast(new_data,old_data,base_x=16,word_line_error = 0.099):


def writeDisturbance(new_data,old_data,base_x=16,word_line_error = 0.099):
    new_data = bin(int('1' + new_data, base=base_x))[3:]
    old_data = bin(int('1' + old_data, base=base_x))[3:]
    new_data = np.array(list(new_data)).astype('bool')
    old_data = np.array(list(old_data)).astype('bool')



    aggressive = ~new_data & (new_data ^ old_data)
    tmp = ~ (new_data ^ aggressive)
    victim_right = (tmp & (shiftArr(aggressive, 1, 0)))
    victim_left = (tmp & (shiftArr(aggressive, -1, 0)))
    fail_cell = failedCell(victim_right,word_line_error) | failedCell(victim_left,word_line_error)
    return np.sum(fail_cell)#,fail_cell#

def writeDisturbanceBitLine(new_data, old_data, adj1_data, adj2_data, base_x=16, bit_line_error = 0.115):
    new_data = bin(int('1' + new_data, base=base_x))[3:]
    old_data = bin(int('1' + old_data, base=base_x))[3:]
    new_data_arr = np.array(list(new_data)).astype('bool')
    old_data_arr = np.array(list(old_data)).astype('bool')

    adj1_data = bin(int('1' + adj1_data, base=base_x))[3:]
    adj1_data_arr = np.array(list(adj1_data)).astype('bool')
    adj2_data = bin(int('1' + adj2_data, base=base_x))[3:]
    adj2_data_arr = np.array(list(adj2_data)).astype('bool')

    aggressive = ~new_data_arr & (new_data_arr ^ old_data_arr)
    victim_adj1 = (~adj1_data_arr) & aggressive
    victim_adj2 = (~adj2_data_arr) & aggressive
    fail_cell_adj1 = failedCell(victim_adj1, bit_line_error)
    fail_cell_adj2 = failedCell(victim_adj2, bit_line_error)
    return fail_cell_adj1,fail_cell_adj2

def writeDisturbanceAgg(new_data, old_data, adj1_data, adj2_data, base_x=16, bit_line_error = 0.115,word_line_error = 0.099):
    new_data = bin(int('1' + new_data, base=base_x))[3:]
    old_data = bin(int('1' + old_data, base=base_x))[3:]
    new_data_arr = np.array(list(new_data)).astype('bool')
    old_data_arr = np.array(list(old_data)).astype('bool')

    adj1_data = bin(int('1' + adj1_data, base=base_x))[3:]
    adj1_data_arr = np.array(list(adj1_data)).astype('bool')
    adj2_data = bin(int('1' + adj2_data, base=base_x))[3:]
    adj2_data_arr = np.array(list(adj2_data)).astype('bool')

    aggressive = ~new_data_arr & (new_data_arr ^ old_data_arr)
    tmp = ~ (new_data_arr ^ aggressive)
    victim_right = (tmp & (shiftArr(aggressive, 1, 0)))
    victim_left = (tmp & (shiftArr(aggressive, -1, 0)))
    fail_cell = victim_left | victim_right#failedCell(victim_right, word_line_error) | failedCell(victim_left, word_line_error)

    victim_adj1 = (~adj1_data_arr) & aggressive
    victim_adj2 = (~adj2_data_arr) & aggressive
    fail_cell_adj1 = victim_adj1#failedCell(victim_adj1, bit_line_error)
    fail_cell_adj2 = victim_adj2#failedCell(victim_adj2, bit_line_error)
    return fail_cell,fail_cell_adj1,fail_cell_adj2
