import math

import numpy as np

pi = math.pi
#正态分布参数估计的pdf
def pdf(x,eql,s):
    sqrt_2pi = math.sqrt(2*pi)
    index = -0.5*((x-eql)/s)**2
    data = (1/(sqrt_2pi*s))*np.exp(index)
    return data

def calculateMRV(rgb,cov,rgb_eql):
    for i in range(len(rgb)):
        cov = cov + np.dot((rgb[i] - rgb_eql).reshape(3, 1), (rgb[i] - rgb_eql).reshape(1, 3))
    return cov/255

def savepara(gr_eql,gr_s,gw_eql,gw_s):
    for i in range(1000):
        if abs(pdf(i / 1000, gr_eql, gr_s) - pdf(i / 100, gw_eql, gw_s)) < 0.00001:
            break
    return i/1000

#无参估计的直方图估计pdf
def histogram(x, array):
    array_sort = np.sort(array)
    N = len(array_sort)
    start = array_sort[0]
    end = array_sort[-1]
    data = []
    index = 25
    V = (end-start)/index 
    a = 0
    for j in range(1, index):
        for i in range(a, N):
            if array_sort[i] >= start+j*V:
                data.append((i-a)/N)
                a = i
                break
    data.append((N-a)/N)
    if x < start:
        return data[0]
    elif x > end:
        return data[-1]
    else:
        for k in range(index):
            if x > start+k*V and x < start+(k+1)*V:
                return data[k]
            

def parzen(x, array, s):
    N = len(array)
    data_sum = 0
    std = s/10
    for i in range(N):
        index = -0.5*((x-array[i])/std)**2
        sqrt_2pi = math.sqrt(2*pi)
        data_sum += (1/(sqrt_2pi*std))*np.exp(index)

    return data_sum/N
         





