import math

import numpy as np

pi = math.pi
def pdf(x,eql,s):
    sqrt_2pi = math.sqrt(2*pi)
    index = -0.5*((x-eql)/s)*((x-eql)/s)
    data = sqrt_2pi*np.exp(index)
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