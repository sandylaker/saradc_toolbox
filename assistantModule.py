import numpy as np
import matplotlib.pyplot as plt
import time

def de2biRange(high,n,low=0):
    '''
    convert an range of decimal integers elementwise to a list of binary codes.
    the returned list has a dimension of two. e.g [[0,0,1],[0,1,1]]
    :param high: the high boundary (exclusive)
    :param n: number of bits for binary conversion
    :param low: the low boundary(inclusive)
    :return: a list of binary codes, each binary code is also a list of integers,
             which has a length of n.
    '''
    deRange = np.arange(low,high)
    biCodes = [np.binary_repr(x,n) for x in deRange]
    biCodes = [[int(i) for i in biCodes[j]] for j in range(len(biCodes))]
    return biCodes

# start_time = time.time()
# bc = de2biDict(4096,12)
# print(bc)
# print(len(bc))
# print('elapsed time: %s seconds'%(time.time()-start_time))