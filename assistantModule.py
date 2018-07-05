import numpy as np
import matplotlib.pyplot as plt
import time

def de2biRange(high,n,low=0):
    deRange = np.arange(low,high)
    biCodes = [np.binary_repr(x,n) for x in deRange]
    biCodes = [[int(i) for i in biCodes[j]] for j in range(len(biCodes))]
    return biCodes

# start_time = time.time()
# bc = de2biDict(4096,12)
# print(bc)
# print(len(bc))
# print('elapsed time: %s seconds'%(time.time()-start_time))