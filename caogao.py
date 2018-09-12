import numpy as np
import matplotlib.pyplot as plt
from saradc import SarAdc
import time
from saradc_differential import SarAdcDifferential as SarAdcDiff
from numba import jit

@jit
def foo(n):
    rst = 0
    for i in range(n):
        rst = rst+1
    return rst

start = time.time()
rst = foo(int(1e7))
print('elapsed time %.5f seconds' %(time.time() - start))







