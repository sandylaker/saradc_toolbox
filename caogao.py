import numpy as np
import matplotlib.pyplot as plt
from saradc_cm import SarAdcCM
from saradc import SarAdc
import time
from assistantModule import bin_array, getDecisionLvls, fastConversion


# adc = SarAdcCM(mismatch=0.1)
# start_time = time.time()
# vin_sampled = adc.sampling()
# dacOut = adc.getAnalogOutput()
# adc.fftPlot()
# print('elapsed time: %s seconds'%(time.time()-start_time))
# plt.show()


adc = SarAdc(mismatch=0.01)
# start_time_1 = time.time()
# f1 = plt.figure(1)
# adc.plotDnlInl()
# print('elapsed time: %s seconds'%(time.time()-start_time_1))
start_time_2 = time.time()
f2 = plt.figure(2)
adc.plotFastDnlInl()
print('elapsed time: %s seconds'%(time.time()-start_time_2))
plt.show()
print('succesfully connected to Github')

# weights = adc.weights
# biArray = np.concatenate(([0]*4,[1],[0]*7))
# print('biArray: ',biArray)
# analogValue = np.inner(biArray,weights)* adc.vref
# print('analogValue: ',analogValue)
# transLvls = np.inner(bin_array(4096,12),weights)* adc.vref
# compareResult = np.array(np.greater_equal(analogValue,transLvls),dtype=np.int64)
# decimalOutput = np.sum(compareResult)-1
# print('decimalOutput: ',decimalOutput)
# rightOutput = np.inner(biArray,np.array([2**(11-i) for i in range(12)]))
# print('Right decimal Output:',rightOutput)

# start_time = time.time()
#
# adc = SarAdc(vref=128,n=7,mismatch=0.01)
# vref = adc.vref
# n= adc.n
# weights = adc.weights
# analogSamples = np.random.uniform(0,128,40)
# conversionResult = fastConversion(analogSamples,weights,n,vref)
# for i in range(len(analogSamples)):
#     print('%5.3f => %d'%(analogSamples[i],conversionResult[i]))
#
# print('elapsed time: %s seconds'%(time.time()-start_time))






