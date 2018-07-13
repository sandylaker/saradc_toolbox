import numpy as np
import matplotlib.pyplot as plt
from saradc_cm import SarAdcCM
from saradc import SarAdc
import time
from saradc_differential import SarAdcDifferential as SarAdcDiff
from assistantModule import bin_array, getDecisionLvls, fastConversion, capArraygenerator


# adc = SarAdcCM(mismatch=0.1)
# start_time = time.time()
# vin_sampled = adc.sampling()
# dacOut = adc.getAnalogOutput()
# adc.fftPlot()
# print('elapsed time: %s seconds'%(time.time()-start_time))
# plt.show()

# # check the dnl plot function in single ended sar adc.
# adc = SarAdc(mismatch=0.01)
# # start_time_1 = time.time()
# # f1 = plt.figure(1)
# # adc.plotDnlInl()
# # print('elapsed time: %s seconds'%(time.time()-start_time_1))
# start_time_2 = time.time()
# f2 = plt.figure(2)
# adc.plotDnlInl(resolution=0.01,method='fast')
# print('elapsed time: %s seconds'%(time.time()-start_time_2))
# plt.show()
# print('succesfully connected to Github!!!')

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

'''compare the time comsumption and results of sar_adc function of single ended
and differential SAR ADC.
'''
saradc_se = SarAdc(vref=1.2,n=12,mismatch=0)    # se stands for  single-ended
saradc_df = SarAdcDiff(vref=1.2,n=12,mismatch=0)    # df stands for differential
analog = np.random.uniform(0,1.2,100)
d_se = []
d_df = []
start_time1 = time.time()
for a in analog:
    d_se += [saradc_se.sar_adc(a)[-1]]
print('transfer time of single ended ADC: %f seconds'%(time.time() - start_time1))
start_time2 = time.time()
for a in analog:
    d_df += [saradc_df.sar_adc(a)[-1]]
print('transfer time of differential ended ADC: %f seconds'%(time.time() - start_time2))
print(np.array_equal(d_se,d_df))





