import numpy as np
import matplotlib.pyplot as plt
from saradc_cm import SarAdcCM
from saradc import SarAdc
import time
from assistantModule import de2biRange


# adc = SarAdcCM(mismatch=0.1)
# start_time = time.time()
# vin_sampled = adc.sampling()
# dacOut = adc.getAnalogOutput()
# adc.fftPlot()
# print('elapsed time: %s seconds'%(time.time()-start_time))
# plt.show()


adc = SarAdc(mismatch=0.00)
#start_time = time.time()
#adc.plotDnlInl(resolution=0.01)
#print('elapsed time: %s seconds'%(time.time()-start_time))
#plt.show()

weights = adc.weights
biArray = np.concatenate(([0]*4,[1],[0]*7))
print('biArray: ',biArray)
analogValue = np.inner(biArray,weights)* adc.vref
print('analogValue: ',analogValue)
transLvls = np.inner(de2biRange(4096,12),weights)* adc.vref
compareResult = np.array(np.greater(analogValue,transLvls),dtype=np.int64)
decimalOutput = np.sum(compareResult)
print('decimalOutput: ',decimalOutput)
rightOutput = np.inner(biArray,np.array([2**(11-i) for i in range(12)]))
print('Right decimal Output:',rightOutput)









