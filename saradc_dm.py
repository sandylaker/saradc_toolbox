import numpy as np
import matplotlib.pyplot as plt
from saradc import SarAdc

#  Debug Mode of SAR ADC Class,inheriting from SAR_ADC

class SarAdcDM(SarAdc):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def dm_plot(self,vin):
        '''
        debug mode plot:plot every step of adc process of given single input
        analog value
        :param vin: input analog value
        :return: two subplots
        '''
        dOutputList = self.sar_adc(vin)
        print('dOutputList',dOutputList)
        vref = self.vref
        vcm = self.vcm
        n = self.n
        weights = self.weights

        x_sequence = list(np.arange(n+1))
        dOutputList = [[int(k) for k in dOutputList[m]] for m in range(len(dOutputList))]
        vdac = [self.dac(dOutputList[i]) for i in range(n)]
        print('vdac',vdac)
        vdac =[0.5*vref] + vdac
        vx = [i - vin for i in vdac]

        # start to plot 2 diagrams, 1st one for dac output, 2nd for voltage at point X
        ax1 = plt.subplot(121)
        ax1.set_title('DAC Output')
        ax1.step(x_sequence,vdac,color='b',label='v_dac',linewidth=0.8,where='post')
        ax1.axhline(vin,linestyle='--',color = 'g',linewidth=0.8,label = 'v_in:%5.3f'%(vin))
        ax1.set_xlabel('Number of DAC Clock')
        ax1.set_ylabel('U/V')
        ax1.grid(linestyle=':')
        ax1.set_xticks(list(np.arange(n + 1)))
        ax1.legend(fontsize='xx-small')

        ax2= plt.subplot(122)
        ax2.set_title('Curve of v_x')
        ax2.step(x_sequence, vx, color='r', label='v_x', linewidth=0.8, where='post')
        ax2.set_xlabel('Number of DAC Clock')
        ax2.set_ylabel('U/V')
        ax2.grid(linestyle=':')
        ax2.set_xticks(list(np.arange(n + 1)))
        ax2.legend(fontsize='xx-small')
        plt.tight_layout()


