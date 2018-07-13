import numpy as np
import matplotlib.pyplot as plt
from saradc import SarAdc

class SarAdcCM(SarAdc):
    '''
    the continuous mode of SAR ADC, which is able to calculate the dynamic specifications
    and plot the FFT.
    '''

    def __init__(self,fftLength=4096,fs=50e6,primeNumber=1193,window = None,**kwargs):
        super().__init__(**kwargs)
        self.fftLength = fftLength  # length of FFT
        self.fs = fs    # sampling frequency
        self.primeNumber = primeNumber      # the number to determine the input frequency by coherent sampling
        self.fin = fs * primeNumber/fftLength   # input frequency by coherent sampling
        if not window:  # add no window
            self.window_boolean = False
            self.window = np.ones(self.fftLength)
        else:
            self.window_boolean = True
            self.window = np.hanning(self.fftLength)

    def sampling(self):
        '''
        get a array of sampled input of a sine wave
        :return: a array of analog values
        '''
        Ts = 1/self.fs      # period of sampling
        t = np.arange(0,self.fftLength*Ts,Ts)       # time array
        sinAmp = self.vref/2    # amplitude of sine wave
        vin_sampled = sinAmp * np.sin(2*np.pi * self.fin * t) + self.vcm
        return vin_sampled

    def getDigitalOutput(self):
        '''
        convert sampled input into a list of digital values
        '''
        dOutput = []
        vin_sampled = self.sampling()
        for i in range(self.fftLength):
            # for a single analog input, list of digital output in n dac clocks
            register = self.sar_adc(vin_sampled[i])
            dOutput += [register[-1]]
        return dOutput

    def getAnalogOutput(self):
        '''
        this is a copy of the DAC block in SarAdc Class,in order to convert a list of digital values(data type String)
        into analog values
        :return: a list of analog values
        '''
        dOutput = self.getDigitalOutput()
        # convert list of string to list of list of int
        dOutput = [[int(i) for i in dOutput[j]] for j in range(len(dOutput))]
        aOutput = [np.inner(x, self.vref/2/2**np.arange(self.n)) for x in dOutput]
        return aOutput

    def getfftOutput(self):
        aOutput = self.getAnalogOutput()
        aOutput_windowed = np.multiply(aOutput,self.window)
        fftOutput = np.fft.rfft(aOutput_windowed)
        return fftOutput

    def fftPlot(self):
        fftOutput = self.getfftOutput()
        print('len(fftOutput)',len(fftOutput))
        snr = self.calSNR(fftOutput)
        enob = (snr -1.76)/6.02
        magnitude = np.abs(fftOutput)
        sinAmp = self.vref/2
        ref = np.sum(self.window)/2 * sinAmp
        s_dbfs = 20 * np.log10(magnitude/ref)

        freq = np.arange((self.fftLength/2)+1) / (float(self.fftLength)/self.fs)
        freq = freq/1e6     # in MHz
        plt.plot(freq[1:],s_dbfs[1:])
        plt.grid()
        plt.xlabel('frequency MHz')
        plt.ylabel('dBFS')
        plt.title('FFT Plot with mismatch %.4f'%self.mismatch)
        plt.text(2,-20,'SNR:%5.3f'%(snr),fontsize = 'small')
        plt.text(2,-25,'ENOB:%5.3f'%(enob),fontsize='small')
        plt.text(2,-30,'fin:%2.3EHz'%(self.fin),fontsize= 'small')

    def calSNR(self,fftOutput):
        if not self.window_boolean:
            signalBins = [self.primeNumber]
            noiseBins = np.hstack((np.arange(1,self.primeNumber),np.arange(self.primeNumber+1,len(fftOutput))))
        else:
            signalBins = np.arange(self.primeNumber,self.primeNumber+3)
            noiseBins = np.hstack((np.arange(1,self.primeNumber),np.arange(self.primeNumber+3,len(fftOutput))))
        signal = np.linalg.norm(fftOutput[signalBins])
        noise = np.linalg.norm(fftOutput[noiseBins])
        #print('noise:',noise)
        snr = 20* np.log10(signal/noise)
        return snr






