import numpy as np
import matplotlib.pyplot as plt

'''
this assistant module contains the functions for the continuous mode of SAR ADC.
The functions are used in the class SarAdc and SarAdcDifferential.
'''

def sampling(adc):
    '''
    get a array of sampled input of a sine wave
    :param: adc : instance of class SarAdc
    :return: a array of analog values
    '''
    Ts = 1 / adc.fs  # period of sampling
    t = np.arange(0, adc.fftLength * Ts, Ts)  # time array
    sinAmp = adc.vref / 2  # amplitude of sine wave
    vin_sampled = sinAmp * np.sin(2 * np.pi * adc.fin * t) + adc.vcm
    return vin_sampled

def getDigitalOutput(adc):
    '''
    convert sampled input into a list of digital values
    @:param adc: instance of class SarAdc
    '''
    dOutput = []
    vin_sampled = sampling(adc)
    for i in range(adc.fftLength):
        # for a single analog input, list of digital output in n dac clocks
        register = adc.sar_adc(vin_sampled[i])
        dOutput += [register[-1]]
    return dOutput

def getAnalogOutput(adc):
    '''
    this is a copy of the DAC block in SarAdc Class,in order to convert a list of digital values(data type String)
    into analog values
    :param adc: instance of class SarAdc
    :return: a list of analog values
    '''
    dOutput = getDigitalOutput(adc)
    # convert list of string to list of list of int
    dOutput = [[int(i) for i in dOutput[j]] for j in range(len(dOutput))]
    aOutput = [np.inner(x, adc.vref/2/2**np.arange(adc.n)) for x in dOutput]
    return aOutput

def getfftOutput(adc):
    '''
    get the output of FFT
    :param adc: instance of class SarAdc
    :return: a array of complex number. rfft function is applied here.
    '''
    aOutput = getAnalogOutput(adc)
    aOutput_windowed = np.multiply(aOutput,adc.window)
    fftOutput = np.fft.rfft(aOutput_windowed)
    return fftOutput

def plotfft(adc):
    '''
    plot the FFT spectrum.
    :param adc: instance of SarAdc
    :return: a plot.
    '''
    fftOutput = getfftOutput(adc)
    print('len(fftOutput)',len(fftOutput))
    SNR = snr(adc,fftOutput)
    enob = (SNR -1.76)/6.02
    magnitude = np.abs(fftOutput)
    sinAmp = adc.vref/2
    ref = np.sum(adc.window)/2 * sinAmp
    s_dbfs = 20 * np.log10(magnitude/ref)

    freq = np.arange((adc.fftLength/2)+1) / (float(adc.fftLength)/adc.fs)
    freq = freq/1e6     # in MHz
    plt.plot(freq[1:],s_dbfs[1:])
    plt.grid()
    plt.xlabel('frequency MHz')
    plt.ylabel('dBFS')
    plt.title('FFT Plot with mismatch %.4f'%adc.mismatch)
    plt.text(2,-20,'SNR:%5.3f'%(SNR),fontsize = 'small')
    plt.text(2,-25,'ENOB:%5.3f'%(enob),fontsize='small')
    plt.text(2,-30,'fin:%2.3EHz'%(adc.fin),fontsize= 'small')

def snr(adc,fftOutput):
    '''
    calculate the SNR
    :param adc: an instance of SAR ADC Class
    :param fftOutput: Output of FFT
    :return: SNR
    '''
    if not adc.window_boolean:
        signalBins = [adc.primeNumber]
        noiseBins = np.hstack((np.arange(1,adc.primeNumber),np.arange(adc.primeNumber+1,len(fftOutput))))
    else:
        signalBins = np.arange(adc.primeNumber,adc.primeNumber+3)
        noiseBins = np.hstack((np.arange(1,adc.primeNumber),np.arange(adc.primeNumber+3,len(fftOutput))))
    signal = np.linalg.norm(fftOutput[signalBins])
    noise = np.linalg.norm(fftOutput[noiseBins])
    #print('noise:',noise)
    SNR = 20* np.log10(signal/noise)
    return SNR
