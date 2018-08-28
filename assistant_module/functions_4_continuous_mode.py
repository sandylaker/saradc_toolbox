import numpy as np
import matplotlib.pyplot as plt
"""
this assistant module contains the functions for the continuous mode of SAR ADC.
The functions are used in the class SarAdc and SarAdcDifferential.
"""


def sampling(adc):
    """
    get a array of sampled input of a sine wave
    :param: adc : instance of class SarAdc
    :return: a array of analog values
    """
    Ts = 1 / adc.fs  # period of sampling
    t = np.arange(0, adc.fft_length * Ts, Ts)  # time array
    sine_amp = adc.vref / 2  # amplitude of sine wave
    vin_sampled = sine_amp * np.sin(2 * np.pi * adc.fin * t) + adc.vcm
    return vin_sampled


def get_digital_output(adc):
    """
    convert sampled input into a list of digital values
    @:param adc: instance of class SarAdc
    """
    d_output = []
    vin_sampled = sampling(adc)
    for i in range(adc.fft_length):
        # for a single analog input, list of digital output in n dac clocks
        register = adc.sar_adc(vin_sampled[i])
        d_output += [register[-1]]
    return d_output


def get_analog_output(adc):
    """
    this is a copy of the DAC block in SarAdc Class,in order to convert a list of digital values(data type String)
    into analog values
    :param adc: instance of class SarAdc
    :return: a list of analog values
    """
    d_output = get_digital_output(adc)
    # convert list of string to list of list of int
    d_output = [[int(i) for i in d_output[j]] for j in range(len(d_output))]
    a_output = [np.inner(x, adc.vref/2/2**np.arange(adc.n)) for x in d_output]
    return a_output


def get_fft_output(adc):
    """
    get the output of FFT
    :param adc: instance of class SarAdc
    :return: a array of complex number. rfft function is applied here.
    """
    a_output = get_analog_output(adc)
    a_output_windowed = np.multiply(a_output,adc.window)
    fft_output = np.fft.rfft(a_output_windowed)
    return fft_output


def plot_fft(adc, ax):
    """
    plot the FFT spectrum.
    :param adc: instance of SarAdc
    :param ax: axes of plot
    :return: a plot.
    """
    fft_output = get_fft_output(adc)
    print('len(fft_output)', len(fft_output))
    SNR = snr(adc, fft_output)
    enob = (SNR - 1.76)/6.02
    magnitude = np.abs(fft_output)
    sine_amp = adc.vref/2
    ref = np.sum(adc.window)/2 * sine_amp
    s_dbfs = 20 * np.log10(magnitude/ref)

    freq = np.arange((adc.fft_length/2)+1) / (float(adc.fft_length)/adc.fs)
    freq_unit = 'Hz'
    if freq[-1] > 1e6:
        freq = freq/1e6     # in MHz
        freq_unit = 'MHz'
    elif (freq[-1] > 1e3) and (freq[-1] < 1e6):
        freq = freq/1e3     # in kHz
        freq_unit = 'kHz'

    ax.plot(freq[1:], s_dbfs[1:])
    ax.set_xlabel(freq_unit)
    ax.set_ylabel('dBFS')
    ax.set_title('FFT Plot with mismatch %.4f' % adc.mismatch)
    ax.text(0.1,  0.9, 'SNR:%5.3f' % SNR, fontsize='small', transform=ax.transAxes)
    ax.text(0.1, 0.85, 'ENOB:%5.3f' % enob, fontsize='small', transform=ax.transAxes)
    ax.text(0.1, 0.80, 'fin:%2.3EHz' % adc.fin, fontsize='small', transform=ax.transAxes)


def snr(adc, fft_output):
    """
    calculate the SNR
    :param adc: an instance of SAR ADC Class
    :param fft_output: Output of FFT
    :return: SNR
    """
    if not adc.window_boolean:
        signal_bins = [adc.prime_number]
        noise_bins = np.hstack((np.arange(1, adc.prime_number), np.arange(adc.prime_number+1, len(fft_output))))
    else:
        signal_bins = np.arange(adc.prime_number, adc.prime_number+3)
        noise_bins = np.hstack((np.arange(1, adc.prime_number), np.arange(adc.prime_number+3, len(fft_output))))
    signal = np.linalg.norm(fft_output[signal_bins])
    noise = np.linalg.norm(fft_output[noise_bins])
    # print('noise:',noise)
    SNR = 20 * np.log10(signal/noise)
    return SNR