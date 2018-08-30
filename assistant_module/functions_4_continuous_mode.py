import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
"""
this assistant module contains the functions for the continuous mode of SAR ADC.
The functions are used in the class SarAdc and SarAdcDifferential.
"""


def sampling(adc, fs, fft_length, prime_number):
    """
    get a array of sampled input of a sine wave
    :param: adc : instance of class SarAdc
    :param: fs: sampling frequency
    :param: fft_length: the length of FFT
    :param: prime_number: the prime number for coherent sampling, it determines the input frequency.
    :return: a array of analog values
    """
    Ts = 1 / fs  # period of sampling
    # the input frequency of sine wave
    fin = prime_number/fft_length * fs
    t = np.arange(0, fft_length * Ts, Ts)  # time array
    sine_amp = adc.vref / 2  # amplitude of sine wave
    vin_sampled = sine_amp * np.sin(2 * np.pi * fin * t) + adc.vcm
    return vin_sampled


def get_digital_output(adc, fs, fft_length, prime_number):
    """
    convert sampled input into a list of digital values
    :param adc: instance of class SarAdc
    :param fs: sampling frequency
    :param fft_length: length of FFT
    :param prime_number: the prime number for coherent sampling, it determines the input frequency.
    :return: a list of digital output.
    """
    d_output = []
    vin_sampled = sampling(adc, fs, fft_length, prime_number)
    for i in range(fft_length):
        # for a single analog input, list of digital output in n dac clocks
        register = adc.sar_adc(vin_sampled[i])
        d_output += [register[-1]]
    return d_output


def get_analog_output(adc, fs, fft_length, prime_number):
    """
    this is a copy of the DAC block in SarAdc Class,in order to convert a list of digital values(data type String)
    into analog values
    :param adc: instance of class SarAdc
    :param fs: sampling frequency
    :param fft_length: length of FFT
    :param prime_number: the prime number for coherent sampling, it determines the input frequency.
    :return: a list of analog values
    """
    d_output = get_digital_output(adc, fs, fft_length, prime_number)
    # convert list of string to list of list of int
    d_output = [[int(i) for i in d_output[j]] for j in range(len(d_output))]
    a_output = [np.inner(x, adc.vref/2/2**np.arange(adc.n)) for x in d_output]
    return a_output


def get_fft_output(adc, fs, fft_length, prime_number, window_bool=False):
    """
    get the output of FFT
    :param adc: instance of class SarAdc
    :param fs: sampling frequency
    :param fft_length: length of FFT
    :param prime_number: the prime number for coherent sampling, it determines the input frequency.
    :param window_bool: boolean, apply hanning window if True.
    :return: a array of complex number. rfft function is applied here.
    """
    a_output = get_analog_output(adc, fs, fft_length, prime_number)
    if window_bool:
        window = np.hanning(fft_length)
    else:
        window = np.ones(fft_length)
    a_output_windowed = np.multiply(a_output, window)
    fft_output = np.fft.rfft(a_output_windowed)
    return fft_output


def plot_fft(adc, ax, fs, fft_length, prime_number, window_bool=False):
    """
    plot the FFT spectrum.
    :param adc: instance of SarAdc
    :param ax: axes of plot
    :param fs: sampling frequency
    :param fft_length: length of FFT
    :param prime_number: the prime number for coherent sampling, it determines the input frequency.
    :param window_bool: boolean, apply hanning window if True.
    :return: a plot.
    """
    fin = prime_number/fft_length * fs
    if window_bool:
        window = np.hanning(fft_length)
    else:
        window = np.ones(fft_length)
    fft_output = get_fft_output(adc, fs, fft_length, prime_number, window_bool)
    print('len(fft_output)', len(fft_output))
    SNR = snr(adc, fft_output, prime_number, window_bool)
    enob = (SNR - 1.76)/6.02

    magnitude = np.abs(fft_output)
    sine_amp = adc.vref/2
    ref = np.sum(window)/2 * sine_amp
    # mask the infinity results and fill them with a negative value
    s_dbfs = 20 * ma.log10(magnitude/ref)
    s_dbfs = s_dbfs.filled(-500)

    freq = np.arange((fft_length/2)+1) / (float(fft_length)/fs)
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
    ax.text(0.05,  0.89, 'SNR:%5.3f' % SNR, fontsize='small', transform=ax.transAxes)
    ax.text(0.05, 0.84, 'ENOB:%5.3f' % enob, fontsize='small', transform=ax.transAxes)
    ax.text(0.05, 0.79, 'fin:%2.3EHz' % fin, fontsize='small', transform=ax.transAxes)


def snr(adc, fft_output, prime_number, window_bool=False):
    """
    calculate the SNR
    :param adc: an instance of SAR ADC Class
    :param fft_output: Output of FFT
    :param window_bool: boolean, apply hanning window if True.
    :return: SNR
    """
    if not window_bool:
        signal_bins = [prime_number]
        noise_bins = np.hstack((np.arange(1, prime_number), np.arange(prime_number+1, len(fft_output))))
    else:
        signal_bins = np.arange(prime_number, prime_number+3)
        noise_bins = np.hstack((np.arange(1, prime_number), np.arange(prime_number+3, len(fft_output))))
    signal = np.linalg.norm(fft_output[signal_bins])
    noise = np.linalg.norm(fft_output[noise_bins])
    # print('noise:',noise)
    SNR = 20 * np.log10(signal/noise)
    return SNR