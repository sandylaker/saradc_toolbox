import numpy as np
import matplotlib.pyplot as plt
from sympy import randprime
from assistant_module import *
import assistant_module.functions_4_continuous_mode as cm
import time


class SarAdc:

    def __init__(self, vref=1.2, n=12, radix=2, mismatch=0.001, structure='conventional',
                 fft_length=4096, fs=50e6, prime_number=1193, window=None,):
        self.vref = vref    # reference voltage
        self.n = n      # resolution of ADC
        self.vcm = vref/2       # common mode voltage
        self.mismatch = mismatch
        self.radix = radix
        self.structure = structure
        # attention: if structure == 'split array', the capacitor array is LSB-Array + attenuator + MSB-Array,
        # in which the position of MSB and LSB are exchanged if comparing with other structures.
        # However in the weights array, the first element corresponds to the MSB and the last element corresponds
        # to the LSB, which accords with the other structures.
        cap_array, weights = cap_array_generator(self.n, self.radix, self.mismatch, structure=structure)
        self.cap_array = cap_array
        self.weights = weights
        # print('cap_array', self.cap_array)
        # print('weights', self.weights)

        self.fft_length = fft_length  # length of FFT
        self.fs = fs  # sampling frequency
        self.prime_number = prime_number  # the number to determine the input frequency by coherent sampling
        self.fin = fs * prime_number / fft_length  # input frequency by coherent sampling
        if not window:  # add no window
            self.window_boolean = False
            self.window = np.ones(self.fft_length)
        else:
            self.window_boolean = True
            self.window = np.hanning(self.fft_length)

    def comparator(self, a, b):
        """
        compute the difference of input and dac output
        :param a: the first input
        :param b: the second input
        :return: the difference of two inputs. data type: float
        """
        difference = a - b
        return difference

    def dac(self, digits):
        """
        convert a list of digit bits into analog value.
        :param digits: a list of ints.
        :return: analog value
        """
        weights = self.weights
        return np.inner(weights, np.array(digits)) * self.vref

    def sar_logic(self, diff):
        """ 
        the logic to determine and update the digit in each dac clock.
        the i-th switch of the circuit are initially set up to 1 in i-th dac clock,
        if the difference <0 , switch are set to 0
        if the difference >=0, switch doesn't switch
        :param diff: difference return from the method comparator
        :return: single digit value
        """
        if diff < 0:
            return 0
        else:
            return 1

    def sar_adc(self, vin):
        """
        the main function of SAR ADC
        :param vin: input analog voltage
        :return: a list, in which each element is a list of string which represents
                 the digital output of each dac clock. e.g. ['100','010','011']
        """
        digital_output_list = []
        digital_output = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            digital_output[i] = 1
            vdac = self.dac(digital_output)
            diff = self.comparator(vin, vdac)
            digital_output[i] = int(self.sar_logic(diff))
            digital_output_list += [''.join(map(str, digital_output))]
        return digital_output_list

    def dnl(self, resolution=0.01, method='fast'):
        """
        calculate the differential non linearity.
        :param resolution: represents the ratio between the error of DNL and ideal LSB.
        :param method: 'fast': fast algorithm, which computes the transition levels directly
                       'iterative': find the transition levels iteratively
                       'code_density': based on the code density theory.
        :return: a list of DNLs.
        """
        if method == 'iterative':
            lsb_ideal = self.vref/(2**self.n)
            step = lsb_ideal * resolution
            ramp = np.arange(0, self.vref, step)

            # digital_output = np.asarray([self.sar_adc(x)[-1] for x in ramp])
            # digital_output = np.asarray([list(x) for x in digital_output], dtype=np.int64)
            # exp_array = np.asarray([2**(self.n-1-i) for i in range(self.n)])
            # digital_output_decimal = np.inner(exp_array, digital_output)

            # if resolution < 0.01, a huge amount of calculation is required,
            # so use for loop to calculate for the sub-array, in order to avoid stackoverflow
            # by calling fast_conversion method directly
            if resolution < 0.01:
                digital_output_decimal = np.array([])
                for i in range(2**self.n):
                    temp = fast_conversion(ramp[int(i/resolution): int((i+1)/resolution)],
                                           weights=self.weights,n=self.n,vref=self.vref)
                    digital_output_decimal = np.concatenate((digital_output_decimal,temp))
            else:
                digital_output_decimal = fast_conversion(ramp, weights=self.weights, n=self.n, vref=self.vref)

            tran_lvls = np.array([], dtype=np.int64)
            miss_code_count = 0
            dnl = []
            for i in range(2**self.n):
                position_array = np.where(digital_output_decimal == i)      # the 'where' method returns a tuple
                if position_array[0].size != 0:
                    tran_lvls = np.append(tran_lvls, position_array[0][0])
                else:
                    miss_code_count += 1
            print('number of misscode:', miss_code_count)
            for i in range(len(tran_lvls)-1):
                dnl += [(tran_lvls[i+1] - tran_lvls[i])*resolution - 1]
            return dnl
        elif method == 'fast':
            decision_lvls = get_decision_lvls(self.weights, self.n, self.vref)
            ideal_lsb = self.vref / (2 ** self.n)
            dnl = np.diff(decision_lvls) / ideal_lsb - 1
            return dnl

        elif method == 'code_density':
            sine_amp = self.vref / 2
            fs = 5e6
            Ts = 1 / fs
            n_record = np.pi * (2 ** (self.n - 1)) * (1.96 ** 2) / (resolution ** 2)
            print('number of records: %s' % ('{:e}'.format(n_record)))
            t = np.arange(0, n_record * Ts, Ts)
            fin = randprime(0, 0.5 * n_record) / n_record * fs
            sine_wave = sine_amp * np.sin(2 * np.pi * fin * t) + self.vcm

            # digital_output = [self.sar_adc(x)[-1] for x in sine_wave]
            # bi2deDict = getbi2deDict(self.n)
            # digital_output_decimal = [bi2deDict[x] for x in digital_output]
            digital_output_decimal = fast_conversion(
                sine_wave, weights=self.weights, n=self.n, vref=self.vref)
            min_hit_num = np.amin(digital_output_decimal)
            max_hit_num = np.amax(digital_output_decimal)
            code_hist, bins = np.histogram(digital_output_decimal, np.arange(min_hit_num, max_hit_num + 1))
            ''''
            plt.hist(digital_output_decimal, np.arange(2 ** self.n))
            print('max(codHist)', np.argmax(code_hist),np.amax(code_hist))
            plt.axis([0,2**self.n, np.amin(code_hist), np.max(code_hist)])
            '''

            code_cum_hist = np.cumsum(code_hist)
            tran_lvl = -np.cos(np.pi * code_cum_hist / sum(code_hist))
            code_hist_lin = np.array(tran_lvl[1:]) - np.array(tran_lvl[:-1])
            trunc = 1
            code_hist_trunc = code_hist_lin[trunc: - trunc]
            actual_lsb = sum(code_hist_trunc) / (len(code_hist_trunc))
            dnl = np.concatenate(([0], code_hist_trunc / actual_lsb - 1))
            return dnl

    def inl(self, dnl):
        inl = np.cumsum(dnl)
        return inl

    def plot_dnl_inl(self, resolution=0.01, method='fast'):
        """
        plot the DNL and INL, here a ramp signal is applied
        :param resolution: the increment of each step of the ramp signal, divided by ideal LSB
        :param method:  'fast': use fast algorithm to find the transition level.
                        'iterative': find the transition level iteratively.
                        'code_density': based on the code density theory.
        :return: two subplots.
        """
        dnl = self.dnl(resolution, method)
        inl = self.inl(dnl)

        ax1 = plt.subplot(211)
        ax1.plot(np.arange(len(dnl)), dnl, linewidth=0.5)
        ax1.grid()
        ax1.set_title('DNL(mismatch: %5.3f)' % self.mismatch)
        ax1.set_xlabel('Digital Output Code')
        ax1.set_ylabel('DNL (LSB)')

        ax2 = plt.subplot(212)
        ax2.plot(np.arange(len(inl)), inl, linewidth=0.5)
        ax2.grid()
        ax2.set_title('INL(mismatch: %5.3f)' % self.mismatch)
        ax2.set_xlabel('Digital Output Code')
        ax2.set_ylabel('INL (LSB)')
        plt.tight_layout()

    def fast_analog_decimal(self, analog_samples):
        """
        use fast conversion algorithm to convert an array of analog values
        into decimal digital values
        :param analog_samples: an array of analog values
        :return: an array of decimal values
        """
        decimal_output = fast_conversion(analog_samples, self.weights, self.n, self.vref)
        return decimal_output

    def sampling(self):
        return cm.sampling(self)

    def get_digital_output(self):
        return cm.get_digital_output(self)

    def get_analog_output(self):
        return cm.get_analog_output(self)

    def get_fft_output(self):
        return cm.get_fft_output(self)

    def plot_fft(self):
        ax = plt.subplot(111)
        cm.plot_fft(self, ax)
        ax.grid(linestyle='-')

    def snr(self):
        return cm.snr(self, self.get_fft_output())

    def plot_energy(self):
        """
        plot the energy consumption of the possible codes with different switching methods
        """
        ax = plt.subplot(111)
        plot_energy(self.n, ax, switch='conventional', marker='v', structure=self.structure)
        plot_energy(self.n, ax, switch='split', marker='D', structure=self.structure)
        ax.legend()
        ax.grid(linestyle='-')

    def plot_burst_mode(self, v_input, switch='conventional'):
        vx_list = []
        weights_ideal = [0.5 ** (i + 1) for i in range(self.n)]
        d_output = [int(x) for x in self.sar_adc(v_input)[-1]]
        if switch == 'conventional' or switch == 'split':
            vx_list += [- v_input + self.vref/2]
            for i in range(1, self.n):
                v_x = vx_list[i - 1] + (2 * d_output[i - 1] - 1) * self.vref * weights_ideal[i]
                vx_list += [v_x]

        # concatenate vip_list(or vin_list) and its last element,
        # in order to show the voltage at point X after last comparision
        ax = plt.subplot(111)
        ax.step(np.arange(self.n+1), np.concatenate((vx_list, [vx_list[-1]])),
                 label=r'$V_{x}$', color='b', linewidth=1.5, where='post')
        ax.axhline(y=0, color='g', ls=':', linewidth=1)
        ax.grid(linestyle=':')
        ax.set_xticks(np.arange(0, self.n+1))
        ax.set_xticklabels(np.concatenate((['sample'], d_output)))
        ax.set_xlabel('Digital Output')
        ax.set_ylabel(r'Voltage ($V$)')
        ax.legend(fontsize='small')
        ax.set_title(r'Voltage at point X with input: %.3f $V$' % v_input)



if __name__ == '__main__':
    start = time.time()
    adc = SarAdc(vref=1.2, n=12, mismatch=0.1)
    # adc.plot_energy()
    # plt.show()
    f1 = plt.figure(1)
    adc.plot_dnl_inl(resolution=0.01, method='iterative')
    f2 = plt.figure(2)
    adc.plot_dnl_inl(method='fast')
    # adc.plot_fft()
    # adc.plot_burst_mode(0.8)
    print('elapsed time: %.5f seconds' % (time.time()-start))
    plt.show()








