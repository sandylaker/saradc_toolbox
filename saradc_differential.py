import numpy as np
import matplotlib.pyplot as plt
from sympy import randprime
from assistant_module import *
from assistant_module import functions_4_continuous_mode as cm
import time

class SarAdcDifferential:
    """
    the sar adc with differential structure
    """

    def __init__(self, vref=1.2, n=12, radix=2, mismatch=0.001, structure='differential'
                 , window=None):
        self.vref = vref
        self.vcm = vref/2
        self.n = n
        self.radix = radix
        self.mismatch = mismatch
        self.structure = structure
        cap_array, weights = cap_array_generator(self.n, self.radix, self.mismatch, structure='differential')
        self.cap_array_p, self.cap_array_n = cap_array
        self.weights_p, self.weights_n = weights
        # print('cap_array positive',  self.cap_array_p)
        # print('cap_array negative', self.cap_array_n)
        # print('weights positive', self.weights_p)
        # print('weights negative', self.weights_n)

        # if not window:  # add no window
        #     self.window_boolean = False
        #     self.window = np.ones(self.fft_length)
        # else:
        #     self.window_boolean = True
        #     self.window = np.hanning(self.fft_length)

    def comparator(self, a, b):
        """
        compute the difference of input and dac output
        :param a: the first input
        :param b: the second input
        :return: the difference of two inputs. data type: float
        """
        difference = a - b
        return difference

    def dac(self, digits, weights):
        """
        convert a list of digit bits into analog value.
        :param digits: a list of ints.
        :param weights: the binary weights of dac
        :return: analog value
        """
        return np.inner(weights, np.array(digits)) * self.vref

    def sar_logic(self, diff):

        if diff <= 0:
            return 1
        else:
            return 0

    def sar_adc(self, vin):
        """
        convert an analog value to digital value, the digital value of each successive approximation
        step is restored in a list ,which is the returned output of this function
        :param vin: single analog input value
        :return: a list of strings, which represents the digital output of each step.
                e.g ['100','110','101']
        """
        v_ip = vin
        v_in = self.vref - vin
        weights_p = self.weights_p
        weights_n = self.weights_n
        d_output_list = []
        d_output = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            d_output[i] = 1
            v_xp = self.dac(d_output, weights_p) - v_ip
            v_xn = self.vref - self.dac(d_output, weights_n) - v_in  # because of complementary state of switches
            diff = self.comparator(v_xp, v_xn)
            d_output[i] = int(self.sar_logic(diff))
            d_output_list += [''.join(map(str, d_output))]
        return d_output_list

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

            # if resolution < 0.01, a huge amount of calculation is required,
            # so use for loop to calculate for the sub-array, in order to avoid stackoverflow
            # by calling fast_conversion method directly
            if resolution < 0.01:
                d_output_decimal_p = np.array([])
                d_output_decimal_n = np.array([])
                for i in range(2**self.n):
                    temp_p = fast_conversion(ramp[int(i/resolution): int((i+1)/resolution)],
                                             weights=self.weights_p, n=self.n, vref=self.vref)
                    temp_n = fast_conversion(ramp[int(i/resolution): int((i+1)/resolution)],
                                             weights=self.weights_n, n=self.n, vref=self.vref)
                    d_output_decimal_p = np.concatenate((d_output_decimal_p, temp_p))
                    d_output_decimal_n = np.concatenate((d_output_decimal_n, temp_n))
            else:
                d_output_decimal_p = fast_conversion(ramp, weights=self.weights_p, n=self.n, vref=self.vref)
                d_output_decimal_n = fast_conversion(ramp, weights=self.weights_n, n=self.n, vref=self.vref)

            tran_lvls_p = np.array([], dtype=np.int64)
            tran_lvls_n = np.array([], dtype=np.int64)
            dnl_p = []
            dnl_n = []
            for i in range(2**self.n):
                position_array_p = np.where(d_output_decimal_p == i)      # the 'where' method returns a tuple
                position_array_n = np.where(d_output_decimal_n == i)
                # print(position_array)
                if position_array_p[0].size != 0:
                    tran_lvls_p = np.append(tran_lvls_p, position_array_p[0][0])
                else:
                    if i == 0:
                        tran_lvls_p = np.append(tran_lvls_p, 0)
                    else:
                        tran_lvls_p = np.append(tran_lvls_p, tran_lvls_p[-1])
                if position_array_n[0].size != 0:
                    tran_lvls_n = np.append(tran_lvls_n, position_array_n[0][0])
                else:
                    if i == 0:
                        tran_lvls_n = np.append(tran_lvls_n, 0)
                    else:
                        tran_lvls_n = np.append(tran_lvls_n, tran_lvls_n[-1])
            for i in range(len(tran_lvls_p)-1):
                dnl_p += [(tran_lvls_p[i+1] - tran_lvls_p[i])*resolution - 1]
            for i in range(len(tran_lvls_n)-1):
                dnl_n += [(tran_lvls_n[i+1] - tran_lvls_n[i])*resolution - 1]
            dnl = np.add(dnl_p, dnl_n)/2
            return dnl
        elif method == 'fast':
            # calculate the dnl of the positive end
            weights_p = self.weights_p
            decsion_lvls_p = get_decision_lvls(weights_p, self.n, self.vref)
            ideal_lsb = self.vref / (2 ** self.n)
            dnl_p = np.diff(decsion_lvls_p) / ideal_lsb - 1

            # calculate the dnl of the negative end
            weights_n = self.weights_n
            decision_lvls_n = get_decision_lvls(weights_n, self.n, self.vref)
            dnl_n = np.diff(decision_lvls_n) / ideal_lsb - 1
            # the dnl of the adc is the average dnl of both ends
            dnl = np.add(dnl_p, dnl_n)/2
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

            d_output = [self.sar_adc(x)[-1] for x in sine_wave]
            bi2deDict = getbi2deDict(self.n)
            d_output_decimal = [bi2deDict[x] for x in d_output]
            min_hit_num = np.amin(d_output_decimal)
            max_hit_num = np.amax(d_output_decimal)
            code_hist, bins = np.histogram(d_output_decimal, np.arange(min_hit_num, max_hit_num + 1))
            ''''
            plt.hist(d_output_decimal, np.arange(2 ** self.n))
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

    def sampling(self, fs, fft_length, prime_number):
        return cm.sampling(self, fs, fft_length, prime_number)

    def get_digital_output(self, fs, fft_length, prime_number):
        return cm.get_digital_output(self, fs, fft_length, prime_number)

    def get_analog_output(self, fs, fft_length, prime_number):
        return cm.get_analog_output(self, fs, fft_length, prime_number)

    def get_fft_output(self, fs, fft_length, prime_number, window_bool=False):
        return cm.get_fft_output(self, fs, fft_length, prime_number, window_bool)

    def plot_fft(self, fs=50e6, fft_length=4096, prime_number=1193, window_bool=False):
        ax = plt.subplot(111)
        cm.plot_fft(self, ax, fs, fft_length, prime_number, window_bool)
        ax.grid(linestyle='-')

    def snr(self, fs, fft_length, prime_number, window_bool=False):
        fft_output = self.get_fft_output(fs, fft_length, prime_number, window_bool)
        return cm.snr(self, fft_output, prime_number, window_bool)

    def plot_energy(self):
        """
        plot the energy consumption of the possible codes with different switching methods
        """
        ax = plt.subplot(111)
        plot_energy(self.n, ax, switch='conventional', marker='v', structure=self.structure)
        plot_energy(self.n, ax, switch='monotonic', marker='s', structure=self.structure)
        plot_energy(self.n, ax, switch='mcs', marker='o', structure=self.structure)
        plot_energy(self.n, ax, switch='split', marker='D', structure=self.structure)
        ax.legend()
        ax.grid(linestyle='-')

    def plot_burst_mode(self, v_input, switch='monotonic'):
        """
        given an certain analog input value, plot the voltages of point X of both sides( the inverting and non-inverting
        input of the op amp) in each conversion step. There are two switching methods for option.
        :param v_input: an analog input value
        :param switch: the method of switching.
                    'monotonic': monotonic capacitor switching:after the ADC turns off the bootstrapped switches,
                        the comparator directly performs the first comparison without switching any capacitor.
                        According to the comparator output, the largest capacitor on the higher voltage potential
                        side is switched to ground and the other one (on the lower side) remains unchanged.
                        The ADC repeats the procedure until the LSB is decided. For each bit cycle, there is only
                        one capacitor switch.
                     'conventional': the conventional switching method, similar to the conventional switching method
                        in the conventional(single-ended) structure.
                     structure( single ended).
                     'mcs': v_cm based (merged capacitor switching) method.
        :return: a step plot
        """
        vxp_list = []
        vxn_list = []
        v_ip = v_input
        v_in = self.vref - v_ip
        d_output = [int(x) for x in self.sar_adc(v_input)[-1]]
        if switch == 'monotonic':
            vxp_list += [v_ip - self.vref]
            vxn_list += [v_in - self.vref]
            for i in range(self.n-1):
                v_xp = vxp_list[-1] - d_output[i]*self.vref/(self.radix**(i+1))
                v_xn = vxn_list[-1] - (1-d_output[i]) * self.vref/(self.radix**(i+1))
                vxp_list += [v_xp]
                vxn_list += [v_xn]

        if switch == 'conventional' or switch == 'split':
            weights_ideal = [0.5**(i+1) for i in range(self.n)]
            v_xp = self.vcm - v_ip + self.vref/2
            v_xn = self.vref - v_xp
            vxp_list += [v_xp]
            vxn_list += [v_xn]
            for i in range(1, self.n):
                # the term (2*d_output[i-1] -1) aims to form the map {1: 1, 0: -1}.
                v_xp = vxp_list[i-1] + (2*d_output[i-1] - 1) * self.vref * weights_ideal[i]
                v_xn = self.vref - v_xp
                vxp_list += [v_xp]
                vxn_list += [v_xn]

        if switch == 'mcs':
            # v_xp converges to 2*v_ip - v_cm
            vxp_list += [v_ip]
            vxn_list += [v_in]
            for i in range(1, self.n):
                v_xp = vxp_list[i-1] + (2*d_output[i-1]-1) * self.vcm/(self.radix ** i)
                v_xn = self.vref - v_xp
                vxp_list += [v_xp]
                vxn_list += [v_xn]

        # concatenate vip_list(or vin_list) and its last element,
        # in order to show the voltage at point X after last comparision
        ax = plt.subplot(111)
        ax.step(np.arange(self.n+1), np.concatenate((vxp_list, [vxp_list[-1]]))
                , label=r'$V_{xp}$', color='b', linewidth=1.5, where='post')
        ax.step(np.arange(self.n+1), np.concatenate((vxn_list, [vxn_list[-1]]))
                , label=r'$V_{xn}$', color='r', linewidth=1.5, where='post')
        if switch == 'conventional' or switch == 'split':
            ax.axhline(y=self.vcm, color='g', ls=':', linewidth=1, label=r'$V_{cm}$')
        ax.grid(linestyle=':')
        ax.set_xticks(np.arange(0, self.n + 1))
        ax.set_xticklabels(np.concatenate((['sample'], d_output)))
        ax.set_xlabel('Digital Output')
        ax.set_ylabel(r'Voltage ($V$)')
        ax.legend(fontsize='small')
        ax.set_title(r'Voltage at point X with input: %.3f $V$' % v_input)

    def plot_continuous_mode(self, fft_length, fs, prime_number):
        ax = plt.subplot(111)
        plot_dac_output(ax, self, fft_length=fft_length, prime_number=prime_number, fs=fs)
        ax.grid(linestyle='-')


if __name__ == '__main__':
    start = time.time()
    adc = SarAdcDifferential(vref=1.2, n=12, mismatch=0.1)
    # adc.plot_energy()
    # f1 = plt.figure(1)
    # adc.plot_dnl_inl(resolution=0.01, method='iterative')
    # f2 = plt.figure(2)
    # adc.plot_dnl_inl(method='fast')
    # adc.plot_burst_mode(v_input=0.8, switch='conventional')
    adc.plot_fft()
    print('elapsed time: %.5f seconds' % (time.time()-start))
    plt.show()



