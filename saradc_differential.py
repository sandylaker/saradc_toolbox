import numpy as np
import matplotlib.pyplot as plt
from sympy import randprime
from assistantModule import *
import functions4ContinuousMode as cm
import time

class SarAdcDifferential:
    '''
    the sar adc with differential structure
    '''

    def __init__(self,vref=1.2, n=12, radix=2, mismatch=0.001, structure='differential',
                 fftLength=4096, fs=50e6, primeNumber=1193, window=None,):
        self.vref = vref
        self.vcm = vref/2
        self.n = n
        self.radix = radix
        self.mismatch = mismatch
        self.structure = structure
        capArray,weights = capArraygenerator(self.n,self.radix,self.mismatch,structure='differential')
        self.capArray_p,self.capArray_n = capArray
        self.weights_p, self.weights_n = weights
        print('capArray positive',self.capArray_p)
        print('capArray negative',self.capArray_n)
        print('weights positive',self.weights_p)
        print('weights negative',self.weights_n)

        self.fftLength = fftLength  # length of FFT
        self.fs = fs  # sampling frequency
        self.primeNumber = primeNumber  # the number to determine the input frequency by coherent sampling
        self.fin = fs * primeNumber / fftLength  # input frequency by coherent sampling
        if not window:  # add no window
            self.window_boolean = False
            self.window = np.ones(self.fftLength)
        else:
            self.window_boolean = True
            self.window = np.hanning(self.fftLength)

    def comparator(self, a, b):
        '''
        compute the difference of input and dac output
        :param a: the first input
        :param b: the second input
        :return: the difference of two inputs. data type: float
        '''
        difference = a - b
        return difference

    def dac(self, digits,weights):
        '''
        convert a list of digit bits into analog value.
        :param digits: a list of ints.
        :return: analog value
        '''
        return np.inner(weights, np.array(digits)) * self.vref

    def sarLogic(self, diff):

        if diff <=0:
            return 1
        else:
            return 0

    def sar_adc(self,vin):
        '''
        convert an analog value to digital value, the digital value of each successive approximation
        step is restored in a list ,which is the returned output of this function
        :param vin: single analog input value
        :return: a list of strings, which represents the digital output of each step.
                e.g ['100','110','101']
        '''
        v_ip= vin
        v_in = self.vref - vin
        weights_p = self.weights_p
        weights_n = self.weights_n
        dOutputList = []
        dOutput = np.zeros(self.n,dtype= int)
        for i in range(self.n):
            dOutput[i] = 1
            v_xp = self.dac(dOutput,weights_p)- v_ip
            v_xn = self.vref- self.dac(dOutput,weights_n) - v_in  # because of complementary state of switches
            diff = self.comparator(v_xp,v_xn)
            dOutput[i] = int(self.sarLogic(diff))
            dOutputList += [''.join(map(str, dOutput))]
        return dOutputList

    def dnl(self,resolution =0.01,method='fast'):
        '''
        calculate the differential non linearity.
        :param resolution: represents the ratio between the error of DNL and ideal LSB.
        :param method: 'fast': fast algorithm, which computes the transition levels directly
                       'iterative': find the transition levels iteratively
                       'code_density': based on the code density theory.
        :return: a list of DNLs.
        '''
        if method == 'iterative':
            lsb_ideal = self.vref/(2**self.n)
            #print('lsb_ideal',lsb_ideal)
            step = lsb_ideal * resolution
            ramp = np.arange(0,self.vref,step)

            dOutput = np.asarray([self.sar_adc(x)[-1] for x in ramp])
            # bi2deDict = self.getbi2deDict()
            # dOutput_decimal = np.array([bi2deDict[x] for x in dOutput])    # only dArray can use 'where' method
            dOutput = np.asarray([list(x) for x in dOutput],dtype= np.int64)
            exp_array = np.asarray([2**(self.n-1-i) for i in range(self.n)])
            dOutput_decimal = np.inner(exp_array,dOutput)
            #print('dOutput_decimal',dOutput_decimal,type(dOutput_decimal))
            transLvls = np.array([],dtype=np.int64)
            misscodeCount = 0
            dnl = []
            for i in range(2**self.n):
                positionArray = np.where(dOutput_decimal== i)      # the 'where' method returns a tuple
                #print(positionArray)
                if positionArray[0].size !=0:
                    transLvls = np.append(transLvls,positionArray[0][0])
                else:
                    misscodeCount+=1
            #print(transLvls)
            print('number of misscode:',misscodeCount)
            for i in range(len(transLvls)-1):
                dnl += [(transLvls[i+1] - transLvls[i])*resolution -1]
            return dnl
        elif method == 'fast':
            # calculate the dnl of the positive end
            weights_p = self.weights_p
            decisionLvls_p = getDecisionLvls(weights_p, self.n, self.vref)
            ideal_lsb = self.vref / (2 ** self.n)
            dnl_p = np.diff(decisionLvls_p) / ideal_lsb - 1

            # calculate the dnl of the negative end
            weights_n = self.weights_n
            decisionLvls_n = getDecisionLvls(weights_n, self.n, self.vref)
            dnl_n = np.diff(decisionLvls_n) / ideal_lsb - 1
            # the dnl of the adc is the average dnl of both ends
            dnl = np.add(dnl_p,dnl_n)/2
            return dnl

        elif method =='code_density':
            sinAmp = self.vref / 2
            fs = 5e6
            Ts = 1 / fs
            nRecord = np.pi * (2 ** (self.n - 1)) * (1.96 ** 2) / (resolution ** 2)
            print('number of records: %s'%('{:e}'.format(nRecord)))
            t = np.arange(0, nRecord * Ts, Ts)
            fin = randprime(0, 0.5 * nRecord) / nRecord * fs
            sineWave = sinAmp * np.sin(2 * np.pi * fin * t) + self.vcm

            dOutput = [self.sar_adc(x)[-1] for x in sineWave]
            bi2deDict = getbi2deDict(self.n)
            dOutput_decimal = [bi2deDict[x] for x in dOutput]
            minHitNum = np.amin(dOutput_decimal)
            maxHitNum = np.amax(dOutput_decimal)
            codeHist, bins = np.histogram(dOutput_decimal, np.arange(minHitNum, maxHitNum + 1))
            ''''
            plt.hist(dOutput_decimal, np.arange(2 ** self.n))
            print('max(codHist)', np.argmax(codeHist),np.amax(codeHist))
            plt.axis([0,2**self.n, np.amin(codeHist), np.max(codeHist)])
            '''

            codeCumHist = np.cumsum(codeHist)
            tranLvl = -np.cos(np.pi * codeCumHist / sum(codeHist))
            codeHistLin = np.array(tranLvl[1:]) - np.array(tranLvl[:-1])
            trunc = 1
            codeHistTrunc = codeHistLin[trunc: - trunc]
            actualLsb = sum(codeHistTrunc) / (len(codeHistTrunc))
            dnl = np.concatenate(([0], codeHistTrunc / actualLsb - 1))
            return dnl

    def inl(self,dnl):
        inl = np.cumsum(dnl)
        return inl

    def plotDnlInl(self,resolution=0.01,method='fast'):
        dnl = self.dnl(resolution,method)
        inl = self.inl(dnl)

        ax1= plt.subplot(211)
        ax1.plot(np.arange(len(dnl)),dnl,linewidth = 0.5)
        ax1.grid()
        ax1.set_title('DNL(mismatch: %5.3f)' % self.mismatch)
        ax1.set_xlabel('Digital Output Code')
        ax1.set_ylabel('DNL (LSB)')

        ax2 = plt.subplot(212)
        ax2.plot(np.arange(len(inl)),inl,linewidth=0.5)
        ax2.grid()
        ax2.set_title('INL(mismatch: %5.3f)' % self.mismatch)
        ax2.set_xlabel('Digital Output Code')
        ax2.set_ylabel('INL (LSB)')
        plt.tight_layout()


    def sampling(self):
        return cm.sampling(self)

    def getDigitalOutput(self):
        return cm.getDigitalOutput(self)

    def getAnalogOutput(self):
        return cm.getAnalogOutput(self)

    def getfftOutput(self):
        return cm.getfftOutput(self)

    def plotfft(self):
        cm.plotfft(self)

    def snr(self):
        return cm.snr(self,self.getfftOutput())

    def plot_energy(self):
        '''
        plot the energy consumption of the possible codes with different switching methods
        '''
        fig,ax = plt.subplots()
        plot_energy(self.n, ax, switch='conventional', marker='v', structure=self.structure)
        plot_energy(self.n, ax, switch='monotonic', marker='s', structure=self.structure)
        plot_energy(self.n, ax, switch='mcs', marker='o', structure=self.structure)
        plot_energy(self.n, ax, switch='split', marker='D',structure=self.structure)
        ax.legend()
        ax.grid()

    def plotBurstMode(self,v_input,switch='monotonic'):
        '''
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
                     'conventional': the conventional switching method,which is the same as in the conventional
                     structure( single ended).
        :return: a step plot
        '''
        vxp_list = []
        vxn_list = []
        v_ip = v_input
        v_in = self.vref - v_ip
        dOutput = [int(x) for x in self.sar_adc(v_input)[-1]]
        if switch == 'monotonic':
            vxp_list += [v_ip]
            vxn_list += [v_in]
            for i in range(self.n-1):
                    v_xp = vxp_list[-1] - dOutput[i]*self.vref/(self.radix**(i+1))
                    v_xn = vxn_list[-1] - (1-dOutput[i])* self.vref/(self.radix**(i+1))
                    vxp_list += [v_xp]
                    vxn_list += [v_xn]
            # computing the v_ip and v_in after last comparision,note that the last capacitor to be switched
            # is the dummy capacitor and the largest capacitor in the mcs method is 2^(n-2)*C_0
            # so that the voltage change is 1/2^(n-1) * vref
            v_xp = vxp_list[-1] - dOutput[-1]*self.vref / (self.radix ** (self.n-1))
            v_xn = vxn_list[-1] - (1-dOutput[-1])* self.vref/(self.radix**(self.n-1))
            vxp_list += [v_xp]
            vxn_list += [v_xn]
        if switch == 'conventional':
            weights_ideal = [0.5**(i+1) for i in range(self.n)]
            v_xp = self.vcm - v_ip + self.vref/2
            v_xn = self.vref - v_xp
            vxp_list += [v_xp]
            vxn_list += [v_xn]
            for i in range(1,self.n):
                v_xp = self.vcm - v_ip + self.vref* np.inner(dOutput[:i+1],weights_ideal[:i+1])
                v_xn = self.vref - v_xp
                vxp_list += [v_xp]
                vxn_list += [v_xn]
            # computing the v_ip and v_in after last comparision,note that the last capacitor to be switched
            # is the dummy capacitor
            v_xp = vxn_list[-1] + dOutput[-1]*self.vref * weights_ideal[-1]
            v_xn = self.vref - v_xp
            vxp_list += [v_xp]
            vxn_list += [v_xn]

        # concatenate vip_list(or vin_list) and its last element,
        # in order to show the voltage at point X after last comparision
        plt.step(np.arange(self.n+2), np.concatenate((vxp_list, [vxp_list[-1]])),
                 label=r'$V_{xp}$', color='b',linewidth=1.5, where='post')
        plt.step(np.arange(self.n+2), np.concatenate((vxn_list, [vxn_list[-1]])),
                 label=r'$V_{xn}$', color='r',linewidth=1.5, where='post')
        plt.axhline(y=self.vcm,color='g', ls=':', linewidth=1, label=r'$V_{cm}$')
        plt.grid(linestyle=':')
        plt.xticks(np.arange(0, self.n+2), np.concatenate((['sample'],dOutput)))
        plt.xlabel('Digital Output')
        plt.ylabel(r'Voltage ($V$)')
        plt.legend(fontsize='small')
        plt.title(r'Voltage at point X with input: %.3f $V$'%v_input)


start = time.time()
adc = SarAdcDifferential(n=10,mismatch=0)
adc.plot_energy()
print('elapsed time: %.5f seconds' % (time.time()-start))
plt.show()


