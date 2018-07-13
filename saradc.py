import numpy as np
import matplotlib.pyplot as plt
from sympy import randprime
from assistantModule import bin_array, getDecisionLvls, fastConversion,getbi2deDict,capArraygenerator
import timeit

class SarAdc:

    def __init__(self,vref=1.2,n=12,radix=2,mismatch=0.001,structure='conventional'):
        self.vref = vref    # reference voltage
        self.n = n      # resolution of ADC
        self.vcm = vref/2       # common mode voltage
        self.mismatch = mismatch
        self.radix = radix
        # attention: if structure == 'split array', the capacitor array is LSB-Array + attenuator + MSB-Array,
        # in which the position of MSB and LSB are exchanged if comparing with other structures.
        # However in the weights array, the first element corresponds to the MSB and the last element corresponds
        # to the LSB, which accords with the other structures.
        capArray,weights = capArraygenerator(self.n,self.radix,self.mismatch,structure= structure)
        self.capArray = capArray
        self.weights = weights
        print('capArray',self.capArray)
        print('weights',self.weights)


    def comparator(self,a,b):
        '''
        compute the difference of input and dac output
        :param a: the first input
        :param b: the second input
        :return: the difference of two inputs. data type: float
        '''
        difference = a - b
        return difference

    def dac(self,digits):
        '''
        convert a list of digit bits into analog value.
        :param digits: a list of ints.
        :return: analog value
        '''
        weights = self.weights
        return np.inner(weights,np.array(digits)) * self.vref

    def sarLogic(self,diff):
        '''
        the logic to determine and update the digit in each dac clock.
        the i-th switch of the circuit are initially set up to 1 in i-th dac clock,
        if the difference <0 , switch are set to 0
        if the difference >=0, switch doesn't switch
        :param diff: difference return from the method comparator
        :param i: i-th clock
        :return: single digit value
        '''
        if diff < 0:
            return 0
        else:
            return 1

    def sar_adc(self,vin):
        '''
        the main function of SAR ADC
        :param vin: input analog voltage
        :return: a list, in which each element is a list of string which represents
                 the digital output of each dac clock. e.g. ['100','010','011']
        '''
        weights = self.weights
        dOutputList = []
        dOutput = np.zeros(self.n,dtype= int)
        for i in range(self.n):
            dOutput[i] = 1
            vdac = self.dac(dOutput)
            diff = self.comparator(vin,vdac)
            dOutput[i] = int(self.sarLogic(diff))
            dOutputList += [''.join(map(str,dOutput))]
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
            decisionLvls = getDecisionLvls(self.weights, self.n, self.vref)
            ideal_lsb = self.vref / (2 ** self.n)
            dnl = np.diff(decisionLvls) / ideal_lsb - 1
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

    def fastAnalogDecimal(self,analogSamples):
        '''
        use fast conversion algorithm to convert an array of analog values
        into decimal digital values
        :param analogSamples: an array of analog values
        :return: an array of decimal values
        '''
        decimalOutput = fastConversion(analogSamples,self.weights,self.n,self.vref)
        return decimalOutput










