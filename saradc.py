import numpy as np
import matplotlib.pyplot as plt
from sympy import randprime
from assistantModule import bin_array, getDecisionLvls, fastConversion
import timeit

class SarAdc:

    def __init__(self,vref=1.2,n=12,radix=2,mismatch=0.001):
        self.vref = vref    # reference voltage
        self.n = n      # resolution of ADC
        self.vcm = vref/2       # common mode voltage
        self.mismatch = mismatch
        self.radix = radix
        '''
        self.capArray = np.multiply(1000,[2048.55480524356, 1026.35602156611,509.426016434429,
                                          254.693367420063, 128.235024722168, 63.7380924592568,
                                          31.9161800386331, 15.7006162247105,7.87203708528868 ,
                                          3.95958184549688, 1.94867831950140, 0.956675748732019,
                                          0.978907660977901])
        self.weights = self.capArray[:-1]/sum(self.capArray)
        '''
        self.capArray = []
        capExp = np.concatenate(([0],np.arange(self.n)),axis=0)     # exponential of capacitance array
        #print('capExponential',capExp)
        for i in capExp:
            cap_i = np.random.normal(self.radix**i,self.mismatch * np.sqrt(self.radix**i))     # good case
            self.capArray += [cap_i]
        capSum = np.sum(self.capArray)
        #  reserve the capArray and abandon the last element
        weights = (np.flip(self.capArray,0)[:-1]) / capSum      # binary weights
        self.weights = weights
        print('capArray',self.capArray)
        print('weights',self.weights)


    def comparator(self,vin,vdac):
        '''
        compute the difference of input and dac output
        :param vin: input voltage
        :param vdac: output of dac block
        :return: difference: data type: float
        '''
        difference =  vin - vdac
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
            digit = self.sarLogic(diff)
            dOutput[i] = int(digit)
            dOutputList += [''.join(map(str,dOutput))]
        return dOutputList

    def getbi2deDict(self):
        '''
        get the binary representation to decimal number dictionary. e.g {'001':1}
        :return: a dictionary
        '''
        bi2deDict = {}
        decimal = np.arange(2**self.n)
        for x in decimal:
            binary = np.binary_repr(x,self.n)
            bi2deDict.update({binary: x})
        return bi2deDict

    # def calDNL(self,resolution=0.1,):
    #
    #     sinAmp = self.vref/2
    #     fs = 5e6
    #     Ts = 1/fs
    #     nRecord = np.pi*(2**(self.n-1))*(1.96**2)/(resolution**2)
    #     print('number of records',nRecord)
    #     t = np.arange(0,nRecord*Ts, Ts)
    #     fin = randprime(0,0.5*nRecord) / nRecord * fs
    #     sineWave = sinAmp* np.sin(2*np.pi* fin* t ) + self.vcm
    #
    #     dOutput = [self.sar_adc(x)[-1] for x in sineWave]
    #     bi2deDict = self.getbi2deDict()
    #     dOutput_decimal = [bi2deDict[x] for x in dOutput]
    #     minHitNum = np.amin(dOutput_decimal)
    #     maxHitNum = np.amax(dOutput_decimal)
    #     codeHist,bins = np.histogram(dOutput_decimal,np.arange(minHitNum,maxHitNum+1))
    #     ''''
    #     plt.hist(dOutput_decimal, np.arange(2 ** self.n))
    #     print('max(codHist)', np.argmax(codeHist),np.amax(codeHist))
    #     plt.axis([0,2**self.n, np.amin(codeHist), np.max(codeHist)])
    #     '''
    #
    #     codeCumHist = np.cumsum(codeHist)
    #     tranLvl = -np.cos(np.pi* codeCumHist/ sum(codeHist))
    #     codeHistLin = np.array(tranLvl[1:]) - np.array(tranLvl[:-1])
    #     trunc = 1
    #     codeHistTrunc = codeHistLin[trunc : - trunc]
    #     actualLsb = sum(codeHistTrunc)/(len(codeHistTrunc))
    #     dnl = np.concatenate(([0], codeHistTrunc/actualLsb -1))
    #
    #     plt.plot(np.arange(len(dnl)), dnl)
    #     plt.grid()
    #     plt.title('DNL (mismatch:%5.3f )'%(self.mismatch))
    #     plt.xlabel('Digital Output Code')
    #     plt.ylabel('DNL (LSB)')

    def dnl(self,resolution =0.01):
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
        for i in range(2**self.n):
            positionArray = np.where(dOutput_decimal== i)      # the 'where' method returns a tuple
            #print(positionArray)
            if positionArray[0].size !=0:
                transLvls = np.append(transLvls,positionArray[0][0])
            else:
                misscodeCount+=1
        #print(transLvls)
        print('number of misscode:',misscodeCount)
        dnl = []
        for i in range(len(transLvls)-1):
            dnl += [(transLvls[i+1] - transLvls[i])*resolution -1]
        return dnl

    def inl(self,dnl):
        inl = np.cumsum(dnl)
        return inl

    def plotDnlInl(self,resolution=0.01):
        dnl = self.dnl(resolution)
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

    def fastDnl(self):
        decisionLvls = getDecisionLvls(self.weights,self.n,self.vref)
        ideal_lsb = self.vref/(2**self.n)
        dnl = np.diff(decisionLvls)/ideal_lsb -1
        return dnl

    def plotFastDnlInl(self):
        dnl = self.fastDnl()
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







