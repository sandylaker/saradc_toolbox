import numpy as np
import matplotlib.pyplot as plt
import time

def capArraygenerator(n=12,radix=2,mismatch=0.01,structure='conventional'):
    '''
    generates an array of capacitors and computes the binary weights in the dac. there are three
    different structures from which to choose. pay attention that in the returned tuple of capacitor array
    and weights array. the shape of the arrays can be two dimensional or one dimensional according to the structures.
    :param n: number of bits
    :param radix: the radix
    :param mismatch: mismatch of the capacitors
    :param structure: 'conventional': the conventional structure of capacitor divider. the amplifier is single ended.
                      'differential': two capacitor dividers are connected to the positive and negative input of op amp.
                                      The amplifier is single ended. The states of the switches in positive array is
                                      complementary to that in negative array.
                      'split': an attenuator capacitor is placed between the LSB and MSB capacitor array.
    :return: a tuple of capacitor array and weights array.
                    if 'conventional': shape of capacitor array:(n,) , MSB to LSB,
                                       shape of weights array: (n,) ,  MSB to LSB.
                    if 'differential': shape of capacitor array:(2,n+1), MSB to LSB,
                                       shape of weights array: (2,n) , MSB to LSB
                    if 'split':        shape of capacitor array: (n+2,) , LSB to MSB
                                       shape of weights array: (n,) , MSB to LSB
    '''
    if structure == 'conventional':
        capExp = np.concatenate(([0], np.arange(n)), axis=0)  # exponential of capacitance array
        capArray =[]
        # print('capExponential',capExp)
        for i in capExp:
            cap_i = np.random.normal(radix ** i, mismatch * np.sqrt(radix ** i))  # good case
            capArray += [cap_i]
        capSum = np.sum(capArray)
        #  reserve the capArray and abandon the last element
        weights = (np.flip(capArray, -1)[:-1]) / capSum  # binary weights
        return capArray,weights
    elif structure == 'differential':
        capExp = np.concatenate(([0], np.arange(n)), axis=0)  # exponential of capacitance array
        capArray = np.array([[],[]])
        for i in capExp:
            cap_i = np.random.normal(radix ** i, mismatch * np.sqrt(radix ** i),size=(2,1)) # good case
            capArray = np.hstack((capArray,cap_i))    # get an (2,n+1) array
        capSum = np.sum(capArray,axis=-1)[:,np.newaxis] # in order to use broadcasting, get an (2,1) array
        weights = (np.flip(capArray,-1)[:,:-1]) / capSum  # get an (2,n) array
        return capArray,weights     # capArray shape(2,n+1); weights shape (2,n)
    elif structure =='split':
        if(n%2==1):
            n = n+1
            print('Warning: split capacitor structure only support even number of bits,'
                  ,'n is automatically set to n+1')
        capExp = np.concatenate(([0],np.arange(n/2)),axis=0)
        capArray = np.array([[],[]])
        for i in capExp:
            cap_i = np.random.normal(radix ** i, mismatch * np.sqrt(radix ** i),size=(2,1)) # good case
            capArray = np.hstack((capArray,cap_i))   # get an (2,n/2) array
        capArray_lsb = capArray[0][:]
        capArray_msb = capArray[1][1:]  # MSB array has no dummy capacitor , shape(n/2,)
        capSum_lsb = np.sum(capArray_lsb)
        capSum_msb = np.sum(capArray_msb)
        cap_attenuator = 1  # ideally it should be capSum_lsb/capSum_msb, but here we set it to 1 directly

        # the series of attenuator capacitor and entire MSB array
        capSum_MA = cap_attenuator * capSum_msb/(cap_attenuator+ capSum_msb)
        # the series of attenuator capacitor and entire LSB array
        capSum_LA = cap_attenuator * capSum_lsb/(cap_attenuator + capSum_lsb)

        # attention: the location of positive input of the amplifier is between attenuator capacitor and MSB array
        # so here we need to multiply with an extra term 'cap_attenuator/(cap_attenuator+capSum_msb)'
        weights_lsb = (np.flip(capArray_lsb,-1)[:-1])/(capSum_lsb + capSum_MA) * (cap_attenuator/(cap_attenuator+capSum_msb))
        weights_msb = (np.flip(capArray_msb,-1))/(capSum_msb + capSum_LA)
        weights = np.concatenate((weights_msb,weights_lsb))

        # attention: in the following step, the concatenated array is LSB-Array + attenuator + MSB-Array,
        # in which the position of MSB and LSB are exchanged if comparing with other structures.
        # However in the weights array, the first element corresponds to the MSB and the last element corresponds
        # to the LSB, which accords with the other structures.
        capArray = np.concatenate((capArray_lsb,[cap_attenuator],capArray_msb))
        return capArray,weights


def getbi2deDict(n):
    '''
    get the binary representation of a n-bit range to decimal number dictionary. e.g {'001':1,'010':2,...}
    :return: a dictionary
    '''
    bi2deDict = {}
    decimal = np.arange(2 ** n)
    for x in decimal:
        binary = np.binary_repr(x, n)
        bi2deDict.update({binary: x})
    return bi2deDict

def bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int64's.
    if arr is a single integer, the shape of returned array would be (1,m) (2-dimensional array)
    if arr is a ndarray-like sequence with shape(n,), the shape of returned array would be (n,m) (2-dimensional array)
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    arr = np.asarray(arr)
    if arr.shape ==():
        arr = np.asarray([arr])
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int64)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int64")

    return ret

# def de2biRange(high,n,low=0):
#     '''
#     convert an range of decimal integers elementwise to a list of binary codes.
#     the returned list has a dimension of two. e.g [[0,0,1],[0,1,1]]
#     :param high: the high boundary (exclusive)
#     :param n: number of bits for binary conversion
#     :param low: the low boundary(inclusive)
#     :return: a list of binary codes, each binary code is also a list of integers,
#              which has a length of n.
#     '''
#     deRange = np.arange(low,high)
#     biCodes = [np.binary_repr(x,n) for x in deRange]
#     biCodes = [[int(i) for i in biCodes[j]] for j in range(len(biCodes))]
#     return biCodes

def getDecisionLvls(weights,n,vref):
    '''
    computes all the decision levels(also called transition points)
    :param weights: binary weights
    :param n: number of bits
    :param vref: reference voltage
    :return: a array of decision levels
    '''
    biCodes = bin_array(np.arange(2**n),n)
    decisionLvls = np.inner(biCodes, weights) * vref
    return decisionLvls

def fastConversion(analogSamples,weights,n,vref):
    '''
    uses the fast conversion algorithm to convert an array of analogSamples into
    decimal digital values.
    :param analogSamples: a array with one dimension
    :param weights: binary weights of adc
    :param n: number of bits
    :param vref: reference voltage of adc
    :return: a array of decimal integers,whose number of dimension is 1 and length
            equal to the length of analogSamples
    '''
    # convert analog input to array and add one dimension
    # use asarray method to handle the case that analogSamples is a single value.
    analogSamples = np.asarray(analogSamples)[:,np.newaxis]    # shape(M,1)
    decLvls = getDecisionLvls(weights,n,vref)[np.newaxis,:]    # shape(1,N)
    # use numpy broadcasting to compare two matrix elementwise
    relationMatrix = np.asarray(np.greater_equal(analogSamples,decLvls),dtype=np.int64)
    # sum each row and minus 1 getting a array with shape(M,)
    conversionResult = np.sum(relationMatrix,axis=-1) -1
    return conversionResult


def getDecisionPath(n):
    '''
    get a array of decision path of the full decision tree.
    :param n: depth of the decision tree, it is equivalent to the resolution of
    the DAC.
    :return: A two-dimensional array,each row of which represents the decision path
    of a possible decision level ( a odd decimal integer).
    '''
    # n = self.n # depth of the decision tree
    # possible decision level before the last comparision
    code_decimal = np.arange(1,2**n,2)
    code_binary = bin_array(code_decimal,n) # binary digits, shape (len(code_decimal),n)
    #store the decision thresholds generated in each conversion
    decisionPath = np.zeros((len(code_decimal),n))
    for i in range(len(code_decimal)):
        code_i = code_decimal[i]
        delta = np.array([2**i for i in range(n-1)])
        D = code_binary[i]
        decisionPath[i,-1] = code_i
        decisionPath[i,0] = 2**(n-1)
        for j in range(n-2,0,-1):
            decisionPath[i,j] = decisionPath[i,j+1] + (-1)**(2-D[j])*delta[n-2-j]
    return decisionPath


def getEnergy(n,switch='conventional',structure='conventional'):
    '''
    get the energy consumption of every code, each code represents the possible decision level before the last
    decision(a odd decimal integer).
    :param n: resolution of DAC
    :param switch: switching method: 'conventional': conventional one-step switching
                                     'monotonic': monotonic capacitor switching, in each transition step, only one
                                                  capacitor in one side is switched.
                                     'mcs': merged capacitor switching
                                     'split': split-capacitor method. The MSB capacitor is split into a copy of the
                                            rest of the capacitor array. When down-switching occurs, only the
                                            corresponding capacitor in the sub-capacitor array is discharged to the
                                            ground
    :param structure: structure of ADC: 'conventional': conventional single-ended structure
                                        'differential': has two arrays of capacitors, the switch states of positive and
                                                        negative side are complementary. The energy consumption is two
                                                        times of that in the conventional structure, if conventional
                                                        switching method is used.
    :return: a ndarray, each element represents the energy consumption of each code.
    '''
    # possible decision level before the last comparision
    code_decimal = np.arange(1, 2 ** n, 2)
    decisionPath = getDecisionPath(n)  # two-dimensional
    # store the switching energy of each code
    sw_energy_sum = np.zeros(len(code_decimal))
    if switch == 'conventional':
        coefficient= 1
        if structure == 'differential':
            # the switching states of both sides are complementary, so that the energy consumption is two times of
            # that in conventional(single-ended) structure.
            coefficient = 2
        for i in range(len(code_decimal)):
            # weight of each decision threshold layer
            weights_ideal = [0.5 ** (i + 1) for i in range(n)]
            #print(code_decimal[i],'=>',decisionPath[i])
            sw_energy = np.zeros(n)
            sw_energy[0] = 0.5 * decisionPath[i,0]

            # calculate the energy for up-switching steps
            sw_up_pos = np.where(decisionPath[i,1:]>decisionPath[i,0:-1])[0]+1 # 1 is the index offset
            # print(code_decimal[i],' sw_up_pos: ',sw_up_pos)
            if not sw_up_pos.size == 0:
                # sw_energy[sw_up_pos] = decisionPath[i,sw_up_pos]*(-1)*(weights_ideal[sw_up_pos])+ 2**(n-1-sw_up_pos)
                # 2**(n-1-sw_up_pos) stands for E_sw = C_up*V_ref^2
                for k in sw_up_pos:
                    # \delta V_x is positive,so *(-1)
                    sw_energy[k] =  decisionPath[i,k]*(-1)*(weights_ideal[k])+2**(n-1-k)

            sw_dn_pos = np.where(decisionPath[i,1:]< decisionPath[i,0:-1])[0]+1
            # print(code_decimal[i],' sw_dn_pos: ',sw_dn_pos)
            if not sw_dn_pos.size == 0:
                # sw_energy[sw_dn_pos] = decisionPath[i,sw_dn_pos]*(-1)*(weights_ideal[sw_dn_pos]) + 2**(n-1-sw_dn_pos)
                for k in sw_dn_pos:
                    sw_energy[k] =  decisionPath[i,k]*(weights_ideal[k]) + 2**(n-1-k)
            # print(code_decimal[i],': ',sw_energy)
            sw_energy_sum[i] = np.sum(sw_energy)
        return coefficient * sw_energy_sum

    if switch == 'monotonic':
        if structure == 'conventional':
            raise Exception('Conventional(single-ended) structure does not support monotonic switching.')
        for i in range(len(code_decimal)):
            # the total capacitance of positive and negative sides
            c_tp = c_tn = 2 ** (n - 1)
            weights_ideal = np.concatenate(([0],[0.5**(j) for j in range(1,n)]))   # vx unchanged in the first step
            sw_energy = np.zeros(n)
            sw_energy[0] = 0

            # define an array to store the switching types(up or down) of each step.
            sw_process = np.zeros(n)
            # find the up-switching and down-switching steps
            sw_up_pos = np.where(decisionPath[i, 1:] > decisionPath[i, 0:-1])[0] + 1  # 1 is the index offset
            sw_dn_pos = np.where(decisionPath[i, 1:] < decisionPath[i, 0:-1])[0] + 1
            sw_process[sw_up_pos],sw_process[sw_dn_pos] = 1, 0
            for k in range(1,n):
                # if up-switching occurs, a capacitor of the p-side will be connected to the ground while n-side remains
                # unchanged; if down-switching occurs, a capacitor of n -side will be connected to the ground while
                # p-side remains unchanged. Attention: here is the range(1,n), when k starts from 1, the first
                # capacitor switched to the ground is 2**(n-2)*C0 ( the MSB capacitor differs from which in the
                # conventional case.
                c_tp = c_tp - 2**(n-1-k) * sw_process[k]
                c_tn = c_tn - 2**(n-1-k) * (1-sw_process[k])
                sw_energy[k] = c_tp * (-1) * (- weights_ideal[k]) * sw_process[k] \
                               + c_tn * (-1) * (- weights_ideal[k]) * (1-sw_process[k])
            sw_energy_sum[i] = np.sum(sw_energy)
        return sw_energy_sum

    if switch == 'mcs':
        if structure == 'conventional':
            raise Exception('Conventional(single-ended) structure does not support monotonic switching.')
        weights_ideal = np.concatenate(([0.5 ** j for j in range(1, n)], [0.5 ** (n - 1)]))
        cap_ideal = np.concatenate(([2 ** (n - 2 - j) for j in range(n - 1)], [1]))
        for i in range(len(code_decimal)):
            sw_energy = np.zeros(n)

            # find the up-switching and down-switching steps
            sw_up_pos = np.where(decisionPath[i, 1:] > decisionPath[i, 0:-1])[0] + 1  # 1 is the index offset
            sw_dn_pos = np.where(decisionPath[i, 1:] < decisionPath[i, 0:-1])[0] + 1
            # connection of bottom plates of positive and negative capacitor arrays.
            # at the sampling phase, all the bottom plates are connected to Vcm = 0.5* Vref
            cap_connect_p = np.full((n, n), 0.5)
            cap_connect_n = np.full((n, n), 0.5)
            # define an array to store the switching types(up or down) of each step.
            sw_process = np.zeros(n)
            sw_process[sw_up_pos], sw_process[sw_dn_pos] = 1.0, 0
            # store the v_x of both sides in each step, here the term v_ip and v_in are subtracted.
            v_xp = np.zeros(n)
            v_xn = np.zeros(n)
            # store the voltage difference between the plates of each capacitor in each step, here the term v_ip- v_cm
            # and v_in - v_cm are subtracted, because when calculating the change of v_cap, these terms are constant and
            # so eliminated.
            v_cap_p = np.zeros((n, n))
            v_cap_n = np.zeros((n, n))

            for k in range(1, n):
                # update the connections of bottom plates
                cap_connect_p[k:, k-1], cap_connect_n[k:, k-1] = 1 - sw_process[k], sw_process[k]

                v_xp[k] = np.inner(cap_connect_p[k], weights_ideal)
                v_xn[k] = np.inner(cap_connect_n[k], weights_ideal)
                # calculate the voltage across the top and bottom plates of capacitors
                v_cap_p[k] = v_xp[k] - cap_connect_p[k]
                v_cap_n[k] = v_xn[k] - cap_connect_n[k]
                # find index of  the capacitors connected to the reference voltage
                c_tp_index = np.where(cap_connect_p[k] == 1.0)[0]
                c_tn_index = np.where(cap_connect_n[k] == 1.0)[0]
                # energy = - V_ref * ∑(c_t[j] * ∆v_cap[j])
                sw_energy_p = -np.inner(cap_ideal[c_tp_index], (v_cap_p[k, c_tp_index]-v_cap_p[k-1, c_tp_index]))
                sw_energy_n = -np.inner(cap_ideal[c_tn_index], (v_cap_n[k, c_tn_index]-v_cap_n[k-1, c_tn_index]))
                sw_energy[k] = sw_energy_p + sw_energy_n
            sw_energy_sum[i] = np.sum(sw_energy)
        return sw_energy_sum

    if switch == 'split':
        coefficient = 1
        if structure == 'differential':
            coefficient = 2
        if n < 2:
            raise Exception("Number of bits must be greater than or equal to 2. ")
        # capacitor array, cap_ideal has the shape of (2,n), in which the first row is the sub-capacitor array of the
        # MSB capacitor, the second row is the main capacitor array(excluding the MSB capacitor)
        cap_ideal = np.repeat(np.concatenate(([2**(n-2-i) for i in range(n-1)], [1]))[np.newaxis, :], 2, axis=0)
        weights_ideal = cap_ideal/(2**n)
        for i in range(len(code_decimal)):
            sw_energy = np.zeros(n)
            sw_energy[0] = 0.5 * decisionPath[i, 0]
            # find the up-switching and down-switching steps
            sw_up_pos = np.where(decisionPath[i, 1:] > decisionPath[i, 0:-1])[0] + 1  # 1 is the index offset
            sw_dn_pos = np.where(decisionPath[i, 1:] < decisionPath[i, 0:-1])[0] + 1
            # define an array to store the switching types(up or down) of each step.
            sw_process = np.zeros(n)
            sw_process[sw_up_pos], sw_process[sw_dn_pos] = 1.0, 0
            # store the bottom plates connection in each step
            cap_connect = np.repeat(np.vstack((np.ones(n), np.zeros(n)))[np.newaxis, :, :], n, axis=0)
            # store the voltage at X point ,here the term v_cm - v_in is subtracted
            v_x = np.zeros(n)
            v_x[0] = np.sum(np.multiply(weights_ideal, cap_connect[0]))
            # the voltage between top plates and bottom plates
            v_cap = np.zeros((n, 2, n))
            v_cap[0] = v_x[0] - cap_connect[0]
            for k in range(1, n):
                # if up-switching: the capacitor with index k-1 in the main capacitor array will be charged to V_ref,
                # and the capacitor with same index remains charged to V_ref; if down-switching: the capacitor
                # with index k-1 in the sub-capacitor array will be discharged to ground, and the capacitor with the
                # same index remains discharged.
                cap_connect[k:, :, k-1] = sw_process[k]
                v_x[k] = np.sum(np.multiply(weights_ideal, cap_connect[k]))
                v_cap[k] = v_x[k] - cap_connect[k]
                # find index of  the capacitors charged to the reference voltage
                c_t_index = np.where(cap_connect[k] == 1.0)  # 2-dimensional index
                # energy = - V_ref * ∑(c_t[j] * ∆v_cap[j])
                # attention that v_cap is 3d-array, the the slicing index should also be 3-dimensional
                sw_energy[k] = - np.inner(cap_ideal[c_t_index], (v_cap[k, c_t_index[0], c_t_index[-1]] -
                                                                 v_cap[k-1, c_t_index[0], c_t_index[-1]]))
            sw_energy_sum[i] = np.sum(sw_energy)
        return coefficient * sw_energy_sum


def plot_energy(n, ax, switch='conventional', structure='conventional', marker='v'):
    """
    plot the energy consumption of all possible decision level before the last comparision.
    :param n: number of bits
    :param ax: Axes of the plot
    :param switch: switching method, 'conventional' or 'monotonic'
    :param structure: structure of ADC
    :param marker: marker of the curve
    :return: a plot of energy comsumption
    """
    # possible decision level before the last comparision
    code_decimal = np.arange(1, 2 ** n, 2)
    sw_energy_sum = getEnergy(n, switch=switch, structure=structure)
    ax.plot(code_decimal, sw_energy_sum, marker=marker, label=switch, markevery=0.05)
    # axis.grid()
    ax.set_xlabel('Output Code')
    ax.set_ylabel(r'Switching Energy ($C_0V_{ref}^2$)')
    ax.set_title('Switching Energy Consumption')

