import numpy as np
import matplotlib.pyplot as plt
import time

def capArraygenerator(n=12,radix=2,mismatch=0.01,structure='single_ended'):
    if structure == 'single_ended':
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
            capArray = np.hstack((capArray,cap_i))    # get an (2,n) array
        capSum = np.sum(capArray,axis=-1)[:,np.newaxis] # in order to use broadcasting, get an (2,1) array
        weights = (np.flip(capArray,-1)[:,:-1]) / capSum  # get an (2,n) array
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
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
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

# weights = np.array([0.5**i for i in range(1,13)])
# decLvls =getDecisionLvls(weights,12,1.2)
# print(len(decLvls))