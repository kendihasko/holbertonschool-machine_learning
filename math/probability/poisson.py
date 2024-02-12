#!/usr/bin/env python3
'''
A class that represemts a poisson distribution.
'''


class Poisson:
    '''
A class that represemts a poisson distribution.
    '''
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)) / len(data)

    def pmf(self, k):
        '''
        A method that calculates the value of the PMF
        for a given number of “successes”
        '''
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        factorial = 1
        for i in range(k):
            factorial *= (k + 1)
            pmf = ((e ** - lambtha) * (lambtha ** k)) / factorial
            return pmf

    def cdf(self, k):
        '''
        A method that calculates the value of the CDF
        for a given number of “successes”
        '''
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
            cdf = 0
            for i in range(k+1):
                cdf += self.pmf(i)
            return cdf
