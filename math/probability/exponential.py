#!/usr/bin/env python3
'''
A script that represents an exponential distribution
'''


class Exponential:
    '''
A class that represents an exponential distribution
    '''
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)

        if self.lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is None:
            data = self.lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (float(sum(data)) / len(data))

    def pdf(self, x):
        '''
        A method that calculates the value of the PDF for a given time period
        '''
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))
