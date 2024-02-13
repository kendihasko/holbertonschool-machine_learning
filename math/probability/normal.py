#!/usr/bin/env python3
'''
A script that represents a normal distribution
'''


class Normal:
    '''
    A class that represents a normal distribution
    '''
    def __init__(self, data=None, mean=0., stddev=1.):

        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = (float(sum(data)) / len(data))
            summation = 0
            for x in data:
                summation += (x - mean) ** 2
            self.stddev = (summation / len(data)) ** (0.5)

    def z_score(self, x):
        '''
        A method that calculates the z-score of a given x-value
        '''
        self.z = (x - self.mean) / self.stddev
        return self.z

    def x_value(self, z):
        '''
        A method that calculates the z-score of a given x-value
        '''
        self.x = (self.stddev * z) + self.mean
        return self.x

    def pdf(self, x):
        '''
        Why calculate pdf if numpy already does it for us???
        '''
        e = 2.7182818285
        pi = 3.1415926536
        mean = self.mean
        stddev = self.stddev

        coefficient = 1 / (stddev * ((2 * pi) ** (1 / 2)))
        power = -0.5 * ((self.z_score(x)) ** 2)
        pdf = coefficient * (e ** power)
        return pdf

    def cdf(self, x):
        '''
        Why calculate pdf if numpy already does it for us???
        '''
        mean = self.mean
        stddev = self.stddev
        pi = 3.1415926536
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        erf = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        erf = erf - ((value ** 7) / 42) + ((value ** 9) / 216)
        erf *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + erf)
        return cdf
