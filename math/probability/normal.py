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
            data = [self.mean, self.stddev]
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = (float(sum(data)) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

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