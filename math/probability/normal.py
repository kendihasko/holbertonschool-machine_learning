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

            sum_squared_diff = sum((x - self.mean) ** 2 for x in data)
            variance = sum_squared_diff / len(data)
            self.stddev = variance ** 0.5

