#!/usr/bin/env python3
"""
Defines function that randomly shears an image
"""


import tensorflow as tf


def change_brightness(image, max_delta):
    """ randomly changes the brightness of an image """
    return tf.image.adjust_brightness(image, max_delta)
