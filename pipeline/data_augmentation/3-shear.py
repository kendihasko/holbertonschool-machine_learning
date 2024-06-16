#!/usr/bin/env python3
"""
Defines function that randomly shears an image
"""


import tensorflow as tf


def shear_image(image, intensity):
    """ randomly shears an image """
    image_nparray = tf.keras.preprocessing.image.img_to_array(image)
    shear_nparray = tf.keras.preprocessing.image.random_shear(image_nparray,
                                                              intensity)
    return tf.keras.preprocessing.image.array_to_img(shear_nparray)
