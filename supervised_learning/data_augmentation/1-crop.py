def crop_image(image, size):
    """ crop an image """
    return tf.image.random_crop(image, size = size)