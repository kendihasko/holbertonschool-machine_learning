def change_brightness(image, max_delta):
    return tf.image.adjust_brightness(image, max_delta)