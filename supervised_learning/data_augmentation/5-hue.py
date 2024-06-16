def change_hue(image, delta):
     return tf.image.adjust_hue(image, delta)