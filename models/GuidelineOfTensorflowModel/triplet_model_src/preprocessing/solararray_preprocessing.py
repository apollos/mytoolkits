import tensorflow as tf
from tensorflow.contrib import image as image_op
import numpy as np

def preprocess_image(image,  output_height=800, output_width=1000, is_training=True):
    image.set_shape([None, None, 3])
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, (output_height, output_width))
    image = tf.squeeze(image, [0])
    if is_training:
        angle = tf.random_uniform([], -0.01 * 3.141592, 0.01 * 3.141592)
        image = tf.image.resize_image_with_crop_or_pad(image, int(1.025 * output_height), int(1.025 * output_width))
        image = image_op.rotate(image, angle)
        image = tf.random_crop(image, [output_height, output_width, 3])
        image = tf.image.random_brightness(image, 30)
        image = tf.image.random_contrast(image, 0.8,1.2)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)
    image = (image/255 - 0.5)*2
    return image