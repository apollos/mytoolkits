import tensorflow as tf
from tensorflow.contrib import image as image_op
import numpy as np

def preprocess_image(image, is_training, output_height=100, output_width=100):
    # if is_training:
    #     Do something to image
    # else:
    #     Do something else to image
    # return image

    image.set_shape([None, None, 3])
    image_shape = tf.shape(image)
    image = 255 - image
    height = image_shape[0]
    width = image_shape[1]
    height_smaller_than_width = tf.less_equal(height, width)

    new_longer_edge = tf.constant(min(output_height, output_width),dtype=tf.int32)

    new_height, new_width = tf.cond(
        height_smaller_than_width,
        lambda: [new_longer_edge*height / width , new_longer_edge],
        lambda: [new_longer_edge, new_longer_edge * width / height])


    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, (new_height, new_width))
    image = tf.squeeze(image, [0])
    if is_training:
        angle = tf.random_uniform([], -0.1 * 3.141592, 0.1 * 3.141592)
        image = tf.image.resize_image_with_crop_or_pad(image, int(1.1 * output_height), int(1.1 * output_width))
        image = image_op.rotate(image, angle)
        image = tf.random_crop(image, [output_height, output_width, 3])
        image = tf.image.random_brightness(image, 10)
        image = tf.image.random_contrast(image, 0.8,1.2)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)
    # image = (image/255 - 0.5)*2
    return image