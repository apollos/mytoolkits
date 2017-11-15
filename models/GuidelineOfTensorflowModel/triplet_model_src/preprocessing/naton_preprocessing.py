import tensorflow as tf
from tensorflow.contrib import image as image_op


def preprocess_image(image, output_height, output_width, is_training):
    image = tf.to_float(image)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [int(output_height * 1.2), int(output_width * 1.2)])
    image = tf.squeeze(image, [0])
    if is_training:
        angle = tf.random_uniform([], 0., 2. * 3.141592)
        image = image_op.rotate(image, angle)
    if is_training:
        image = tf.random_crop(image, [output_height, output_width, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)
    image = image/128.-1
    return image
