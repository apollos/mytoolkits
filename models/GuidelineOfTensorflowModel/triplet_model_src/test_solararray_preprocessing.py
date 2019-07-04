import os,sys
import numpy as np
import cv2
import tensorflow as tf
from preprocessing import solararray_preprocessing

def test():
    image_dir = '/Users/kaiwu/Documents/POC/xiwenzh_solar_array/ELcustomerdata/Q0/'
    image_files = [image_dir+f for f in os.listdir(image_dir)]

    filename = tf.placeholder(tf.string)
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, 3)
    is_train = True
    face = solararray_preprocessing.preprocess_image(image, is_train)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for file in image_files:
        print("read file:", file)
        img = sess.run(face, feed_dict={filename:file})
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(np.min(img), np.max(img))
        img = img - np.min(img)
        img[img>255]=255
        cv2.imshow("image", np.uint8(img))
        cv2.waitKey(0)

test()
quit()