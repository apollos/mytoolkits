import os,sys
import cv2
import tensorflow as tf
from preprocessing import face_preprocessing

def test():
    image_dir = '/Users/kaiwu/Documents/POC/FaceRecognition/imgs/'
    image_files = [image_dir+f for f in os.listdir(image_dir)]

    filename = tf.placeholder(tf.string)
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, 3)
    is_train = tf.placeholder(tf.bool)
    face = face_preprocessing.preprocess_image(image, is_train, 128, 128)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for file in image_files:
        print("read file:", file)
        img = sess.run(face, feed_dict={filename:file, is_train:False})
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face", img)
        cv2.waitKey(0)
    pass

test()
quit()

def test2():
    image_dir = '/Users/kaiwu/Documents/POC/FaceRecognition/imgs/'
    image_files = [image_dir+f for f in os.listdir(image_dir)]

    for file in image_files:
        print("read file:", file)
        image = cv2.imread(file)
        face = face_preprocessing.direct_preprocess(image)
        cv2.imshow("Image", face)
        cv2.waitKey(0)
    pass

test2()