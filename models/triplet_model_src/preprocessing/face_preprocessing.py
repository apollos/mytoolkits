from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import detect_face
import numpy as np
import cv2
#Align face
def guard(x,N):
    x[x<0] = 0
    x[x>N-1] = N-1
    return [int(i) for i in x]

def transform(x, y, trans_rot):
    # x,y position
    # trans_rot rotation matrix
    xx = trans_rot[0,0]*x + trans_rot[0,1]*y + trans_rot[0,2]
    yy = trans_rot[1,0]*x + trans_rot[1,1]*y + trans_rot[1,2]
    return xx, yy

def align(img, f5pt, crop_size, ec_mc_y, ec_y):
    f5pt = f5pt.reshape(2,5).T
    ang_tan = (f5pt[0,1]-f5pt[1,1])/(f5pt[0,0]-f5pt[1,0])
    ang = np.arctan(ang_tan) / np.pi * 180

    center = (0.5*img.shape[0], 0.5*img.shape[1])
    rot = cv2.getRotationMatrix2D(center, ang, 1.0)
    img_rot = cv2.warpAffine(img, rot, (img.shape[1], img.shape[0]))

    #eye center
    x = (f5pt[0,0]+f5pt[1,0])/2
    y = (f5pt[0,1]+f5pt[1,1])/2

    [xx, yy] = transform(x, y, rot)
    eyec = np.round([xx, yy])

    #mouth center
    x = (f5pt[3,0]+f5pt[4,0])/2
    y = (f5pt[3,1]+f5pt[4,1])/2
    [xx, yy] = transform(x, y, rot)
    mouthc = np.round([xx, yy])

    resize_scale = ec_mc_y/(mouthc[1]-eyec[1])

    img_resize = cv2.resize(img_rot, None, fx=resize_scale, fy=resize_scale)

    eyec2 = (eyec - np.array([img_rot.shape[1]/2., img_rot.shape[0]/2.])) * resize_scale +\
            np.array([img_resize.shape[1]/2., img_resize.shape[0]/2.])
    eyec2 = np.round(eyec2)

    img_crop = np.zeros((crop_size, crop_size, 3), dtype=img_resize.dtype)

    crop_x = eyec2[0] - np.floor(crop_size / 2.)
    crop_x_end = crop_x + crop_size - 1

    crop_y = eyec2[1] - ec_y
    crop_y_end = crop_y + crop_size - 1

    box = np.concatenate((guard(np.array([crop_x, crop_x_end]), img_resize.shape[1]), \
                          guard(np.array([crop_y, crop_y_end]), img_resize.shape[0])))

    crop_y = int(crop_y)
    crop_x = int(crop_x)
    img_crop[box[2]-crop_y:box[3]-crop_y, box[0]-crop_x:box[1]-crop_x,:] = img_resize[box[2]:box[3],box[0]:box[1],:]
    return img_crop

def face_detector():
    def worker(image):
        #print('worker input image', image.shape)
        threshold_0 = 0.6
        threshold_1 = 0.7
        threshold_2 = 0.7
        factor = 0.709
        minsize = 0.1
        crop_size = 144
        ec_mc_y = 48
        ec_y = 48
        height, width, channels = image.shape
        minsize = minsize * min(height, width)
        threshold = [threshold_0, threshold_1, threshold_2]  # three steps's threshold
        image_copy = np.zeros(image.shape, dtype=image.dtype)
        # if using opencv to read image, pls exchange the order of channels
        # image_copy[..., 0] = image[..., 2]
        # image_copy[..., 1] = image[..., 1]
        # image_copy[..., 2] = image[..., 0]
        image_copy[:] = image
        bounding_boxes, points = detect_face.detect_face(image_copy, minsize, pnet, rnet, onet, threshold, factor)
        if points.ndim != 2:
            image = cv2.resize(image, (crop_size,crop_size))
            return image
        else:
            face = align(image, points[:, 0], crop_size, ec_mc_y, ec_y)
            return face


    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        sess.run(tf.global_variables_initializer())
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    worker.pnet = pnet
    worker.rnet = rnet
    worker.onet = onet
    return worker

def preprocess_image(image, is_training, output_height=128, output_width=128):
    detector = face_detector()
    face = tf.py_func(detector, [image], tf.uint8)

    face = tf.cond(is_training, lambda: tf.random_crop(face, [128, 128, 3]),\
                   lambda: tf.image.resize_image_with_crop_or_pad(face, 128, 128))

    if output_width != 128 or output_width != 128:
        face = tf.expand_dims(face, 0)
        face = tf.image.resize_bilinear(face, [output_height, output_width])
        face = tf.squeeze(face, [0])

    return face

def direct_preprocess(image):
    detector = face_detector()
    face = detector(image)
    return face



