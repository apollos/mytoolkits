""" Tensorflow implementation of the lightened model found at
https://github.com/AlfredXiangWu/LightCNN
"""
###############################################################################
# Licensed Materials - Property of IBM
# 5725-Y38
# @ Copyright IBM Corp. 2017 All Rights Reserved
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
###############################################################################

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten

def mfm(data,  out_channels, kernel_size=3, stride=1, padding=1, type=1, name='conv'):
    if type == 1:
        data = tf.layers.conv2d(inputs=data, filters=2*out_channels,
                                kernel_size=kernel_size, strides=stride,
                                padding='same', name = name)
        data_a, data_b = tf.split(data, 2, axis=3)
        data = tf.maximum(data_a, data_b)
    else:
        print(data)
        data = flatten(data)
        data = fully_connected(inputs=data,  num_outputs=2 * out_channels, activation_fn=None, scope  = name)
        data_a, data_b = tf.split(data, 2, axis=1)
        data = tf.maximum(data_a, data_b)
    return data

def group(data, in_channels, out_channels, kernel_size, stride, padding, name=''):
    conv_a = mfm(data, in_channels, 1, 1, 0, name=name+'.conv_a')
    conv = mfm(conv_a, out_channels, kernel_size, stride, padding, name=name+'.conv')
    return conv

def resblock(data, out_channels, name=''):
    conv1 = mfm(data, out_channels, kernel_size=3, stride=1, padding=1, name=name+'.conv1')
    conv2 = mfm(conv1, out_channels, kernel_size=3, stride=1, padding=1, name=name+'.conv2')
    out = data + conv2
    return out

def block(data, num_blocks,  out_channels, name=''):
    for i in range(num_blocks):
        data = resblock(data, out_channels, name=name+'.'+str(i))
    return data

def lightened_model(images, scope='LightenedNet'):
    layers = [1,2,3,4]
    end_points={}
    with tf.variable_scope(scope, 'LightenedNet'):
        conv1 = mfm(images, 48, 5, 1, 2, name='module.conv1')
        end_points['conv1'] = conv1
        pool1 = max_pool2d(conv1, kernel_size=[2,2], stride=[2,2], padding='VALID')
        block1 = block(pool1, layers[0], 48, name='module.block1')
        end_points['block1'] = block1
        group1 = group(block1, 48, 96, 3, 1, 1, name='module.group1')
        end_points['group1'] = group1
        pool2 = max_pool2d(group1, kernel_size=[2,2], stride=[2,2], padding='VALID')
        block2 = block(pool2, layers[1], 96, name='module.block2')
        end_points['block2'] = block2
        group2 = group(block2, 96, 192, 3, 1, 1, name='module.group2')
        end_points['group2'] = group2
        pool3 = max_pool2d(group2, kernel_size=[2,2], stride=[2,2], padding='VALID')
        block3 = block(pool3, layers[2], 192, name='module.block3')
        end_points['block3'] = block3
        group3 = group(block3, 192, 128, 3, 1, 1, name='module.group3')
        end_points['group3'] = group3
        block4 = block(group3, layers[3], 128, name='module.block4')
        end_points['block4'] = block4
        group4 = group(block4, 128, 128, 3, 1, 1, name='module.group4')
        end_points['group4'] = group4
        pool4 = max_pool2d(group4, kernel_size=[2,2], stride=[2,2], padding='VALID')
        fc = mfm(pool4, 256, type=0, name='module.fc')
        end_points['fc'] = fc
    return fc, end_points
lightened_model.default_image_size = 128