from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, cv2
os.environ['CUDA_VISIBLE_DEVICES']=''

from datetime import datetime
import os.path
import time
import sys
import glob
import json
import numpy as np
from six.moves import xrange
import tensorflow as tf
import tf_parameter_mgr
import inference_helper
from nets import nets_factory
from preprocessing import preprocessing_factory

FLAGS = tf.app.flags.FLAGS

# Choose your image preprocessing
tf.app.flags.DEFINE_string('preprocessing_type', 'naton', 'image processing type')
# Choose the network structure
tf.app.flags.DEFINE_string('network_type', 'inception_v3', 'image feature extraction network type')
# Set the number of classes
tf.app.flags.DEFINE_integer('number_classes', 301, 'number of classes')
# prediction parameters
tf.app.flags.DEFINE_string('input_dir', '',
                           """Directory where to put the predicted images.""")
tf.app.flags.DEFINE_string('output_dir', '',
                           """Directory where to put the inference result.""")
tf.app.flags.DEFINE_string('output_file', '',
                           """File name of the inference result.""")
tf.app.flags.DEFINE_string('model', '',
                           """The pre-trained model referring to the checkpoint.""")
tf.app.flags.DEFINE_string('label_file', '',
                           """Labels of the image classes.""")
tf.app.flags.DEFINE_float('prob_thresh', 0.1,
                          """The prediction probability threshold to display.""")
tf.app.flags.DEFINE_bool('validate', False,
                         """Evaluating this model with validation dataset or not.""")


BATCH_SIZE = 128

def image_input_file():
    input_dir = os.path.expanduser(FLAGS.input_dir)
    imagefiles = glob.iglob(input_dir + '/*.*')
    imagefiles = [im_f for im_f in imagefiles
                  if im_f.endswith(".jpg") or im_f.endswith(".jpeg") or im_f.endswith(".png")]
    filename_queue = tf.train.string_input_producer(imagefiles, num_epochs=1, shuffle=False)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    input_image = tf.image.decode_image(value, channels=3)
    return key, input_image, tf.constant(-1, dtype=tf.int32)


def image_input_bin(filename_queue):
    image_bytes = 32 * 32 * 3
    label_bytes = 1
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.reshape(tf.strided_slice(record_bytes, [0], [label_bytes]), []), tf.int32)
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
        [3, 32, 32])
    input_image = tf.transpose(depth_major, [1, 2, 0])
    return key, input_image, label


def image_input_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    label = tf.cast(features['label'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    input_image = tf.reshape(image, tf.stack([height, width, depth]))
    return key, input_image, label


def image_input_dataset():
    datafiles = tf_parameter_mgr.getValData()
    filename_queue = tf.train.string_input_producer(datafiles, num_epochs=1, shuffle=False)
    if datafiles[0].endswith('.bin'):
        return image_input_bin(filename_queue)
    else:
        return image_input_tfrecord(filename_queue)


def predict():
    if FLAGS.validate:
        input_filename, input_image, input_label = image_input_dataset()
    else:
        input_filename, input_image, input_label = image_input_file()

    preprocessing_type = FLAGS.preprocessing_type
    preprocessor = preprocessing_factory.get_preprocessing(preprocessing_type, is_training=False)
    network_type = FLAGS.network_type
    default_size = nets_factory.get_default_size(network_type)
    input_image = preprocessor(input_image, output_height=default_size, output_width=default_size)
    input_image.set_shape([default_size, default_size, 3])

    filename_batch, image_batch, label_batch = tf.train.batch([input_filename, input_image, input_label],
                                                              batch_size=BATCH_SIZE,
                                                              allow_smaller_final_batch=True)
    network = nets_factory.get_network_fn(network_type, num_classes=FLAGS.number_classes, is_training=False)
    logits, _ = network(image_batch)

    proba = tf.nn.softmax(logits)

    prediction = np.ndarray([0, proba.get_shape().as_list()[1]], np.float32)
    imagenames = np.ndarray([0], dtype=np.object)
    true_label = np.ndarray([0], dtype=np.object)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.model)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        tf.train.start_queue_runners(sess)
        start = time.time()
        while True:
            try:
                p, fn, l = sess.run([proba, filename_batch, label_batch])
                prediction = np.append(prediction, p, 0)
                imagenames = np.append(imagenames, fn, 0)
                true_label = np.append(true_label, l, 0)
            except tf.errors.OutOfRangeError:
                break
        print("Predictions are Finished in %.2f s." % (time.time() - start))
    if FLAGS.validate:
        inference_helper.writeClassificationResult(os.path.join(FLAGS.output_dir, FLAGS.output_file),
                                                   imagenames, prediction, ground_truth=true_label)
    else:
        inference_helper.writeClassificationResult(os.path.join(FLAGS.output_dir, FLAGS.output_file),
                                                   imagenames, prediction,
                                                   prob_thresh=FLAGS.prob_thresh,
                                                   label_file=FLAGS.label_file)

def main(argv=None):
    predict()


if __name__ == '__main__':
    tf.app.run()

