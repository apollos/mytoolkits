import os
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
 
def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.
 
    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
 
 
def extract_features(image_paotths, verbose=False):
    """
    extract_features computed the inception bottleneck feature for a list of images
 
    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))
 
    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
 
        for i, image_path in enumerate(image_paths):
            if verbose:
                print('Processing %s...' % (image_path))
 
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image)
 
            image_data = gfile.FastGFile(image_path, 'rb').read()
            feature = sess.run(flattened_tensor, {
                'DecodeJpeg/contents:0': image_data
            })
            features[i, :] = np.squeeze(feature)
 
    return features 
