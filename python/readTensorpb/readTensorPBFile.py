import tensorflow as tf
from tensorflow.python.platform import gfile

pf_file_path = "/home/yu/workspace/Data/weights/tensorflow/classify_image_graph_def.pb"
with tf.Session() as sess:
    model_filename = pf_file_path
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR= "logs/"
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
