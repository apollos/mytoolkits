"""
For testing through python, change and run this code.
"""

import argparse
import numpy as np
import tensorflow as tf



def create_graph(modelFullPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(imagePath, labelsFullPath, modelFullPath):
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph(modelFullPath)

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imagePath',
        type=str,
        default='/home/yu/workspace/Data/solar_panel/caffe/test_image/Q1_A07170100716745_H.jpg',
        help='Path to folders of to be predicted image.',
        dest="imagePath"
    )
    parser.add_argument(
        '--labelsFullPath',
        type=str,
        default='/home/yu/workspace/Data/solar_panel/caffe/weight/output_labels.txt',
        help='Path to labels',
        dest="labelsFullPath"
    )
    parser.add_argument(
        '--modelFullPath',
        type=str,
        default='/home/yu/workspace/Data/weights/tensorflow/solar_panel/output_graph.pb',
        help='Where to save the trained graph.',
        dest="modelFullPath"
    )
    results = parser.parse_args()
    run_inference_on_image(results.imagePath, results.labelsFullPath, results.modelFullPath)