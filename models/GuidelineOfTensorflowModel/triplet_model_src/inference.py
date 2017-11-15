from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, re, glob, json
import tensorflow as tf
import numpy as np
from preprocessing import face_preprocessing

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', '',
                           """Directory where to put the inference result.""")

tf.app.flags.DEFINE_string('output_file', 'test.json',
                           """File name of the inference result.""")

tf.app.flags.DEFINE_float('prob_thresh', 1.,
                          """The prediction probability threshold to display.""")

tf.app.flags.DEFINE_string('input_dir', '/Users/kaiwu/Documents/POC/TripletLoss/face/test/',
                           """Directory where to put the predicted(query) images.""")

tf.app.flags.DEFINE_string('gallery_image_dir', '/Users/kaiwu/Documents/POC/TripletLoss/face/train/',
                           """"Directory where to put the gallery images""")

tf.app.flags.DEFINE_string('model', '/Users/kaiwu/Documents/POC/FaceRecognition/src/initial_model/',
                           """The pre-trained model referring to the checkpoint.""")

tf.app.flags.DEFINE_float('image_height', 128, """image height""")

tf.app.flags.DEFINE_float('image_width', 128, """image height""")


tf.app.flags.DEFINE_integer('img_loader_type', 4, 'type of image file loader')
tf.app.flags.DEFINE_string('input_placeholder_name', 'Placeholder:0', 'input_placeholder_name')
tf.app.flags.DEFINE_string('feature_name', 'Maximum_29:0', 'feature_name')
tf.app.flags.DEFINE_bool('l2normalize', True, 'if the embedding network doesn\'t normalize the feature, normalize it!')


def img_preprocess_m1(image):
    image = tf.image.decode_image(image, channels=3)
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.image_height, FLAGS.image_width])
    image = tf.squeeze(image, [0])

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape((FLAGS.image_height, FLAGS.image_width, 3))
    return image


def img_preprocess_m2(image):
    image = tf.image.decode_image(image, channels=3)
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.image_height, FLAGS.image_width)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape((FLAGS.image_height, FLAGS.image_width, 3))
    return image


def img_preprocess_m3(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [FLAGS.image_height, FLAGS.image_width])
    image = tf.squeeze(image, [0])
    image = tf.image.per_image_standardization(image)
    image.set_shape((FLAGS.image_height, FLAGS.image_width, 3))
    return image

#For face recognition
def img_preprocess_m4(image):
    image = tf.image.decode_image(image, 3)
    is_train = tf.constant(False)
    image = face_preprocessing.preprocess_image(image, is_train, FLAGS.image_height, FLAGS.image_width)
    image = tf.image.rgb_to_grayscale(image)
    return image


def load_model(model, sess):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), clear_devices=True)
        print("start to restore")
        saver.restore(sess, os.path.join(model_exp, ckpt_file))
        print("finish restoring")


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def extract_features(imagefiles):
    print(len(imagefiles))
    filequeue = tf.train.string_input_producer(imagefiles, capacity=128, shuffle=False, num_epochs=1)
    reader = tf.WholeFileReader()
    label, image = reader.read(filequeue)
    if FLAGS.img_loader_type == 1:
        image = img_preprocess_m1(image)
    elif FLAGS.img_loader_type == 2:
        image = img_preprocess_m2(image)
    elif FLAGS.img_loader_type == 3:
        image = img_preprocess_m3(image)
    elif FLAGS.img_loader_type == 4:
        image = img_preprocess_m4(image)
    else:
        raise Exception("Unknown Preprocessing Type %d" % (FLAGS.img_loader_type))

    image_batch, label_batch = tf.train.batch([image, label], batch_size=4, num_threads=1,
                                              allow_smaller_final_batch=True)

    feat_array = np.zeros((len(imagefiles), 256))
    lbl_array = []
    with tf.Session() as sess:
        try:
            coord = tf.train.Coordinator()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            load_model(FLAGS.model, sess)
            # graph = tf.get_default_graph()
            # for op in graph.get_operations():
            #     print(op.type, op.name)
            #for n in tf.get_default_graph().as_graph_def().node:
            #    print(n.name)
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            images_placeholder = tf.get_default_graph().get_tensor_by_name(FLAGS.input_placeholder_name)
            embeddings = tf.get_default_graph().get_tensor_by_name(FLAGS.feature_name)
            #If the network doesn't normalize the features, normalize the feature here
            if FLAGS.l2normalize:
                embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10)
            index = 0
            while True:
                imgs, lbs = sess.run([image_batch, label_batch])
                feats = sess.run(embeddings, feed_dict={images_placeholder: imgs})
                feat_array[index:index + feats.shape[0]] = feats
                lbl_array += list(lbs)
                index += feats.shape[0]

        except tf.errors.OutOfRangeError:
            print('Finish feature extraction!')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)
            sess.close()
    return feat_array, lbl_array


def main(argv=None):
    gallery_imagefiles = glob.iglob(FLAGS.gallery_image_dir + '/*/*.*')
    gallery_images = [im_f for im_f in gallery_imagefiles
                      if im_f.endswith(".jpg") or im_f.endswith(".jpeg") or im_f.endswith(".png")or im_f.endswith(".bmp")]

    imagefiles = glob.iglob(FLAGS.input_dir + '/*.*')
    imagefiles = [im_f for im_f in imagefiles
                  if im_f.endswith(".jpg") or im_f.endswith(".jpeg") or im_f.endswith(".png") or  im_f.endswith(".bmp")]


    # extract gallery images' features
    number_of_gallery = len(gallery_images)
    number_of_query = len(imagefiles)

    all_feat_array, all_labels = extract_features(gallery_images + imagefiles)
    gallery_feat_array = all_feat_array[:number_of_gallery]
    query_feat_array = all_feat_array[number_of_gallery:]

    output_dict = {"type": "comparison", "result": []}
    cos_dist = 1-np.dot(query_feat_array, gallery_feat_array.transpose())

    for i in range(number_of_query):
        idx = np.argmin(cos_dist[i])
        dist = cos_dist[i,idx]
        similar = (dist < FLAGS.prob_thresh)
        output_dict['result'].append({'Query': os.path.basename(imagefiles[i]),
                                      'Target': os.path.basename(gallery_images[idx]),
                                      'Distance': float(dist),
                                      'isSimilar': bool(similar)})

    with open(os.path.join(FLAGS.output_dir, FLAGS.output_file), 'w') as f:
        json.dump(output_dict, f, indent=4)

if __name__ == '__main__':
    tf.app.run()