# Copyright (c) 2017 IBM (CN)
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.contrib import slim
import tf_parameter_mgr
import monitor_cb
from monitor_cb import CMonitor

from nets import nets_factory
from preprocessing import preprocessing_factory
from sklearn.metrics import pairwise_distances as pdist
from tensorflow.python.framework import ops

flags = tf.app.flags

flags.DEFINE_string("train_dir", "./", "current working directory")
# data reading args
flags.DEFINE_string("train_data", "/u/kaiwu/OCR/train/", "training images' path")
flags.DEFINE_integer("num_input_threads", 4, "number of parallel input threads")

flags.DEFINE_string("test_data", '/u/kaiwu/OCR/test/', "test images' path")
flags.DEFINE_integer("test_num_batches", 1, "number of batches of test")
flags.DEFINE_integer("test_interval", 10, "test interval")

# training args
flags.DEFINE_integer("classes_per_batch", 192, "Number of classes per batch.")
flags.DEFINE_integer("images_per_class", 64, "Number of images per class.")
flags.DEFINE_integer("batch_size", 128, "Number of images per batch")

# network args
# Choose your image preprocessing
flags.DEFINE_string('preprocessing_type', 'ocr', 'image processing type')
# Choose the network structure
flags.DEFINE_string('network_type', 'inception_v3', 'image feature extraction network type')


flags.DEFINE_string("end_point_name", "Mixed_6c", "end point name for feature embedding")
flags.DEFINE_string("pretrained_model_name", '', "pretrained model name")
#flags.DEFINE_string("weights", 'initial/', "weight file directory")
flags.DEFINE_string("weights", 'train2/', "weight file directory")
flags.DEFINE_string("fixed_parameters_names", "var2fixed.txt",
                    "fixed parameters' names' list filename")
flags.DEFINE_string("restore_parameters_names", "var2restore.txt",
                    "restore parameters' names'  list filename")

flags.DEFINE_float("keep_probability", 0.95, "keep probability of the body network's output")
flags.DEFINE_integer("feature_dim", 128, "embedding features' dimension")

# distributed training args
flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of parameter server host:port; if empty, run local')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of worker host:port')
flags.DEFINE_string('job_name', '', 'The job this process will run, either "ps" or "worker"')
flags.DEFINE_integer('task_id', 0, 'The task index for this process')

flags.DEFINE_float("margin", 0.5, "margin of triplet loss")
flags.DEFINE_bool('log_device_placement', False, 'log_device_placement')

flags.DEFINE_integer('max_to_keep', 3, 'number of checkpoints')
FLAGS = flags.FLAGS


def get_restore_fixed_var_list():
    var2restore = []
    var2fixed = []
    fixed_parameters_names = FLAGS.fixed_parameters_names
    restore_parameters_names = FLAGS.restore_parameters_names

    if fixed_parameters_names != None:
        try:
            with open(fixed_parameters_names) as file:
                param = file.readline()
                var2fixed = param.split(',')
                var2fixed = [var+":0" for var in var2fixed]
        except Exception as e:
            print("Error while reading", fixed_parameters_names)

    if restore_parameters_names != None:
        try:
            with open(restore_parameters_names) as file:
                param = file.readline()
                var2restore = param.split(',')
                var2restore = [var + ":0" for var in var2restore]
        except Exception as e:
            print("Error while reading", restore_parameters_names)

    return var2restore, var2fixed

def is_image_file(image_filename):
    if image_filename.endswith(".jpg") or image_filename.endswith(".jpeg") \
            or image_filename.endswith(".png") or image_filename.endswith(".bmp"):
        return True
    else:
        return False


def random_sample_images(img_files, classes_per_batch, images_per_class):
    def sample_images():
        sampled_imgs = []
        sampled_lbls = []
        has_sampled = 0
        idx = range(len(labels))
        np.random.shuffle(idx)
        label = 0
        while has_sampled < num_images:
            if label >= len(labels):
                raise Exception("Unreasonable setting of images_per_class and classes_per_batch, pls set these params according to the available data!")
            d = labels[idx[label]]
            np.random.shuffle(img_files[d])
            for i in range(min(images_per_class, len(img_files[d]))):
                if has_sampled >= num_images:
                    break
                sampled_imgs.append(img_files[d][i])
                sampled_lbls.append(idx[label])
                has_sampled = has_sampled + 1
            label = label + 1
        # print('sampled_lbls',sampled_lbls)
        # print('idx',idx)
        # print('labels', labels)
        # print('sampled_imgs',sampled_imgs, 'sampled_lbls',sampled_lbls)
        return sampled_imgs, sampled_lbls

    labels = dict(enumerate(img_files.keys()))
    num_images = classes_per_batch * images_per_class
    sample_images.num_images = num_images
    sample_images.img_files = img_files
    sample_images.labels = labels
    sample_images.classes_per_batch = classes_per_batch
    sample_images.images_per_class = images_per_class
    return sample_images


def get_image_files(is_training=True):
    classes_per_batch = FLAGS.classes_per_batch
    images_per_class = FLAGS.images_per_class
    if is_training:
        data_dir = tf_parameter_mgr.getTrainData(False)
    else:
        data_dir = tf_parameter_mgr.getTestData(False)
    print('data_dir [', data_dir,']')
    img_files = {}
    for d in os.listdir(data_dir):
        if not os.path.isdir(data_dir + "/" + d):
            continue
        files = os.listdir(data_dir + "/" + d)
        class_imgs = []
        for f in files:
            if is_image_file(f):
                class_imgs.append(data_dir + "/" + d + "/" + f)
        if len(class_imgs) > 0:
            img_files[d] = class_imgs
    if classes_per_batch >= len(img_files):
        classes_per_batch = len(img_files)
    # Artifically restrict the classes_per_batch can be divided by 3 for the following model construction
    if classes_per_batch % 3 != 0:
        classes_per_batch -= (classes_per_batch % 3)

    image_sampler = random_sample_images(img_files, classes_per_batch, images_per_class)
    data, label = tf.py_func(image_sampler, [], [tf.string, tf.int64])
    return data, label, classes_per_batch * images_per_class


def get_data(is_training):
    with tf.name_scope('Data'):
        # Train Data
        i_train, l_train, num_images_train = get_image_files(is_training=True)
        input_queue_train = data_flow_ops.FIFOQueue(capacity=num_images_train * 4, dtypes=[tf.string, tf.int64],
                                                    shapes=[[], []])
        input_queue_train_enqueue = input_queue_train.enqueue_many([i_train, l_train])
        input_queue_train_qr = tf.train.QueueRunner(input_queue_train, [input_queue_train_enqueue])
        tf.add_to_collection("queue_runners", input_queue_train_qr)
        filename_train, label_train = input_queue_train.dequeue()
        file_content_train = tf.read_file(filename_train)
        file_content_train = tf.image.decode_image(file_content_train)

        preprocessing_type = FLAGS.preprocessing_type
        preprocessor_train = preprocessing_factory.get_preprocessing(preprocessing_type, is_training=True)
        default_size = 100
        image_train = preprocessor_train(file_content_train, output_height=default_size, output_width=default_size)
        image_train.set_shape([default_size, default_size, 3])
        # Unfortunately, we have to restrict num_threads=1 here.
        image_batch_train, label_batch_train = tf.train.batch([image_train, label_train], batch_size=num_images_train,
                                                              num_threads=1, capacity=num_images_train * 2)

        # Test Data
        i_test, l_test, num_images_test = get_image_files(is_training=False)
        input_queue_test = data_flow_ops.FIFOQueue(capacity=num_images_test * 4, dtypes=[tf.string, tf.int64],
                                                   shapes=[[], []])
        input_queue_test_enqueue = input_queue_test.enqueue_many([i_test, l_test])
        input_queue_test_qr = tf.train.QueueRunner(input_queue_test, [input_queue_test_enqueue])
        tf.add_to_collection("queue_runners", input_queue_test_qr)
        filename_test, label_test = input_queue_test.dequeue()
        file_content_test = tf.read_file(filename_test)
        file_content_test = tf.image.decode_image(file_content_test)

        preprocessor_test = preprocessing_factory.get_preprocessing(preprocessing_type, is_training=False)
        image_test = preprocessor_test(file_content_test, output_height=default_size, output_width=default_size)
        image_test.set_shape([default_size, default_size, 3])
        # Unfortunately, we have to restrict num_threads=1 here.
        image_batch_test, label_batch_test = tf.train.batch([image_test, label_test], batch_size=num_images_test,
                                                            num_threads=1, capacity=num_images_test * 2)


        image_batch = tf.cond(is_training, lambda: tf.identity(image_batch_train), lambda: tf.identity(image_batch_test), name='image_batch')
        label_batch = tf.cond(is_training, lambda: tf.identity(label_batch_train), lambda: tf.identity(label_batch_test), name='label_batch')
    return image_batch, label_batch


def get_loss_op(embeddings):
    with tf.name_scope('Loss'):
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, FLAGS.feature_dim]), 3, 1)
        margin = FLAGS.margin
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
        basic_loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        tf.add_to_collection("losses", basic_loss)
        total_loss = tf.add_n(tf.get_collection("losses"), name='total_loss')

    return total_loss


def sample_triplet(embeddings_value, labels_value):
    # collect anchor-positive pairs
    ap_pairs = []
    nsamples = labels_value.shape[0]
    for i in range(nsamples):
        for j in range(i):
            if labels_value[i] == labels_value[j]:
                ap_pairs.append([i, j])
    max_numberof_op_pairs = 2048
    total_possible_ap_pairs = len(ap_pairs)
    keep_prob = float(max_numberof_op_pairs) / total_possible_ap_pairs
    triplet = []
    distance = pdist(embeddings_value)
    num_selected_ap = 0
    num_selected_triplet = 0
    total_loss = 0
    # sample triplet
    for ap in ap_pairs:
        if np.random.random() >= keep_prob:
            continue
        positive_label = labels_value[ap[0]]
        all_negative_index = np.where(labels_value != positive_label)[0]
        d = distance[ap[0], all_negative_index]
        all_negative = np.where(d - distance[ap[0], ap[1]] < FLAGS.margin)[0]
        nrof_all_negative = all_negative.shape[0]
        if nrof_all_negative > 0:
            rnd_index = np.random.randint(nrof_all_negative)
            neg_idx = all_negative_index[all_negative[rnd_index]]
            triplet.append([ap[0], ap[1], neg_idx])
            total_loss += FLAGS.margin + distance[ap[0], ap[1]] - d[all_negative[rnd_index]]
            num_selected_triplet += 1
        num_selected_ap += 1
    np.random.shuffle(triplet)
    rtriplet = []
    for t in triplet:
        rtriplet = rtriplet + t
    return rtriplet, total_loss, num_selected_triplet, num_selected_ap


def train_step(data_batch_val, label_batch_val, session, is_training_placeholder, data_placeholder, embeddings,
               train_op, istep=0):
    # Step 1 get embeddings of data
    nrof_examples = data_batch_val.shape[0]
    batch_size = FLAGS.batch_size
    if batch_size%3 != 0:
        batch_size += (3-batch_size%3)
    num_sub_batches = int(np.ceil(float(nrof_examples) / batch_size))
    embeddings_value = np.zeros((nrof_examples, FLAGS.feature_dim))
    start_time = time.time()
    for isub_batch in range(num_sub_batches):
        start_idx = isub_batch * batch_size
        end_idx = min((isub_batch + 1) * batch_size, nrof_examples)
        embeddings_value[start_idx:end_idx] = session.run(embeddings,
                                                          feed_dict={
                                                              data_placeholder: data_batch_val[start_idx:end_idx],
                                                              is_training_placeholder: True})
        for var in tf.get_collection("monitored variables"):
            val = session.run(var)
            print('Emb: ', var.name, val[:10])
    duration = time.time() - start_time
    start_time = time.time()
    print("Batch [%d]'s feature embedding: Time %3f seconds"%(istep, duration))

    triplet, loss, num_selected_triplet, num_selected_ap = sample_triplet(embeddings_value, label_batch_val)
    duration = time.time() - start_time
    start_time = time.time()
    print("Batch [%d]'s selected #triplet=%4d, total_loss=%4f, #selected_triplet=%4d, #selected_ap=%4d. Time %3d seconds"
          %(istep, len(triplet)/3, loss, num_selected_triplet, num_selected_ap, duration))
    # Step 2 forward_backward triplet loss
    nrof_triplet_examples = len(triplet)
    num_sub_batches = int(np.ceil(float(nrof_triplet_examples) / batch_size))
    print("Batch [%d]'s Train:"%(istep))
    start_time = time.time()
    for isub_batch in range(num_sub_batches):
        start_idx = isub_batch * batch_size
        end_idx = min((isub_batch + 1) * batch_size, nrof_triplet_examples)
        sample_idx = triplet[start_idx:end_idx]
        duration = time.time() - start_time
        print("    [%d/%d] Time %3f seconds"%(isub_batch,num_sub_batches, duration))

        session.run(train_op, feed_dict={data_placeholder: data_batch_val[sample_idx], is_training_placeholder: True})
        for var in tf.get_collection("monitored variables"):
            val = session.run(var)
            # print('Train:', var.name, val[:10])
    duration = time.time() - start_time
    start_time = time.time()
    print("Batch [%d]'s Train: Time %3f seconds" % (istep, duration))
    average_loss = loss / num_selected_triplet
    average_accuracy = 1-float(num_selected_triplet) / num_selected_ap
    return average_loss, average_accuracy

def test_step(data_batch_val, label_batch_val, session, is_training_placeholder, data_placeholder, embeddings):
    nrof_examples = data_batch_val.shape[0]
    batch_size = FLAGS.batch_size
    if batch_size%3 != 0:
        batch_size += (3-batch_size%3)
    num_sub_batches = int(np.ceil(float(nrof_examples) / batch_size))
    embeddings_value = np.zeros((nrof_examples, FLAGS.feature_dim))
    for isub_batch in range(num_sub_batches):
        start_idx = isub_batch * batch_size
        end_idx = min((isub_batch + 1) * batch_size, nrof_examples)
        embeddings_value[start_idx:end_idx] = session.run(embeddings,
                                                          feed_dict={
                                                              data_placeholder: data_batch_val[start_idx:end_idx],
                                                              is_training_placeholder: False})
        for var in tf.get_collection("monitored variables"):
            val = session.run(var)
            # print('Test:', var.name, val[:10])
    triplet, loss, num_selected_triplet, num_selected_ap = sample_triplet(embeddings_value, label_batch_val)
    average_loss = loss / num_selected_triplet
    average_accuracy = 1-float(num_selected_triplet) / num_selected_ap
    return average_loss, average_accuracy


def train():
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    print('Ps hosts are: %s' % ps_hosts)
    print('Worker hosts are: %s' % worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        server.join()

    var2restore, var2fixed = get_restore_fixed_var_list()
    is_chief = (FLAGS.task_id == 0)
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster)):
        global_step = tf.contrib.framework.get_or_create_global_step()
        global_step_increment_op = tf.assign_add(global_step, 1)
        is_training = tf.placeholder_with_default(False, shape=[],name='is_training')
        image_batch, label_batch = get_data(is_training)

        network_type = FLAGS.network_type
        embedding_network = nets_factory.get_network_fn(network_type, num_classes=21,
                                                        is_training=is_training, final_endpoint=FLAGS.end_point_name)
        shape = [d for d in image_batch.shape.dims]
        shape[0] = tf.Dimension(None)
        #shape[1] = tf.Dimension(None)
        #shape[2] = tf.Dimension(None)
        data_placeholder = tf.placeholder(tf.float32, shape=shape, name='data_placeholder')
        embeddings, end_points = embedding_network(data_placeholder)
        embeddings = slim.flatten(embeddings)
        if FLAGS.keep_probability < 1:
            embeddings = slim.dropout(embeddings, FLAGS.keep_probability, is_training=is_training,
                                     scope='Dropout')
        embeddings = slim.fully_connected(embeddings, FLAGS.feature_dim, activation_fn=None)
        embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10, name="embedding")

        total_loss = get_loss_op(embeddings)
        lr = tf_parameter_mgr.getLearningRate(global_step)
        opt = tf_parameter_mgr.getOptimizer(lr)

        # Exclude fixed variables
        trainable_var = []
        for var in ops.get_collection("trainable_variables") + ops.get_collection("trainable_resource_variables"):
            if var.name not in var2fixed:
                trainable_var.append(var)
        grads = opt.compute_gradients(total_loss, var_list=trainable_var)
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=None)
        with tf.control_dependencies([apply_gradient_op] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.no_op(name='train')

    log_dir = os.path.join(FLAGS.train_dir, 'log')
    accuracy_placeholder = tf.placeholder(tf.float32,[])
    loss_placeholder = tf.placeholder(tf.float32,[])
    monitor = CMonitor(log_dir, tf_parameter_mgr.getTestInterval(), tf_parameter_mgr.getMaxSteps())
    monitor.SummaryScalar('train accuracy', accuracy_placeholder)
    monitor.SummaryScalar('test accuracy', accuracy_placeholder)
    monitor.SummaryScalar('train loss', loss_placeholder)
    monitor.SummaryScalar('test loss', loss_placeholder)

    has_weights_monitored = False
    has_bias_monitored = False
    has_activation_monitored = False

    if is_chief:
        graph = tf.get_default_graph()
        all_ops = graph.get_operations()
        for op in all_ops:
            if op.type == 'VariableV2':
                output_tensor = graph.get_tensor_by_name(op.name + ':0')
                if op.name.endswith('/weights'):
                    if has_weights_monitored:
                        continue
                    monitor.SummaryHist("weight", output_tensor, op.name.replace('/', ''))
                    monitor.SummaryNorm2("weight", output_tensor, op.name.replace('/', ''))
                    has_weights_monitored = True
                elif op.name.endswith('/biases'):
                    if has_bias_monitored:
                        continue
                    monitor.SummaryHist("bias", output_tensor, op.name.replace('/', ''))
                    has_bias_monitored = True
            # elif op.type == 'Relu':
            #     if has_activation_monitored:
            #         continue
            #     output_tensor = graph.get_tensor_by_name(op.name + ':0')
            #     monitor.SummaryHist("activation", output_tensor, op.name.replace('/', ''))
            #     has_activation_monitored = True

    train_summary = tf.summary.merge_all(monitor_cb.DLMAO_TRAIN_SUMMARIES)
    test_summary = tf.summary.merge_all(monitor_cb.DLMAO_TEST_SUMMARIES)
    summaryWriter = tf.summary.FileWriter(log_dir)

    hooks = []

    pretrained_model = FLAGS.weights
    FLAGS.batch_size = tf_parameter_mgr.getTrainBatchSize()

    var_list = []
    for var in tf.global_variables()+tf.local_variables():
        if var.name in var2restore:
            var_list.append(var)



    saver = None
    if pretrained_model != None and len(var_list) > 0:
        saver = tf.train.Saver(var_list=var_list)
    with tf.train.MonitoredTrainingSession(master=server.target, hooks=hooks, is_chief=is_chief, checkpoint_dir=FLAGS.train_dir,
                                            config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
                                           save_summaries_steps=None, save_summaries_secs=None) as mon_sess:
    # scaffold = tf.train.Scaffold()
    # if is_chief:
    #     session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, checkpoint_dir=FLAGS.train_dir, master=server.target)
    #     hooks.append(tf.train.CheckpointSaverHook(FLAGS.train_dir, save_secs=120, scaffold=scaffold))
    # else:
    #     session_creator = tf.train.WorkerSessionCreator(scaffold=scaffold, master=server.target)
    #
    # with tf.train.MonitoredSession(
    #         session_creator=session_creator, hooks=hooks) as mon_sess:

        print("Before Initializatin ...")
        for var in tf.get_collection("monitored variables"):
            val = mon_sess.run(var)
            print(var.name, val.flatten()[:10])
        if pretrained_model != None and saver != None:
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print("Restore pre-trained checkpoint:", ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                print("Successfully restore checkpoint:", ckpt.model_checkpoint_path)
            else:
                files = os.listdir(pretrained_model)
                for f in files:
                    if f.endswith('ckpt'):
                        model_checkpoint_path = pretrained_model+f
                        #try:
                        saver.restore(mon_sess, model_checkpoint_path)
                        print("Successfully restore checkpoint:", model_checkpoint_path)
                        #except Exception as e:
                        #    print("Fail to restore ", model_checkpoint_path,'with message',e)
                        break

        print("After Initializatin ...")
        for var in tf.get_collection("monitored variables"):
            val = mon_sess.run(var)
            print(var.name, val.flatten()[:10])

        steps = 0
        while steps < tf_parameter_mgr.getMaxSteps():
            data_batch_val, label_batch_val = mon_sess.run([image_batch, label_batch], feed_dict={is_training: True})
            loss_train, accuracy_train = train_step(data_batch_val, label_batch_val, mon_sess, is_training,
                                                    data_placeholder, embeddings, train_op, istep=steps)
            summary, _, s_v, lr_v = mon_sess.run([train_summary, global_step_increment_op, global_step, lr], feed_dict={accuracy_placeholder: accuracy_train, loss_placeholder: loss_train})
            summaryWriter.add_summary(summary, steps)
            summaryWriter.flush()
            print('global_step', s_v, 'learning rate', lr_v)
            # for var in tf.get_collection('monitored variables'):
            #     val = mon_sess.run(var)
            #     print(var.name, val.flatten()[:10])
            if steps % FLAGS.test_interval == 0:
                data_batch_val, label_batch_val = mon_sess.run([image_batch, label_batch],
                                                               feed_dict={is_training: False})
                loss_test, accuracy_test = test_step(data_batch_val, label_batch_val, mon_sess, is_training,
                                                     data_placeholder, embeddings)
                summary = mon_sess.run(test_summary,
                                       feed_dict={accuracy_placeholder: accuracy_test, loss_placeholder: loss_test})
                summaryWriter.add_summary(summary, steps)
                summaryWriter.flush()
            steps += 1

if __name__ == '__main__':
    train()
