from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tf_parameter_mgr
import monitor_cb
from monitor_cb import CMonitor


from datetime import datetime
import os.path
import time
import sys
import glob
import numpy as np
import tensorflow as tf
from preprocessing import preprocessing_factory
from nets import nets_factory
from tensorflow.python.ops import variables
# Choose your image preprocessing
tf.app.flags.DEFINE_string('preprocessing_type', 'cifarnet', 'image processing type')
# Choose the network structure
tf.app.flags.DEFINE_string('network_type', 'cifarnet', 'image feature extraction network type')
# Set the number of classes
tf.app.flags.DEFINE_integer('number_classes', 10, 'number of classes')

tf.app.flags.DEFINE_string('weights', '', 'initialize with pretrained model weights')

tf.app.flags.DEFINE_string('train_dir', 'train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('test_interval', 32, 'test_interval')
tf.app.flags.DEFINE_integer('eval_topk', 1, 'accuracy of evaluation of top-k')

# Parameters for distributed training, no need to modify
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')
tf.app.flags.DEFINE_bool('log_device_placement', False, 'log_device_placement')

FLAGS = tf.app.flags.FLAGS
FLAGS.batch_size = tf_parameter_mgr.getTrainBatchSize()

def get_train_op(total_loss, global_step, return_grad=False):
    lr = tf_parameter_mgr.getLearningRate(global_step)
    # Compute gradients.
    opt = tf_parameter_mgr.getOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.no_op(name='train')
    if return_grad:
        return apply_gradient_op, grads
    return train_op

def get_loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss

def get_accuracy(logits, labels):
    top_k_op = tf.nn.in_top_k(logits, labels, FLAGS.eval_topk)
    correct = np.sum(top_k_op)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def get_data(is_train=True):
    batch_size = FLAGS.batch_size
    if is_train:
        filenames = tf_parameter_mgr.getTrainData()
    else:
        filenames = tf_parameter_mgr.getTestData()

    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    label = features['label']
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    image = tf.reshape(image, tf.stack([height, width, depth]))

    preprocessing_type = FLAGS.preprocessing_type
    preprocessor = preprocessing_factory.get_preprocessing(preprocessing_type, is_training=is_train)
    network_type = FLAGS.network_type
    default_size = nets_factory.get_default_size(network_type)
    image = preprocessor(image, output_height=default_size, output_width=default_size)
    image.set_shape([default_size, default_size, 3])

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=4, capacity=32)
    return image_batch, label_batch


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

    is_chief = (FLAGS.task_id == 0)

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster)):
        global_step = tf.contrib.framework.get_or_create_global_step()
        is_training = tf.placeholder_with_default(False, shape=[])
        i_train, l_train = get_data(is_train=True)
        i_test, l_test = get_data(is_train=False)
        images, labels = tf.cond(is_training, lambda: (i_train, l_train), lambda: (i_test, l_test))
        network_type = FLAGS.network_type
        embedding_network = nets_factory.get_network_fn(network_type, num_classes=FLAGS.number_classes,
                                                        is_training=is_training)
        logits, end_points = embedding_network(images)
        total_loss = get_loss(logits, labels)
        accuracy = get_accuracy(logits, labels)
        train_op = get_train_op(total_loss, global_step)
    # setup MAO and Tensorboard monitoring
    print('FLAGS.train_dir', FLAGS.train_dir)
    log_dir = os.path.join(FLAGS.train_dir, 'log')
    # graph = tf.get_default_graph()
    monitor = CMonitor(log_dir, tf_parameter_mgr.getTestInterval(), tf_parameter_mgr.getMaxSteps())
    monitor.SummaryScalar('train accuracy', accuracy)
    monitor.SummaryScalar('test accuracy', accuracy)
    
    monitor.SummaryScalar('train loss', total_loss)
    monitor.SummaryScalar('test loss', total_loss)

    # if is_chief:
    #     graph = tf.get_default_graph()
    #     all_ops = graph.get_operations()
    #     for op in all_ops:
    #         if op.type == 'VariableV2':
    #             output_tensor = graph.get_tensor_by_name(op.name + ':0')
    #             if op.name.endswith('/weights'):
    #                 monitor.SummaryHist("weight", output_tensor, op.name.replace('/',''))
    #                 monitor.SummaryNorm2("weight", output_tensor, op.name.replace('/',''))
    #             elif op.name.endswith('/biases'):
    #                 monitor.SummaryHist("bias", output_tensor, op.name.replace('/',''))
    #         elif op.type == 'Relu':
    #             output_tensor = graph.get_tensor_by_name(op.name + ':0')
    #             monitor.SummaryHist("activation", output_tensor, op.name.replace('/',''))
    #     monitor.SummaryGradient('weight', total_loss)
    #     monitor.SummaryGradient('bias', total_loss)
    #     monitor.SummaryGWRatio()

    train_summary = tf.summary.merge_all(monitor_cb.DLMAO_TRAIN_SUMMARIES)
    test_summary = tf.summary.merge_all(monitor_cb.DLMAO_TEST_SUMMARIES)
    summaryWriter = tf.summary.FileWriter(log_dir)

    class _LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._next_trigger_step = FLAGS.test_interval
            self._trigger = True

        def before_run(self, run_context):
            args = {'global_step': global_step}
            if self._trigger:
                args['train_summary'] = train_summary

            return tf.train.SessionRunArgs(args)

        def after_run(self, run_context, run_values):
            gs = run_values.results['global_step']
            if self._trigger:
                self._trigger = False
                summaryWriter.add_summary(run_values.results['train_summary'], gs)
                summary = run_context.session.run(test_summary, feed_dict={is_training: False})
                summaryWriter.add_summary(summary, gs)
                summaryWriter.flush()
            if gs >= self._next_trigger_step:
                self._next_trigger_step += FLAGS.test_interval
                self._trigger = True


    hooks = [tf.train.StopAtStepHook(last_step=tf_parameter_mgr.getMaxSteps()),
             tf.train.NanTensorHook(total_loss)]

    if is_chief:
        hooks.append(_LoggerHook())

    pretrained_model = FLAGS.weights
    variables_saved = variables._all_saveable_objects()
    print("Variables stored in the model:")
    variables_to_restore = []
    for var in variables_saved:
        if var.op.name.startswith('global_step') or var.op.name.startswith('InceptionV3/Logits'):
            print('remove', var.op.name, var.op)
            continue
        else:
            variables_to_restore.append(var)

    print("------------------------------")
    print('will restore ', variables_to_restore)
    saver = tf.train.Saver(var_list=variables_to_restore)

    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, checkpoint_dir=FLAGS.train_dir, hooks=hooks,
            config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement), save_summaries_steps=None, save_summaries_secs=None) as mon_sess:
        if pretrained_model != None:
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print("Restore pre-trained checkpoint:", ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                print("Successfully restore checkpoint:", ckpt.model_checkpoint_path)
            else:
                files = os.listdir(pretrained_model)
                for f in files:
                    if f.endswith('ckpt'):
                        model_checkpoint_path = pretrained_model+"/"+f
                        try:
                            saver.restore(mon_sess, model_checkpoint_path)
                            print("Successfully restore checkpoint:", model_checkpoint_path)
                        except Exception as e:
                            print("Fail to restore ",model_checkpoint_path,'with message',e)
                        break
        steps = 0

        while not mon_sess.should_stop():
            mon_sess.run(train_op, feed_dict={is_training: True})
            steps += 1
            if steps % 100 == 0: print('%d stpes executed on worker %d.' % (steps, FLAGS.task_id))
            print('%d stpes executed on worker %d.' % (steps, FLAGS.task_id))

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
