# MIT License
# 
# Copyright (c) 2016 David Sandberg
# Copyright (c) 2017 IBM (CN) 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform 
import os,sys

#retrieve imgs
def get_input_imgs(images_dir):
    print("get_input_imgs", images_dir)
    image_filename_extension_list = ['.jpg', '.jepg', '.png','.bmp']
    inputClasses = os.listdir(images_dir)
    inputImgs = []
    total_num_imgs = 0
    for c, name in enumerate(inputClasses):
        imgDir = images_dir + "/"+name+"/"
        if not os.path.isdir(imgDir):
            continue
        imgs = os.listdir(imgDir)
        filelist = []
        for img in imgs:
            filename = imgDir+img
            if os.path.splitext(filename)[1] in image_filename_extension_list:
                filelist.append(filename)
            else:
                print("Skip file: ", filename)
        if len(filelist) > 0:
            inputImgs.append(filelist)
            total_num_imgs += len(filelist)

    return inputImgs, total_num_imgs

def sample_images(inputImgs, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
    nrof_classes = len(inputImgs)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    image_paths = []
    num_per_class = []
    
    i = 0
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(inputImgs[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [inputImgs[class_index][j] for j in idx]
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1
        #Avoid problems for small datasets, i.e., number of persons is too small
        i = i%nrof_classes
    #print("sample_images: ", image_paths, num_per_class)
    return image_paths, num_per_class


def get_loss(anchor, positive, negative , FLAGS):
    margin = FLAGS.margin
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss



def get_train_op(loss, update_gradient_vars, learning_rate, global_step, FLAGS):
    # Compute gradients.
    if FLAGS.optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif FLAGS.optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate)
    elif FLAGS.optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate)
    elif FLAGS.optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate)
    else:
        raise ValueError('Invalid optimization algorithm')
    
    if  FLAGS.synchronize_mode and FLAGS.ps_hosts:
        replicas_to_aggregate = len(FLAGS.worker_hosts.split(','))
        opt = tf.train.SyncReplicasOptimizer(
          opt=opt, replicas_to_aggregate=replicas_to_aggregate)
        
    train_op = opt.minimize(loss, global_step= global_step, var_list=update_gradient_vars)
    return train_op, opt

def eval_triplets(embeddings, nrof_images_per_class, image_paths, FLAGS):
    """
    Select the triplets for testing
    """
    avg_loss = 0
    trip_idx = 0
    emb_start_idx = 0
    num_ap_pairs = 0
    triplets = []
    nrof_triplets = 0
    max_numberof_op_pairs = 2000
    total_possible_ap_pairs = 0
    for i in xrange(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            total_possible_ap_pairs += nrof_images-1-j
    keep_prob = float(min(max_numberof_op_pairs, total_possible_ap_pairs))/total_possible_ap_pairs
    
    for i in xrange(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            for pair in xrange(j, nrof_images): 
                if np.random.random() > keep_prob:
                    continue
                p_idx = emb_start_idx + pair
                candidate_list = range(0, emb_start_idx) + range(emb_start_idx + nrof_images, embeddings.shape[0])
                n_idx = candidate_list[np.random.randint(len(candidate_list))]
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[n_idx]))
                
                num_ap_pairs += 1
                if neg_dist_sqr - pos_dist_sqr < FLAGS.margin:
                    nrof_triplets += 1
                    avg_loss += pos_dist_sqr + FLAGS.margin - neg_dist_sqr
                triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    
        emb_start_idx += nrof_images
    
    return triplets, num_ap_pairs, nrof_triplets, avg_loss/num_ap_pairs


def sample_triplets(embeddings, nrof_images_per_class, image_paths, FLAGS):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_ap_pairs = 0
    triplets = []
    max_numberof_op_pairs = 2000
    total_possible_ap_pairs = 0
    for i in xrange(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            total_possible_ap_pairs += nrof_images-1-j
    keep_prob = float(min(max_numberof_op_pairs, total_possible_ap_pairs))/total_possible_ap_pairs
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    for i in xrange(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                if np.random.random() >= keep_prob:
                    continue
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.Infinity
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < FLAGS.margin)[0]  # VGG Face selection
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
#                     print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
#                         (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1
                num_ap_pairs += 1

        emb_start_idx += nrof_images
    np.random.shuffle(triplets)
    return triplets, num_ap_pairs, len(triplets)

def all_pairwise_error(embedding_array, num_per_class, FLAGS):
    dist = squareform(pdist(embedding_array))
    np.fill_diagonal(dist, np.Infinity)
    num_correct = 0
    start_idx = 0
    for i in xrange(FLAGS.people_per_batch):
        nrof_images = int(num_per_class[i])
        for j in xrange(0, nrof_images):
            index = np.argpartition(dist[start_idx+j], FLAGS.test_topk)[:FLAGS.test_topk]
            if np.sum((index<start_idx + nrof_images)*(index>=start_idx)) > 0:
                num_correct += 1
    
    return float(num_correct)/embedding_array.shape[0]

def get_pretrained_model_filename(FLAGS):
    #FLAGS.weights = '/tmp/tt/initial/'
    if FLAGS.weights:
        model_filename = tf.train.latest_checkpoint(FLAGS.weights)
        if model_filename != None:
            return model_filename
        else:
            if os.path.isdir(FLAGS.weights):
                #Simple try to find the actual weight file
                files = os.listdir(FLAGS.weights)
                for f in files:
                    if os.path.isfile(FLAGS.weights +"/"+f):
                        model_filename = FLAGS.weights +"/"+f
        return model_filename
    else:
        return ''