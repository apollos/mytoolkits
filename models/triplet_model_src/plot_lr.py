import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_lr_policy_exponential(max_steps, base_lr, lr_decay_rate, lr_decay_steps, lr_staircase):
    global_step=tf.placeholder(tf.int64)
    lr = tf.train.exponential_decay(learning_rate = base_lr, global_step=global_step, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate, staircase=lr_staircase)
    lrs = []
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps):
        v = sess.run(lr, feed_dict={global_step:i})
        lrs.append(v)
    plt.plot(np.arange(max_steps), lrs)
    plt.show()

plot_lr_policy_exponential(max_steps=10000,
                           base_lr=0.01,
                           lr_decay_rate=0.5,
                           lr_decay_steps=3000,
                           lr_staircase=False
                           )


