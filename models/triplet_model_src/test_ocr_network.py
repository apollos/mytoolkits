import tensorflow as tf
from nets import nets_factory
def test():
    image = tf.placeholder(tf.float32, [None, 100, 100, 3])
    is_training = tf.placeholder(tf.bool, [])
    net = nets_factory.get_network_fn('inception_v3', is_training = is_training, final_endpoint='Mixed_6c')
    embedding, end_points =net(image)
    print(embedding, embedding.shape)
    writer = tf.summary.FileWriter('/tmp/my-model/')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    writer.close()


test()
quit()