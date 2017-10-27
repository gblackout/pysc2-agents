import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import slim

"""
    you can define different networks here, but do include the self.loss class attribute
"""


class Network:
    def __init__(self, xshape, yshape, options):
        self.w = tf.get_variable(name='w', initializer=tf.constant(-1.0))
        self.x = tf.placeholder(tf.float32, xshape, name="x")
        self.y = tf.placeholder(tf.float32, yshape, name="y")

        self.loss = tf.reduce_mean(tf.square(self.y - self.x * self.w))


class AtariFCN:
    def __init__(self, entropy_coef, a_size):

        self.image_size = [84, 84, 4]

        with tf.name_scope('input'):
            # NHWC: n float tensor; where each c in C is the grayscale frame with height H and width W
            self.inputs = tf.placeholder(shape=[None] + self.image_size, dtype=tf.float32, name='state')
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32, name='target_value')
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantage')

        with tf.name_scope('conv1'):
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.inputs, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')

        with tf.name_scope('conv2'):
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')

        with tf.name_scope('dense1'):
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.relu)

        # Output layers for policy and value estimations
        with tf.name_scope('policy_output'):
            self.policy = slim.fully_connected(hidden, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=xavier_initializer(),
                                               biases_initializer=None)
        with tf.name_scope('value_output'):
            self.value = slim.fully_connected(hidden, 1,
                                              activation_fn=None,
                                              weights_initializer=xavier_initializer(),
                                              biases_initializer=None)

        with tf.name_scope('responsible_outputs'):
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        with tf.name_scope('loss'):
            self.entropy = tf.negative(tf.reduce_sum(self.policy * tf.log(self.policy)), name='entropy')

            self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])), name='value_loss')
            self.policy_loss = tf.negative(tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages),
                                           name='policy_loss')

            self.loss = tf.add(self.value_loss, self.policy_loss - entropy_coef * self.entropy, name='loss')