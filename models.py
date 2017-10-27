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
    def __init__(self, entropy_coef, a_size=3):

        self.image_size = [84, 84, 4]

        # NHWC
        self.inputs = tf.placeholder(shape=[None] + self.image_size, dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

        self.conv1 = slim.conv2d(activation_fn=tf.nn.relu,
                                 inputs=self.inputs, num_outputs=16,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID')

        self.conv2 = slim.conv2d(activation_fn=tf.nn.relu,
                                 inputs=self.conv1, num_outputs=32,
                                 kernel_size=[4, 4], stride=[2, 2], padding='VALID')

        hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.relu)

        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(hidden, a_size,
                                           activation_fn=tf.nn.softmax,
                                           weights_initializer=xavier_initializer(),
                                           biases_initializer=None)

        self.value = slim.fully_connected(hidden, 1,
                                          activation_fn=None,
                                          weights_initializer=xavier_initializer(),
                                          biases_initializer=None)

        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        # Loss functions
        self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
        self.loss = self.value_loss + self.policy_loss - entropy_coef * self.entropy