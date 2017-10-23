import tensorflow as tf


"""
    you can define different networks here, but do include the self.loss class attribute
"""


class Network:
    def __init__(self, xshape, yshape, options):
        self.w = tf.get_variable(name='w', initializer=tf.constant(-1.0))
        self.x = tf.placeholder(tf.float32, xshape, name="x")
        self.y = tf.placeholder(tf.float32, yshape, name="y")

        self.loss = tf.reduce_mean(tf.square(self.y - self.x * self.w))