import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import sleep
import scipy.signal
import scipy.misc
import os


# Copies one set of variables to another.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Model():
    def __init__(self, scope, opt):
        self.scope = scope

        with tf.variable_scope(scope):
            self.w = tf.get_variable(name='w', initializer=tf.constant(-1.0))
            self.x = tf.placeholder(tf.float32, [None], name="x")
            self.y = tf.placeholder(tf.float32, [None], name="y")

            self.loss = tf.reduce_mean(tf.square(self.y - self.x * self.w))

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

            # Apply local gradients to global network
            self.get_local_grads_op = opt.compute_gradients(self.loss, local_vars)

            # two main opertation
            self.update_global_op = opt.apply_gradients(zip(self.get_local_grads_op, global_vars))
            self.update_local_op = update_target_graph('global', scope)

    def work(self, sess):

        trainx = np.array([1, 2, 3, 4, 5, 6])
        trainy = np.array([1, 2, 3, 4, 5, 6])

        feed_dict = {self.x:trainx, self.y:trainy}

        while not coord.should_stop():
            # sync
            sess.run(self.update_local_op)

            # print global loss
            print self.scope, '=====>', sess.run(self.loss, feed_dict)

            sess.run(self.update_global_op, feed_dict)






if __name__ == '__main__':

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    num_workers = multiprocessing.cpu_count()

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=1e-6)
        master_network = Model('global', opt)  # Generate global network

        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Model('worker_%d'%i, opt))

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)

        coord.join(worker_threads)