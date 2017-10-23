import tensorflow as tf
from models import Network
import numpy as np
from utils import update_target_graph


class Worker:
    def __init__(self, scope, opt, env, coord, global_vars, options):
        """
        
        :param scope: 
        :param opt: 
        :param env: 
        :param coord: 
        :param global_vars:
            collection of variables from global graph; set this to None for master,
            and pass its local_vars to init all workers.
        :param options: 
        """

        self.scope = scope
        self.max_episode = options.max_episode
        self.env = env
        self.batch_size = options.batch_size
        self.coord = coord

        with tf.variable_scope(scope):
            self.model = Network([None], [None], options) # TODO change here for different net

            # Get gradients from local network using local losses
            self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            global_vars = self.local_vars if global_vars is None else global_vars

            # Apply local gradients to global network
            self.get_local_grads_op = tf.gradients(self.model.loss, self.local_vars)

            # two main opertation
            self.update_global_op = opt.apply_gradients(zip(self.get_local_grads_op, global_vars))  # TODO can be danger
            self.update_local_op = update_target_graph(global_vars, self.local_vars)

    def work(self, sess):
        # initial sync
        sess.run(self.update_local_op)

        episode_cnt = 0
        while not (self.coord.should_stop() and episode_cnt < self.max_episode):

            for buf in self.run_episode():

                # TODO change here for different data preprocessing
                trainx = np.array([1, 2, 3, 4, 5, 6])
                trainy = np.array([1, 2, 3, 4, 5, 6])
                feed_dict = {self.model.x: trainx, self.model.y: trainy}

                sess.run(self.update_global_op, feed_dict)

                # sync after update
                sess.run(self.update_local_op)

            episode_cnt += 1

    def run_episode(self):
        # TODO change here for way of running episode
        buf = []

        while not self.env.is_episode_finished():
            self.step()

            buf.append(None)

            if len(buf) == self.batch_size:
                yield buf
                buf = []

        if len(buf) > 0:
            yield buf

    def step(self):
        # TODO change here for different way of taking step
        pass
