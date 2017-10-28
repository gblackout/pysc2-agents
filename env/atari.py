import gym
import tensorflow as tf
import numpy as np


class AtariEnv:
    def __init__(self):
        self.env = gym.envs.make('Breakout-v0')
        self.action_space = range(self.env.action_space.n)

        self._state_buf = None
        self.state_process_op = self._make_process_op()

    def step(self, sess, *args, **kwargs):

        next_state, reward, done, info = self.env.step(*args, **kwargs)

        self._update_buf(next_state, sess)

        reward = max(min(reward, 1), -1)

        return np.copy(self._state_buf), reward, done, info

    def reset(self, sess):
        state = self.env.reset()
        self._update_buf(state, sess, init=True)
        return np.copy(self._state_buf)

    def _update_buf(self, state, sess, init=False):
        buf = sess.run(self.state_process_op, {self.input_state: state})

        if init:
            self._state_buf = np.concatenate([buf]*4, axis=0)
        else:
            self._state_buf = np.roll(self._state_buf, -1, axis=0)
            self._state_buf[-1, :, :] = buf[0, :, :]

    def _make_process_op(self):
        """
            210-160-3 atari RGB image
        :return:
            1-84-84 float grayscale matrix
        """
        with tf.device("/cpu:0"):
            with tf.variable_scope("state_preprocess"):
                self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
                output = tf.image.rgb_to_grayscale(self.input_state)
                output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
                output = tf.image.resize_images(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                output = tf.to_float(output) / 255.0
                output = tf.transpose(output, perm=[2, 1, 0])

        return output

if __name__ == '__main__':
    AtariEnv()