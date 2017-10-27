import gym
import tensorflow as tf
import numpy as np


class AtariEnv:
    def __init__(self):
        self.env = gym.envs.make('Breakout-v0')
        self.action_space = range(self.env.action_space.n)

        self._state_buf = None
        # TODO assuming this op is set outside to be pinned to gpu device
        self.state_process_op = self._make_process_op()

    def step(self, sess, *args, **kwargs):

        next_state, reward, done, info = self.env.step(*args, **kwargs)
        self._update_buf(sess, next_state)

        return np.copy(self._state_buf), reward, done, info

    def reset(self, sess):
        state = self.env.reset()
        self._update_buf(state, sess, init=True)
        return np.copy(self._state_buf)

    def _update_buf(self, sess, state, init=False):
        buf = sess.run(self.state_process_op, {self.input_state: state})

        if init:
            self._state_buf = np.stack([buf]*4, axis=2)
        else:
            self._state_buf = np.roll(self._state_buf, -1, axis=2)
            self._state_buf[:, :, -1] = buf

    def _make_process_op(self):
        with tf.variable_scope("state_preprocess"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            output = tf.image.rgb_to_grayscale(self.input_state)
            output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
            output = tf.image.resize_images(output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.squeeze(output)

        return output

if __name__ == '__main__':
    AtariEnv()