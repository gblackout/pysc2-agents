import tensorflow as tf
from models import AtariFCN
import numpy as np
from utils import update_target_graph
from env.atari import AtariEnv


class Worker:
    def __init__(self, scope, opt, env, coord, global_vars, options):
        """
        
        :param scope: 
        :param opt: 
        :type param env: AtariEnv
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
        self.discount_factor = options.discount_factor
        self.recorder = RewardMointor(scope)
        self.episode_cnt = 0

        with tf.variable_scope(scope):
            self.model = AtariFCN(options.entropy_coef, len(self.env.action_space))

            # Get gradients from local network using local losses
            self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            global_vars = self.local_vars if global_vars is None else global_vars

            # Apply local gradients to global network
            self.get_local_grads_op = tf.gradients(self.model.loss, self.local_vars)

            # two main opertation
            self.get_local_clipped_grads_op, self.grad_norms = tf.clip_by_global_norm(self.get_local_grads_op,
                                                                                      options.clip_norm)
            # TODO can be danger
            self.update_global_op = opt.apply_gradients(zip(self.get_local_clipped_grads_op, global_vars))

            self.update_local_op = update_target_graph(global_vars, self.local_vars)

    def work(self, sess):
        # initial sync
        sess.run(self.update_local_op)

        # creating batch buf and reuse them to avoid re-allocating mem
        # NHWC, +1 for saving the last state with no reward and action
        states_buf = np.zeros([self.batch_size + 1] + self.model.image_size, dtype=np.float32)
        action_buf = np.zeros(self.batch_size, dtype=np.int32)
        reward_buf = np.zeros(self.batch_size, dtype=np.float32)

        while not (self.coord.should_stop() and self.episode_cnt < self.max_episode):

            for done, buf_ind in self.run_episode(states_buf, action_buf, reward_buf, sess):
                self.update(states_buf[:buf_ind + 1] if done else states_buf,
                            action_buf[:buf_ind] if done else action_buf,
                            reward_buf[:buf_ind] if done else reward_buf,
                            done, sess)

            self.episode_cnt += 1

    def update(self, states_buf, action_buf, reward_buf, done, sess):

        cumu_reward_buf = np.zeros(states_buf.shape[0], dtype=np.float32)  # batch_size + 1
        state_vals = self.predict_value(states_buf, sess)

        if not done:
            cumu_reward_buf[-1] = state_vals[-1]

        # TODO can do without for loop
        for i in range(cumu_reward_buf.shape[0] - 1)[::-1]:  # batch_size
            cumu_reward_buf[i] = reward_buf[i] + cumu_reward_buf[i + 1] * self.discount_factor

        advantage_buf = cumu_reward_buf[:-1] - state_vals[:-1]

        feed_dict = {
            self.model.target_v: cumu_reward_buf[:-1],
            self.model.inputs: states_buf[:-1],
            self.model.actions: action_buf,
            self.model.advantages: advantage_buf,
        }

        # update global with local gradient
        _ = sess.run([self.update_global_op], feed_dict=feed_dict)

        # sync after update
        sess.run(self.update_local_op)

    def run_episode(self, states_buf, action_buf, reward_buf, sess):
        """
        run a complete episode and fill the buffers; each time when buf is full it yields boolean indicating
        whether episode ends and a ind pointing to the end of filled buf. The latter is used to determine the size
        when episode ends before buf is full
        
        """
        curr_state = self.env.reset(sess)
        done = False
        curr_buf_ind = 0

        episode_reward = 0.0
        episode_steps = 0

        while not done:

            act_probs = self.predict_action_prob(curr_state, sess)
            action = np.random.choice(self.env.action_space, p=act_probs)
            next_state, reward, done, _ = self.env.step(sess, action)

            states_buf[curr_buf_ind] = curr_state
            action_buf[curr_buf_ind] = action
            reward_buf[curr_buf_ind] = reward

            curr_buf_ind += 1
            curr_state = next_state

            episode_reward += reward
            episode_steps += 1

            if curr_buf_ind == self.batch_size:
                states_buf[curr_buf_ind] = curr_state  # put in final state
                yield done, curr_buf_ind
                curr_buf_ind = 0

        self.recorder.add_record(episode_reward, episode_steps, self.episode_cnt)

        if curr_buf_ind > 0:
            states_buf[curr_buf_ind] = curr_state  # put in final state
            yield done, curr_buf_ind

    def predict_action_prob(self, state, sess):
        res = sess.run(self.model.policy, {self.model.inputs: [state]})
        return res[0]

    def predict_value(self, state, sess):
        res = sess.run(self.model.value, {self.model.inputs: state})
        return res

    def predict_value_single(self, state, sess):
        return self.predict_value([state], sess)[0]


# TODO ad-hoc
class RewardMointor:
    def __init__(self, name, log_freq=5):
        self.summary_writer = tf.summary.FileWriter(name)
        self.episode_reward_ls = []
        self.episode_length_ls = []
        self.log_freq = log_freq

    def add_record(self, episode_reward, episode_length, episode_count):
        self.episode_length_ls.append(episode_length)
        self.episode_reward_ls.append(episode_reward)

        if len(self.episode_reward_ls) == self.log_freq:
            summary = tf.Summary()
            summary.value.add(tag='Perf/Reward', simple_value=float(np.mean(self.episode_reward_ls)))
            summary.value.add(tag='Perf/Length', simple_value=float(np.mean(self.episode_length_ls)))
            self.summary_writer.add_summary(summary, episode_count)
            self.episode_reward_ls = []
            self.episode_length_ls = []