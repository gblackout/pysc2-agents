from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import importlib
import threading
from os.path import join as joinpath
from utils import makedir, get_output_folder

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.lib import app
import gflags as flags
import tensorflow as tf

from run_loop import run_loop

COUNTER = 0
LOCK = threading.Lock()

FLAGS = flags.FLAGS


# path
flags.DEFINE_string("output_path", "./out", "Path for snapshot.")

# resources setup
flags.DEFINE_string("device", "0,1", "Device for training.") # default two GPUs
flags.DEFINE_integer("parallel", 8, "How many instances to run in parallel.")

# game setup
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_enum("agent_race", "T", sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")

flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

# model setup
flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")

# training setup
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", 1e5, "Total steps for training.") # max episode for all agents
flags.DEFINE_integer("max_agent_steps", 60, "Total agent steps.") # max step per episode
flags.DEFINE_integer("snapshot_step", 1e3, "Step for snapshot.") # save snapshot per snapshot_step episode
flags.DEFINE_integer("max_to_keep", 10, "max snapshot to keep")

# debugging
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

FLAGS(sys.argv)

if FLAGS.training:
    PARALLEL = FLAGS.parallel
    MAX_AGENT_STEPS = FLAGS.max_agent_steps
    DEVICE = ['/gpu:' + dev for dev in FLAGS.device.split(',')]
else:
    PARALLEL = 1
    MAX_AGENT_STEPS = 1e5
    DEVICE = ['/cpu:0']

# setup all pathes
output_dir = get_output_folder(FLAGS.output_path, '%s-%s' % (FLAGS.map, FLAGS.net))
LOG = joinpath(output_dir, 'summary')
SNAPSHOT = joinpath(output_dir, 'checkpoint')

makedir(output_dir)
makedir(LOG)
makedir(SNAPSHOT)


def run_thread(agent, map_name, visualize):

    with sc2_env.SC2Env(
            map_name=map_name,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=visualize) as env:

        env = available_actions_printer.AvailableActionsPrinter(env)
        replay_buffer = []

        # for each episode
        for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):

            if FLAGS.training:

                replay_buffer.append(recorder)

                if is_done:

                    counter = 0
                    with LOCK:
                        global COUNTER
                        COUNTER += 1
                        counter = COUNTER

                    # Learning rate schedule
                    learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)

                    # update agent
                    agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)

                    # clear buffer ? why
                    replay_buffer = []

                    # save model for every snapshot_step
                    if counter % FLAGS.snapshot_step == 1:
                        agent.save_model(SNAPSHOT, counter)

                    # stop if max_steps reached
                    if counter >= FLAGS.max_steps:
                        break

            elif is_done:
                obs = recorder[-1].observation
                score = obs["score_cumulative"][0]
                print('Your score is ' + str(score) + '!')

        if FLAGS.save_replay:
            env.save_replay(agent.name)


def _main(unused_argv):
    """
        Run agents
    """

    # ======================================================================
    #                                config
    # ======================================================================

    # sw
    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    # Assert the map exists.
    maps.get(FLAGS.map)

    # Setup agents
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    agents = []
    # init agents
    for i in range(PARALLEL):
        agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
        # TODO why assigning to different device?
        agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net, FLAGS.max_to_keep)
        agents.append(agent)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # ======================================================================
    #                                training
    # ======================================================================
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(LOG)
    for i in range(PARALLEL):
        agents[i].setup(sess, summary_writer)

    agent.initialize()
    if not FLAGS.training or FLAGS.continuation:
        global COUNTER
        COUNTER = agent.load_model(SNAPSHOT)

    # Run threads
    threads = []
    for i in range(PARALLEL - 1):
        t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False))
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(5)

    # last one used to render ?
    run_thread(agents[-1], FLAGS.map, FLAGS.render)

    for t in threads:
        t.join()

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    app.really_start(_main)
