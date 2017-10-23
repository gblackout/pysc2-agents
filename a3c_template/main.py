import threading
import multiprocessing
import tensorflow as tf
from time import sleep
from optparse import OptionParser
from worker import Worker


def init_env():
    return None # TODO change here for different env


def get_session(options):
    config = tf.ConfigProto(allow_soft_placement=options.allow_soft_placement,
                            log_device_placement=options.log_device_placement)
    config.gpu_options.allow_growth = options.allow_growth
    return tf.Session(config=config)


def main(options):
    num_workers = multiprocessing.cpu_count()
    global_scope = 'global'

    opt = tf.train.AdamOptimizer(learning_rate=1e-2)
    coord = tf.train.Coordinator()

    # global network on cpu
    with tf.device("/cpu:0"):
        master_network = Worker(scope=global_scope,
                                opt=opt,
                                env=init_env(),
                                coord=coord,
                                global_vars=None,
                                options=options)

    # local network on gpu
    workers = []
    for i in range(num_workers):
        with tf.device("/gpu:%d" % (i % options.num_gpu)):
            workers.append(Worker(scope='worker_%d' % i,
                                  opt=opt,
                                  env=init_env(),
                                  coord=coord,
                                  global_vars=master_network.local_vars,
                                  options=options))

    # run threads
    with get_session(options) as sess:

        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)

        coord.join(worker_threads)


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option('--mode', dest='mode', type='int', default=0)

    parser.add_option('--logname', dest='logname', type='string', default='')
    parser.add_option('--file_labels_fn', dest='file_labels_fn', type='string', default='../hyper_label_index')
    parser.add_option('max_episode', dest='max_episode', type='int', default=500)
    parser.add_option('batch_size', dest='batch_size', type='int', default=60)

    parser.add_option('num_gpu', dest='num_gpu', type='int', default=2)
    parser.add_option('--soft_placement', dest='allow_soft_placement', action='store_true')
    parser.add_option('--no_soft_placement', dest='allow_soft_placement', action='store_false')
    parser.set_defaults(allow_soft_placement=True)
    parser.add_option('--log_device_placement', dest='log_device_placement', action='store_true')
    parser.add_option('--no_log_device_placement', dest='log_device_placement', action='store_false')
    parser.set_defaults(log_device_placement=False)
    parser.add_option('--allow_growth', dest='allow_growth', action='store_true')
    parser.add_option('--no_allow_growth', dest='allow_growth', action='store_false')
    parser.set_defaults(allow_growth=True)

    FLAGS, _ = parser.parse_args()

    main(FLAGS)
