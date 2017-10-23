from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
from os.path import join as joinpath
import tensorflow as tf


# Copies one set of variables to another.
def update_target_graph(f, t):
    """
    
    :param f:
        can be string (scope) or list of collections
    :param t: 
        can be string (scope) or list of collections
    :return: 
        op that copy values from f to t
    """

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, f) if type(f) is str else f
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, t) if type(t) is str else t

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def linesep(q, sep='='):
    tmp = 80 - len(q)
    slen = tmp / 2 + (tmp % 2 == 0)

    print('\n', sep * slen, q, sep * slen, '\n')
    sys.stdout.flush()


def makedir(_path, remove_old=False):
    if os.path.isdir(_path):
        if not remove_old:
            raise Exception('old folder exists at %s please use remove_old flag to remove' % _path)
        shutil.rmtree(_path)

    os.mkdir(_path)


def get_output_folder(parent_dir, run_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    run_name: str
      string description for the experiment which is used as name of this sub-folder

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """

    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)

    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if (not os.path.isdir(joinpath(parent_dir, folder_name))) or (run_name not in folder_name):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = joinpath(parent_dir, run_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def rmfile(_path):
    if os.path.isfile(_path):
        os.remove(_path)
    elif os.path.isdir(_path):
        raise ValueError('remove target at %s is a dir' % _path)
    else:
        raise ValueError('remove target at %s not exists' % _path)
