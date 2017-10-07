from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import features
import os, sys, shutil
from os.path import join as joinpath


# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


def preprocess_minimap(minimap):
    layers = []
    assert minimap.shape[0] == len(features.MINIMAP_FEATURES)

    for i in range(len(features.MINIMAP_FEATURES)):
        if i == _MINIMAP_PLAYER_ID:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)

        elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)

        else:
            layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)

            for j in range(features.MINIMAP_FEATURES[i].scale):
                indy, indx = (minimap[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)

    return np.concatenate(layers, axis=0)


def preprocess_screen(screen):
    layers = []
    assert screen.shape[0] == len(features.SCREEN_FEATURES)

    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)

        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)

        else:
            layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)

            for j in range(features.SCREEN_FEATURES[i].scale):
                indy, indx = (screen[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)

    return np.concatenate(layers, axis=0)


def minimap_channel():
    c = 0
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == _MINIMAP_PLAYER_ID:
            c += 1
        elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            c += 1
        else:
            c += features.MINIMAP_FEATURES[i].scale
    return c


def screen_channel():
    c = 0
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
            c += 1
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            c += 1
        else:
            c += features.SCREEN_FEATURES[i].scale
    return c


def linesep(q, sep='='):
    tmp = 80 - len(q)
    slen = tmp / 2 + (tmp % 2 == 0)

    print('\n', sep*slen, q, sep*slen, '\n')
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