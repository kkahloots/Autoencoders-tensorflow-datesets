import os
import sys
import json
import numpy as np

def _save_args(args, exp_name, log_dir):
    print('Saving Model Arguments ...')
    my_file = log_dir + '/' + exp_name + '.json'
    with open(my_file, 'w') as fp:
        json.dump(args, fp)


def _load_args(exp_name, log_dir):
    my_file = log_dir + '/' + exp_name + '.json'
    with open(my_file, 'r') as fp:
        args = json.load(fp)
    return args


def save_args(args, exp_name, log_dir, exclude=[]):
    for ex in exclude:
        my_file = log_dir + '/' + 'extra' + '/'+ ex +'.npy'

        np.save(my_file, args[ex])
        del args[ex]
    _save_args(args, exp_name, log_dir)

def load_args(exp_name, log_dir, include=[]):
    args = _load_args(exp_name, log_dir)
    for inc in include:
        try:
            my_file = log_dir + '/' + 'extra' + '/'+ inc +'.npy'
            args[inc] = np.load(my_file)
        except:
            pass

    return args


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        sys.exit(-1)