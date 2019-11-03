import tensorflow as tf
import numpy as np
from types import FunctionType
from utils.codes import Models, Losses, Kernels, properties

z_dim = 32

class default_config:
    model_name = Models.AE
    graph_type = Models.AE
    dataset_name = ''
    #
    latent_dim = z_dim
    latent_mean = np.zeros(z_dim)
    latent_std = np.ones(z_dim)
    samples = None

    num_layers = 3
    hidden_dim = 500
    l2 = 1e-6
    batch_size = 1000
    batch_norm = True
    learning_rate = 1e-3
    dropout = 0.25
    isConv = False
    epochs = int(10e10)
    restore = False
    plot = False
    trace = False
    y_index = 'label'
    colab = False
    colabpath = ''
    early_stopping = True
    log_dir = 'log_dir'
    checkpoint_dir = 'checkpoint_dir'
    config_dir = 'config_dir'
    act_out = tf.nn.softplus
    transfer_fct = tf.nn.relu
    kinit = tf.contrib.layers.xavier_initializer()
    bias_init = tf.constant_initializer(0.0)
    reuse = False
    isBuilt=False
    isTrained=False

# # Stopping tolerance
    tolerance = 1e-6
    min_lr = 1e-8
    epsilon = 1e-8
    save_epoch = 5
    colab_save = 5
    seeds = 987654321

#
    reconst_loss = Losses.MLE
    div_cost = Losses.KLD
#
    prior_reconst_loss = Losses.MLE
    prior_div_cost = Losses.KLD
#
# ####### Beta VAE
    beta = 10.0
#
# ####### AnnVAE
    ann_gamma = 100
    c_max = 25
    itr_thd = 1000
#
# ####### DIPIVAE
    lambda_d = 5
    d_factor = 5
#
# ####### BayesianVAE
# #Monte Carlo sampling
    ntrain_batches = 1000
    ntest_batches = 1000
    MC_samples = 10
#
# # Embedding configuration
    n_components = 2
    n_cluster_dim = 10
    perplexity = 30
    exaggeration = 15
    df = 1
    e_div_cost = Losses.KLD
    kernel_mode = Kernels.StudentT

class config:
    def __init__(self):
        keys = list()
        items = list()
        ddict = dict(default_config.__dict__)
        for key, item in ddict.items():
            if key in properties(default_config):
                keys.append(key)
                items.append(item)
        ddict =  dict(zip(keys, items))
        for k, v in ddict.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def update(self, newvals):
        self.__dict__.update(newvals)

    def keys(self):
        keys = list()
        for key, item in self.__dict__.items():
            if type(item) != FunctionType:
                keys.append(key)
        return keys

    def dict(self):
        keys = list()
        items = list()
        for key, item in self.__dict__.items():
            if key not in ['kinit', 'bias_init', 'act_out', 'transfer_fct']:
                if type(item) != FunctionType:
                    keys.append(key)
                    items.append(item)
        return dict(zip(keys, items))


