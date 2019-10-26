
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

""" 
------------------------------------------------------------------------------
Embedding.py Embedding Model's training and testing
------------------------------------------------------------------------------
"""

import sys
sys.path.append('..')

import os
import gc
import copy
import numpy as np
import dask.array as da
import tensorflow as tf

from tqdm import tqdm
import utils.file_utils as file_utils
import utils.data_utils as data_utils

from utils.models_names import get_model_name
from utils.configuration import config
from utils.logger import Logger
from utils.early_stopping import EarlyStopping

from graphs.AE_Factory import Factory
from bases.base_model import BaseModel

from openTSNE.affinity import PerplexityBasedNN

class Embedding(BaseModel):
    '''
    ------------------------------------------------------------------------------
                                         SET ARGUMENTS
    -------------------------------------------------------------------------------
    '''
    def __init__(self, **kwrds):
        self.config = copy.deepcopy(config())
        for key in kwrds.keys():
            assert key in self.config.keys(), '{} is not a keyword, \n acceptable keywords: {}'. \
                format(key, self.config.keys())

            self.config[key] = kwrds[key]

        self.experiments_root_dir = 'experiments'
        file_utils.create_dirs([self.experiments_root_dir])
        self.config.model_name = get_model_name(self.config.graph_type, self.config)
        self.config.checkpoint_dir = os.path.join(self.experiments_root_dir + '/' + self.config.checkpoint_dir + '/',
                                                  self.config.model_name)
        self.config.config_dir = os.path.join(self.experiments_root_dir + '/' + self.config.config_dir + '/',
                                              self.config.model_name)
        self.config.log_dir = os.path.join(self.experiments_root_dir + '/' + self.config.log_dir + '/',
                                           self.config.model_name)

        log_dir_subfolders = ['reconst', 'latent2d', 'latent3d', 'samples', 'total_random', 'pretoss_random', 'interpolate']
        config_dir_subfolders = ['extra']

        file_utils.create_dirs([self.config.checkpoint_dir, self.config.config_dir, self.config.log_dir])
        file_utils.create_dirs([self.config.log_dir + '/' + dir_ + '/' for dir_ in log_dir_subfolders])
        file_utils.create_dirs([self.config.config_dir + '/' + dir_ + '/' for dir_ in config_dir_subfolders])

        load_config = {}
        try:
            load_config = file_utils.load_args(self.config.model_name, self.config.config_dir)
            self.config.update(load_config)
            self.config.update({key: config[key] for key in ['kinit', 'bias_init', 'act_out', 'transfer_fct', 'push_colab']})
            print('Loading previous configuration ...')
        except:
            print('Unable to load previous configuration ...')

        file_utils.save_args(self.config.dict(), self.config.model_name, self.config.config_dir, ['latent_mean', 'latent_std', 'push_colab')

        if hasattr(self.config, 'height'):
            try:
                self.config.restore = True
                self.build_model(self.config.height, self.config.width, self.config.num_channels)
            except:
                self.config.isBuilt = False
        else:
            self.config.isBuilt = False

    '''
    ------------------------------------------------------------------------------
                                         EPOCH FUNCTIONS
    -------------------------------------------------------------------------------
    '''
    def _train(self, data_train, session, logger):
        losses = list()
        for _ in tqdm(range(data_train.num_batches(self.config.batch_size))):
            batch_x  = next(data_train.next_batch(self.config.batch_size))
            affinities_train = PerplexityBasedNN(batch_x.reshape([-1, self.model_graph.x_flat_dim]),
                                                 perplexity=self.config.perplexity, metric="euclidean",
                                                 n_jobs=-2, random_state=42)

            batch_p = da.from_array(affinities_train.P.toarray() * self.config.exaggeration, chunks=100)
            loss_curr = self.model_graph.train_epoch(session, batch_x, batch_p)

            losses.append(loss_curr)
            cur_it = self.model_graph.global_step_tensor.eval(session)
            summaries_dict = dict(zip(self.model_graph.losses, np.mean(np.vstack(losses), axis=0)))

            logger.summarize(cur_it, summarizer='iter_train', log_dict=summaries_dict)
        losses = np.mean(np.vstack(losses), axis=0)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = dict(zip(self.model_graph.losses, losses))

        logger.summarize(cur_it, summarizer='epoch_train', log_dict=summaries_dict)
        return losses


    def fit(self, X, y=None):
        print('\nProcessing data...')

        self.data_train = data_utils.process_data(X, y, test_size=0)
        if self.config.plot:
            self.data_plot = self.data_train

        self.config.num_batches = self.data_train.num_batches(self.config.batch_size)

        if not self.config.isBuilt:
            self.config.restore=True
            self.build_model(self.data_train.height, self.data_train.width, self.data_train.num_channels)
        else:
            assert (self.config.height == self.data_train.height) and (self.config.width == self.data_train.width) and \
                   (self.config.num_channels == self.data_train.num_channels), \
                    'Wrong dimension of data. Expected shape {}, and got {}'.format((self.config.height,self.config.width, \
                                                                                     self.config.num_channels), \
                                                                                    (self.data_train.height,
                                                                                     self.data_train.width, \
                                                                                     self.data_train.num_channels) \
                                                                                    )

        ''' 
         -------------------------------------------------------------------------------
                                        TRAIN THE MODEL
        ------------------------------------------------------------------------------------- 
        '''
        print('\nTraining a model...')
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(self.config.seeds)
            self.session = session
            logger = Logger(self.session, self.config.log_dir)
            saver = tf.train.Saver()

            early_stopper = EarlyStopping(name='total loss', decay_fn=self.decay_fn)

            if(self.config.restore and self.load(self.session, saver) ):
                load_config = file_utils.load_args(self.config.model_name, self.config.config_dir)
                self.config.update(load_config)

                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(self.session), self.config.epochs+1, 1):
                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch

                losses_tr = self._train(self.data_train, self.session, logger)

                if np.isnan(losses_tr[0]):
                    print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    for lname, lval in zip(self.model_graph.losses, losses_tr):
                        print(lname, lval)
                    sys.exit()

                train_msg = 'TRAIN: \n'
                for lname, lval in zip(self.model_graph.losses, losses_tr):
                    train_msg += str(lname) + ': ' + str(lval) + ' | '

                print(train_msg)
                print()

                if (cur_epoch == 1) or ((cur_epoch % self.config.save_epoch == 0) and (cur_epoch != 0)):
                    gc.collect()
                    self.save(self.session, saver, self.model_graph.global_step_tensor.eval(self.session))
                    if self.config.plot:
                        self.plot_latent(cur_epoch)

                self.session.run(self.model_graph.increment_cur_epoch_tensor)

                # Early stopping
                if (self.config.early_stopping and early_stopper.stop(losses_tr[0])):
                    print('Early Stopping!')
                    break


                if cur_epoch % self.config.colab_save == 0:
                    if self.config.colab:
                        self.push_colab()

            self.save(self.session, saver, self.model_graph.global_step_tensor.eval(self.session))
            if self.config.plot:
                self.plot_latent(cur_epoch)

            if self.config.colab:
                self.push_colab()

        return

    '''  
    ------------------------------------------------------------------------------
                                     SET NETWORK PARAMS
     ------------------------------------------------------------------------------
    '''
    def build_model(self, height, width, num_channels):
        self.config.height = height
        self.config.width = width
        self.config.num_channels = num_channels

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model_graph = Factory(self.config)
            print(self.model_graph)

            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print('\nNumber of trainable paramters', self.trainable_count)
            self.test_graph()

        ''' 
         -------------------------------------------------------------------------------
                        GOOGLE COLAB 
        ------------------------------------------------------------------------------------- 
        '''
        if self.config.colab:
            self.push_colab()
            self.config.push_colab = self.push_colab

        self.config.isBuilt=True
        file_utils.save_args(self.config.dict(), self.config.model_name, self.config.config_dir, ['latent_mean', 'latent_std', 'push_colab'])

    def test_graph(self):
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(self.config.seeds)
            self.session = session
            logger = Logger(self.session, self.config.log_dir)
            saver = tf.train.Saver()

            if (self.config.restore and self.load(self.session, saver)):
                load_config = file_utils.load_args(self.config.model_name, self.config.config_dir)
                self.config.update(load_config)
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

