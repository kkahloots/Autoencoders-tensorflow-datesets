
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

""" 
------------------------------------------------------------------------------
TrainerAE.py autoencoder Model's training and testing
------------------------------------------------------------------------------
"""

import os
import sys
import gc
from os import listdir
from os.path import isfile, join

sys.path.append('..')
import time
import copy
import itertools
import numpy as np
import dask.array as da
from dask.delayed import delayed
import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm

import utils.file_utils as file_utils

from utils.models_names import get_model_name
from utils.configuration import config
from utils.logger import Logger
from utils.early_stopping import EarlyStopping

from graphs.AE_Factory import Factory
from bases.base_model import BaseModel
from dask_ml.preprocessing import MinMaxScaler

class TrainerAE(BaseModel):
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

        if self.config.colab:
            self.google2colab()
            time.sleep(10)

        self.latent_data = None
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
        #try:
        load_config = file_utils.load_args(self.config.model_name, self.config.config_dir, ['latent_mean', 'latent_std', 'samples'])
        #   print('Loading previous configuration ...')
        #except:
        #    print('Unable to load previous configuration ...')

        self.config.update(load_config)

        file_utils.save_args(self.config.dict(), self.config.model_name, self.config.config_dir, ['latent_mean', 'latent_std', 'samples'])

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
    def _train(self, data_train, session, logger, num_batches):
        losses = list()
        iterator = data_train.make_one_shot_iterator()
        for t in tqdm(range(num_batches)):
            batch = session.run(iterator.get_next())

            loss_curr = self.model_graph.train_epoch(session, da.from_array(batch['image']/255, chunks=100))

            losses.append(loss_curr)
            cur_it = self.model_graph.global_step_tensor.eval(session)
            summaries_dict = dict(zip(self.model_graph.losses, np.mean(np.vstack(losses), axis=0)))

            logger.summarize(cur_it, summarizer='iter_train', log_dict=summaries_dict)

        losses = np.mean(np.vstack(losses), axis=0)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = dict(zip(self.model_graph.losses, losses))

        logger.summarize(cur_it, summarizer='epoch_train', log_dict=summaries_dict)
        return losses

    def _test(self, data_test, session, logger, num_batches):
        losses = list()
        iterator = data_test.make_one_shot_iterator()
        for t in tqdm(range(num_batches)):
            batch = session.run(iterator.get_next())
            loss_curr = self.model_graph.test_epoch(session, da.from_array(batch['image']/255, chunks=100))

            losses.append(loss_curr)
            cur_it = self.model_graph.global_step_tensor.eval(session)
            summaries_dict = dict(zip(self.model_graph.losses, np.mean(np.vstack(losses), axis=0)))

            logger.summarize(cur_it, summarizer='iter_test', log_dict=summaries_dict)

        losses = np.mean(np.vstack(losses), axis=0)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = dict(zip(self.model_graph.losses, losses))

        logger.summarize(cur_it, summarizer='epoch_test', log_dict=summaries_dict)
        return losses

    '''
    ------------------------------------------------------------------------------
                                         EPOCH FUNCTIONS
    -------------------------------------------------------------------------------
    '''
    def fit(self, dataset):
        assert str(dataset.__class__).split('.')[0].replace("<class '", '') + '.' + str(dataset.__class__).split('.')[1] \
               == "tensorflow_datasets.image", 'The dataset type is not image tensorflow_datasets'

        self.data_train = dataset.as_dataset(split=tfds.Split.TRAIN, shuffle_files=True, batch_size=self.config.batch_size)
        self.data_test = dataset.as_dataset(split=tfds.Split.TEST, shuffle_files=True, batch_size=self.config.batch_size)

        width = dataset.info.features['image'].shape[0]
        height = dataset.info.features['image'].shape[1]
        num_channels  = dataset.info.features['image'].shape[2]

        self.config.ntrain_batches = dataset.info.splits['train'].num_examples // self.config.batch_size
        self.config.ntest_batches = dataset.info.splits['test'].num_examples // self.config.batch_size

        if not self.config.isBuilt:
            self.config.restore=True
            self.build_model(height, width, num_channels)
            self.build_model(height, width, num_channels)
        else:
            assert (self.config.height == height) and (self.config.width == width) and \
                   (num_channels == num_channels), \
                'Wrong dimension of data. Expected shape {}, and got {}'.\
                format((self.config.height, self.config.width, self.config.num_channels),  (height, width, num_channels))

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
            self.saver = tf.train.Saver()
            early_stopper = EarlyStopping(name='total loss', decay_fn=self.decay_fn)

            if(self.config.restore and self.load(self.session, self.saver) ):
                load_config = file_utils.load_args(self.config.model_name, self.config.config_dir,
                                                   ['latent_mean', 'latent_std', 'samples'])
                self.config.update(load_config)

                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            if self.config.plot:
                if self.config.samples is None:
                    print('\nFinding the unique categories...')
                    y_uniqs = list()
                    iterator = self.data_train.make_one_shot_iterator()
                    for t in tqdm(range(self.config.ntrain_batches)):
                        batch = session.run(iterator.get_next())
                        y, _ = tf.unique(batch[self.config.y_index])
                        y_uniqs += y.eval().tolist()

                    self.y_uniqs = np.unique(y_uniqs)
                    y_uniqs = self.y_uniqs[:10]
                    y_uniqs = np.array(list(itertools.repeat(y_uniqs, 10))).flatten()[:10]

                    print('\nSampling from the unique categories...')
                    samples = dict(zip(y_uniqs, itertools.repeat(list(), len(y_uniqs))))
                    iterator = self.data_train.make_one_shot_iterator()
                    for t in tqdm(range(self.config.ntrain_batches)):
                        batch = session.run(iterator.get_next())
                        for yi in y_uniqs:
                            if len(samples[yi]) <= 10:
                                samples[yi] = samples[yi] + da.from_array(
                                    tf.boolean_mask(mask=batch[self.config.y_index]==yi, tensor=batch['image']).eval(),
                                                    chunks=10).compute().tolist()
                            samples[yi] = samples[yi][:10]
                    self.config.samples = da.from_array(da.vstack(samples.values()), chunks=10).compute()

            if not self.config.isTrained:
                for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(self.session), self.config.epochs+1, 1):
                    print('EPOCH: ', cur_epoch)
                    self.current_epoch = cur_epoch

                    losses_tr = self._train(self.data_train, self.session, logger, self.config.ntrain_batches)
                    if np.isnan(losses_tr[0]):
                        print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                        for lname, lval in zip(self.model_graph.losses, losses_tr):
                            print(lname, lval)
                        sys.exit()

                    losses_test = self._test(self.data_test, self.session, logger, self.config.ntest_batches)
                    train_msg = 'TRAIN: \n'
                    for lname, lval in zip(self.model_graph.losses, losses_tr):
                        train_msg += str(lname) + ': ' + str(lval) + ' | '

                    eval_msg = 'TEST: \n'
                    for lname, lval in zip(self.model_graph.losses, losses_test):
                        eval_msg += str(lname) + ': ' + str(lval) + ' | '

                    print(train_msg)
                    print(eval_msg)
                    print()

                    if (cur_epoch == 1) or ((cur_epoch % self.config.save_epoch == 0) and (cur_epoch != 0)):
                        self.save_model()
                        if self.config.plot:
                            self.plot_latent(cur_epoch)
                            self.plot_reconst(cur_epoch)

                    self.session.run(self.model_graph.increment_cur_epoch_tensor)

                    # Early stopping
                    if (self.config.early_stopping and early_stopper.stop(losses_test[0])):
                        print('Early Stopping!')
                        break

                    if cur_epoch % self.config.colab_save == 0:
                        if self.config.colab:
                            self.push_colab()

                self.config.isTrained = True
                self.save_model()

                if self.config.plot:
                    self.plot_latent(cur_epoch)
                    self.plot_reconst(cur_epoch)

            if self.config.colab:
                self.push_colab()

    def save_model(self):
        self.save(self.session, self.saver, self.model_graph.global_step_tensor.eval(self.session))
        self.compute_distribution(self.data_train, self.session, self.config.ntrain_batches)
        file_utils.save_args(self.config.dict(), self.config.model_name, self.config.config_dir,
                             ['latent_mean', 'latent_std', 'samples'])
        gc.collect()

    def compute_distribution(self, images, session, num_batches):
        self.generate_latent(images, session, num_batches)
        print("Computing the latent's distribution ... ")
        self.model_graph.config.latent_mean = self.latent_data['latent'].mean(axis=0).compute()
        self.model_graph.config.latent_std = self.latent_data['latent'].std(axis=0).compute()

    def generate_latent(self, images, session, num_batches):
        print("Generating latent space ... ")
        latents = list()
        labels = list()
        iterator = images.make_one_shot_iterator()
        for t in tqdm(range(num_batches)):
            batch = session.run(iterator.get_next())
            latents_batch = self.model_graph.encode(session, da.from_array(batch['image']/255, chunks=100))
            y_index = list(batch.keys()).index(self.config.y_index)-1
            label_batch = da.from_array(np.array([batch[k] for k in batch.keys() if k !='image']), chunks=100)

            latents.append(latents_batch)
            labels.append(label_batch)
        latents = da.from_array(delayed(np.vstack(latents)).compute(), chunks=100)
        labels = da.from_array(delayed(np.vstack(labels)).compute(), chunks=100)

        self.latent_data = {'latent': latents.reshape((-1, self.config.latent_dim)),
                            'label': labels.reshape((-1, 1)),
                            'y_index': y_index}

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
        file_utils.save_args(self.config.dict(), self.config.model_name, self.config.config_dir, ['latent_mean', 'latent_std', 'samples'])

    def test_graph(self):
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(self.config.seeds)
            self.session = session
            self.saver = tf.train.Saver()

            if (self.config.restore and self.load(self.session, self.saver)):
                load_config = file_utils.load_args(self.config.model_name, self.config.config_dir,
                                                   ['latent_mean', 'latent_std', 'samples'])
                self.config.update(load_config)

                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            print('random latent batch ...')
            samples = self.model_graph._sampling_reconst(session, std_scales=np.ones(self.config.latent_dim))[0]
            print('random latent shape {}'.format(samples.shape))

    def _sampling_reconst(self, std_scales, random_latent=None):
        def aux_fun(session, rand_samp):
            return self.model_graph._sampling_reconst(session=session, std_scales=std_scales, random_latent=rand_samp)

        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(self.config.seeds)
            self.session = session
            self.saver = tf.train.Saver()

            if (self.config.restore and self.load(self.session, self.saver)):
                load_config = file_utils.load_args(self.config.model_name, self.config.config_dir,
                                                   ['latent_mean', 'latent_std', 'samples'])
                self.config.update(load_config)

                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            samples = list()
            if random_latent is None:
                while True:
                    samples.append(self.model_graph._sampling_reconst(session=session, std_scales=std_scales)[0])
                    if len(samples) >= (100//self.config.batch_size)+1:
                        samples = da.vstack(samples)
                        samples = samples[:100]
                        break

            else:
                samples = self.batch_function(aux_fun, random_latent)

        scaler = MinMaxScaler()
        return scaler.fit_transform(samples.flatten().reshape(-1, 1).astype(np.float32)).reshape(samples.shape)
