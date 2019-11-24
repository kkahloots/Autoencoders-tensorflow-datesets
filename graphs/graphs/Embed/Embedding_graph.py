"""
Embedding_graph.py:
Tensorflow Graph for the Deep Embedding Autoencoder
"""
__author__ = "Khalid M. Kahloot"
__copyright__ = "Copyright 2019, Only for professionals"
__paper__ = "https://xifengguo.github.io/papers/IJCAI17-IDEC.pdf"

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

import bases.losses as losses
import bases.kernels as kernels
from bases.base_graph import BaseGraph
from keras_radam.training import RAdamOptimizer


'''
This is the Main EmbeddingGraph.
'''


class EmbeddingGraph(BaseGraph):

    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.config.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.config.batch_size, self.config.width, self.config.height, self.config.num_channels],
                                          name='x_batch')
            self.x_batch_flat = tf.reshape(self.x_batch, [-1, self.x_flat_dim])
            self.joint_probabilities_batch = tf.placeholder(tf.float32, [self.config.batch_size, self.config.batch_size], name='joint_probabilities_batch')

            self.lr = tf.placeholder_with_default(self.config.learning_rate, shape=None, name='lr')


    ''' 
    ------------------------------------------------------------------------------
                                     GRAPH FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''

    def create_graph(self):
        print('\n[*] Defining encoder...')
        with tf.variable_scope('encoder', reuse=self.config.reuse):
            Qlatent_x = self.create_encoder(input_=self.x_batch if self.config.isConv else self.x_batch_flat,
                                            hidden_dim=self.config.hidden_dim,
                                            output_dim=self.config.n_components,
                                            num_layers=self.config.num_layers,
                                            transfer_fct=self.config.transfer_fct,
                                            act_out=None,
                                            reuse=self.config.reuse,
                                            kinit=self.config.kinit,
                                            bias_init=self.config.bias_init,
                                            drop_rate=self.config.dropout,
                                            prefix='en_',
                                            isConv=self.config.isConv)

            self.latent = Qlatent_x.output

    '''  
    ------------------------------------------------------------------------------
                                     LOSSES
    ------------------------------------------------------------------------------ 
    '''
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.variable_scope('e_divergence_cost', reuse=self.config.reuse):
            kerneled_embedding = kernels.get_kernel(X=self.latent, batch_size=self.config.batch_size, param=self.config.df,
                                                    epsilon=self.config.epsilon, kernel=self.config.kernel_mode)
            self.e_divergence_cost = losses.get_distributions_div_cost(self.joint_probabilities_batch, kerneled_embedding, self.config.e_div_cost)
        self.e_div_cost_m = tf.reduce_mean(self.e_divergence_cost)

        with tf.variable_scope("L2_loss", reuse=self.config.reuse):
            tv = tf.trainable_variables()
            self.L2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

        with tf.variable_scope('embedding_loss', reuse=self.config.reuse):
            self.embedding_loss = tf.add(tf.reduce_mean(self.e_divergence_cost), self.config.l2 * self.L2_loss, name='embedding_loss')

        with tf.variable_scope("optimizer", reuse=self.config.reuse):
            self.optimizer = RAdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.embedding_loss, global_step=self.global_step_tensor)

        self.losses = ['Embedding_loss', 'E_diverg_{}'.format(self.config.e_div_cost), 'Regul_L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x, p):
        tensors = [self.train_step, self.embedding_loss, self.e_divergence_cost, self.L2_loss]
        feed_dict = {self.x_batch: x, self.joint_probabilities_batch: p}
        _, loss, e_div, L2_loss = session.run(tensors, feed_dict=feed_dict)
        return loss, e_div, L2_loss

    def test_epoch(self, session, x, p):
        tensors = [self.embedding_loss, self.e_divergence_cost, self.L2_loss]
        feed_dict = {self.x_batch: x, self.joint_probabilities_batch: p}
        loss, e_div, L2_loss = session.run(tensors, feed_dict=feed_dict)
        return loss, e_div, L2_loss
