"""
AnnVAE_graph.py:
Tensorflow Graph for the Annealed Variational Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://arxiv.org/abs/1804.03599"

import tensorflow as tf
import bases.losses as losses
from bases.base_graph import BaseGraph
from keras_radam.training import RAdamOptimizer


'''
This is the Main Annealed VAEGraph.
'''
class AnnVAEGraph(BaseGraph):
    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()
    
    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.config.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.config.batch_size, self.config.width, self.config.height, self.config.num_channels], name='x_batch')
            self.x_batch_flat = tf.reshape(self.x_batch , [-1,self.x_flat_dim])
            
            self.latent_batch = tf.placeholder(tf.float32, [self.config.batch_size, self.config.latent_dim], name='px_batch')
            self.lr = tf.placeholder_with_default(self.config.learning_rate, shape=None, name='lr')

            self.sample_batch = tf.random_normal((self.config.batch_size, self.config.latent_dim), -1, 1, dtype=tf.float32)

    ''' 
    ------------------------------------------------------------------------------
                                     GRAPH FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''
    def create_graph(self):
        print('\n[*] Defining encoders...')
        with tf.variable_scope('encoder_mean', reuse=self.config.reuse):
            Qlatent_x_mean = self.create_encoder(input_=self.x_batch if self.config.isConv else self.x_batch_flat,
                            hidden_dim=self.config.hidden_dim,
                            output_dim=self.config.latent_dim,
                            num_layers=self.config.num_layers,
                            transfer_fct=self.config.transfer_fct,
                            act_out=None, 
                            reuse=self.config.reuse,
                            kinit=self.config.kinit,
                            bias_init=self.config.bias_init,
                            drop_rate=self.config.dropout,
                            prefix='enmean_',
                            isConv=self.config.isConv)
        
            self.encoder_mean = Qlatent_x_mean.output

            with tf.variable_scope('encoder_var', reuse=self.config.reuse):
                Qlatent_x_var = self.create_encoder(input_=self.x_batch if self.config.isConv else self.x_batch_flat,
                                                 hidden_dim=self.config.hidden_dim,
                                                 output_dim=self.config.latent_dim,
                                                 num_layers=self.config.num_layers,
                                                 transfer_fct=self.config.transfer_fct,
                                                 act_out=tf.nn.softplus,
                                                 reuse=self.reuse,
                                                 kinit=self.config.kinit,
                                                 bias_init=self.config.bias_init,
                                                 drop_rate=self.config.dropout,
                                                 prefix='envar_',
                                                 isConv=self.config.isConv)

            self.encoder_var = Qlatent_x_var.output

        print('\n[*] Reparameterization trick...')
        self.encoder_logvar = tf.log(self.encoder_var + self.config.epsilon)
        eps = tf.random_normal((self.config.batch_size, self.config.latent_dim), 0, 1, dtype=tf.float32)
        self.latent = tf.add(self.encoder_mean, tf.multiply(tf.sqrt(self.encoder_var), eps))

        self.latent_batch = self.latent
            
        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder_mean', reuse=self.config.reuse):
            Px_latent_mean = self.create_decoder(input_=self.latent_batch,
                                            hidden_dim=self.config.hidden_dim,
                                            output_dim=self.x_flat_dim,
                                            num_layers=self.config.num_layers,
                                            transfer_fct=self.config.transfer_fct,
                                            act_out=tf.nn.sigmoid,
                                            reuse=self.config.reuse,
                                            kinit=self.config.kinit,
                                            bias_init=self.config.bias_init,
                                            drop_rate=self.config.dropout,
                                            prefix='de_',
                                            isConv=self.config.isConv)
        
            self.x_recons_flat = Px_latent_mean.output
        self.x_recons = tf.reshape(self.x_recons_flat , [-1,self.config.width, self.config.height, self.config.num_channels])

    '''  
    ------------------------------------------------------------------------------
                                     LOSSES
    ------------------------------------------------------------------------------ 
    '''
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.name_scope('reconstruct'):
            self.reconstruction = losses.get_reconst_loss(self.x_batch_flat, self.x_recons_flat, self.config.reconst_loss)
        self.loss_reconstruction_m = tf.reduce_mean(self.reconstruction)

        with tf.variable_scope('L2_loss', reuse=self.config.reuse):
            tv = tf.trainable_variables()
            self.L2_loss = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        with tf.variable_scope('encoder_loss', reuse=self.config.reuse):
            self.ae_loss = tf.add(tf.reduce_mean(self.reconstruction), self.config.l2*self.L2_loss, name='encoder_loss')

        with tf.variable_scope('divergence_cost', reuse=self.config.reuse):
            self.divergence_cost = losses.get_self_divergence(self.encoder_mean, self.encoder_logvar, self.config.div_cost)
        self.div_cost_m = tf.reduce_mean(self.divergence_cost)

        with tf.variable_scope('vae_loss', reuse=self.config.reuse):
            self.vae_loss = tf.add(self.ae_loss, self.div_cost_m)

        with tf.variable_scope('annvae_loss', reuse=self.config.reuse):
            c = self.anneal(self.config.c_max, self.global_step_tensor, self.config.itr_thd)
            self.anneal_reg = self.config.ann_gamma * tf.math.abs(self.div_cost_m - c)
            self.annvae_loss = tf.add(self.ae_loss, self.anneal_reg)

        with tf.variable_scope("optimizer" ,reuse=self.config.reuse):
            self.optimizer = RAdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.annvae_loss, global_step=self.global_step_tensor)

        self.losses = ['ELBO_AnnVAE', 'VAE', 'AE', 'Recons_{}'.format(self.config.reconst_loss),
                       'Regul_anneal_reg', 'Div_{}'.format(self.config.div_cost),
                       'Regul_L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step,  self.annvae_loss, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.anneal_reg, self.div_cost_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, loss, vaeloss, aeloss, recons, anneal_reg, div_cost, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, vaeloss, aeloss, recons, anneal_reg, div_cost, L2_loss
    
    def test_epoch(self, session, x):
        tensors = [self.annvae_loss, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.anneal_reg, self.div_cost_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        loss, vaeloss, aeloss, recons, anneal_reg, div_cost, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, vaeloss, aeloss, recons, anneal_reg, div_cost, L2_loss


    def anneal(self, c_max, step, iteration_threshold):
      """Anneal function for anneal_vae (https://arxiv.org/abs/1804.03599).
      Args:
        c_max: Maximum capacity.
        step: Current step.
        iteration_threshold: How many iterations to reach c_max.
      Returns:
        Capacity annealed linearly until c_max.
      """
      return tf.math.minimum(c_max * 1.,
                             c_max * 1. * tf.to_float(step) / iteration_threshold)