"""
AnnBayVAE_graph.py:
Tensorflow Graph for the Annealed Bayesian Variational Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://arxiv.org/pdf/1606.05908.pdf"

import tensorflow as tf
import bases.losses as losses
from bases.base_graph import BaseGraph
from keras_radam.training import RAdamOptimizer


'''
This is the Main Annealed Bayesian VAEGraph.
'''
class AnnBayVAEGraph(BaseGraph):

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
        print('\n[*] Defining _sampling_reconst from X...')
        self.sample_flat = tf.multiply(tf.ones([self.config.MC_samples, self.config.batch_size, self.x_flat_dim]), self.x_batch_flat)
        sample_flat_shape = self.sample_flat.get_shape().as_list()
        if self.config.isConv:
            sample_input = tf.reshape(self.sample_flat , [-1, self.config.width, self.config.height, self.config.num_channels])
            print('\n[*] _sampling_reconst shape {}'.format(sample_input.shape))
        else:
            print('\n[*] _sampling_reconst shape {}'.format(sample_flat_shape))

        ####################################################################################
        print('\n[*] Defining prior ...')
        print('\n[*] Defining prior encoder...')
        with tf.variable_scope('prior_encoder', reuse=self.config.reuse):
            Qlatent_sample = self.create_encoder(input_=sample_input if self.config.isConv else self.sample_flat,
                                                 hidden_dim=self.config.hidden_dim,
                                                 output_dim=self.config.latent_dim,
                                                 num_layers=self.config.num_layers,
                                                 transfer_fct=self.config.transfer_fct,
                                                 act_out=None,
                                                 reuse=self.config.reuse,
                                                 kinit=self.config.kinit,
                                                 bias_init=self.config.bias_init,
                                                 drop_rate=self.config.dropout,
                                                 prefix='prior_en_',
                                                 isConv=self.config.isConv)

            self.prior_mean = Qlatent_sample.output
            self.prior_var = tf.nn.sigmoid(self.prior_mean)

            self.latent_sample = self.prior_mean

        print('\n[*] Defining prior decoder...')
        with tf.variable_scope('prior_decoder', reuse=self.config.reuse):
            Psample_latent = self.create_decoder(input_=self.config.latent_sample,
                                                hidden_dim=self.config.hidden_dim,
                                                output_dim=sample_flat_shape[-1],
                                                num_layers=self.config.num_layers,
                                                transfer_fct=self.config.transfer_fct,
                                                act_out=tf.nn.sigmoid,
                                                reuse=self.config.reuse,
                                                kinit=self.config.kinit,
                                                bias_init=self.config.bias_init,
                                                drop_rate=self.config.dropout,
                                                prefix='prior_de_',
                                                isConv=self.config.isConv)
            self.sample_recons_flat = Psample_latent.output
            if self.config.isConv:
                self.sample_recons_flat = tf.reshape(self.sample_recons_flat, sample_flat_shape)
            print('\n[*] _sampling_reconst tsne_cost shape {}'.format(self.sample_recons_flat.shape))

        ####################################################################################
        print('\n[*] Defining posterior ...')
        print('\n[*] Defining posterior encoders...')
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
                                                prefix='post_enmean_',
                                                isConv=self.config.isConv)

            self.encoder_mean = Qlatent_x_mean.output

        with tf.variable_scope('encoder_var', reuse=self.config.reuse):
            Qlatent_x_var = self.create_encoder(input_=self.x_batch if self.config.isConv else self.x_batch_flat,
                                                 hidden_dim=self.config.hidden_dim,
                                                 output_dim=self.config.latent_dim,
                                                 num_layers=self.config.num_layers,
                                                 transfer_fct=self.config.transfer_fct,
                                                 act_out=tf.nn.softplus,
                                                 reuse=self.config.reuse,
                                                 kinit=self.config.kinit,
                                                 bias_init=self.config.bias_init,
                                                 drop_rate=self.config.dropout,
                                                 prefix='post_envar_',
                                                 isConv=self.config.isConv)

            self.encoder_var = Qlatent_x_var.output

        print('\n[*] Reparameterization trick...')
        self.encoder_logvar = tf.log(self.encoder_var + self.config.epsilon)
        eps = tf.random_normal((self.config.batch_size, self.config.latent_dim), 0, 1, dtype=tf.float32)
        self.latent = tf.add(self.encoder_mean, tf.multiply(tf.sqrt(self.encoder_var), eps))

        self.latent_batch = self.latent

        print('\n[*] Defining posterior encoders...')
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
                                            prefix='post_de_',
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
        with tf.name_scope('prior_recons'):
            self.prior_recons = losses.get_reconst_loss(self.sample_flat, self.sample_recons_flat, self.config.prior_reconst_loss)
        self.prior_recons_m = tf.reduce_mean(self.prior_recons)

        with tf.name_scope('reconstruct'):
            self.reconstruction = losses.get_reconst_loss(self.x_batch_flat, self.x_recons_flat, self.config.reconst_loss)
        self.loss_reconstruction_m = tf.reduce_mean(self.reconstruction)

        with tf.variable_scope('L2_loss', reuse=self.config.reuse):
            tv = tf.trainable_variables()
            self.L2_loss = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv if 'post_' in v.name])
        
        with tf.variable_scope('encoder_loss', reuse=self.config.config.reuse):
            self.ae_loss = tf.add(tf.reduce_mean(self.reconstruction), self.config.l2*self.L2_loss, name='encoder_loss')

        with tf.variable_scope('divergence_cost', reuse=self.config.config.reuse):
            self.divergence_cost = losses.get_self_divergence(self.encoder_mean, self.encoder_logvar, self.config.div_cost)
        self.div_cost_m = tf.reduce_mean(self.divergence_cost)

        with tf.variable_scope('vae_loss', reuse=self.config.reuse):
            self.vae_loss = tf.add(self.ae_loss, self.div_cost_m)

        with tf.variable_scope('annvae_loss', reuse=self.config.reuse):
            c = self.anneal(self.config.c_max, self.global_step_tensor, self.config.itr_thd)
            self.anneal_reg = self.config.ann_gamma * tf.math.abs(self.div_cost_m - c)
            self.annvae_loss = tf.add(self.ae_loss, self.anneal_reg)

        with tf.variable_scope('bayae_loss', reuse=self.config.reuse):
            if self.config.isConv:
                self.bay_div = -1 * losses.get_divergence(self.encoder_mean, self.encoder_var, \
                                                          tf.reshape(self.prior_mean, [self.config.MC_samples, self.config.batch_size, self.config.latent_dim]), \
                                                          tf.reshape(self.prior_var, [self.config.MC_samples, self.config.batch_size, self.config.latent_dim]),
                                                          self.config.prior_div_cost)
            else:
                self.bay_div = -1 * losses.get_divergence(self.encoder_mean, self.encoder_var, \
                                                          self.prior_mean, self.prior_var,
                                                          self.config.prior_div_cost)
            self.bayae_loss = tf.add(tf.cast(self.config.ntrain_batches, 'float32') * self.ae_loss, self.bay_div, name='bayae_loss')
            self.bayvae_loss = tf.add(tf.cast(self.config.ntrain_batches, 'float32') * self.vae_loss, self.bay_div, name='bayvae_loss')
            self.annbayvae_loss = tf.add(tf.cast(self.config.ntrain_batches, 'float32') * self.annvae_loss, self.bay_div,
                                      name='bayvae_loss')

        with tf.variable_scope('optimizer', reuse=self.config.reuse):
            self.optimizer = RAdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.annbayvae_loss, global_step=self.global_step_tensor)

        self.losses = ['ELBO_AnnBayVAE', 'BayVAE', 'BayAE', 'AE', 'Recons_{}'.format(self.config.reconst_loss),
                       'Div_{}'.format(self.config.div_cost),
                       'Regul_anneal_reg', 'Regul_L2', 'prior_recons_{}'.format(self.config.prior_reconst_loss),
                       'bayesian_div_{}'.format(self.config.prior_div_cost)]


    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step, self.annbayvae_loss, self.bayvae_loss, self.bayae_loss, self.ae_loss, self.loss_reconstruction_m,
                   self.div_cost_m, self.anneal_reg, self.L2_loss,
                   self.prior_recons_m, self.bay_div]
        feed_dict = {self.x_batch: x}
        _, annbayvae_loss, bayvae_loss, bayae_loss, aeloss, recons, div, ann_reg, L2_loss, prior_recons, bay_div = session.run(tensors, feed_dict=feed_dict)
        return annbayvae_loss, bayvae_loss, bayae_loss, aeloss, recons, div, ann_reg, L2_loss, prior_recons, bay_div

    def test_epoch(self, session, x):
        tensors = [self.annbayvae_loss, self.bayvae_loss, self.bayae_loss, self.ae_loss, self.loss_reconstruction_m,
                   self.div_cost_m, self.anneal_reg, self.L2_loss,
                   self.prior_recons_m, self.bay_div]
        feed_dict = {self.x_batch: x}
        annbayvae_loss, bayvae_loss, bayae_loss, aeloss, recons, div, ann_reg, L2_loss, prior_recons, bay_div = session.run(tensors, feed_dict=feed_dict)
        return annbayvae_loss, bayvae_loss, bayae_loss, aeloss, recons, div, ann_reg, L2_loss, prior_recons, bay_div