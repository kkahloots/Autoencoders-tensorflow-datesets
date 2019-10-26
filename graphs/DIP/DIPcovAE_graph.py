"""
DIPcovAE_graph.py:
Disentangled Inferred Prior - covar
Tensorflow Graph for the DIP covar Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://openreview.net/pdf?id=H1kG7GZAW"

import tensorflow as tf
import bases.losses as losses
from bases.base_graph import BaseGraph
from keras_radam.training import RAdamOptimizer


'''
This is the Main DIPcovAEGraph.
'''
class DIPcovAEGraph(BaseGraph):

    def extra_sittings(self):
        self.d = self.config.d_factor * self.config.lambda_d

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
            self.encoder_var = tf.nn.sigmoid(self.encoder_mean)
        self.latent = self.encoder_mean
        self.latent_batch = self.latent
            
        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder_mean', reuse=self.config.reuse):
            Px_latent_mean = self.create_decoder(input_=self.config.latent_batch,
                                            hidden_dim=self.config.hidden_dim,
                                            output_dim=self.config.x_flat_dim,
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

        with tf.variable_scope('dipae_loss', reuse=self.config.reuse):
            self.covar_reg = self.regularizer(self.encoder_mean, self.encoder_var)
            self.dipae_loss = tf.add(self.ae_loss, self.covar_reg)

        with tf.variable_scope("optimizer" ,reuse=self.config.reuse):
            self.optimizer = RAdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.dipae_loss, global_step=self.global_step_tensor)

        self.losses = ['ELBO_DIPcovAE', 'Regul_Covariance_Prior', 'AE', 'Recons_{}'.format(self.config.reconst_loss),
                       'Regul_L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step, self.dipae_loss, self.covar_reg, self.ae_loss,
                   self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, loss, covar_reg, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, covar_reg, aeloss, recons, L2_loss
    
    def test_epoch(self, session, x):
        tensors = [self.dipae_loss, self.covar_reg, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        loss, covar_reg, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, covar_reg, aeloss, recons, L2_loss


    '''  
    ------------------------------------------------------------------------------
                                         DIP OPERATIONS
    ------------------------------------------------------------------------------ 
    '''

    def compute_covariance_latent_mean(self, latent_mean):
        """
        :param latent_mean:
        :return:
        Computes the covariance of latent_mean.
        Uses cov(latent_mean) = E[latent_mean*latent_mean^T] - E[latent_mean]E[latent_mean]^T.
        Args:
          latent_mean: Encoder mean, tensor of size [batch_size, num_latent].
        Returns:
          cov_latent_mean: Covariance of encoder mean, tensor of size [latent_dim, latent_dim].
        """
        exp_latent_mean_latent_mean_t = tf.reduce_mean(
            tf.expand_dims(latent_mean, 2) * tf.expand_dims(latent_mean, 1), axis=0)
        expectation_latent_mean = tf.reduce_mean(latent_mean, axis=0)

        cov_latent_mean = tf.subtract(exp_latent_mean_latent_mean_t,
          tf.expand_dims(expectation_latent_mean, 1) * tf.expand_dims(expectation_latent_mean, 0))
        return cov_latent_mean

    def regularize_diag_off_diag_dip(self, covariance_matrix, lambda_od, lambda_d):
        """
        Compute on and off diagonal regularizers for DIP-VAE models.
        Penalize deviations of covariance_matrix from the identity matrix. Uses
        different weights for the deviations of the diagonal and off diagonal entries.
        Args:
            covariance_matrix: Tensor of size [num_latent, num_latent] to covar_reg.
            lambda_od: Weight of penalty for off diagonal elements.
            lambda_d: Weight of penalty for diagonal elements.
        Returns:
            dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
        """
        covariance_matrix_diagonal = tf.diag_part(covariance_matrix)
        covariance_matrix_off_diagonal = covariance_matrix - tf.diag(covariance_matrix_diagonal)
        dip_regularizer = tf.add(
              lambda_od * tf.reduce_sum(covariance_matrix_off_diagonal**2),
              lambda_d * tf.reduce_sum((covariance_matrix_diagonal - 1)**2))

        return dip_regularizer

    def regularizer(self, latent_mean, latent_logvar):
        cov_latent_mean = self.compute_covariance_latent_mean(latent_mean)

        # Eq 6 page 4
        # mu = z_mean is [batch_size, num_latent]
        # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
        cov_dip_regularizer = self.regularize_diag_off_diag_dip(cov_latent_mean, self.config.lambda_d, self.d)

        return cov_dip_regularizer