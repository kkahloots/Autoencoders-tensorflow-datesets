

import tensorflow as tf
import dask.array as da
import numpy as np

from networks.dense_net import DenseNet
from networks.conv_net import ConvNet3
from networks.deconv_net import DeconvNet3


class BaseGraph:
    def __init__(self, configuration):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

        self.config = configuration
        self.x_flat_dim = self.config.width * self.config.height * self.config.num_channels
        self.extra_sittings()
        self.build_graph()

    def extra_sittings(self):
        pass

    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()

    def create_inputs(self):
        raise NotImplementedError

    def create_graph(self):
        raise NotImplementedError

    def create_loss_optimizer(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def test_epoch(self):
        raise NotImplementedError

    '''  
    ------------------------------------------------------------------------------
                                     GENERATE LATENT and RECONSTRUCT
    ------------------------------------------------------------------------------ 
    '''
    def reconst_loss(self, session, x):
        tensors = [self.reconstruction]
        feed = {self.x_batch: x}
        return session.run(tensors, feed_dict=feed)

    def decay_lr(self, session):
        self.lr = tf.multiply(0.1, self.lr)
        nlr = session.run(self.lr)

        if nlr > self.config.min_lr:
            print('decaying learning rate ... ')

            tensors = [self.lr]
            feed_dict = {self.lr: nlr}
            nlr = session.run(tensors, feed_dict=feed_dict)[0]
            nlr = session.run(self.lr)
            nlr = round(nlr, 8)
            print('new learning rate: {}'.format(nlr))

    '''  
    ------------------------------------------------------------------------------
                                         GRAPH OPERATIONS
    ------------------------------------------------------------------------------ 
    '''
    def encode(self, session, inputs):
        tensors = [self.latent]
        feed_dict = {self.x_batch: inputs}
        return session.run(tensors, feed_dict=feed_dict)

    def decode(self, session, latent):
        tensors = [self.x_recons]
        feed_dict = {self.latent: latent}
        return session.run(tensors, feed_dict=feed_dict)

    def _sampling_reconst(self, session, std_scales, random_latent=None):

        if random_latent is None:
            random_latent = list()
            for m, sig, sc in zip(self.config.latent_mean, self.config.latent_std, std_scales):
                random_latent.append(session.run(tf.random_normal((self.config.batch_size, 1), m, sc * sig,
                                                                                dtype=tf.float32)))
            random_latent = da.hstack(random_latent)
        else:
            try:
                random_latent = random_latent.compute()
            except:
                pass
            for m, sig, sc, ic in zip(self.config.latent_mean, self.config.latent_std, std_scales, range(random_latent.shape[0])):
                random_latent[:, ic] =  m + (sc* np.sqrt(sig) * random_latent[:, ic])

        tensors = [self.x_recons]
        feed_dict = {self.latent_batch: random_latent}

        return session.run(tensors, feed_dict=feed_dict)


    '''  
    ------------------------------------------------------------------------------
                                     ENCODER-DECODER
    ------------------------------------------------------------------------------ 
    '''

    def create_encoder(self, input_, hidden_dim, output_dim, num_layers, transfer_fct, \
                       act_out, reuse, kinit, bias_init, drop_rate, prefix, isConv):
        latent_ = DenseNet(input_=input_,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           num_layers=num_layers,
                           transfer_fct=transfer_fct,
                           act_out=act_out,
                           reuse=reuse,
                           kinit=kinit,
                           bias_init=bias_init,
                           drop_rate=drop_rate,
                           prefix=prefix) if not isConv else \
            ConvNet3(input_=input_,
                     hidden_dim=hidden_dim,
                     output_dim=output_dim,
                     num_layers=num_layers,
                     transfer_fct=transfer_fct,
                     act_out=act_out,
                     reuse=reuse,
                     kinit=kinit,
                     bias_init=bias_init,
                     drop_rate=drop_rate,
                     prefix=prefix)
        return latent_

    def create_decoder(self, input_, hidden_dim, output_dim, num_layers, transfer_fct, \
                       act_out, reuse, kinit, bias_init, drop_rate, prefix, isConv):
        recons_ = DenseNet(input_=input_,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           num_layers=num_layers,
                           transfer_fct=transfer_fct,
                           act_out=act_out,
                           reuse=reuse,
                           kinit=kinit,
                           bias_init=bias_init,
                           drop_rate=drop_rate,
                           prefix=prefix) if not isConv else \
            DeconvNet3(input_=input_,
                       num_layers=num_layers,
                       hidden_dim=hidden_dim,
                       output_dim=output_dim,
                       width=self.config.width,
                       height=self.config.height,
                       nchannels=self.config.num_channels,
                       transfer_fct=transfer_fct,
                       act_out=act_out,
                       reuse=reuse,
                       kinit=kinit,
                       bias_init=bias_init,
                       drop_rate=drop_rate,
                       prefix=prefix)
        return recons_



