"""
Created on Thu Sep 13 00:32:57 2018

@author: pablosanchez
"""
import tensorflow as tf
from networks.dense_net import DenseNet

class DeconvNet(object):
    def __init__(self,  width, height, nchannels, reuse, transfer_fct=tf.nn.relu,
                 act_out=tf.nn.sigmoid, drop_rate=0.2, batch_norm=True, kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0), prefix=''):
        
        self.width = width
        self.height = height
        self.nchannels = nchannels

        self.transfer_fct = transfer_fct
        self.act_out = act_out
        self.reuse = reuse
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        
        self.kinit= kinit
        self.bias_init = bias_init
        self.prefix = prefix

    def build(self, input_):
        raise NotImplementedError
 
    def deconv_layer(self, input_, filters, k_size, stride, padding, name, act_func=tf.nn.relu):
        deconv = tf.layers.conv2d_transpose(input_, 
                                            filters,
                                            k_size, 
                                            strides=stride, 
                                            padding=padding, 
                                            activation=act_func, 
                                            kernel_initializer=self.kinit, 
                                            bias_initializer=self.bias_init, 
                                            name=self.prefix+'_'+ name if self.prefix !='' else name,
                                            reuse=self.reuse)
        print('[*] Layer (',deconv.name, ') output shape:', deconv.get_shape().as_list())
    
        return deconv

class DeconvNet3(DeconvNet):
    def __init__(self, input_, output_dim, hidden_dim, num_layers, width, height, nchannels, reuse, transfer_fct=tf.nn.relu,
                 act_out=tf.nn.sigmoid, drop_rate=0.2, batch_norm=True, kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0), prefix=''):
        super().__init__(width, height, nchannels,reuse, transfer_fct, act_out, drop_rate, batch_norm, kinit, bias_init)
        
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.prefix = prefix
        self.output = self.build(input_)

    def build(self, input_):
        aux_size = self.width//2//2
        aux_size_2 = self.height//2//2
        initial_n_channels = 64
        out_dense_dim = aux_size*aux_size_2*initial_n_channels
        hidden_dim = input_.get_shape()[-1].value*3
        
        dense = DenseNet(input_=input_,
                         hidden_dim=hidden_dim, 
                         output_dim=out_dense_dim, 
                         num_layers=self.num_layers, 
                         transfer_fct=self.transfer_fct,
                         act_out=self.transfer_fct, 
                         reuse=self.reuse, 
                         kinit=self.kinit,
                         bias_init=self.bias_init,
                         drop_rate=self.drop_rate,
                         batch_norm=self.batch_norm,
                         prefix=self.prefix)
        x = dense.output
        x = tf.reshape(x, [-1,aux_size,aux_size_2,initial_n_channels])
        
        x = dense.output
        x = tf.reshape(x, [-1,aux_size,aux_size_2,initial_n_channels])
        x = self.deconv_layer(input_=x, 
                              filters=64, 
                              k_size=4,  #[4, 4]
                              stride=2, 
                              padding='SAME', 
                              name= self.prefix+ '_' + 'deconv_1' if self.prefix !='' else 'deconv_1',
                              act_func=self.transfer_fct)
        
        if self.batch_norm:
            x = tf.layers.batch_normalization(x,name='batch_norm_1')
            
        x = self.deconv_layer(input_=x, 
                            filters=32, 
                            k_size=4,  #[4, 4]
                            stride=2, 
                            padding='SAME',
                            name=self.prefix + '_' + 'deconv_2' if self.prefix != '' else 'deconv_2',
                            act_func=self.transfer_fct)
                            
        if self.batch_norm:
            x = tf.layers.batch_normalization(x,name='batch_norm_2')
            
        for i in range(self.num_layers):
            x = self.deconv_layer(input_=x, 
                                filters=self.nchannels, 
                                k_size=4,  #[4, 4]
                                stride=1, 
                                padding='SAME',
                                name=self.prefix + '_' + 'deconv_'+str(i+3) if self.prefix != '' else 'deconv_'+str(i+3),
                                act_func=self.transfer_fct)
            if self.batch_norm:
                x = tf.layers.batch_normalization(x,name='batch_norm_'+str(i+3))
        
        x = self.deconv_layer(input_=x, 
                                filters=self.nchannels, 
                                k_size=4,  #[4, 4]
                                stride=1, 
                                padding='SAME',
                                name=self.prefix + '_' + 'deconv_' + str(i + 4) if self.prefix != '' else 'deconv_' + str(
                                  i + 4),
                                act_func=self.act_out)
                                
        x = tf.layers.flatten(x)
        return x
        
