
"""
Created on Thu Sep 13 00:32:57 2018

@author: pablosanchez
"""
import tensorflow as tf
from networks.dense_net import DenseNet

class ConvNet(object):
    def __init__(self, hidden_dim, output_dim, reuse, transfer_fct=tf.nn.relu,
                 act_out=tf.nn.sigmoid, drop_rate=0.2, batch_norm=True, kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0), prefix=''):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transfer_fct = transfer_fct
        self.act_out = act_out
        self.reuse = reuse
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        
        self.kinit= kinit
        self.bias_init = bias_init
        self.prefix= prefix

    def build(self, input_):
        raise NotImplementedError

    # filetrs == num_channels
    # padding == 'SAME' 'VALID'
    def conv_layer(self, input_, filters, k_size, stride, padding, name, act_func=tf.nn.relu):
        conv = tf.layers.conv2d(inputs=input_, 
                                filters=filters, 
                                kernel_size=k_size, 
                                strides=stride, 
                                padding=padding, 
                                activation=act_func, 
                                kernel_initializer=self.kinit, 
                                bias_initializer=self.bias_init, 
                                name=name, 
                                reuse=self.reuse)
        
        print('[*] Layer (',conv.name, ') output shape:', conv.get_shape().as_list())

        return conv 
    def max_pool(self, input_, pool_size, strides, name):
    
        # Pooling Layer #1 output = [batch_size,14, 14,32]
        pool = tf.layers.max_pooling2d(inputs=input_, 
                                       pool_size=pool_size, 
                                       strides=strides, 
                                       name=name )
        print('[*] Layer (',pool.name, ') output shape:', pool.get_shape().as_list())
    
        return pool  


class ConvNet3(ConvNet):
    def __init__(self, input_, hidden_dim, output_dim, reuse, num_layers=3, transfer_fct=tf.nn.relu,
                 act_out=tf.nn.sigmoid, drop_rate=0.2, batch_norm=True, kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0), prefix=''):
        super().__init__(hidden_dim, output_dim, reuse, transfer_fct, act_out, drop_rate,batch_norm, kinit, bias_init, prefix)
        
        self.num_layers=num_layers
        self.output = self.build(input_)


    def build(self, input_):
        output = None
        x = input_
        for i in range(self.num_layers+2):
            x = self.conv_layer(input_=x, 
                                filters=32, 
                                k_size=4,  #[4, 4]
                                stride=2, 
                                padding='SAME',
                                name=self.prefix + '_' + 'conv_' + str(
                                    i + 1) if self.prefix != '' else 'conv_' + str(i + 1),
                                act_func=self.transfer_fct)
                                
            if self.batch_norm:
                x = tf.layers.batch_normalization(x,name='batch_norm_'+str(i+1))
        
        x = tf.layers.flatten(x)
        
        dense = DenseNet(input_=x,
                         hidden_dim=self.hidden_dim, 
                         output_dim=self.output_dim, 
                         num_layers=self.num_layers, 
                         transfer_fct=self.transfer_fct,
                         act_out=self.act_out, 
                         reuse=self.reuse, 
                         kinit=self.kinit,
                         bias_init=self.bias_init,
                         drop_rate=self.drop_rate,
                         batch_norm=self.batch_norm,
                         prefix=self.prefix)
        x = dense.output
        return x
    
class ConvNet3Gauss(ConvNet):
    def __init__(self, input_, hidden_dim, output_dim, reuse, transfer_fct=tf.nn.relu,
                 act_out_mean=None,act_out_var=tf.nn.softplus, drop_rate=0.2,  batch_norm=True, kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0), prefix=''):
        super().__init__(hidden_dim, output_dim, reuse, transfer_fct, act_out_mean, drop_rate, batch_norm, kinit, bias_init)
        
        self.act_out_mean = act_out_mean
        self.act_out_var = act_out_var
        self.mean, self.var = self.build(input_)

    def build(self, input_):
        output = None
        x = self.conv_layer(input_=input_, 
                            filters=32, 
                            k_size=4,  #[4, 4]
                            stride=2, 
                            padding='SAME', 
                            name=self.prefix + '_'+'conv_1' if self.prefix !='' else 'conv_1',
                            act_func=self.transfer_fct)
                            
        if self.batch_norm:
            x = tf.layers.batch_normalization(x,name='batch_norm_1')                      
        
        x = self.conv_layer(input_=x, 
                            filters=64, 
                            k_size=4,  #[4, 4]
                            stride=2, 
                            padding='SAME',
                            name=self.prefix + '_' + 'conv_2' if self.prefix != '' else 'conv_2',
                            act_func=self.transfer_fct)
        
        if self.batch_norm:
            x = tf.layers.batch_normalization(x,name='batch_norm_2')                      
                            
        x = self.conv_layer(input_=x, 
                            filters=64, 
                            k_size=4,  #[4, 4]
                            stride=2, 
                            padding='SAME',
                            name=self.prefix + '_' + 'conv_3' if self.prefix != '' else 'conv_3',
                            act_func=self.transfer_fct)
        
        x = tf.layers.flatten(x)
        
        
        dense = DenseNet(input_=x,
                         hidden_dim=self.hidden_dim, 
                         output_dim=self.hidden_dim, 
                         num_layers=1, 
                         transfer_fct=self.transfer_fct,
                         act_out=self.transfer_fct, 
                         reuse=self.reuse, 
                         kinit=self.kinit,
                         bias_init=self.bias_init,
                         drop_rate=self.drop_rate,
                         batch_norm=self.batch_norm,
                         prefix=self.prefix)

        x = dense.output
        with tf.variable_scope('mean', reuse=self.reuse):
            dense_mean = DenseNet(input_=x,
                             hidden_dim=self.hidden_dim, 
                             output_dim=self.output_dim, 
                             num_layers=1, 
                             transfer_fct=self.transfer_fct,
                             act_out=self.act_out_mean, 
                             reuse=self.reuse, 
                             kinit=self.kinit,
                             bias_init=self.bias_init,
                             drop_rate=self.drop_rate,
                             batch_norm=self.batch_norm,
                             prefix=self.prefix)

        with tf.variable_scope('var', reuse=self.reuse):
            dense_var = DenseNet(input_=x,
                             hidden_dim=self.hidden_dim, 
                             output_dim=self.output_dim, 
                             num_layers=1, 
                             transfer_fct=self.transfer_fct,
                             act_out=self.act_out_var, 
                             reuse=self.reuse, 
                             kinit=self.kinit,
                             bias_init=self.bias_init,
                             drop_rate=self.drop_rate,
                             batch_norm=self.batch_norm,
                             prefix=self.prefix)
        return dense_mean.output, dense_var.output

        

 
