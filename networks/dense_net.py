
import tensorflow as tf

class DenseNet(object):
    def __init__(self, input_, hidden_dim, output_dim, num_layers, reuse, transfer_fct=tf.nn.relu,
                 act_out=tf.nn.sigmoid, drop_rate=0.2, batch_norm=True, kinit=tf.contrib.layers.xavier_initializer(),
                 bias_init=tf.constant_initializer(0.0), prefix=''):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
        self.num_layers = num_layers
        self.transfer_fct = transfer_fct
        self.act_out = act_out
        self.reuse = reuse
        self.drop_rate = drop_rate
        self.prefix = prefix
        self.batch_norm = batch_norm
        
        self.kinit= kinit
        self.bias_init = bias_init
        self.output = None
        if(num_layers >0):
            self.output = self.build(input_)

    def build(self, input_):
        output = None
        h = dict()
        print("")
        if(self.num_layers==1):         
            output = self.dense(input_=input_, 
                                output_dim=self.output_dim, 
                                name=self.prefix+'dense_1',
                                act_func=self.act_out)
        else:
            h['H1'] = self.dense_dropout(input_=input_,
                                     output_dim=self.hidden_dim,
                                     name=self.prefix+'dense_1',
                                     act_func=self.transfer_fct)
        
            
        for i in range(2, self.num_layers + 1):
            if(i == self.num_layers):
                output = self.dense(input_=h['H' + str(i - 1)], 
                                output_dim=self.output_dim, 
                                name=self.prefix+'dense_' + str(i),
                                act_func=self.act_out)
               
            else:
                h['H' + str(i)] = self.dense_dropout(input_=h['H' + str(i - 1)],
                                                 output_dim=self.hidden_dim,
                                                 name=self.prefix+'dense_' + str(i),
                                                 act_func=self.transfer_fct)
                if self.batch_norm:
                    h['H' + str(i)] = self.dense_batch_normalization(input_=h['H' + str(i)], name=self.prefix+'H' + str(i))
                
        return output
 
    
    def dense(self, input_, output_dim, name, act_func=tf.nn.relu):
        
        h = tf.layers.dense(inputs=input_, units=output_dim, activation=act_func, 
                            kernel_initializer=self.kinit, name=name, reuse=self.reuse, 
                            bias_initializer=self.bias_init)
        print('[*] Layer (', h.name, ') output shape:', h.get_shape().as_list())

        return h

    def dense_batch_normalization(self, input_, name):
        h = tf.layers.batch_normalization(input_,name=name+'_batch_norm')
        
        print('[*] Layer (', h.name, ') output shape:', h.get_shape().as_list())    
        
        return h
        
    def dense_dropout(self, input_, output_dim, name, act_func=tf.nn.relu):
        
        h = self.dense(input_, output_dim, name, act_func)        
        h = tf.layers.dropout(h,rate=self.drop_rate,name=name+'_dropout')
        
        print('[*] Layer (', h.name, ') output shape:', h.get_shape().as_list())    
        
        return h