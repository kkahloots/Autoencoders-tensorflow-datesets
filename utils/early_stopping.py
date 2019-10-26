
"""
Created on Tue Sep 11 20:52:46 2018

@author: pablosanchez
"""
from utils.configuration import default_config as config

class EarlyStopping(object):
    def __init__(self, name='', patience=15, min_delta = config.tolerance, decay_fn=None):
        self.name = name
        self.patience = patience
        self.min_delta = min_delta
        self.patience_cnt = 0
        self.prev_loss_val = 200000000
        self.decay_fn = decay_fn

    def stop(self, loss_val):
        if(self.prev_loss_val - loss_val> self.min_delta):
            self.patience_cnt = 0
            self.prev_loss_val = loss_val
            
        else:
            self.patience_cnt += 1
            print(self.name + ' Patience count: ', self.patience_cnt)
            
        if self.decay_fn is not None:
            if self.patience_cnt == self.patience//2 + 1:
                self.decay_fn()

        if(self.patience_cnt > self.patience):
            return True
        else:
            return False
        
    