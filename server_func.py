import numpy as np
import os

def get_inner_folds(client_data_path, outer_fold, inner_fold, tr, val, te):
	all_tr = np.concatenate((tr,val))
	new_val = np.loadtxt(os.path.join(client_data_path,'fold{}_te_client_ids.lst'.format(inner_fold)),dtype=int,delimiter=',')
	tr = np.setdiff1d(all_tr, new_val, True)
	return tr, new_val

class StepDecay:
    def __init__(self, init_lr=0.01, factor=0.25, drop_every=10, n_warmup_steps=500):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.init_lr = init_lr
        self.factor = factor
        self.drop_every = drop_every
        self.n_warmup_steps = n_warmup_steps
    def __call__(self, epoch):
        if epoch < self.n_warmup_steps:
            return self.init_lr
        # compute the learning rate for the current epoch
        exp = np.floor((epoch-self.n_warmup_steps) / self.drop_every)
        alpha = self.init_lr * (self.factor ** exp)
        # return the learning rate
        return float(alpha)