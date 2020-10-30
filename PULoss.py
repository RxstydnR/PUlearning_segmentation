# Keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K


class PULoss(tf.keras.losses.Loss):
    """ PU Learning loss function.
    
    Example)
        "uPU": PULoss(prior, loss=loss_type, nnpu=False)
        "nnPU": PULoss(prior, loss=loss_type, nnpu=True, gamma=0.1, beta=0)
    """
    
    def __init__(self, prior, loss=(lambda x: K.sigmoid(-x)), gamma=1, beta=0.05, nnPU=False):
        super().__init__() 
        self.prior = prior
        self.gamma = gamma       # gamma parameter of nnPU
        self.beta = beta         # beta parameter of nnPU
        self.loss_func = loss    # the name of a loss function（'logistic'(softmax) or 'sigmoid')
        self.nnPU = nnPU         # uPU? or nnPU?
        self.positive = 1        # positive label value
        self.unlabeled = -1      # negative label value
        self.min_count = K.constant(1.)
        self.check_params()
                
    def check_params(self):
        assert (0<self.prior<1), "The class prior should be in (0,1)"
        assert (0<=self.beta<=self.prior) , 'Set beta above 0. '
        assert (0<=self.gamma<=1) , 'Set gamma between 0~1.'

    @tf.function
    def call(self, target, inp): 
        
        positive, unlabeled = (target==self.positive,target==self.unlabeled)
        positive, unlabeled = K.cast(positive, 'float32'), K.cast(unlabeled, 'float32') # bool to number(float)
        
        n_positive, n_unlabeled = K.max([self.min_count, K.sum(positive)]), K.max([self.min_count, K.sum(unlabeled)])
        
        y_positive = self.loss_func(positive*inp)       # Rp+ の l
        y_positive_inv = self.loss_func(-positive*inp)  # Rp- の l
        y_unlabeled = self.loss_func(-unlabeled*inp)    # Ru- の l
        
        # πp * Rp+
        positive_risk = self.prior * K.sum(y_positive) / n_positive
        # (-πp * Rp-) + (Ru-)
        negative_risk = (- self.prior * K.sum(y_positive_inv) / n_positive) + (K.sum(y_unlabeled) / n_unlabeled)

        if (negative_risk < -self.beta) and (self.nnPU):
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk
        