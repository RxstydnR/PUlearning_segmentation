# Keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
import numpy as np

# [Temperature in softmax](https://qiita.com/nkriskeeic/items/db3b4b5e835e63a7f243)

class PULoss(tf.keras.losses.Loss):
    
    def __init__(self, prior, loss="sigmoid", gamma=1, beta=0, nnPU=False, temp=1):
        super().__init__() 
        self.prior = prior
        self.gamma = gamma       # gamma parameter of nnPU
        self.beta = beta         # beta parameter of nnPU
        self.nnPU = nnPU         # uPU? or nnPU?
        self.T = temp
        self.positive = 1        # positive label value
        self.unlabeled = -1      # negative label value 
        self.min_count = K.constant(1.)
        self.check_params()
        
        assert loss in ["softmax", "sigmoid"], 'Loss must be softmax or sigmoid.'
        if loss=="softmax":
            self.loss = (lambda x: K.softmax(x/self.T))
        elif loss=="sigmoid":
            self.loss = (lambda x: K.sigmoid(x/self.T))
                
    def check_params(self):
        assert (0<self.prior<1), "The class prior should be in (0,1)"
        assert (0<=self.beta<=self.prior) , 'Set beta above 0. '
        assert (0<=self.gamma<=1) , 'Set gamma between 0~1.'

    # @tf.function
    def call(self, y_true, y_pred): 

        positive = K.equal(y_true, self.positive)
        unlabeled = K.equal(y_true, self.unlabeled)

        positive_idx = K.cast(positive, 'float32')
        unlabeled_idx = K.cast(unlabeled, 'float32')
        
        n_positive = K.max([self.min_count, K.sum(positive_idx)])
        n_unlabeled = K.max([self.min_count, K.sum(unlabeled_idx)])
        
        # Loss when model predicts positive. If prediction is under 0 (i.e. Negative), loss is accumulated because loss function gets -x.
        # πp * Rp+ (False Negative): 
        positive_risk = self.prior * K.sum(self.loss(-positive_idx * y_pred)) / K.cast(n_positive,'float32')
        
        # -πp * Rp- (False Positive) + Ru- (False Positive in Unlabel data)
        negative_risk = (- self.prior * K.sum(self.loss(positive_idx*y_pred)) / K.cast(n_positive,'float32')) + (K.sum(self.loss(unlabeled_idx * y_pred)) / K.cast(n_unlabeled,'float32'))
        
        if (negative_risk < -self.beta) and (self.nnPU):
            # nnPU learning loss
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk


class PNULoss(tf.keras.losses.Loss):
    """ PNU Learning loss function.
    
        PNU is combined method with PN and PU, or PN and NU.
        PNU (PNPU or PNNU) is said to be better than PUNU Learning in the paper.

        PNU classification with η = −1, 0, +1 corresponds to NU, PN, and PU classification.
        As η gets larger/smaller, the effect of the positive/negative classes is more emphasized.（η が1に近いほどPositiveの影響が大きくなり、-1に近いほどNegativeの影響が大きくなる）
        If η is equel to -1, that means NU Learning.
        If η is equel to  1, that means PU Learning.

        You have to find the best η value using validation experiment.

        Use Example)
            "PNU": PNULoss(prior=0.4, eta=0.5)
    """
    def __init__(self, prior, eta, loss="sigmoid", temp=1.0): # (lambda x: K.softmax(-x))
        super().__init__() 
        self.prior = prior
        self.eta = eta
        self.T = temp
        self.positive = 1.0   # positive label value
        self.negative = -1.0  # negative label value
        self.unlabeled = 0.0  # unlabeled label value
        self.check_params()

        assert loss in ["softmax", "sigmoid"], 'Loss must be softmax or sigmoid.'
        if loss=="softmax":
            self.loss = (lambda x: K.softmax(x/self.T))
        elif loss=="sigmoid":
            self.loss = (lambda x: K.sigmoid(x/self.T))
                
    def check_params(self):
        assert (0<self.prior<1),"The class prior should be in (0,1)"
        assert (-1<=self.eta<=1),'Eta must be in [-1,1)].'

    # @tf.function
    def call(self, y_true, y_pred): 
                
        positive = K.equal(y_true, self.positive)
        negative = K.equal(y_true, self.negative)
        unlabeled = K.equal(y_true, self.unlabeled)
        
        positive_idx = K.cast(positive, 'float32')
        negative_idx = K.cast(negative, 'float32')
        unlabeled_idx = K.cast(unlabeled, 'float32')
        
        pred_p = tf.gather_nd(y_pred, tf.where(positive))
        pred_n = tf.gather_nd(y_pred, tf.where(negative))
        pred_u = tf.gather_nd(y_pred, tf.where(unlabeled))

        n_positive = K.max([1, K.sum(positive_idx)])
        n_negative = K.max([1, K.sum(negative_idx)])
        n_unlabeled = K.max([1, K.sum(unlabeled_idx)])
            
        # Prior probability of "Positive" and "Negative"
        th_p, th_n = self.prior, 1-self.prior
                
        # false_neg = K.sum(self.loss(-positive_idx * y_pred)) / K.cast(n_positive,'float32')
        # false_pos = K.sum(self.loss( negative_idx * y_pred)) / K.cast(n_negative,'float32')
        false_neg = K.sum(self.loss(-pred_p)) / K.cast(n_positive,'float32')
        false_pos = K.sum(self.loss( pred_n)) / K.cast(n_negative,'float32')
        
        # Equation number 1 in the paper 
        # θp*Rp + θn*Rn
        pn_risk = (th_p * false_neg) + (th_n * false_pos)
        
        if self.eta >= 0: # "PU"+PN
            
            # unlabeled false positive 
            # false_pos_u = K.sum(self.loss(unlabeled_idx * y_pred)) / K.cast(n_unlabeled,'float32')
            false_pos_u = K.sum(self.loss(pred_u)) / K.cast(n_unlabeled,'float32')
            
            # Equation number 4 in the paper 
            # θp*Rp + Ru,n = 2*θp*Rp + Ru,n - θp
            # xu_risk = (2 * th_p * false_neg) + false_pos_u - th_p
            xu_risk = (th_p * false_neg) + K.max([0, false_pos_u + th_p*false_neg - th_p])
            
            # η and γ are different in the equation in the paepr,
            # but η (eta) is only used to decide which PNPU or PNNU is adopted in this program.
            # Therefore, let eta have roles of deciding PNPU and PNNU, and of γ.
            risk = ((1-self.eta) * pn_risk) + (self.eta * xu_risk)
        
        else: # "NU"+PN
            # unlabeled false negative
            # false_neg_u = K.sum(self.loss(-unlabeled_idx * y_pred)) / K.cast(n_unlabeled,'float32')
            false_neg_u = K.sum(self.loss(-pred_u)) / K.cast(n_unlabeled,'float32')

            # Equation number 7 in the paper
            # θn*Rn + Ru,p = 2*θn*Rn + Ru,p - θn
            # xu_risk = (2 * th_n * false_pos) + false_neg_u - th_n 
            xu_risk = (th_n * false_pos) + K.max([0, false_neg_u + th_n*false_pos - th_n])
            
            risk = ((1+self.eta) * pn_risk) + (-self.eta * xu_risk)
            
        return risk