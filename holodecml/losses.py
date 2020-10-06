import logging
import tensorflow as tf
from typing import List, Dict
import tensorflow.keras.backend as K

logger = logging.getLogger(__name__)


class SymmetricCrossEntropy:
    
    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self.a = a
        self.b = b
    
    def __call__(self, *args, **kwargs) -> float:
        bce = tf.keras.losses.CategoricalCrossentropy()
        kld = tf.keras.losses.KLDivergence()
        return self.a * bce(*args, **kwargs) + self.b * kld(*args, **kwargs)
    
    
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def wmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def R2(y_true, y_pred):
    """ Is actually 1 - R2
    """
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res/(SS_tot + K.epsilon())

def keras_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def attention_net_loss(y_true, y_pred):
    # y_true and y_pred will have shape (batch_size x max_num_particles x 5)
    loss_real = tf.reduce_mean(tf.abs(y_true[y_true[:, :, -1] > 0] - y_pred[y_true[:, :, -1] > 0]))
    loss_bce = binary_crossentropy(y_true[:,:,-1], 
                                   y_pred[:,:,-1])
    loss_total = loss_real + loss_bce
    return loss_total

