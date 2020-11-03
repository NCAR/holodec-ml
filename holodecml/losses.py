import logging
import tensorflow as tf
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
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res / (SS_tot + K.epsilon())


def keras_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def attention_net_loss(y_true, y_pred):
    # y_true and y_pred will have shape (batch_size x max_num_particles x 5)
    print(y_true)
    print(y_pred)
    loss_real = tf.reduce_mean(tf.abs(y_true[y_true[:, :, -1] > 0] - y_pred[y_true[:, :, -1] > 0]))
    print("loss_real", tf.shape(loss_real))
    loss_bce = tf.keras.losses.binary_crossentropy(y_true[:, :, -1],
                                                   y_pred[:, :, -1])
    print("loss_bce", tf.shape(loss_bce))
    print(loss_bce)
    loss_total = loss_real + loss_bce
    return loss_total


def attention_net_validation_loss(y_true, y_pred):
    loss_dist = tf.zeros((), dtype=tf.float32)
    loss_diam = tf.zeros((), dtype=tf.float32)
    loss_prob = tf.zeros((), dtype=tf.float32)
    loss_bce = tf.zeros((), dtype=tf.float32)
    
    print("validation loss", y_true.shape, y_pred.shape)
    print(tf.shape(y_true))

    for h in range(tf.shape(y_true)[0]):
        print("h", h)
        y_true_h = y_true[h:h + 1][y_true[h:h + 1, :, -1] > 0]
        dist_x = (y_true_h[:, 0:1] - tf.transpose(y_pred)[0:1, :, h]) ** 2
        dist_y = (y_true_h[:, 1:2] - tf.transpose(y_pred)[1:2, :, h]) ** 2
        dist_z = (y_true_h[:, 2:3] - tf.transpose(y_pred)[2:3, :, h]) ** 2
        dist_squared = dist_x + dist_y + dist_z
        loss_dist_h = tf.math.reduce_sum(tf.math.reduce_min(dist_squared, axis=1))
        loss_dist += loss_dist_h
        print("done with loss_dist")

        max_idx = tf.cast(tf.math.argmin(dist_squared, axis=1), dtype=tf.int32)
        print("done with first max_idx")
        print(max_idx)
        print("range", tf.range(tf.shape(dist_squared)[0]))
        max_idx_2d = tf.stack((tf.range(tf.shape(dist_squared)[0]), max_idx), axis=-1)
        print("done with second max_idx_2d")
        print(max_idx_2d)

        dist_d = (y_true_h[:, 3:4] - tf.transpose(y_pred)[3:4, :, h]) ** 2
        print("trying to calculate loss_diam_h")
#         loss_diam_h = tf.gather_nd(dist_d, max_idx_2d)
        loss_diam_h = tf.math.reduce_sum(tf.gather_nd(dist_d, max_idx_2d))
        print("done calculating loss_diam_h")  
        print(loss_diam_h)
        loss_diam = tf.math.add(loss_diam, loss_diam_h)

        dist_p = (y_true_h[:, 4:5] - tf.transpose(y_pred)[4:5, :, h]) ** 2
        loss_prob_h = tf.math.reduce_sum(tf.gather_nd(dist_p, max_idx_2d))
        loss_prob = tf.math.add(loss_prob, loss_prob_h)
        print("done calculating loss_prob")

        print("about to calculate loss_bce_h", max_idx)        
        y_pred_h_bce = y_pred[h, :, -1]
        print("y_pred_h_bce", y_pred_h_bce.shape)
        loss_bce_h = tf.keras.losses.binary_crossentropy(y_true_h[:, -1],
                                                         tf.gather(y_pred_h_bce,max_idx))
        print("done calculating loss_bce_h")
        loss_bce = tf.math.add(loss_bce, loss_bce_h)

    valid_error = loss_dist + loss_diam + loss_bce
    print(tf.shape(valid_error))

    return valid_error
