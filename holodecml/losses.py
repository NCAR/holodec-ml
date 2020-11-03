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
    loss_real = tf.reduce_mean(tf.abs(y_true[y_true[:, :, -1] > 0] - y_pred[y_true[:, :, -1] > 0]))
    loss_bce = tf.keras.losses.binary_crossentropy(y_true[:, :, -1].flatten(),
                                                   y_pred[:, :, -1].flatten())
    loss_total = loss_real + loss_bce
    return loss_total


def attention_net_validation_loss(y_true, y_pred):
    loss_dist = tf.zeros((), dtype=tf.float64)
    loss_diam = tf.zeros((), dtype=tf.float64)
    loss_prob = tf.zeros((), dtype=tf.float64)
    loss_bce = tf.zeros((), dtype=tf.float64)

    for h in range(y_true.shape[0]):
        y_true_h = y_true[h:h + 1][y_true[h:h + 1, :, -1] > 0]
        dist_x = (y_true_h[:, 0:1] - tf.transpose(y_pred)[0:1, :, h]) ** 2
        dist_y = (y_true_h[:, 1:2] - tf.transpose(y_pred)[1:2, :, h]) ** 2
        dist_z = (y_true_h[:, 2:3] - tf.transpose(y_pred)[2:3, :, h]) ** 2
        dist_squared = dist_x + dist_y + dist_z
        loss_dist_h = tf.math.reduce_sum(tf.math.reduce_min(dist_squared, axis=1))
        loss_dist += loss_dist_h

        max_idx = tf.math.argmin(dist_squared, axis=1)
        max_idx = tf.stack((tf.constant(range(dist_squared.shape[0]), dtype=tf.int64), max_idx), axis=-1)

        dist_d = (y_true_h[:, 3:4] - tf.transpose(y_pred)[3:4, :, h]) ** 2
        loss_diam_h = tf.math.reduce_sum(tf.gather_nd(dist_d, max_idx))
        loss_diam = tf.math.add(loss_diam, loss_diam_h)

        dist_p = (y_true_h[:, 4:5] - tf.transpose(y_pred)[4:5, :, h]) ** 2
        loss_prob_h = tf.math.reduce_sum(tf.gather_nd(dist_p, max_idx))
        loss_prob = tf.math.add(loss_prob, loss_prob_h)

        loss_bce_h = tf.keras.losses.binary_crossentropy(y_true_h[:, -1],
                                                         y_pred[h, max_idx[:, 1], -1])
        loss_bce = tf.math.add(loss_bce, loss_bce_h)

    valid_error = loss_dist + loss_diam + loss_bce

    return valid_error
