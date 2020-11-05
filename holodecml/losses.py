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
    print("ATTENTION_NET_LOSS")
    loss_real = tf.reduce_mean(tf.abs(y_true[y_true[:, :, -1] > 0] - y_pred[y_true[:, :, -1] > 0]))
    print(f"loss_real: {tf.shape(loss_real)}")
    loss_bce = tf.keras.losses.binary_crossentropy(tf.reshape(y_true[:, :, -1],[-1]),
                                                   tf.reshape(y_pred[:, :, -1],[-1]))
    print(f"loss_bce: {tf.shape(loss_bce)}")
    loss_total = loss_real + loss_bce
    return loss_total

def attention_net_validation_loss(y_true, y_pred):
    print("ATTENTION_NET_VALIDATION_LOSS")
    loss_dist = tf.zeros((), dtype=tf.float32)
    loss_diam = tf.zeros((), dtype=tf.float32)
    loss_prob = tf.zeros((), dtype=tf.float32)
    loss_bce = tf.zeros((), dtype=tf.float32)

    for h in range(tf.shape(y_true)[0]):
        y_true_h = y_true[h:h + 1][y_true[h:h + 1, :, -1] > 0]
        dist_x = (y_true_h[:, 0:1] - tf.transpose(y_pred)[0:1, :, h]) ** 2
        dist_y = (y_true_h[:, 1:2] - tf.transpose(y_pred)[1:2, :, h]) ** 2
        dist_z = (y_true_h[:, 2:3] - tf.transpose(y_pred)[2:3, :, h]) ** 2
        dist_squared = dist_x + dist_y + dist_z
        loss_dist_h = tf.math.reduce_sum(tf.math.reduce_min(dist_squared, axis=1))
        loss_dist = loss_dist + loss_dist_h

        max_idx = tf.cast(tf.math.argmin(dist_squared, axis=1), dtype=tf.int32)
        max_idx_2d = tf.stack((tf.range(tf.shape(dist_squared)[0]), max_idx), axis=-1)

        dist_d = (y_true_h[:, 3:4] - tf.transpose(y_pred)[3:4, :, h]) ** 2
        loss_diam_h = tf.math.reduce_sum(tf.gather_nd(dist_d, max_idx_2d))
        loss_diam = loss_diam + loss_diam_h

        dist_p = (y_true_h[:, 4:5] - tf.transpose(y_pred)[4:5, :, h]) ** 2
        loss_prob_h = tf.math.reduce_sum(tf.gather_nd(dist_p, max_idx_2d))
        loss_prob = loss_prob + loss_prob_h

        y_pred_h_bce = y_pred[h, :, -1]
        loss_bce_h = tf.keras.losses.binary_crossentropy(y_true_h[:, -1],
                                                         tf.gather(y_pred_h_bce,max_idx))
        loss_bce = loss_bce + loss_bce_h

    loss_dist = loss_dist/tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    loss_diam = loss_diam/tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    loss_prob = loss_prob/tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    loss_bce = loss_bce/tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

    # print(f"loss_dist: {loss_dist}\ttf.shape(loss_dist): {tf.shape(loss_dist)}")
    # print(f"loss_diam: {loss_diam}\ttf.shape(loss_diam): {tf.shape(loss_diam)}")
    # print(f"loss_dist: {loss_prob}\ttf.shape(loss_prob): {tf.shape(loss_prob)}")
    # print(f"loss_dist: {loss_bce}\ttf.shape(loss_bce): {tf.shape(loss_bce)}")
    valid_error = loss_dist + loss_diam + loss_bce

    return valid_error
