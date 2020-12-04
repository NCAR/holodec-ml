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


def noisy_true_particle_loss(y_true, y_pred):
    # y_true and y_pred will have shape (batch_size x max_num_particles x 5)
    loss_real = tf.reduce_mean(tf.abs(y_true[y_true[:, :, -1] > 0] - y_pred[y_true[:, :, -1] > 0]))
    loss_bce = tf.keras.losses.binary_crossentropy(tf.reshape(y_true[:, :, -1],[-1]),
                                                   tf.reshape(y_pred[:, :, -1],[-1]))
    loss_total = loss_real + loss_bce
    return loss_total

def random_particle_distance_loss(y_true, y_pred):
    loss_xy = tf.zeros((), dtype=tf.float32)
    loss_z = tf.zeros((), dtype=tf.float32)
    loss_d = tf.zeros((), dtype=tf.float32)
    
    for h in range(tf.shape(y_pred)[0]):
        y_pred_h = y_pred[h]
        print("y_pred_h.shape", y_pred_h.get_shape())
        y_true_h = y_true[h]
        print("y_true_h.shape", y_true_h.shape)
        real_idx = tf.argmin(y_true_h[:, -1], axis=0)
        if real_idx == 0:
            real_idx = tf.cast(tf.shape(y_true_h)[0], dtype=tf.int64)
        print("real_idx.shape", real_idx.get_shape())
        y_true_h = y_true_h[:real_idx]
        print("y_true_h.shape", y_true_h.get_shape())
        
        dist_x = (y_pred_h[:, 0:1] - tf.transpose(y_true_h)[0:1, :]) ** 2
        dist_y = (y_pred_h[:, 1:2] - tf.transpose(y_true_h)[1:2, :]) ** 2
        dist_xy = dist_x + dist_y
        print(f"dist_xy.shape: {dist_xy.shape}")
        loss_xy_h = tf.math.reduce_sum(tf.math.reduce_min(dist_xy, axis=1))
        loss_xy = loss_xy + loss_xy_h

        # determine index of true particle closest to each predicted particle
        max_idx = tf.cast(tf.math.argmin(dist_xy, axis=1), dtype=tf.int32)
        max_idx_2d = tf.stack((tf.range(tf.shape(dist_xy)[0]), max_idx), axis=-1)

        loss_z_h = (y_pred_h[:, 2:3] - tf.transpose(y_true_h)[2:3, :]) ** 2
        loss_z_h = tf.math.reduce_sum(tf.gather_nd(loss_z_h, max_idx_2d))
        loss_z = loss_z + loss_z_h
        
        loss_d_h = (y_pred_h[:, 3:4] - tf.transpose(y_true_h)[3:4, :]) ** 2
        loss_d_h = tf.math.reduce_sum(tf.gather_nd(loss_d_h, max_idx_2d))
        loss_d = loss_d + loss_d_h

    loss_xy = loss_xy/tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    loss_z = loss_z/tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    loss_d = loss_d/tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)

    valid_error = loss_xy + loss_z + loss_d
    print(f"ERROR SHAPE: {valid_error.shape}")

    return valid_error 
