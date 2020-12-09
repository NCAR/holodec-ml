import tensorflow as tf
import numpy as np
from holodecml.losses import noisy_true_particle_loss, random_particle_distance_loss


true_bad = np.array([[[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0]],
                     [[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0]]])
pred_bad = np.array([[[10, 10, 10, 10, 0.1], [20, 20, 20, 20, 0.1], [50, 50, 50, 50, 0.1],
                      [60, 60, 60, 60, 0.9], [70, 70, 70, 70, 0.9]],
                     [[10, 10, 10, 10, 0.1], [20, 20, 20, 20, 0.1], [50, 50, 50, 50, 0.1],
                      [60, 60, 60, 60, 0.9], [70, 70, 70, 70, 0.9]]])

true_good = np.array([[[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0]],
                      [[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0]]])
pred_good = np.array([[[60, 60, 60, 60, 0.1], [31, 31, 31, 31, 0.9], [2, 2, 2, 2, 0.9], [39, 39, 39, 39, 0.9],
                       [70, 70, 70, 70, 0.1]],
                      [[60, 60, 60, 60, 0.1], [31, 31, 31, 31, 0.9], [2, 2, 2, 2, 0.9], [39, 39, 39, 39, 0.9],
                       [70, 70, 70, 70, 0.1]]])

true_perfect = np.array([[[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0]],
                         [[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0]]])
pred_perfect = np.array([[[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0],
                          [60, 60, 60, 60, 0.9], [70, 70, 70, 70, 0.9]],
                         [[1, 1, 1, 1, 1], [30, 30, 30, 30, 1], [40, 40, 40, 40, 1], [1, 1, 1, 1, 0],
                          [60, 60, 60, 60, 0.9], [70, 70, 70, 70, 0.9]]])

def test_random_particle_distance_loss():

    assert random_particle_distance_loss(true_perfect, pred_perfect) == 0.0
    assert random_particle_distance_loss(true_good, pred_good) < random_particle_distance_loss(true_bad, pred_bad)
    return

# def test_attention_net_loss():
#     assert attention_net_loss(true_good, pred_good).shape == ()
#     return

if __name__ == '__main__':
    test_random_particle_distance_loss()
