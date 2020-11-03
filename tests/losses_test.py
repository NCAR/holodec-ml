from holodecml.losses import attention_net_loss, attention_net_validation_loss
import unittest


class TestLosses(unittest.TestCase):
    def test_attention_net_validation_loss(self):
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

        assert attention_net_validation_loss(true_perfect, pred_perfect) == 0.0
        assert attention_net_validation_loss(true_good, pred_good) < attention_net_validation_loss(true_bad, pred_bad)
        assert attention_net_validation_loss(true_good, pred_good).shape == ()

        return

    def test_attention_net_loss(self):
        assert attention_net_loss(true_good, pred_good).shape == ()
        return
