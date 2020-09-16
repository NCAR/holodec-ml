import tensorflow as tf


class TruePositives(tf.metrics.Metric):
    def __init__(self, name="true_pos", **kwargs):
        super(TruePositives, self).__init__(name=name, **kwargs)
        self.true_pos = self.add_weight(name="tp", dtype="float64",
                                        initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.true_pos.assign(cm[1,1] / tf.keras.backend.sum(cm))

    def result(self):
        return self.true_pos

    def reset_states(self):
        self.true_pos.assign(0.0)

class FalsePositives(tf.metrics.Metric):
    def __init__(self, name="false_pos", **kwargs):
        super(FalsePositives, self).__init__(name=name, **kwargs)
        self.false_pos = self.add_weight(name="fp", dtype="float64",
                                         initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.false_pos.assign(cm[0,1]/tf.keras.backend.sum(cm))

    def result(self):
        return self.false_pos

    def reset_states(self):
        self.false_pos.assign(0.0)

class FalseNegatives(tf.metrics.Metric):
    def __init__(self, name="false_neg", **kwargs):
        super(FalseNegatives, self).__init__(name=name, **kwargs)
        self.false_neg = self.add_weight(name="fn", dtype="float64",
                                         initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.false_neg.assign(cm[1,0]/tf.keras.backend.sum(cm))

    def result(self):
        return self.false_neg

    def reset_states(self):
        self.false_neg.assign(0.0)

class TrueNegatives(tf.metrics.Metric):
    def __init__(self, name="true_neg", **kwargs):
        super(TrueNegatives, self).__init__(name=name, **kwargs)
        self.true_neg = self.add_weight(name="tn", dtype="float64",
                                        initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1 / y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.true_neg.assign(cm[0,0] / tf.keras.backend.sum(cm))

    def result(self):
        return self.true_neg

    def reset_states(self):
        self.true_neg.assign(0.0)
