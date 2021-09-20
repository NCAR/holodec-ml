import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import tensorflow as tf
import pandas as pd 
import numpy as np

from hagelslag.evaluation.ProbabilityMetrics import *
from hagelslag.evaluation.MetricPlotter import *


class DistributedROC(DistributedROC):
    
    def binary_metrics(self, tol = 1e-12):
        
        TP = self.contingency_tables["TP"]
        FP = self.contingency_tables["FP"]
        TN = self.contingency_tables["TN"]
        FN = self.contingency_tables["FN"]
        
        metrics = {
            "precision": TP / (TP + FP + tol),
            "recall": TP / (TP + FN + tol),
            "specificity": TN / (TN + FP + tol),
            "accuracy": (TP + TN) / (TP + FP + FN + TN + tol)
        }
        metrics["f1"] = 2 * (metrics["recall"] * metrics["precision"]) / (metrics["recall"] + metrics["precision"] + tol)
        
        return metrics
    
    def from_str(self, in_str):
        """
        Read the DistributedROC string and parse the contingency table values from it.
        Args:
            in_str (str): The string output from the __str__ method
        """
        parts = in_str.split(";")
        for part in parts:
            var_name, value = part.split(":")
            if var_name == "Obs_Threshold":
                self.obs_threshold = float(value)
            elif var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
                self.contingency_tables = pd.DataFrame(columns=self.contingency_tables.columns,
                                                       data=np.zeros((self.thresholds.size,
                                                                     self.contingency_tables.columns.size)))
            elif var_name in self.contingency_tables.columns:
                self.contingency_tables[var_name] = np.array(value.split(), dtype=int)

    def __str__(self):
        """
        Output the information within the DistributedROC object to a string.
        """
        out_str = "Obs_Threshold:{0:0.8f}".format(self.obs_threshold) + ";"
        out_str += "Thresholds:" + " ".join(["{0:0.8f}".format(t) for t in self.thresholds]) + ";"
        for col in self.contingency_tables.columns:
            out_str += col + ":" + " ".join(["{0:d}".format(t) for t in self.contingency_tables[col]]) + ";"
        out_str = out_str.rstrip(";")
        return out_str
    

# class TruePositives(tf.metrics.Metric):
#     def __init__(self, name="true_pos", **kwargs):
#         super(TruePositives, self).__init__(name=name, **kwargs)
#         self.true_pos = self.add_weight(name="tp", dtype="float64",
#                                         initializer="zeros")
        
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
#         pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
#                                                  1, 0))
#         cm = tf.math.confusion_matrix(true, pred)
#         self.true_pos.assign(cm[1,1] / tf.keras.backend.sum(cm))

#     def result(self):
#         return self.true_pos

#     def reset_states(self):
#         self.true_pos.assign(0.0)

# class FalsePositives(tf.metrics.Metric):
#     def __init__(self, name="false_pos", **kwargs):
#         super(FalsePositives, self).__init__(name=name, **kwargs)
#         self.false_pos = self.add_weight(name="fp", dtype="float64",
#                                          initializer="zeros")
        
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
#         pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
#                                                  1, 0))
#         cm = tf.math.confusion_matrix(true, pred)
#         self.false_pos.assign(cm[0,1]/tf.keras.backend.sum(cm))

#     def result(self):
#         return self.false_pos

#     def reset_states(self):
#         self.false_pos.assign(0.0)

# class FalseNegatives(tf.metrics.Metric):
#     def __init__(self, name="false_neg", **kwargs):
#         super(FalseNegatives, self).__init__(name=name, **kwargs)
#         self.false_neg = self.add_weight(name="fn", dtype="float64",
#                                          initializer="zeros")
        
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
#         pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
#                                                  1, 0))
#         cm = tf.math.confusion_matrix(true, pred)
#         self.false_neg.assign(cm[1,0]/tf.keras.backend.sum(cm))

#     def result(self):
#         return self.false_neg

#     def reset_states(self):
#         self.false_neg.assign(0.0)

# class TrueNegatives(tf.metrics.Metric):
#     def __init__(self, name="true_neg", **kwargs):
#         super(TrueNegatives, self).__init__(name=name, **kwargs)
#         self.true_neg = self.add_weight(name="tn", dtype="float64",
#                                         initializer="zeros")
        
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
#         pred = tf.keras.backend.flatten(tf.where(y_pred > 1 / y_pred.shape[1],
#                                                  1, 0))
#         cm = tf.math.confusion_matrix(true, pred)
#         self.true_neg.assign(cm[0,0] / tf.keras.backend.sum(cm))

#     def result(self):
#         return self.true_neg

#     def reset_states(self):
#         self.true_neg.assign(0.0)
