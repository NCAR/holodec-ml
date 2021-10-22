import pandas as pd
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DistributedROC(DistributedROC):

    def binary_metrics(self, tol=1e-12):

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
        metrics["f1"] = 2 * (metrics["recall"] * metrics["precision"]) / \
            (metrics["recall"] + metrics["precision"] + tol)

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
                self.contingency_tables[var_name] = np.array(
                    value.split(), dtype=int)

    def __str__(self):
        """
        Output the information within the DistributedROC object to a string.
        """
        out_str = "Obs_Threshold:{0:0.8f}".format(self.obs_threshold) + ";"
        out_str += "Thresholds:" + \
            " ".join(["{0:0.8f}".format(t) for t in self.thresholds]) + ";"
        for col in self.contingency_tables.columns:
            out_str += col + ":" + \
                " ".join(["{0:d}".format(t)
                         for t in self.contingency_tables[col]]) + ";"
        out_str = out_str.rstrip(";")
        return out_str
