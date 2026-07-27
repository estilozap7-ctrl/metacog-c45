"""
=========================================================
MetaCog-C45 Framework
Classification Metrics
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Implementa las principales métricas de evaluación para
problemas de clasificación binaria.
=========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .confusion_matrix import ConfusionMatrix


class ClassificationMetrics:

    """
    Calcula métricas de clasificación a partir de una
    matriz de confusión.
    """

    def __init__(self, confusion_matrix):

        if not isinstance(confusion_matrix, ConfusionMatrix):

            raise TypeError(
                "Se esperaba un objeto ConfusionMatrix."
            )

        self.cm = confusion_matrix

    # =====================================================
    # Accuracy
    # =====================================================

    @property
    def accuracy(self):

        return (
            self.cm.tp + self.cm.tn
        ) / self.cm.total

    # =====================================================
    # Precision
    # =====================================================

    @property
    def precision(self):

        denominator = self.cm.tp + self.cm.fp

        if denominator == 0:
            return 0.0

        return self.cm.tp / denominator

    # =====================================================
    # Recall (Sensibilidad)
    # =====================================================

    @property
    def recall(self):

        denominator = self.cm.tp + self.cm.fn

        if denominator == 0:
            return 0.0

        return self.cm.tp / denominator

    # =====================================================
    # Specificity
    # =====================================================

    @property
    def specificity(self):

        denominator = self.cm.tn + self.cm.fp

        if denominator == 0:
            return 0.0

        return self.cm.tn / denominator

    # =====================================================
    # F1 Score
    # =====================================================

    @property
    def f1_score(self):

        p = self.precision
        r = self.recall

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    # =====================================================
    # Balanced Accuracy
    # =====================================================

    @property
    def balanced_accuracy(self):

        return (
            self.recall +
            self.specificity
        ) / 2

    # =====================================================
    # Error Rate
    # =====================================================

    @property
    def error_rate(self):

        return 1 - self.accuracy

    # =====================================================
    # Matthews Correlation Coefficient
    # =====================================================

    @property
    def mcc(self):

        tp = self.cm.tp
        tn = self.cm.tn
        fp = self.cm.fp
        fn = self.cm.fn

        numerator = (tp * tn) - (fp * fn)

        denominator = np.sqrt(

            (tp + fp) *

            (tp + fn) *

            (tn + fp) *

            (tn + fn)

        )

        if denominator == 0:
            return 0.0

        return numerator / denominator

    # =====================================================
    # DataFrame
    # =====================================================

    def dataframe(self):

        return pd.DataFrame({

            "Metric":[

                "Accuracy",

                "Precision",

                "Recall",

                "Specificity",

                "F1-Score",

                "Balanced Accuracy",

                "Error Rate",

                "Matthews CC"

            ],

            "Value":[

                self.accuracy,

                self.precision,

                self.recall,

                self.specificity,

                self.f1_score,

                self.balanced_accuracy,

                self.error_rate,

                self.mcc

            ]

        })

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        print("="*60)

        print("CLASSIFICATION METRICS")

        print("="*60)

        print(self.dataframe())

        print("="*60)