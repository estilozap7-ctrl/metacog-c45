"""
=========================================================
PyC45 Framework
Confusion Matrix
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Implementa la matriz de confusión para problemas de
clasificación binaria.
=========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConfusionMatrix:
    """
    Matriz de confusión para clasificación binaria.
    """

    def __init__(self):

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    # =====================================================
    # Construcción
    # =====================================================

    def fit(self, y_true, y_pred):

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        self.tp = np.sum((y_true == 1) & (y_pred == 1))
        self.tn = np.sum((y_true == 0) & (y_pred == 0))
        self.fp = np.sum((y_true == 0) & (y_pred == 1))
        self.fn = np.sum((y_true == 1) & (y_pred == 0))

        return self

    # =====================================================
    # Constructor alternativo
    # =====================================================

    @classmethod
    def from_predictions(cls, y_true, y_pred):

        cm = cls()

        cm.fit(y_true, y_pred)

        return cm

    # =====================================================
    # Matriz
    # =====================================================

    @property
    def matrix(self):

        return np.array([

            [self.tn, self.fp],

            [self.fn, self.tp]

        ])

    # =====================================================
    # DataFrame
    # =====================================================

    def dataframe(self):

        return pd.DataFrame(

            self.matrix,

            index=[
                "Real 0",
                "Real 1"
            ],

            columns=[
                "Pred 0",
                "Pred 1"
            ]

        )

    # =====================================================
    # Total de registros
    # =====================================================

    @property
    def total(self):

        return self.tp + self.tn + self.fp + self.fn

    # =====================================================
    # Resumen
    # =====================================================

    def summary(self):

        print("=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)

        print(self.dataframe())

        print()

        print(f"TP : {self.tp}")
        print(f"TN : {self.tn}")
        print(f"FP : {self.fp}")
        print(f"FN : {self.fn}")

        print(f"Total : {self.total}")

        print("=" * 60)

    # =====================================================
    # Representación
    # =====================================================

    def __repr__(self):

        return (
            f"ConfusionMatrix("
            f"TP={self.tp}, "
            f"TN={self.tn}, "
            f"FP={self.fp}, "
            f"FN={self.fn})"
        )