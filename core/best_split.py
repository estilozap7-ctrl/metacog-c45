"""
=========================================================
PyC45 Framework
Best Split Finder
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Busca automáticamente el mejor atributo y el mejor
punto de corte utilizando Gain Ratio.
=========================================================
"""

from __future__ import annotations

import numpy as np

from .gain_ratio import GainRatioCalculator


class BestSplitFinder:
    """
    Encuentra la mejor división del dataset.
    """

    def __init__(self):

        self.best_feature = None
        self.best_threshold = None
        self.best_gain_ratio = -1

    # =====================================================
    # Posibles puntos de corte
    # =====================================================

    @staticmethod
    def candidate_thresholds(values):

        values = np.sort(np.unique(values))

        thresholds = []

        for i in range(len(values)-1):

            threshold = (
                values[i] + values[i+1]
            ) / 2

            thresholds.append(threshold)

        return thresholds

    # =====================================================
    # Buscar mejor división
    # =====================================================

    def find_best_split(self, X, y):

        self.best_feature = None
        self.best_threshold = None
        self.best_gain_ratio = -1

        # recorrer todas las variables
        for feature in X.columns:

            thresholds = self.candidate_thresholds(
                X[feature]
            )

            # probar todos los cortes
            for threshold in thresholds:

                mask = X[feature] <= threshold

                y_left = y[mask]

                y_right = y[~mask]

                # evitar nodos vacíos
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain_ratio = GainRatioCalculator.calculate(
                    y,
                    y_left,
                    y_right
                )

                if gain_ratio > self.best_gain_ratio:

                    self.best_gain_ratio = gain_ratio

                    self.best_feature = feature

                    self.best_threshold = threshold

        return (
            self.best_feature,
            self.best_threshold,
            self.best_gain_ratio
        )