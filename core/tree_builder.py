"""
=========================================================
PyC45 Framework
Tree Builder
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Construye recursivamente el árbol de decisión C4.5.
=========================================================
"""

from __future__ import annotations

import numpy as np

from .decision_node import DecisionNode
from .entropy import EntropyCalculator
from .best_split import BestSplitFinder


class TreeBuilder:

    def __init__(
        self,
        max_depth=20,
        min_samples_split=5,
        min_gain_ratio=0.001
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio

        self.splitter = BestSplitFinder()

    # =====================================================
    # Construcción recursiva
    # =====================================================

    def build(self, X, y, depth=0):

        # --------------------------------------------
        # Nodo puro
        # --------------------------------------------

        if EntropyCalculator.is_pure(y):

            counts = EntropyCalculator.class_counts(y)
            maj = EntropyCalculator.majority_class(y)

            return DecisionNode(

                prediction=maj,

                samples=len(y),

                depth=depth,

                entropy=0.0,

                is_leaf=True,

                class_counts=counts,

                probability=1.0

            )

        # --------------------------------------------
        # Muy pocos registros
        # --------------------------------------------

        if len(y) < self.min_samples_split:

            counts = EntropyCalculator.class_counts(y)
            maj = EntropyCalculator.majority_class(y)

            return DecisionNode(

                prediction=maj,

                samples=len(y),

                depth=depth,

                entropy=EntropyCalculator.calculate(y),

                is_leaf=True,

                class_counts=counts,

                probability=counts.get(maj, 0) / len(y) if len(y) > 0 else 0.0

            )

        # --------------------------------------------
        # Profundidad máxima
        # --------------------------------------------

        if depth >= self.max_depth:

            counts = EntropyCalculator.class_counts(y)
            maj = EntropyCalculator.majority_class(y)

            return DecisionNode(

                prediction=maj,

                samples=len(y),

                depth=depth,

                entropy=EntropyCalculator.calculate(y),

                is_leaf=True,

                class_counts=counts,

                probability=counts.get(maj, 0) / len(y) if len(y) > 0 else 0.0

            )

        # --------------------------------------------
        # Buscar mejor división
        # --------------------------------------------

        feature, threshold, gain_ratio = \
            self.splitter.find_best_split(X, y)

        # --------------------------------------------
        # Ganancia insuficiente
        # --------------------------------------------

        if gain_ratio < self.min_gain_ratio or feature is None:

            counts = EntropyCalculator.class_counts(y)
            maj = EntropyCalculator.majority_class(y)

            return DecisionNode(

                prediction=maj,

                samples=len(y),

                depth=depth,

                entropy=EntropyCalculator.calculate(y),

                gain_ratio=gain_ratio,

                is_leaf=True,

                class_counts=counts,

                probability=counts.get(maj, 0) / len(y) if len(y) > 0 else 0.0

            )

        # --------------------------------------------
        # Dividir dataset
        # --------------------------------------------

        mask = X[feature] <= threshold

        X_left = X.loc[mask]

        y_left = y.loc[mask]

        X_right = X.loc[~mask]

        y_right = y.loc[~mask]

        # --------------------------------------------
        # Crear nodo
        # --------------------------------------------

        counts = EntropyCalculator.class_counts(y)
        maj = EntropyCalculator.majority_class(y)

        node = DecisionNode(

            feature=feature,

            threshold=threshold,

            gain_ratio=gain_ratio,

            entropy=EntropyCalculator.calculate(y),

            samples=len(y),

            depth=depth,

            class_counts=counts,

            probability=counts.get(maj, 0) / len(y) if len(y) > 0 else 0.0,

            prediction=maj

        )

        # --------------------------------------------
        # Construcción recursiva
        # --------------------------------------------

        node.left = self.build(

            X_left,

            y_left,

            depth + 1

        )

        node.right = self.build(

            X_right,

            y_right,

            depth + 1

        )

        return node