"""
=========================================================
PyC45 Framework
C45 Classifier (Prediction Alias)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Clase principal del Framework PyC45.
=========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .tree_builder import TreeBuilder
from .tree_utils import TreeUtils
from .decision_node import DecisionNode


class C45Classifier:

    """
    Implementación del algoritmo C4.5
    """

    # =====================================================
    # Constructor
    # =====================================================

    def __init__(
        self,
        max_depth=20,
        min_samples_split=5,
        min_gain_ratio=0.001
    ):

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.min_gain_ratio = min_gain_ratio

        self.root = None

        self.builder = TreeBuilder(

            max_depth=max_depth,

            min_samples_split=min_samples_split,

            min_gain_ratio=min_gain_ratio

        )

    # =====================================================
    # Entrenamiento
    # =====================================================

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.root = self.builder.build(X, y)

        return self

    # =====================================================
    # Predicción individual
    # =====================================================

    def _predict_sample(self, node, sample):

        while not node.is_leaf:

            if sample[node.feature] <= node.threshold:

                node = node.left

            else:

                node = node.right

        return node.prediction

    # =====================================================
    # Predicción
    # =====================================================

    def predict(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        predictions = []

        for _, sample in X.iterrows():

            predictions.append(

                self._predict_sample(

                    self.root,

                    sample

                )

            )

        return np.array(predictions)

    # =====================================================
    # Probabilidad de predicción
    # =====================================================

    def _predict_proba_sample(self, node, sample):

        while not node.is_leaf:

            if sample[node.feature] <= node.threshold:

                node = node.left

            else:

                node = node.right

        counts = node.class_counts or {}
        total = sum(counts.values())
        if total == 0:
            return np.array([0.0, 0.0])  # fallback for binary classification
        
        # En caso de clasificación binaria estándar, devolvemos las probabilidades de cada clase
        # Si las clases en el dataset no están ordenadas, ordenamos las llaves
        classes = sorted(list(counts.keys()))
        if len(classes) == 1:
            # solo una clase presente
            only_class = classes[0]
            if only_class == 0:
                return np.array([1.0, 0.0])
            else:
                return np.array([0.0, 1.0])
        
        return np.array([counts.get(c, 0) / total for c in classes])

    def predict_proba(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        probas = []

        for _, sample in X.iterrows():

            probas.append(

                self._predict_proba_sample(

                    self.root,

                    sample

                )

            )

        return np.array(probas)

    # =====================================================
    # Accuracy
    # =====================================================

    def score(self, X, y):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        predictions = self.predict(X)

        accuracy = np.mean(predictions == y)

        return accuracy

    # =====================================================
    # Información del árbol
    # =====================================================

    def summary(self):

        print("=" * 60)

        print("PyC45 CLASSIFIER")

        print("=" * 60)

        print(
            "Número de nodos:",
            TreeUtils.count_nodes(self.root)
        )

        print(
            "Número de hojas:",
            TreeUtils.count_leaves(self.root)
        )

        print(
            "Profundidad:",
            TreeUtils.depth(self.root)
        )

        print("=" * 60)

    # =====================================================
    # Mostrar árbol
    # =====================================================

    def print_tree(self):

        TreeUtils.print_tree(self.root)