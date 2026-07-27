"""
=========================================================
MetaCog-C45 Framework
Feature Importance
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Calcula y visualiza la importancia de las variables
utilizando el Gain Ratio acumulado del árbol.
=========================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class FeatureImportance:

    def __init__(self):

        self.importance = {}

    # =====================================================
    # Recorrido del árbol
    # =====================================================

    def _visit(self, node):

        if node is None:
            return

        if not node.is_leaf:

            feature = node.feature

            if feature not in self.importance:

                self.importance[feature] = 0.0

            self.importance[feature] += node.gain_ratio

        self._visit(node.left)
        self._visit(node.right)

    # =====================================================
    # Calcular importancia
    # =====================================================

    def fit(self, root):

        self.importance = {}

        self._visit(root)

        total = sum(self.importance.values())

        if total > 0:

            for feature in self.importance:

                self.importance[feature] /= total

        return self

    # =====================================================
    # DataFrame
    # =====================================================

    def dataframe(self):

        df = pd.DataFrame({

            "Feature": list(self.importance.keys()),

            "Importance": list(self.importance.values())

        })

        df = df.sort_values(

            "Importance",

            ascending=False

        )

        df.reset_index(

            drop=True,

            inplace=True

        )

        return df

    # =====================================================
    # Gráfico
    # =====================================================

    def plot(self, figsize=(10,6)):

        df = self.dataframe()

        plt.figure(figsize=figsize)

        plt.barh(

            df["Feature"],

            df["Importance"]

        )

        plt.gca().invert_yaxis()

        plt.xlabel("Importancia")

        plt.ylabel("Variable")

        plt.title("Feature Importance")

        plt.grid(axis="x", alpha=.3)

        plt.show()