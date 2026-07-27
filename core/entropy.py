"""
=========================================================
MetaCog-C45 Framework
Entropy Calculator
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45
Descripción:
Implementa el cálculo de la Entropía de Shannon para
problemas de clasificación.
=========================================================
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd


class EntropyCalculator:
    """
    Calculadora de Entropía de Shannon.
    """

    # =====================================================
    # Validación
    # =====================================================

    @staticmethod
    def validate(y):

        if y is None:
            raise ValueError("El vector de clases es None.")

        if len(y) == 0:
            raise ValueError("El vector de clases está vacío.")

    # =====================================================
    # Conteo de clases
    # =====================================================

    @staticmethod
    def class_counts(y):

        values, counts = np.unique(
            y,
            return_counts=True
        )

        return dict(zip(values, counts))

    # =====================================================
    # Probabilidades
    # =====================================================

    @staticmethod
    def probabilities(y):

        _, counts = np.unique(
            y,
            return_counts=True
        )

        probabilities = counts / counts.sum()

        return probabilities

    # =====================================================
    # Clase Mayoritaria
    # =====================================================

    @staticmethod
    def majority_class(y):

        EntropyCalculator.validate(y)

        values, counts = np.unique(
            y,
            return_counts=True
        )

        return values[np.argmax(counts)]

    # =====================================================
    # Entropía
    # =====================================================

    @staticmethod
    def calculate(y):

        EntropyCalculator.validate(y)

        probabilities = EntropyCalculator.probabilities(y)

        entropy = 0.0

        for p in probabilities:

            if p > 0:

                entropy -= p * np.log2(p)

        return float(entropy)

    # =====================================================
    # Pureza
    # =====================================================

    @staticmethod
    def purity(y):

        probabilities = EntropyCalculator.probabilities(y)

        return float(np.max(probabilities))

    # =====================================================
    # Nodo puro
    # =====================================================

    @staticmethod
    def is_pure(y):

        return EntropyCalculator.calculate(y) == 0

    # =====================================================
    # Resumen
    # =====================================================

    @staticmethod
    def summary(y):

        EntropyCalculator.validate(y)

        values, counts = np.unique(
            y,
            return_counts=True
        )

        probabilities = counts / counts.sum()

        df = pd.DataFrame({

            "Clase": values,

            "Frecuencia": counts,

            "Probabilidad": np.round(
                probabilities,
                4
            )

        })

        print("="*50)
        print("RESUMEN DE ENTROPÍA")
        print("="*50)

        print(df)

        print()

        print(
            f"Entropía : "
            f"{EntropyCalculator.calculate(y):.6f}"
        )

        print(
            f"Pureza   : "
            f"{EntropyCalculator.purity(y):.6f}"
        )

        print(
            f"Nodo puro: "
            f"{EntropyCalculator.is_pure(y)}"
        )

        return df