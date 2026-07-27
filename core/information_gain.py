"""
=========================================================
MetaCog-C45 Framework
Information Gain Calculator
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Implementa el cálculo de la Ganancia de Información y
el Split Information utilizados por el algoritmo C4.5.
=========================================================
"""

from __future__ import annotations

import numpy as np

from .entropy import EntropyCalculator


class InformationGainCalculator:
    """
    Calculadora de Ganancia de Información.
    """

    # =====================================================
    # Validación
    # =====================================================

    @staticmethod
    def validate(y_parent, y_left, y_right):

        EntropyCalculator.validate(y_parent)
        EntropyCalculator.validate(y_left)
        EntropyCalculator.validate(y_right)

    # =====================================================
    # Entropía ponderada
    # =====================================================

    @staticmethod
    def weighted_entropy(y_left, y_right):

        total = len(y_left) + len(y_right)

        w_left = len(y_left) / total
        w_right = len(y_right) / total

        entropy_left = EntropyCalculator.calculate(y_left)
        entropy_right = EntropyCalculator.calculate(y_right)

        return (
            w_left * entropy_left +
            w_right * entropy_right
        )

    # =====================================================
    # Ganancia de Información
    # =====================================================

    @staticmethod
    def information_gain(y_parent, y_left, y_right):

        InformationGainCalculator.validate(
            y_parent,
            y_left,
            y_right
        )

        parent_entropy = EntropyCalculator.calculate(
            y_parent
        )

        weighted = InformationGainCalculator.weighted_entropy(
            y_left,
            y_right
        )

        gain = parent_entropy - weighted

        return float(gain)

    # =====================================================
    # Split Information
    # =====================================================

    @staticmethod
    def split_information(y_left, y_right):

        total = len(y_left) + len(y_right)

        p_left = len(y_left) / total
        p_right = len(y_right) / total

        split = 0.0

        for p in [p_left, p_right]:

            if p > 0:

                split -= p * np.log2(p)

        return float(split)

    # =====================================================
    # Resumen
    # =====================================================

    @staticmethod
    def summary(y_parent, y_left, y_right):

        parent_entropy = EntropyCalculator.calculate(
            y_parent
        )

        entropy_left = EntropyCalculator.calculate(
            y_left
        )

        entropy_right = EntropyCalculator.calculate(
            y_right
        )

        weighted = InformationGainCalculator.weighted_entropy(
            y_left,
            y_right
        )

        gain = InformationGainCalculator.information_gain(
            y_parent,
            y_left,
            y_right
        )

        split = InformationGainCalculator.split_information(
            y_left,
            y_right
        )

        print("=" * 60)
        print("INFORMATION GAIN")
        print("=" * 60)

        print(f"Entropía Padre     : {parent_entropy:.6f}")
        print(f"Entropía Izquierda : {entropy_left:.6f}")
        print(f"Entropía Derecha   : {entropy_right:.6f}")
        print(f"Entropía Ponderada : {weighted:.6f}")
        print(f"Information Gain   : {gain:.6f}")
        print(f"Split Information  : {split:.6f}")

        print("=" * 60)

        return {
            "parent_entropy": parent_entropy,
            "weighted_entropy": weighted,
            "information_gain": gain,
            "split_information": split
        }