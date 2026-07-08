"""
=========================================================
PyC45 Framework
Gain Ratio Calculator
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Implementa el cálculo del Gain Ratio utilizado
por el algoritmo C4.5.

Gain Ratio = Information Gain / Split Information
=========================================================
"""

from __future__ import annotations

from .information_gain import InformationGainCalculator


class GainRatioCalculator:
    """
    Calculadora del Gain Ratio.
    """

    # =====================================================
    # Gain Ratio
    # =====================================================

    @staticmethod
    def calculate(y_parent, y_left, y_right):

        gain = InformationGainCalculator.information_gain(
            y_parent,
            y_left,
            y_right
        )

        split = InformationGainCalculator.split_information(
            y_left,
            y_right
        )

        # Evitar división por cero
        if split == 0:
            return 0.0

        return gain / split

    # =====================================================
    # Resumen
    # =====================================================

    @staticmethod
    def summary(y_parent, y_left, y_right):

        gain = InformationGainCalculator.information_gain(
            y_parent,
            y_left,
            y_right
        )

        split = InformationGainCalculator.split_information(
            y_left,
            y_right
        )

        ratio = GainRatioCalculator.calculate(
            y_parent,
            y_left,
            y_right
        )

        print("=" * 60)
        print("GAIN RATIO")
        print("=" * 60)

        print(f"Information Gain : {gain:.6f}")
        print(f"Split Information: {split:.6f}")
        print(f"Gain Ratio       : {ratio:.6f}")

        print("=" * 60)

        return {
            "information_gain": gain,
            "split_information": split,
            "gain_ratio": ratio
        }