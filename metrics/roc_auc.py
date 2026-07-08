"""
=========================================================
PyC45 Framework
ROC and AUC Evaluation
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : PyC45

Descripción:
Implementa el cálculo y visualización de la curva ROC y el área bajo la curva (AUC).
=========================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class ROCAUCCalculator:
    """
    Calculadora de la curva ROC y el Área Bajo la Curva (AUC) para clasificación binaria.
    """

    @staticmethod
    def calculate(y_true, y_probs):
        """
        Calcula las tasas de falsos positivos (FPR) y verdaderos positivos (TPR),
        además del valor de AUC usando la integración por el método del trapecio.

        Parámetros:
        -----------
        y_true : array-like de forma (n_samples,)
            Etiquetas reales de clase (0 o 1).
        y_probs : array-like de forma (n_samples,)
            Probabilidades de predicción para la clase positiva (1).

        Retorna:
        --------
        dict
            Diccionario que contiene 'fpr', 'tpr', 'thresholds' y el valor 'auc'.
        """
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)

        # Si todas las etiquetas son iguales, AUC no está definido o es 0.5/0.0
        if len(np.unique(y_true)) < 2:
            return {
                "fpr": np.array([0.0, 1.0]),
                "tpr": np.array([0.0, 1.0]),
                "thresholds": np.array([0.5, 0.5]),
                "auc": 0.5
            }

        # Ordenar las probabilidades en orden descendente
        desc_score_indices = np.argsort(y_probs)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        y_probs_sorted = y_probs[desc_score_indices]

        # Encontrar umbrales únicos
        distinct_value_indices = np.where(np.diff(y_probs_sorted))[0]
        threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

        # Sumas acumuladas de verdaderos positivos (TP) y falsos positivos (FP)
        tps_cum = np.cumsum(y_true_sorted)
        fps_cum = (1 + np.arange(y_true_sorted.size)) - tps_cum

        # Obtener los acumulados en los índices de umbrales
        tps = tps_cum[threshold_indices]
        fps = fps_cum[threshold_indices]

        # Agregar el punto de origen (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        thresholds = np.r_[y_probs_sorted[0] + 1.0, y_probs_sorted[threshold_indices]]

        total_positives = tps[-1]
        total_negatives = fps[-1]

        tpr = tps / total_positives if total_positives > 0 else np.zeros_like(tps)
        fpr = fps / total_negatives if total_negatives > 0 else np.zeros_like(fps)

        # Calcular el AUC mediante la regla del trapecio
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1]) / 2.0

        # AUC debe ser positivo y acotado por [0.0, 1.0]
        auc = float(np.clip(auc, 0.0, 1.0))

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc
        }

    @staticmethod
    def plot(roc_data, title="Curva ROC - PyC45", figsize=(8, 6)):
        """
        Grafica la curva ROC a partir de los datos de evaluación calculados.

        Parámetros:
        -----------
        roc_data : dict
            Diccionario devuelto por el método calculate.
        title : str
            Título de la gráfica.
        figsize : tuple
            Tamaño de la figura.
        """
        plt.figure(figsize=figsize)
        plt.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            color="#E67E22",
            lw=2.5,
            label=f"Curva ROC (AUC = {roc_data['auc']:.4f})"
        )
        plt.plot([0, 1], [0, 1], color="#2C3E50", lw=1.5, linestyle="--", alpha=0.7)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("Tasa de Falsos Positivos (FPR)", fontsize=10)
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=10)
        plt.title(title, fontsize=12, fontweight="bold", pad=15)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.show()
