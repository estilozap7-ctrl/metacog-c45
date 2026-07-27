"""
=========================================================
MetaCog-C45 Framework
Best Split Finder
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Busca automáticamente el mejor atributo y el mejor
punto de corte utilizando Gain Ratio.

Soporta dos modos de retorno:
  - Clásico: retorna el split único de máximo GR.
  - Top-K: retorna los K mejores splits y el mapa de
    GR por variable, requerido por DecisionValidator.
=========================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from .gain_ratio import GainRatioCalculator


class BestSplitFinder:
    """
    Encuentra la mejor división del dataset.

    Soporta búsqueda codiciosa clásica (Top-1) y búsqueda
    Top-K de candidatos para el Decision Core de MetaCog-C45.
    """

    def __init__(self):
        self.best_feature = None
        self.best_threshold = None
        self.best_gain_ratio = -1

    # =====================================================
    # Posibles puntos de corte
    # =====================================================

    @staticmethod
    def candidate_thresholds(values: np.ndarray) -> List[float]:
        """
        Calcula los puntos de corte candidatos como promedios de valores adyacentes.

        Parámetros
        ----------
        values : np.ndarray
            Vector de valores de la característica.

        Retorna
        -------
        List[float]
            Lista de umbrales candidatos.
        """
        unique_vals = np.sort(np.unique(values))
        if len(unique_vals) <= 1:
            return []
        return ((unique_vals[:-1] + unique_vals[1:]) / 2.0).tolist()

    # =====================================================
    # Búsqueda de top-K splits candidatos (MetaCog-C45)
    # =====================================================

    def find_top_k_splits(
        self,
        X,
        y,
        k: int = 3
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Encuentra los K mejores splits candidatos y el mapa de GR por variable.

        Evalúa todos los umbrales posibles de todas las variables y retorna
        los K splits de mayor Gain Ratio (sin duplicar variables), además
        del mejor GR individual de cada variable para el CompetitiveEstimator.

        Parámetros
        ----------
        X : pd.DataFrame
            Variables predictoras locales del nodo u.
        y : pd.Series
            Etiquetas de clase locales.
        k : int, por defecto 3
            Cantidad máxima de candidatos a retornar en el Top-K.

        Retorna
        -------
        Tuple[List[Dict], Dict[str, float]]
            - top_k: Lista de diccionarios con los mejores K splits ordenados
              de mayor a menor GR. Cada elemento contiene 'feature', 'threshold',
              'gain_ratio'.
            - feature_best_gr: Diccionario {feature: best_gain_ratio} con el
              mejor GR encontrado para cada variable (insumo para CompetitiveEstimator).
        """
        # Mejor split encontrado por variable
        feature_best_split: Dict[str, Dict] = {}
        feature_best_gr: Dict[str, float] = {}

        # Convertir a numpy array una sola vez para evitar overhead de pandas indexers
        y_values = y.values if hasattr(y, 'values') else np.asarray(y)

        for feature in X.columns:
            feat_values = X[feature].values
            thresholds = self.candidate_thresholds(feat_values)

            best_gr_feature = -1.0
            best_thresh_feature = None

            for threshold in thresholds:
                mask = feat_values <= threshold
                y_left = y_values[mask]
                y_right = y_values[~mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain_ratio = GainRatioCalculator.calculate(y_values, y_left, y_right)

                if gain_ratio > best_gr_feature:
                    best_gr_feature = gain_ratio
                    best_thresh_feature = threshold

            if best_thresh_feature is not None:
                feature_best_split[feature] = {
                    "feature": feature,
                    "threshold": best_thresh_feature,
                    "gain_ratio": best_gr_feature
                }
                feature_best_gr[feature] = best_gr_feature

        # Ordenar por GR descendente y retornar Top-K
        all_splits = sorted(
            feature_best_split.values(),
            key=lambda x: x["gain_ratio"],
            reverse=True
        )
        top_k = all_splits[:k]

        return top_k, feature_best_gr

    # =====================================================
    # Buscar mejor división (modo clásico, backward-compat)
    # =====================================================

    def find_best_split(
        self,
        X,
        y
    ) -> Tuple[Optional[str], Optional[float], float]:
        """
        Busca el split de máximo Gain Ratio (modo clásico C4.5).

        Es un alias de ``find_top_k_splits(k=1)`` para mantener
        compatibilidad hacia atrás con el clasificador clásico C4.5.

        Parámetros
        ----------
        X : pd.DataFrame
            Variables predictoras locales del nodo u.
        y : pd.Series
            Etiquetas de clase locales.

        Retorna
        -------
        Tuple[Optional[str], Optional[float], float]
            (best_feature, best_threshold, best_gain_ratio)
        """
        top_k, _ = self.find_top_k_splits(X, y, k=1)

        if not top_k:
            return None, None, -1.0

        best = top_k[0]
        self.best_feature = best["feature"]
        self.best_threshold = best["threshold"]
        self.best_gain_ratio = best["gain_ratio"]

        return self.best_feature, self.best_threshold, self.best_gain_ratio