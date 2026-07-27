"""
=========================================================
PyC45 Framework
Competitive Estimator
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : PyC45
Descripción:
Evalúa la competencia local entre atributos candidatos
basándose en las diferencias relativas de sus Gain Ratios.
=========================================================
"""

from __future__ import annotations
from typing import Dict, Any


class CompetitiveEstimator:
    """
    Estima la competencia local de atributos en un nodo u.
    """

    def __init__(self, epsilon: float = 1e-6, competitive_threshold: float = 0.8):
        self.epsilon = epsilon
        self.competitive_threshold = competitive_threshold

    def estimate_competition(
        self,
        feature_gain_ratios: Dict[str, float],
        best_feature: str | None
    ) -> dict:
        """
        Calcula la brecha absoluta de gain ratio y el índice de competitividad CI(u).

        Parámetros:
        -----------
        feature_gain_ratios : Dict[str, float]
            Mapeo de nombre de característica a su mejor Gain Ratio en el nodo.
        best_feature : str | None
            El nombre del atributo seleccionado como óptimo.

        Retorna:
        --------
        dict:
            Metadatos conteniendo 'delta_gr', 'competitiveness_index' (CI) y 'is_competitive'.
        """
        if not feature_gain_ratios or best_feature is None or best_feature not in feature_gain_ratios:
            return {
                "delta_gr": 0.0,
                "competitiveness_index": 0.0,
                "is_competitive": False,
                "alternate_feature": None
            }

        gr_star = feature_gain_ratios[best_feature]
        
        # Si el mejor Gain Ratio es 0, no hay competencia real
        if gr_star <= 0.0:
            return {
                "delta_gr": 0.0,
                "competitiveness_index": 0.0,
                "is_competitive": False,
                "alternate_feature": None
            }

        # Buscar el mejor competidor de variable diferente
        alternate_feature = None
        gr_alternate = -1.0

        for feat, gr in feature_gain_ratios.items():
            if feat != best_feature:
                if gr > gr_alternate:
                    gr_alternate = gr
                    alternate_feature = feat

        # Si no hay alternativos válidos
        if alternate_feature is None or gr_alternate < 0.0:
            return {
                "delta_gr": gr_star,
                "competitiveness_index": 0.0,
                "is_competitive": False,
                "alternate_feature": None
            }

        delta_gr = float(gr_star - gr_alternate)
        
        # CI = 1.0 - (GR* - GR') / (GR* + epsilon)
        # Si GR* y GR' son idénticos, delta_gr = 0, entonces CI = 1.0 (máxima competencia)
        # Si GR' = 0, delta_gr = GR*, entonces CI = 1.0 - (GR* / (GR* + epsilon)) -> ≈ 0.0 (nula competencia)
        ci = 1.0 - (delta_gr / (gr_star + self.epsilon))
        ci = float(max(0.0, min(1.0, ci)))

        is_comp = ci >= self.competitive_threshold

        return {
            "delta_gr": delta_gr,
            "competitiveness_index": ci,
            "is_competitive": is_comp,
            "alternate_feature": alternate_feature
        }
