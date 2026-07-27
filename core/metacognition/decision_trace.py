"""
=========================================================
PyC45 Framework
Metacognitive Decision Trace (Enriched)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : PyC45
Descripción:
Clase/Dataclass que registra el historial cognitivo completo y reproducible
de una decisión del árbol de decisión.
=========================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Dict, List, Any


@dataclass
class DecisionTrace:
    """
    Representa una traza de auditoría cognitiva detallada y reproducible
    de la decisión de división de un nodo.
    """
    node_id: str
    depth: int
    n_samples: int
    class_distribution: Dict[Any, int]
    feature_selected: str
    threshold_selected: float | None
    gain_ratio: float
    ns: float  # Node Stability
    ts: float  # Threshold Survival
    p_star: float
    
    # Parámetros del subsistema de memoria y decisión
    competitiveness_index: float = 0.0
    is_competitive: bool = False
    alternate_feature: str | None = None
    scs: float = 0.0
    verdict: str = "ACCEPT"
    justification: str = ""
    uncertainty: float = 0.0
    top_k_candidates: List[Dict[str, Any]] = field(default_factory=list)
    algorithm_version: str = "MetaCog-C45 v1.0"
    
    # Estructuras secundarias de simulación
    p_vector: Dict[str, float] = field(default_factory=dict)
    bootstrap_selections: List[str] = field(default_factory=list)
    bootstrap_thresholds: List[float] = field(default_factory=list)
    
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """
        Retorna la representación de la traza en un diccionario plano para serialización.
        """
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "n_samples": self.n_samples,
            "class_distribution": self.class_distribution,
            "feature_selected": self.feature_selected,
            "threshold_selected": self.threshold_selected,
            "gain_ratio": self.gain_ratio,
            "ns": self.ns,
            "ts": self.ts,
            "p_star": self.p_star,
            "competitiveness_index": self.competitiveness_index,
            "is_competitive": self.is_competitive,
            "alternate_feature": self.alternate_feature,
            "scs": self.scs,
            "verdict": self.verdict,
            "justification": self.justification,
            "uncertainty": self.uncertainty,
            "top_k_candidates": self.top_k_candidates,
            "algorithm_version": self.algorithm_version,
            "p_vector": self.p_vector,
            "bootstrap_selections": self.bootstrap_selections,
            "bootstrap_thresholds": self.bootstrap_thresholds,
            "timestamp": self.timestamp,
        }
