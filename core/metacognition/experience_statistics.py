"""
=========================================================
PyC45 Framework
ExperienceStatistics
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : PyC45
Descripción:
Módulo encargado de computar estadísticas consolidadas y métricas
de aprendizaje del árbol, incluyendo la importancia metacognitiva
de características (MFI).
=========================================================
"""

from __future__ import annotations
from typing import Dict, List, Any

from .meta_memory import MetaMemory
from .decision_trace import DecisionTrace


class ExperienceStatistics:
    """
    Analizador estadístico de la experiencia acumulada en la MetaMemory.
    """

    @staticmethod
    def calculate_average_ns(traces: List[DecisionTrace]) -> float:
        """
        Retorna el promedio de la Estabilidad de Nodo (NS).
        """
        if not traces:
            return 0.0
        return float(sum(t.ns for t in traces) / len(traces))

    @staticmethod
    def calculate_average_ts(traces: List[DecisionTrace]) -> float:
        """
        Retorna el promedio de la Supervivencia del Umbral (TS).
        """
        if not traces:
            return 0.0
        return float(sum(t.ts for t in traces) / len(traces))

    @staticmethod
    def calculate_metacognitive_feature_importance(
        traces: List[DecisionTrace],
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calcula la Importancia Metacognitiva de Características (MFI).

        MFI(X_j) = sum_{u in splits(X_j)} n_u * NS(u) * TS(u)
        """
        mfi: Dict[str, float] = {}
        
        if not traces:
            return mfi

        # Calcular importancia bruta
        for trace in traces:
            feat = trace.feature_selected
            # Peso del nodo = n_samples * ns * ts
            score = float(trace.n_samples * trace.ns * trace.ts)
            mfi[feat] = mfi.get(feat, 0.0) + score

        # Normalizar para que la suma sea 1.0 (opcional)
        if normalize:
            total = sum(mfi.values())
            if total > 0.0:
                mfi = {feat: float(score / total) for feat, score in mfi.items()}
            else:
                mfi = {feat: 0.0 for feat in mfi}

        return mfi

    @classmethod
    def get_summary(cls, memory: MetaMemory) -> Dict[str, Any]:
        """
        Genera un reporte resumido completo de la experiencia.
        """
        traces = memory.get_all_traces()
        if not traces:
            return {
                "total_decisions": 0,
                "avg_ns": 0.0,
                "avg_ts": 0.0,
                "mfi_normalized": {}
            }

        return {
            "total_decisions": len(traces),
            "avg_ns": cls.calculate_average_ns(traces),
            "avg_ts": cls.calculate_average_ts(traces),
            "mfi_normalized": cls.calculate_metacognitive_feature_importance(traces, normalize=True)
        }
