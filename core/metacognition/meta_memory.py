"""
=========================================================
MetaCog-C45 Framework
MetaMemory and MemoryIndex
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : MetaCog-C45
Descripción:
Repositorio inteligente encargado de almacenar y recuperar
DecisionTraces de manera estructurada e indexada.
=========================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any

from .decision_trace import DecisionTrace


class MemoryIndex:
    """
    Índice inteligente para la recuperación de experiencias similares
    basado en métricas de características y perfiles de datos.
    """

    def __init__(self, w1: float = 0.5, w2: float = 0.5, epsilon: float = 1e-6):
        self.w1 = w1
        self.w2 = w2
        self.epsilon = epsilon
        self.by_feature: Dict[str, List[DecisionTrace]] = {}
        self.by_depth: Dict[int, List[DecisionTrace]] = {}

    def index_trace(self, trace: DecisionTrace) -> None:
        """
        Indexa un nuevo DecisionTrace.
        """
        # Indexación por característica
        feat = trace.feature_selected
        if feat not in self.by_feature:
            self.by_feature[feat] = []
        self.by_feature[feat].append(trace)

        # Indexación por profundidad
        depth = trace.depth
        if depth not in self.by_depth:
            self.by_depth[depth] = []
        self.by_depth[depth].append(trace)

    def clear(self) -> None:
        """
        Limpia los índices.
        """
        self.by_feature.clear()
        self.by_depth.clear()

    def _normalize_distribution(self, class_dist: Dict[Any, int]) -> Dict[Any, float]:
        """
        Normaliza la distribución de frecuencias de clases a probabilidades.
        """
        total = sum(class_dist.values())
        if total <= 0:
            return {}
        return {c: float(count / total) for c, count in class_dist.items()}

    def _compute_distance(
        self,
        n_q: int,
        p_q: Dict[Any, float],
        trace: DecisionTrace
    ) -> float:
        """
        Calcula la distancia total ponderada entre el query y un trace.
        """
        # 1. Distancia de tamaño de nodo
        n_t = trace.n_samples
        d_size = abs(n_q - n_t) / (max(n_q, n_t) + self.epsilon)

        # 2. Distancia de distribución de clases (Euclídea)
        p_t = self._normalize_distribution(trace.class_distribution)
        
        # Unificar llaves de clase
        all_classes = set(p_q.keys()).union(set(p_t.keys()))
        d_class_sq = 0.0
        for c in all_classes:
            prob_q = p_q.get(c, 0.0)
            prob_t = p_t.get(c, 0.0)
            d_class_sq += (prob_q - prob_t) ** 2
        d_class = np.sqrt(d_class_sq)

        return float(self.w1 * d_size + self.w2 * d_class)

    def find_nearest_neighbors(
        self,
        n_samples: int,
        class_distribution: Dict[Any, int],
        all_traces: List[DecisionTrace],
        k: int = 3
    ) -> List[DecisionTrace]:
        """
        Encuentra las K trazas más cercanas en base al perfil del nodo.
        """
        if not all_traces:
            return []

        p_q = self._normalize_distribution(class_distribution)

        # Calcular distancias para todas las trazas
        scored_traces = []
        for trace in all_traces:
            dist = self._compute_distance(n_samples, p_q, trace)
            scored_traces.append((dist, trace))

        # Ordenar por distancia ascendente
        scored_traces.sort(key=lambda x: x[0])

        return [trace for _, trace in scored_traces[:k]]


class MetaMemory:
    """
    Repositorio central de conocimiento metacognitivo del árbol de decisión.
    """

    def __init__(self, w1: float = 0.5, w2: float = 0.5):
        self.traces: List[DecisionTrace] = []
        self.index = MemoryIndex(w1=w1, w2=w2)

    def store(self, trace: DecisionTrace) -> None:
        """
        Guarda una nueva traza de decisión y la indexa.
        """
        self.traces.append(trace)
        self.index.index_trace(trace)

    def get_all_traces(self) -> List[DecisionTrace]:
        """
        Retorna la lista completa de trazas almacenadas.
        """
        return self.traces

    def query_by_feature(self, feature: str) -> List[DecisionTrace]:
        """
        Recupera trazas de decisión que usaron la característica indicada.
        """
        return self.index.by_feature.get(feature, [])

    def query_by_depth(self, depth: int) -> List[DecisionTrace]:
        """
        Recupera trazas de decisión en una profundidad específica.
        """
        return self.index.by_depth.get(depth, [])

    def query_similar(
        self,
        n_samples: int,
        class_distribution: Dict[Any, int],
        k: int = 3
    ) -> List[DecisionTrace]:
        """
        Realiza una búsqueda de vecinos más cercanos sobre la memoria.
        """
        return self.index.find_nearest_neighbors(
            n_samples=n_samples,
            class_distribution=class_distribution,
            all_traces=self.traces,
            k=k
        )

    def clear(self) -> None:
        """
        Limpia completamente el repositorio de memoria y sus índices.
        """
        self.traces.clear()
        self.index.clear()

    def size(self) -> int:
        """
        Retorna la cantidad total de trazas en memoria.
        """
        return len(self.traces)
