"""
=========================================================
MetaCog-C45 Framework
Decision Node
=========================================================

Autor : Luis Alberto Buelvas Cogollo
Proyecto : PyC45
Descripción:
Representa un nodo del árbol de decisión C4.5.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DecisionNode:
    """
    Nodo del árbol de decisión.

    Un nodo puede ser:

    • Nodo interno
    • Nodo hoja
    """

    # ==========================
    # División
    # ==========================

    feature: Optional[str] = None
    threshold: Optional[float] = None

    # ==========================
    # Calidad del nodo
    # ==========================

    gain_ratio: float = 0.0

    information_gain: float = 0.0

    entropy: float = 0.0

    # ==========================
    # Predicción
    # ==========================

    prediction: Optional[int] = None

    class_counts: Optional[dict] = None

    probability: float = 0.0

    # ==========================
    # Información del árbol
    # ==========================

    samples: int = 0

    depth: int = 0

    # ==========================
    # Hijos
    # ==========================

    left: Optional["DecisionNode"] = None

    right: Optional["DecisionNode"] = None

    # ==========================
    # Tipo de nodo
    # ==========================

    is_leaf: bool = False

    # ==========================
    # Referencia metacognitiva
    # Opcional: ID del DecisionTrace
    # asociado en MetaMemory.
    # Solo disponible en modo 'metacog'.
    # ==========================

    trace_id: Optional[str] = None

    # ===================================================

    def is_leaf_node(self):

        return self.is_leaf

    # ===================================================

    def has_children(self):

        return (
            self.left is not None
            or
            self.right is not None
        )

    # ===================================================

    def __str__(self):

        if self.is_leaf:

            return (
                f"Leaf("
                f"class={self.prediction}, "
                f"samples={self.samples})"
            )

        return (
            f"Node("
            f"feature={self.feature}, "
            f"threshold={self.threshold}, "
            f"gain_ratio={self.gain_ratio:.4f})"
        )

    __repr__ = __str__