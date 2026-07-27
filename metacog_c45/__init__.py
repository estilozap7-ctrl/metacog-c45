"""
=========================================================
MetaCog-C45  —  Public API Package
metacog_c45/__init__.py
=========================================================

Este módulo es la capa adaptadora pública del framework.
NO contiene lógica algorítmica — sólo re-exporta los
símbolos esenciales de los paquetes internos (core/,
metrics/, visualization/) para que los usuarios puedan
hacer:

    from metacog_c45 import C45Classifier

en lugar de:

    from core.classifier import C45Classifier

Los módulos internos permanecen intactos en sus ubicaciones
originales. Esta capa es una fachada (Facade Pattern) de
distribución, no una refactorización del código.
=========================================================
"""

# ── Versión del paquete ────────────────────────────────
from metacog_c45.version import (
    __version__,
    __author__,
    __description__,
    __license__,
)

# ── API pública: Clasificador principal ───────────────
from core.classifier import C45Classifier

# ── API pública: Poda ─────────────────────────────────
from core.pruning import DecisionTreePruner

# ── API pública: Capa metacognitiva ───────────────────
from core.metacognition.reflection_engine import ReflectionEngine
from core.metacognition.meta_memory import MetaMemory
from core.metacognition.decision_core import (
    DecisionValidator,
    SplitConfidenceScore,
    DecisionAudit,
)

# ── API pública: Métricas ─────────────────────────────
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator

# ── API pública: Visualización ────────────────────────
from visualization.feature_importance import FeatureImportance
from visualization.tree_plot import TreePlotter

# ── Símbolos exportados públicamente ──────────────────
__all__ = [
    # Versión
    "__version__",
    "__author__",
    "__description__",
    # Clasificador principal
    "C45Classifier",
    "DecisionTreePruner",
    # Metacognición
    "ReflectionEngine",
    "MetaMemory",
    "DecisionValidator",
    "SplitConfidenceScore",
    "DecisionAudit",
    # Métricas
    "ConfusionMatrix",
    "ClassificationMetrics",
    "ROCAUCCalculator",
    # Visualización
    "FeatureImportance",
    "TreePlotter",
]
