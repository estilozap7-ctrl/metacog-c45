"""
=========================================================
MetaCog-C45 Framework
Tree Builder (Dual Mode: Classic | MetaCog-C45)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Construye recursivamente el árbol de decisión.

Soporta dos modos de inducción:
  - 'classic': Algoritmo C4.5 estándar greedy sin cambios.
  - 'metacog': MetaCog-C45, integra el Decision Core
    (DecisionValidator, DecisionAudit, MetaMemory).

En ambos modos la lógica matemática base (Entropía, IG,
GR) es compartida. La bifurcación ocurre únicamente en
la política de decisión post-búsqueda.
=========================================================
"""

from __future__ import annotations

from typing import Optional, Any
import numpy as np

from .decision_node import DecisionNode
from .entropy import EntropyCalculator
from .best_split import BestSplitFinder


class TreeBuilder:
    """
    Constructor recursivo del árbol de decisión.

    Parámetros
    ----------
    max_depth : int, por defecto 20
        Profundidad máxima del árbol.
    min_samples_split : int, por defecto 5
        Mínimo de muestras para realizar una partición.
    min_gain_ratio : float, por defecto 0.001
        Ganancia mínima exigida en modo clásico.
    mode : str, por defecto 'classic'
        Modo de inducción. 'classic' activa el algoritmo C4.5
        estándar; 'metacog' activa el núcleo MetaCog-C45.
    top_k : int, por defecto 3
        Número de candidatos Top-K evaluados por el Decision
        Core. Solo relevante en modo 'metacog'.
    validator : DecisionValidator o None, por defecto None
        Instancia inyectada del validador metacognitivo.
        Solo relevante en modo 'metacog'.
    memory : MetaMemory o None, por defecto None
        Repositorio de experiencia metacognitiva inyectado.
        Solo relevante en modo 'metacog'.
    total_samples : int, por defecto 0
        Número total de muestras en la raíz del árbol.
        Requerido internamente por el SCS dinámico.
    """

    def __init__(
        self,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_gain_ratio: float = 0.001,
        mode: str = "classic",
        top_k: int = 3,
        validator: Any = None,
        memory: Any = None,
        total_samples: int = 0
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio
        self.mode = mode
        self.top_k = top_k
        self.validator = validator
        self.memory = memory
        self.total_samples = total_samples

        self.splitter = BestSplitFinder()

    # =====================================================
    # Factoría de hojas (compartida por ambos modos)
    # =====================================================

    def _make_leaf(self, y, depth: int) -> DecisionNode:
        """
        Crea un nodo hoja con la clase mayoritaria del subconjunto local.
        """
        counts = EntropyCalculator.class_counts(y)
        maj = EntropyCalculator.majority_class(y)
        return DecisionNode(
            prediction=maj,
            samples=len(y),
            depth=depth,
            entropy=EntropyCalculator.calculate(y),
            is_leaf=True,
            class_counts=counts,
            probability=counts.get(maj, 0) / len(y) if len(y) > 0 else 0.0
        )

    # =====================================================
    # Factoría de nodo interno (compartida por ambos modos)
    # =====================================================

    def _make_split_node(
        self,
        feature: str,
        threshold: float,
        gain_ratio: float,
        y,
        depth: int,
        trace_id: Optional[str] = None
    ) -> DecisionNode:
        """
        Crea un nodo interno con el split seleccionado.
        """
        counts = EntropyCalculator.class_counts(y)
        maj = EntropyCalculator.majority_class(y)
        return DecisionNode(
            feature=feature,
            threshold=threshold,
            gain_ratio=gain_ratio,
            entropy=EntropyCalculator.calculate(y),
            samples=len(y),
            depth=depth,
            class_counts=counts,
            probability=counts.get(maj, 0) / len(y) if len(y) > 0 else 0.0,
            prediction=maj,
            trace_id=trace_id
        )

    # =====================================================
    # Condiciones de parada comunes (compartidas)
    # =====================================================

    def _check_stopping_conditions(
        self,
        y,
        depth: int
    ) -> Optional[DecisionNode]:
        """
        Evalúa las condiciones de parada estructurales.
        Retorna un nodo hoja si alguna aplica, None en caso contrario.
        """
        if EntropyCalculator.is_pure(y):
            return self._make_leaf(y, depth)

        if len(y) < self.min_samples_split:
            return self._make_leaf(y, depth)

        if depth >= self.max_depth:
            return self._make_leaf(y, depth)

        return None

    # =====================================================
    # Construcción recursiva — modo clásico
    # =====================================================

    def _build_classic(self, X, y, depth: int) -> DecisionNode:
        """
        Construye el árbol en modo clásico C4.5 (greedy, sin metacognición).
        """
        # Condiciones de parada comunes
        leaf = self._check_stopping_conditions(y, depth)
        if leaf is not None:
            return leaf

        # Buscar el mejor split greedy (Top-1)
        feature, threshold, gain_ratio = self.splitter.find_best_split(X, y)

        # Ganancia insuficiente
        if gain_ratio < self.min_gain_ratio or feature is None:
            return self._make_leaf(y, depth)

        # Crear nodo y dividir
        node = self._make_split_node(feature, threshold, gain_ratio, y, depth)
        mask = X[feature] <= threshold
        node.left = self._build_classic(X.loc[mask], y.loc[mask], depth + 1)
        node.right = self._build_classic(X.loc[~mask], y.loc[~mask], depth + 1)

        return node

    # =====================================================
    # Construcción recursiva — modo MetaCog-C45
    # =====================================================

    def _build_metacog(self, X, y, depth: int, node_counter: list) -> DecisionNode:
        """
        Construye el árbol en modo MetaCog-C45, integrando el Decision Core.
        """
        # Importación diferida para evitar dependencias circulares
        from .metacognition.decision_core import DecisionPolicy, DecisionAudit

        # Condiciones de parada comunes
        leaf = self._check_stopping_conditions(y, depth)
        if leaf is not None:
            return leaf

        # System 1: obtener Top-K candidatos y mapa de GR por variable
        top_k_candidates, _ = self.splitter.find_top_k_splits(X, y, k=self.top_k)

        # ID único del nodo para la traza
        node_counter[0] += 1
        node_id = f"node_{node_counter[0]}_d{depth}"

        # System 2: validación metacognitiva del subespacio Top-K
        outcome = self.validator.validate(
            X_u=X,
            y_u=y,
            top_k_candidates=top_k_candidates,
            depth=depth,
            max_depth=self.max_depth,
            total_samples=self.total_samples,
            competitive_estimator=self.validator._competitive_estimator,
            reflection_engine=self.validator._reflection_engine
        )

        # Auditoría: construir la traza y persistirla en MetaMemory
        trace = DecisionAudit.audit_decision(
            node_id=node_id,
            X_u=X,
            y_u=y,
            depth=depth,
            outcome=outcome
        )
        if self.memory is not None:
            self.memory.store(trace)

        # Aplicar política de decisión
        if outcome.verdict in (DecisionPolicy.COLLAPSE,):
            # Colapso: nodo hoja por baja confianza o ganancia insuficiente
            leaf = self._make_leaf(y, depth)
            leaf.trace_id = node_id
            return leaf

        # ACCEPT / REVIEW / DEFER: proceder con el split seleccionado
        feature = outcome.feature_selected
        threshold = outcome.threshold_selected
        gain_ratio = outcome.gain_ratio

        node = self._make_split_node(feature, threshold, gain_ratio, y, depth, trace_id=node_id)
        mask = X[feature] <= threshold
        node.left = self._build_metacog(X.loc[mask], y.loc[mask], depth + 1, node_counter)
        node.right = self._build_metacog(X.loc[~mask], y.loc[~mask], depth + 1, node_counter)

        return node

    # =====================================================
    # Punto de entrada principal (despacho por modo)
    # =====================================================

    def build(self, X, y, depth: int = 0) -> DecisionNode:
        """
        Construye el árbol de decisión desde el nodo raíz.

        Parámetros
        ----------
        X : pd.DataFrame
            Variables predictoras.
        y : pd.Series
            Etiquetas de clase.
        depth : int, por defecto 0
            Profundidad inicial (siempre 0 en la llamada raíz).

        Retorna
        -------
        DecisionNode
            Nodo raíz del árbol construido.
        """
        if self.mode == "metacog":
            if self.validator is None:
                raise ValueError(
                    "mode='metacog' requiere una instancia de DecisionValidator inyectada."
                )
            # Contador mutable para IDs únicos de nodos
            node_counter = [0]
            return self._build_metacog(X, y, depth, node_counter)
        else:
            return self._build_classic(X, y, depth)