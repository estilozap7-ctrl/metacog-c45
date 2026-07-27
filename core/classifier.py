"""
=========================================================
MetaCog-C45 Framework
C45Classifier — Dual Mode (Classic | MetaCog-C45)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Clase pública principal del framework PyC45 / MetaCog-C45.
API compatible con scikit-learn (fit, predict, predict_proba,
score).

Soporta dos modos de inducción mediante el parámetro 'mode':
  - 'classic' (por defecto): C4.5 estándar. Comportamiento
    100% idéntico al PyC45 original. Sin cambios en la API.
  - 'metacog': MetaCog-C45. Activa el Decision Core completo.
    Expone atributos memory_ y experience_stats_ tras el fit.
=========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .tree_builder import TreeBuilder
from .tree_utils import TreeUtils
from .decision_node import DecisionNode


class C45Classifier:
    """
    Clasificador de árbol de decisión basado en el algoritmo C4.5.

    Compatible con la API de scikit-learn (``fit``, ``predict``,
    ``predict_proba``, ``score``).

    Parámetros
    ----------
    max_depth : int, por defecto 20
        Profundidad máxima del árbol.
    min_samples_split : int, por defecto 5
        Mínimo de muestras para dividir un nodo.
    min_gain_ratio : float, por defecto 0.001
        Gain Ratio mínimo aceptado en modo 'classic'.
    mode : str, por defecto 'classic'
        Modo de inducción.
        - ``'classic'``: C4.5 estándar, sin metacognición.
        - ``'metacog'``: MetaCog-C45 con Decision Core.
    B : int, por defecto 50
        Número de réplicas bootstrap. Solo relevante en modo 'metacog'.
    random_state : int o None, por defecto None
        Semilla de aleatoriedad. Solo relevante en modo 'metacog'.
    top_k : int, por defecto 3
        Número de candidatos Top-K evaluados. Solo en modo 'metacog'.
    theta_accept : float, por defecto 0.5
        Umbral de aceptación del SCS. Solo en modo 'metacog'.
    theta_reject : float, por defecto 0.2
        Umbral de colapso del SCS. Solo en modo 'metacog'.
    theta_comp : float, por defecto 0.75
        Umbral de competencia (CI). Solo en modo 'metacog'.

    Atributos (post-fit)
    --------------------
    root : DecisionNode
        Nodo raíz del árbol construido.
    memory_ : MetaMemory o None
        Repositorio de experiencia metacognitiva. Solo en modo 'metacog'.
    experience_stats_ : dict o None
        Resumen estadístico del proceso de inducción. Solo en modo 'metacog'.
    """

    def __init__(
        self,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_gain_ratio: float = 0.001,
        mode: str = "classic",
        B: int = 50,
        random_state: int | None = None,
        top_k: int = 3,
        theta_accept: float = 0.5,
        theta_reject: float = 0.2,
        theta_comp: float = 0.75,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        gamma: float = 0.5,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        enable_bootstrap_cache: bool = True
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio
        self.mode = mode
        self.B = B
        self.random_state = random_state
        self.top_k = top_k
        self.theta_accept = theta_accept
        self.theta_reject = theta_reject
        self.theta_comp = theta_comp
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma = gamma
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.enable_bootstrap_cache = enable_bootstrap_cache

        self.root: DecisionNode | None = None

        # Espacio global de clases
        self.classes_ = None
        self.n_classes_ = 0

        # Atributos metacognitivos (solo disponibles en modo 'metacog')
        self.memory_ = None
        self.experience_stats_ = None

    # =====================================================
    # Entrenamiento
    # =====================================================

    def fit(self, X, y):
        """
        Entrena el clasificador sobre el dataset de entrenamiento.

        Parámetros
        ----------
        X : array-like o pd.DataFrame
            Variables predictoras.
        y : array-like o pd.Series
            Etiquetas de clase.

        Retorna
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Detectar y registrar el espacio de clases único en fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        total_samples = len(y)

        if self.mode == "metacog":
            # Importaciones diferidas para modo metacog
            from .metacognition import (
                ReflectionEngine,
                CompetitiveEstimator,
                MetaMemory,
                DecisionValidator,
                ExperienceStatistics,
                SplitConfidenceScore
            )

            memory = MetaMemory()
            reflection_engine = ReflectionEngine(B=self.B, random_state=self.random_state)
            competitive_estimator = CompetitiveEstimator()
            scs_calculator = SplitConfidenceScore(
                alpha_0=self.alpha_0,
                beta_0=self.beta_0,
                gamma=self.gamma,
                lambda_1=self.lambda_1,
                lambda_2=self.lambda_2
            )
            validator = DecisionValidator(
                theta_accept=self.theta_accept,
                theta_reject=self.theta_reject,
                theta_comp=self.theta_comp,
                scs_calculator=scs_calculator,
                min_samples_split=self.min_samples_split,
                competitive_estimator=competitive_estimator,
                reflection_engine=reflection_engine,
                enable_bootstrap_cache=self.enable_bootstrap_cache
            )

            builder = TreeBuilder(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain_ratio=self.min_gain_ratio,
                mode="metacog",
                top_k=self.top_k,
                validator=validator,
                memory=memory,
                total_samples=total_samples
            )

            self.root = builder.build(X, y)
            self.memory_ = memory
            self.experience_stats_ = ExperienceStatistics.get_summary(memory)

        else:
            # Modo clásico: idéntico al PyC45 original
            builder = TreeBuilder(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain_ratio=self.min_gain_ratio,
                mode="classic"
            )
            self.root = builder.build(X, y)

        return self

    # =====================================================
    # Predicción individual
    # =====================================================

    def _predict_sample(self, node, sample):
        while not node.is_leaf:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    # =====================================================
    # Predicción
    # =====================================================

    def predict(self, X):
        """
        Predice las clases del conjunto de entrada.

        Parámetros
        ----------
        X : array-like o pd.DataFrame
            Variables predictoras.

        Retorna
        -------
        np.ndarray
            Vector de predicciones.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return np.array([self._predict_sample(self.root, sample) for _, sample in X.iterrows()])

    # =====================================================
    # Probabilidad de predicción
    # =====================================================

    def _predict_proba_sample(self, node, sample):
        while not node.is_leaf:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        counts = node.class_counts or {}
        total = sum(counts.values())
        if self.classes_ is None:
            # Fallback en caso de que no se haya llamado a fit()
            return np.array([0.0, 0.0])
            
        if total == 0:
            return np.zeros(self.n_classes_)
            
        # Generar vector denso mapeado al orden de self.classes_
        proba = []
        for c in self.classes_:
            proba.append(counts.get(c, 0.0) / total)
        return np.array(proba)

    def predict_proba(self, X):
        """
        Predice las probabilidades de clase para el conjunto de entrada.

        Parámetros
        ----------
        X : array-like o pd.DataFrame
            Variables predictoras.

        Retorna
        -------
        np.ndarray
            Matriz de probabilidades de forma (n_samples, n_classes).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return np.array([self._predict_proba_sample(self.root, sample) for _, sample in X.iterrows()])

    # =====================================================
    # Exactitud
    # =====================================================

    def score(self, X, y) -> float:
        """
        Calcula la exactitud del clasificador sobre el conjunto dado.

        Parámetros
        ----------
        X : array-like o pd.DataFrame
            Variables predictoras.
        y : array-like o pd.Series
            Etiquetas verdaderas.

        Retorna
        -------
        float
            Exactitud del clasificador en [0, 1].
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        return float(np.mean(self.predict(X) == y))

    # =====================================================
    # Información del árbol
    # =====================================================

    def summary(self):
        """
        Imprime un resumen de la estructura del árbol entrenado.
        """
        print("=" * 60)
        print(f"{'MetaCog-C45' if self.mode == 'metacog' else 'PyC45'} CLASSIFIER  [mode='{self.mode}']")
        print("=" * 60)
        print("Número de nodos:", TreeUtils.count_nodes(self.root))
        print("Número de hojas:", TreeUtils.count_leaves(self.root))
        print("Profundidad:", TreeUtils.depth(self.root))
        if self.mode == "metacog" and self.experience_stats_:
            print("-" * 60)
            print("avg_NS:", round(self.experience_stats_.get("avg_ns", 0.0), 4))
            print("avg_TS:", round(self.experience_stats_.get("avg_ts", 0.0), 4))
            print("MFI:", self.experience_stats_.get("mfi_normalized", {}))
        print("=" * 60)

    # =====================================================
    # Mostrar árbol
    # =====================================================

    def print_tree(self):
        """
        Imprime visualmente la estructura del árbol de decisión.
        """
        TreeUtils.print_tree(self.root)