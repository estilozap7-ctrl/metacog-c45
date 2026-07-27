"""
=========================================================
MetaCog-C45 Framework
Reflection Engine and Metacognitive Modules (Stabilized)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : MetaCog-C45
Descripción:
Implementa el ReflectionEngine y las estrategias de evaluación
de estabilidad (NodeStability y ThresholdSurvival) con soporte
para random_state, bootstrap estratificado adaptativo y normalización
de entropía por espacio activo (d_eff).
=========================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from .base import BaseMetacognitiveModule
from core.best_split import BestSplitFinder


def check_random_state(seed: Any) -> np.random.RandomState:
    """
    Valida y resuelve el generador de números aleatorios según el estándar scikit-learn.
    """
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, np.random.Generator):
        # Adaptador para Generator usando su BitGenerator
        return np.random.RandomState(seed.bit_generator)
    else:
        raise ValueError(f"{seed} no es una semilla o instancia válida de RandomState.")


class NodeStabilityModule(BaseMetacognitiveModule):
    """
    Evalúa la estabilidad de la selección del atributo óptimo (Node Stability - NS)
    normalizando por el espacio de características activo (d_eff) para evitar sesgos.
    """

    def evaluate(
        self,
        X_u: pd.DataFrame,
        y_u: pd.Series,
        candidate: dict,
        alternate: dict | None = None
    ) -> float:
        p_star = candidate.get('p_star', 0.0)
        p_vector = candidate.get('p_vector', {})
        
        # Calcular d_eff: cantidad de variables elegidas al menos una vez
        d_eff = sum(1 for p in p_vector.values() if p > 0.0)
        
        if d_eff <= 1:
            return float(p_star)

        # Calcular la Entropía de Shannon de la distribución de selección
        entropy = 0.0
        for p in p_vector.values():
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalizar contra el espacio de variables activas
        norm_entropy = entropy / np.log2(d_eff)
        ns_score = p_star * (1.0 - norm_entropy)
        
        return float(np.clip(ns_score, 0.0, 1.0))


class ThresholdSurvivalModule(BaseMetacognitiveModule):
    """
    Evalúa la estabilidad del umbral de decisión continuo (Threshold Survival - TS).
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def evaluate(
        self,
        X_u: pd.DataFrame,
        y_u: pd.Series,
        candidate: dict,
        alternate: dict | None = None
    ) -> float:
        feature = candidate.get('feature')
        bootstrap_thresholds = candidate.get('bootstrap_thresholds', [])
        
        if len(bootstrap_thresholds) < 2:
            return 1.0
            
        sigma_theta_sq = float(np.var(bootstrap_thresholds, ddof=1))
        
        feat_data = X_u[feature].dropna()
        if len(feat_data) < 2:
            var_local = 0.0
        else:
            var_local = float(np.var(feat_data, ddof=1))
            
        # Fórmula normalizada para evitar desequilibrio dimensional
        ts_score = np.exp(-sigma_theta_sq / (var_local * (1.0 + self.epsilon) + self.epsilon))
        
        return float(np.clip(ts_score, 0.0, 1.0))


class ReflectionEngine:
    """
    Orquestador metacognitivo que realiza simulaciones bootstrap locals
    con soporte de reproducibilidad y bootstrap estratificado adaptativo.
    """

    def __init__(
        self, 
        B: int = 50, 
        epsilon: float = 1e-6, 
        random_state: Any = None,
        imbalance_threshold: float = 0.15
    ):
        self.B = B
        self.epsilon = epsilon
        self.random_state = random_state
        self.imbalance_threshold = imbalance_threshold
        self.ns_module = NodeStabilityModule()
        self.ts_module = ThresholdSurvivalModule(epsilon=epsilon)
        self.finder = BestSplitFinder()

    def _get_bootstrap_indices(self, y_u: pd.Series, rng: np.random.RandomState) -> np.ndarray:
        """
        Retorna índices de remuestreo bootstrap. Usa estratificación si hay alto desbalance.
        """
        n_samples = len(y_u)
        
        # Calcular proporciones de clase
        counts = y_u.value_counts(normalize=True)
        is_imbalanced = False
        if len(counts) == 2:
            min_prop = counts.min()
            if min_prop <= self.imbalance_threshold:
                is_imbalanced = True
                
        if is_imbalanced:
            # Bootstrap estratificado
            indices_stratified = []
            for c in counts.index:
                c_indices = np.where(y_u == c)[0]
                n_c = len(c_indices)
                if n_c > 0:
                    drawn = rng.choice(c_indices, size=n_c, replace=True)
                    indices_stratified.extend(drawn)
            indices_stratified = np.array(indices_stratified)
            rng.shuffle(indices_stratified)
            return indices_stratified
        else:
            # Bootstrap estándar
            return rng.choice(n_samples, size=n_samples, replace=True)

    def reflect(
        self,
        X_u: pd.DataFrame,
        y_u: pd.Series,
        candidate: dict,
        alternate: dict | None = None
    ) -> dict:
        feature_star = candidate.get('feature')
        
        if feature_star is None or len(y_u) < 5 or len(X_u.columns) == 0:
            return {
                "NS": 1.0,
                "TS": 1.0,
                "p_star": 1.0,
                "p_vector": {},
                "bootstrap_selections": [],
                "bootstrap_thresholds": []
            }

        rng = check_random_state(self.random_state)
        bootstrap_selections = []
        bootstrap_thresholds = []

        # Bucle de perturbación bootstrap
        for _ in range(self.B):
            indices = self._get_bootstrap_indices(y_u, rng)
            X_b = X_u.iloc[indices].reset_index(drop=True)
            y_b = y_u.iloc[indices].reset_index(drop=True)

            try:
                feat_b, thresh_b, _ = self.finder.find_best_split(X_b, y_b)
                if feat_b is not None:
                    bootstrap_selections.append(feat_b)
                    if feat_b == feature_star:
                        bootstrap_thresholds.append(thresh_b)
            except Exception:
                continue

        total_valid = len(bootstrap_selections)
        if total_valid == 0:
            return {
                "NS": 1.0,
                "TS": 1.0,
                "p_star": 1.0,
                "p_vector": {},
                "bootstrap_selections": [],
                "bootstrap_thresholds": []
            }

        unique_feats, counts = np.unique(bootstrap_selections, return_counts=True)
        p_vector = {feat: float(count / total_valid) for feat, count in zip(unique_feats, counts)}
        p_star = float(p_vector.get(feature_star, 0.0))

        eval_payload = {
            "feature": feature_star,
            "p_star": p_star,
            "p_vector": p_vector,
            "bootstrap_thresholds": bootstrap_thresholds
        }

        ns_score = self.ns_module.evaluate(X_u, y_u, eval_payload, alternate)
        ts_score = self.ts_module.evaluate(X_u, y_u, eval_payload, alternate)

        return {
            "NS": ns_score,
            "TS": ts_score,
            "p_star": p_star,
            "p_vector": p_vector,
            "bootstrap_selections": bootstrap_selections,
            "bootstrap_thresholds": bootstrap_thresholds
        }

    def reflect_node(
        self,
        X_u: pd.DataFrame,
        y_u: pd.Series,
        candidates: List[dict]
    ) -> List[dict]:
        """
        Realiza una única simulación bootstrap de B réplicas a nivel de nodo
        y evalúa la estabilidad de múltiples candidatos de manera eficiente.
        """
        if not candidates or len(y_u) < 5 or len(X_u.columns) == 0:
            return [
                {
                    "NS": 1.0,
                    "TS": 1.0,
                    "p_star": 1.0,
                    "p_vector": {},
                    "bootstrap_selections": [],
                    "bootstrap_thresholds": []
                }
                for _ in candidates
            ]

        rng = check_random_state(self.random_state)
        bootstrap_selections = []
        bootstrap_runs = []

        # Bucle de perturbación bootstrap único para el nodo
        for _ in range(self.B):
            indices = self._get_bootstrap_indices(y_u, rng)
            X_b = X_u.iloc[indices].reset_index(drop=True)
            y_b = y_u.iloc[indices].reset_index(drop=True)

            try:
                feat_b, thresh_b, _ = self.finder.find_best_split(X_b, y_b)
                if feat_b is not None:
                    bootstrap_selections.append(feat_b)
                    bootstrap_runs.append((feat_b, thresh_b))
            except Exception:
                continue

        total_valid = len(bootstrap_selections)
        
        # Calcular el vector de probabilidades global p_vector
        if total_valid > 0:
            unique_feats, counts = np.unique(bootstrap_selections, return_counts=True)
            p_vector = {feat: float(count / total_valid) for feat, count in zip(unique_feats, counts)}
        else:
            p_vector = {}

        results = []
        for cand in candidates:
            feature_star = cand.get('feature')
            if feature_star is None or total_valid == 0:
                results.append({
                    "NS": 1.0,
                    "TS": 1.0,
                    "p_star": 1.0 if feature_star is not None else 0.0,
                    "p_vector": p_vector,
                    "bootstrap_selections": bootstrap_selections,
                    "bootstrap_thresholds": []
                })
                continue
                
            p_star = float(p_vector.get(feature_star, 0.0))
            # Filtrar los umbrales bootstrap solo para la variable de este candidato
            bootstrap_thresholds = [thresh_b for feat_b, thresh_b in bootstrap_runs if feat_b == feature_star]
            
            eval_payload = {
                "feature": feature_star,
                "p_star": p_star,
                "p_vector": p_vector,
                "bootstrap_thresholds": bootstrap_thresholds
            }
            
            ns_score = self.ns_module.evaluate(X_u, y_u, eval_payload, None)
            ts_score = self.ts_module.evaluate(X_u, y_u, eval_payload, None)
            
            results.append({
                "NS": ns_score,
                "TS": ts_score,
                "p_star": p_star,
                "p_vector": p_vector,
                "bootstrap_selections": bootstrap_selections,
                "bootstrap_thresholds": bootstrap_thresholds
            })
            
        return results

