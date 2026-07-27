"""
=========================================================
PyC45 Framework
Decision Core Subsystem (Production Ready)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : PyC45
Descripción:
Implementa el motor de decisión del paradigma MetaCog-C45,
incluyendo el Split Confidence Score (SCS) dinámico,
políticas formales de decisión y auditoría de decisiones.
=========================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Any, Tuple
import numpy as np

from .decision_trace import DecisionTrace
from .meta_memory import MetaMemory
from .competitive_estimator import CompetitiveEstimator
from .reflection_engine import ReflectionEngine, check_random_state


class DecisionPolicy(str, Enum):
    """
    Veredictos formales de decisión metacognitiva.

    Atributos
    ----------
    ACCEPT : str
        La división se consolida de forma definitiva.
    REVIEW : str
        Existe alta competencia y requiere revisión cognitiva.
    DEFER : str
        Consolidación diferida con regularización/poda preventiva.
    COLLAPSE : str
        Falta de estabilidad o ganancia, el nodo colapsa a hoja.
    """
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    DEFER = "DEFER"
    COLLAPSE = "COLLAPSE"


class SplitConfidenceScore:
    """
    Calculador de la Utilidad Metacognitiva del Split (SCS).

    Calcula la confianza de una propuesta de división combinando ganancia,
    estabilidad de variable, estabilidad de umbral y colinealidad/competencia
    mediante una formulación matemática Cobb-Douglas.

    Parámetros
    ----------
    alpha_0 : float, por defecto 1.0
        Elasticidad base para la estabilidad de la variable (NS).
    beta_0 : float, por defecto 1.0
        Elasticidad base para la estabilidad de umbral continuo (TS).
    gamma : float, por defecto 0.5
        Elasticidad para el término de no competencia (1 - CI).
    lambda_1 : float, por defecto 1.0
        Sensibilidad de amplificación de elasticidades por profundidad del nodo.
    lambda_2 : float, por defecto 1.0
        Sensibilidad de amplificación de elasticidades por reducción de muestras.
    epsilon : float, por defecto 1e-6
        Parámetro de regularización numérica contra divisiones por cero.
    """

    def __init__(
        self,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        gamma: float = 0.5,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        epsilon: float = 1e-6
    ):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma = gamma
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epsilon = epsilon

    def calculate_elasticities(
        self,
        depth: int,
        max_depth: int,
        n_samples: int,
        total_samples: int
    ) -> Tuple[float, float]:
        """
        Determina las elasticidades alfa y beta dinámicamente para el nodo u.

        Calcula los exponentes según la profundidad del nodo y la fracción
        relativa de datos remanentes para ajustar la aversión al riesgo del modelo.

        Parámetros
        ----------
        depth : int
            Profundidad actual del nodo u en el árbol.
        max_depth : int
            Límite de profundidad máxima permitido.
        n_samples : int
            Cantidad de muestras locales en el nodo u.
        total_samples : int
            Cantidad total de muestras en la raíz del árbol.

        Retorna
        -------
        Tuple[float, float]
            El par de elasticidades escaladas (alpha, beta).
        """
        max_depth_safe = max(1, max_depth)
        total_samples_safe = max(1, total_samples)
        
        # Proporción local de muestras
        rho = n_samples / total_samples_safe
        
        # Factor multiplicativo de amplificación por escasez de datos y profundidad
        factor = 1.0 + self.lambda_1 * (depth / max_depth_safe) + self.lambda_2 * (1.0 - rho)
        
        alpha = self.alpha_0 * factor
        beta = self.beta_0 * factor
        
        return float(alpha), float(beta)

    def calculate_scs(
        self,
        gr_normalized: float,
        ns: float,
        ts: float,
        ci: float,
        alpha: float,
        beta: float
    ) -> float:
        """
        Calcula el Split Confidence Score mediante la utilidad Cobb-Douglas.

        SCS = GR_norm * (NS ** alpha) * (TS ** beta) * ((1 - CI) ** gamma)

        Parámetros
        ----------
        gr_normalized : float
            Gain Ratio del candidato normalizado en el intervalo [0, 1].
        ns : float
            Node Stability (estabilidad de variable) ∈ [0, 1].
        ts : float
            Threshold Survival (estabilidad de umbral) ∈ [0, 1].
        ci : float
            Competitiveness Index (índice de competitividad) ∈ [0, 1].
        alpha : float
            Elasticidad escalada de la estabilidad de la variable.
        beta : float
            Elasticidad escalada de la estabilidad de umbral.

        Retorna
        -------
        float
            Split Confidence Score del candidato ∈ [0, 1].
        """
        ns_safe = max(0.0, min(1.0, ns))
        ts_safe = max(0.0, min(1.0, ts))
        ci_term = max(0.0, min(1.0, 1.0 - ci))
        
        scs = gr_normalized * (ns_safe ** alpha) * (ts_safe ** beta) * (ci_term ** self.gamma)
        return float(max(0.0, min(1.0, scs)))


@dataclass
class DecisionOutcome:
    """
    Resultado final estructurado y explicable del validador de decisión.

    Parámetros
    ----------
    feature_selected : str o None
        Atributo ganador del subespacio Top-K. None si se colapsa a hoja.
    threshold_selected : float o None
        Umbral óptimo seleccionado. None si se colapsa a hoja.
    gain_ratio : float
        Gain Ratio empírico del split. 0.0 si se colapsa a hoja.
    scs : float
        Split Confidence Score del split óptimo.
    verdict : DecisionPolicy
        Veredicto de decisión metacognitivo.
    justification : str
        Justificación explicativa legible por humanos (XAI).
    ns : float, por defecto 1.0
        Estabilidad de nodo calculada para el atributo.
    ts : float, por defecto 1.0
        Estabilidad de umbral continua calculada.
    p_star : float, por defecto 1.0
        Frecuencia de selección bootstrap del atributo.
    competitiveness_index : float, por defecto 0.0
        Grado de competencia local con el mejor alternativo.
    is_competitive : bool, por defecto False
        Indica si la competencia supera el umbral crítico.
    alternate_feature : str o None, por defecto None
        Nombre de la variable competidora descartada.
    uncertainty : float, por defecto 0.0
        Incertidumbre cognitiva residual (1.0 - SCS).
    top_k_candidates : List[Dict[str, Any]], por defecto factory
        Métricas e índices detallados para todos los candidatos evaluados.
    reproducibility_meta : Dict[str, Any], por defecto factory
        Metadatos de la corrida (semilla, B, elasticidades aplicadas).
    """
    feature_selected: str | None
    threshold_selected: float | None
    gain_ratio: float
    scs: float
    verdict: DecisionPolicy
    justification: str
    ns: float = 1.0
    ts: float = 1.0
    p_star: float = 1.0
    competitiveness_index: float = 0.0
    is_competitive: bool = False
    alternate_feature: str | None = None
    uncertainty: float = 0.0
    top_k_candidates: List[Dict[str, Any]] = field(default_factory=list)
    reproducibility_meta: Dict[str, Any] = field(default_factory=dict)


class DecisionValidator:
    """
    Orquestador de validación del subespacio de candidatos Top-K del nodo u.

    Parámetros
    ----------
    theta_accept : float, por defecto 0.5
        Umbral mínimo de SCS para aceptar directamente un split.
    theta_reject : float, por defecto 0.2
        Umbral mínimo de SCS para evitar el colapso del nodo.
    theta_comp : float, por defecto 0.75
        Umbral crítico de competitividad local (CI) para forzar revisión.
    scs_calculator : SplitConfidenceScore o None, por defecto None
        Instancia del calculador de SCS Cobb-Douglas.
    min_samples_split : int, por defecto 5
        Cantidad mínima de muestras requerida para permitir una partición.
    """

    def __init__(
        self,
        theta_accept: float = 0.5,
        theta_reject: float = 0.2,
        theta_comp: float = 0.75,
        scs_calculator: SplitConfidenceScore | None = None,
        min_samples_split: int = 5,
        competitive_estimator: CompetitiveEstimator | None = None,
        reflection_engine: ReflectionEngine | None = None,
        enable_bootstrap_cache: bool = True
    ):
        self.theta_accept = theta_accept
        self.theta_reject = theta_reject
        self.theta_comp = theta_comp
        self.scs_calculator = scs_calculator or SplitConfidenceScore()
        self.min_samples_split = min_samples_split
        # Dependencias inyectadas opcionales (usadas por TreeBuilder en modo metacog)
        self._competitive_estimator = competitive_estimator or CompetitiveEstimator()
        self._reflection_engine = reflection_engine or ReflectionEngine()
        self.enable_bootstrap_cache = enable_bootstrap_cache

    def validate(
        self,
        X_u: Any,
        y_u: Any,
        top_k_candidates: List[Dict[str, Any]],
        depth: int,
        max_depth: int,
        total_samples: int,
        competitive_estimator: CompetitiveEstimator,
        reflection_engine: ReflectionEngine
    ) -> DecisionOutcome:
        """
        Orquesta la validación del nodo u aplicando la política de decisión sobre Top-K.

        Calcula el SCS de cada split del Top-K mediante bootstrap y estimaciones
        de competencia y retorna el veredicto consolidado.

        Parámetros
        ----------
        X_u : pd.DataFrame o np.ndarray
            Variables predictoras locales del nodo u.
        y_u : pd.Series o np.ndarray
            Clase objetivo local.
        top_k_candidates : List[Dict[str, Any]]
            Lista de diccionarios que representan los mejores splits candidatos.
        depth : int
            Profundidad del nodo en el árbol.
        max_depth : int
            Límite de profundidad del árbol.
        total_samples : int
            Cantidad total de muestras en la raíz del árbol.
        competitive_estimator : CompetitiveEstimator
            Instancia del estimador de competencia.
        reflection_engine : ReflectionEngine
            Instancia del motor de reflexión bootstrap.

        Retorna
        -------
        DecisionOutcome
            Resultado estructurado del análisis metacognitivo y el veredicto final.
        """
        n_samples = len(y_u)
        
        # Caso de colapso rápido por datos escasos o nodo puro
        if n_samples < self.min_samples_split or len(np.unique(y_u)) <= 1 or not top_k_candidates:
            return DecisionOutcome(
                feature_selected=None,
                threshold_selected=None,
                gain_ratio=0.0,
                scs=0.0,
                verdict=DecisionPolicy.COLLAPSE,
                justification="Muestras locales insuficientes o clase pura. Nodo colapsado a hoja.",
                uncertainty=1.0,
                top_k_candidates=top_k_candidates
            )

        # 1. Normalización local del Gain Ratio de los candidatos
        max_gr = max(float(c.get("gain_ratio", 0.0)) for c in top_k_candidates)
        for c in top_k_candidates:
            c_gr = float(c.get("gain_ratio", 0.0))
            c["gr_normalized"] = c_gr / (max_gr + 1e-9)

        # Mapa de Gain Ratio por característica para el estimador competitivo
        gain_ratios_map = {c["feature"]: float(c.get("gain_ratio", 0.0)) for c in top_k_candidates}

        # 2. Obtener elasticidades dinámicas del nodo
        alpha, beta = self.scs_calculator.calculate_elasticities(depth, max_depth, n_samples, total_samples)

        # 3. Evaluar el SCS de cada candidato en el subespacio Top-K
        evaluated_candidates = []
        if self.enable_bootstrap_cache:
            gammas = reflection_engine.reflect_node(X_u, y_u, top_k_candidates)
            for cand, gamma in zip(top_k_candidates, gammas):
                feat = cand["feature"]
                comp_meta = competitive_estimator.estimate_competition(gain_ratios_map, feat)
                ci = comp_meta["competitiveness_index"]
                ns = gamma["NS"]
                ts = gamma["TS"]
                scs = self.scs_calculator.calculate_scs(cand["gr_normalized"], ns, ts, ci, alpha, beta)
                eval_cand = {
                    **cand,
                    "ns": ns,
                    "ts": ts,
                    "p_star": gamma.get("p_star", 0.0),
                    "competitiveness_index": ci,
                    "is_competitive": comp_meta["is_competitive"],
                    "alternate_feature": comp_meta["alternate_feature"],
                    "scs": scs,
                    "bootstrap_selections": gamma.get("bootstrap_selections", []),
                    "bootstrap_thresholds": gamma.get("bootstrap_thresholds", []),
                    "p_vector": gamma.get("p_vector", {})
                }
                evaluated_candidates.append(eval_cand)
        else:
            for cand in top_k_candidates:
                feat = cand["feature"]
                comp_meta = competitive_estimator.estimate_competition(gain_ratios_map, feat)
                ci = comp_meta["competitiveness_index"]
                gamma = reflection_engine.reflect(X_u, y_u, cand)
                ns = gamma["NS"]
                ts = gamma["TS"]
                scs = self.scs_calculator.calculate_scs(cand["gr_normalized"], ns, ts, ci, alpha, beta)
                eval_cand = {
                    **cand,
                    "ns": ns,
                    "ts": ts,
                    "p_star": gamma.get("p_star", 0.0),
                    "competitiveness_index": ci,
                    "is_competitive": comp_meta["is_competitive"],
                    "alternate_feature": comp_meta["alternate_feature"],
                    "scs": scs,
                    "bootstrap_selections": gamma.get("bootstrap_selections", []),
                    "bootstrap_thresholds": gamma.get("bootstrap_thresholds", []),
                    "p_vector": gamma.get("p_vector", {})
                }
                evaluated_candidates.append(eval_cand)

        # 4. Seleccionar el mejor candidato basándose en SCS máximo
        evaluated_candidates.sort(key=lambda x: x["scs"], reverse=True)
        best_cand = evaluated_candidates[0]
        
        scs_star = best_cand["scs"]
        ci_star = best_cand["competitiveness_index"]
        feat_star = best_cand["feature"]
        thresh_star = best_cand["threshold"]
        gr_star = best_cand["gain_ratio"]
        alt_feat = best_cand["alternate_feature"]
        
        # 5. Aplicación formal de políticas de decisión
        if scs_star >= self.theta_accept:
            if ci_star < self.theta_comp:
                verdict = DecisionPolicy.ACCEPT
                justification = (
                    f"División consolidada exitosamente en '{feat_star}'. "
                    f"Alta confianza SCS: {scs_star:.4f} y baja competencia (CI: {ci_star:.4f})."
                )
            else:
                verdict = DecisionPolicy.REVIEW
                justification = (
                    f"Alta confianza en '{feat_star}' (SCS: {scs_star:.4f}) "
                    f"pero con alta competencia con '{alt_feat}' (CI: {ci_star:.4f}). "
                    f"Se requiere revisión cognitiva."
                )
        elif self.theta_reject <= scs_star < self.theta_accept:
            verdict = DecisionPolicy.DEFER
            justification = (
                f"Confianza moderada en '{feat_star}' (SCS: {scs_star:.4f}). "
                f"División aceptada con validación diferida (pruning preventivo)."
            )
        else:
            verdict = DecisionPolicy.COLLAPSE
            justification = (
                f"Confianza del split óptimo '{feat_star}' extremadamente baja (SCS: {scs_star:.4f} "
                f"< theta_reject = {self.theta_reject}). Nodo colapsado a hoja."
            )

        # Metadatos de reproducibilidad
        reproducibility = {
            "depth": depth,
            "max_depth": max_depth,
            "n_samples": n_samples,
            "total_samples": total_samples,
            "alpha_applied": alpha,
            "beta_applied": beta,
            "gamma_applied": self.scs_calculator.gamma,
            "seed": reflection_engine.random_state,
            "B": reflection_engine.B
        }

        return DecisionOutcome(
            feature_selected=feat_star if verdict != DecisionPolicy.COLLAPSE else None,
            threshold_selected=thresh_star if verdict != DecisionPolicy.COLLAPSE else None,
            gain_ratio=gr_star if verdict != DecisionPolicy.COLLAPSE else 0.0,
            scs=scs_star,
            verdict=verdict,
            justification=justification,
            ns=best_cand["ns"],
            ts=best_cand["ts"],
            p_star=best_cand["p_star"],
            competitiveness_index=ci_star,
            is_competitive=best_cand["is_competitive"],
            alternate_feature=alt_feat,
            uncertainty=float(1.0 - scs_star),
            top_k_candidates=evaluated_candidates,
            reproducibility_meta=reproducibility
        )


class DecisionAudit:
    """
    Factoría auditora de decisiones de MetaCog-C45.
    
    Genera el DecisionTrace inyectable a la memoria.
    """

    @staticmethod
    def audit_decision(
        node_id: str,
        X_u: Any,
        y_u: Any,
        depth: int,
        outcome: DecisionOutcome,
        random_state: Any = None
    ) -> DecisionTrace:
        """
        Construye la traza reproducible de auditoría de la decisión.

        Parámetros
        ----------
        node_id : str
            Identificador único del nodo.
        X_u : pd.DataFrame o np.ndarray
            Dataset de entrada local en el nodo u.
        y_u : pd.Series o np.ndarray
            Vector de etiquetas local del nodo u.
        depth : int
            Profundidad del nodo u.
        outcome : DecisionOutcome
            Salida del validador metacognitivo.
        random_state : Any, opcional
            Semilla utilizada para el bootstrap.

        Retorna
        -------
        DecisionTrace
            Objeto traza completamente estructurado.
        """
        p_vector = {}
        bootstrap_selections = []
        bootstrap_thresholds = []

        if outcome.top_k_candidates:
            # Buscar el elemento óptimo seleccionado
            chosen_details = outcome.top_k_candidates[0]
            p_vector = chosen_details.get("p_vector", {})
            bootstrap_selections = chosen_details.get("bootstrap_selections", [])
            bootstrap_thresholds = chosen_details.get("bootstrap_thresholds", [])

        trace = DecisionTrace(
            node_id=node_id,
            depth=depth,
            n_samples=len(y_u),
            class_distribution=dict(y_u.value_counts()) if hasattr(y_u, "value_counts") else {},
            feature_selected=outcome.feature_selected or "LEAF",
            threshold_selected=outcome.threshold_selected,
            gain_ratio=outcome.gain_ratio,
            ns=outcome.ns,
            ts=outcome.ts,
            p_star=outcome.p_star,
            competitiveness_index=outcome.competitiveness_index,
            is_competitive=outcome.is_competitive,
            alternate_feature=outcome.alternate_feature,
            scs=outcome.scs,
            verdict=outcome.verdict.value,
            justification=outcome.justification,
            uncertainty=outcome.uncertainty,
            top_k_candidates=outcome.top_k_candidates,
            p_vector=p_vector,
            bootstrap_selections=bootstrap_selections,
            bootstrap_thresholds=bootstrap_thresholds,
            timestamp=time.time()
        )
        
        return trace
