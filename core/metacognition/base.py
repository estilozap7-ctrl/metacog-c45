"""
=========================================================
PyC45 Framework
Base Metacognitive Module Interface
=========================================================

Autor      : Luis Alberto Buelvas Cogollo / Antigravity
Proyecto   : PyC45
Descripción:
Define la clase base abstracta para todos los módulos de
reflexión metacognitiva que se inyectan en el framework.
=========================================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BaseMetacognitiveModule(ABC):
    """
    Clase base abstracta para los módulos metacognitivos de MetaCog-C45.
    """

    @abstractmethod
    def evaluate(
        self,
        X_u: pd.DataFrame,
        y_u: pd.Series,
        candidate: dict,
        alternate: dict | None = None
    ) -> float:
        """
        Ejecuta la evaluación metacognitiva local sobre un nodo candidato y
        retorna un valor de confianza o estabilidad acotado en el intervalo [0.0, 1.0].

        Parámetros:
        -----------
        X_u : pd.DataFrame
            Variables predictoras locales del nodo.
        y_u : pd.Series
            Clase objetivo local del nodo.
        candidate : dict
            Diccionario que contiene la información del split candidato óptimo.
            Debe incluir las llaves 'feature', 'threshold' y 'gain_ratio'.
        alternate : dict, opcional
            Diccionario del split del competidor más cercano (segundo mejor split).
            Debe incluir las llaves 'feature', 'threshold' y 'gain_ratio'.

        Retorna:
        --------
        float:
            Valor de confianza o consistencia acotado en el intervalo [0.0, 1.0].
        """
        pass
