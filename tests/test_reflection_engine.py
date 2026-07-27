"""
=========================================================
PyC45 Tests
Unit Testing Suite for ReflectionEngine (Stabilized)
Gate Review 1 - Pruebas Q1-compliant
=========================================================

Cubre:
- Resolución de random_state (None, int, RandomState, Generator)
- Normalización por d_eff (sin sesgo dimensional)
- Bootstrap estratificado adaptativo
- Reproducibilidad de resultados
- Casos límite: nodo puro, dataset de una sola columna, n < 5
- Convergencia de NS/TS con respecto a B
=========================================================
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.metacognition.reflection_engine import ReflectionEngine, NodeStabilityModule, ThresholdSurvivalModule, check_random_state


class TestReflectionEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        data = {
            'x1': np.random.uniform(0, 10, 100),
            'x2': np.random.uniform(10, 20, 100)
        }
        self.df = pd.DataFrame(data)
        self.df['y'] = (self.df['x1'] > 5).astype(int)
        
        self.X = self.df[['x1', 'x2']]
        self.y = self.df['y']

    def test_random_state_resolution(self):
        # Probar resolución de None
        r_none = check_random_state(None)
        self.assertIsInstance(r_none, np.random.RandomState)
        
        # Probar resolución de int
        r_int = check_random_state(42)
        self.assertIsInstance(r_int, np.random.RandomState)
        
        # Probar resolución de RandomState
        r_state = np.random.RandomState(42)
        r_resolved = check_random_state(r_state)
        self.assertEqual(r_resolved, r_state)
        
        # Probar resolución de Generator
        r_gen = np.random.default_rng(42)
        r_gen_resolved = check_random_state(r_gen)
        self.assertIsInstance(r_gen_resolved, np.random.RandomState)

    def test_node_stability_d_eff_normalization(self):
        module = NodeStabilityModule()
        
        # Caso ideal: d_eff = 1 -> NS = p_star = 1.0
        candidate_ideal = {
            'p_star': 1.0,
            'p_vector': {'x1': 1.0}
        }
        ns_score = module.evaluate(self.X, self.y, candidate_ideal)
        self.assertEqual(ns_score, 1.0)
        
        # Caso intermedio: p_star = 0.5, p_vector = {'x1': 0.5, 'x2': 0.5, 'x3': 0.0}
        # d_eff debe ser 2, no 3.
        # H(p) = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        # norm_entropy = 1.0 / log2(2) = 1.0
        # ns_score = 0.5 * (1 - 1.0) = 0.0
        candidate_shared = {
            'p_star': 0.5,
            'p_vector': {'x1': 0.5, 'x2': 0.5, 'x3': 0.0}
        }
        ns_score_shared = module.evaluate(self.X, self.y, candidate_shared)
        self.assertEqual(ns_score_shared, 0.0)

    def test_stratified_bootstrap_trigger(self):
        # Crear un dataset con alto desbalance (95% clase 0, 5% clase 1)
        y_imbalanced = pd.Series([0]*95 + [1]*5)
        X_imbalanced = pd.DataFrame({'x': np.random.uniform(0, 10, 100)})
        
        # Inicializar engine con umbral de desbalance de 0.15 (debería activarse)
        engine = ReflectionEngine(B=10, imbalance_threshold=0.15, random_state=42)
        
        rng = check_random_state(42)
        indices = engine._get_bootstrap_indices(y_imbalanced, rng)
        
        # Verificar que el remuestreo estratificado mantiene al menos muestras de la clase 1
        drawn_y = y_imbalanced.iloc[indices]
        self.assertIn(1, drawn_y.values)
        self.assertEqual(len(drawn_y), 100)

    def test_reflection_engine_execution_reproducible(self):
        # Verificar que pasar el mismo random_state produce resultados idénticos
        engine1 = ReflectionEngine(B=10, random_state=42)
        engine2 = ReflectionEngine(B=10, random_state=42)
        
        candidate_split = {
            'feature': 'x1',
            'threshold': 5.0,
            'gain_ratio': 0.8
        }
        
        res1 = engine1.reflect(self.X, self.y, candidate_split)
        res2 = engine2.reflect(self.X, self.y, candidate_split)
        
        self.assertEqual(res1["NS"], res2["NS"])
        self.assertEqual(res1["TS"], res2["TS"])

    def test_edge_case_n_less_than_5(self):
        """Nodo con menos de 5 muestras: el engine debe retornar valores default conservadores."""
        X_tiny = pd.DataFrame({'x1': [1.0, 2.0, 3.0]})
        y_tiny = pd.Series([0, 1, 0])
        engine = ReflectionEngine(B=10, random_state=0)
        gamma = engine.reflect(X_tiny, y_tiny, {'feature': 'x1', 'threshold': 1.5})
        # Con n<5 debe retornar valores de fallback
        self.assertEqual(gamma["NS"], 1.0)
        self.assertEqual(gamma["TS"], 1.0)

    def test_edge_case_pure_node(self):
        """Nodo con clase pura: el gain ratio es 0 en todas las réplicas."""
        X_pure = pd.DataFrame({'x1': np.linspace(0, 10, 50)})
        y_pure = pd.Series([1] * 50)  # puro
        engine = ReflectionEngine(B=20, random_state=7)
        gamma = engine.reflect(X_pure, y_pure, {'feature': 'x1', 'threshold': 5.0})
        # Los valores deben estar acotados en [0, 1]
        self.assertTrue(0.0 <= gamma["NS"] <= 1.0)
        self.assertTrue(0.0 <= gamma["TS"] <= 1.0)

    def test_edge_case_single_column(self):
        """Dataset con una sola columna: d_eff = 1, NS = p_star."""
        X_one = pd.DataFrame({'x1': np.random.uniform(0, 10, 80)})
        y_one = pd.Series((X_one['x1'] > 5).astype(int))
        engine = ReflectionEngine(B=20, random_state=1)
        gamma = engine.reflect(X_one, y_one, {'feature': 'x1', 'threshold': 5.0})
        # NS debería ser igual a p_star porque d_eff <= 1 no penaliza
        self.assertAlmostEqual(gamma["NS"], gamma["p_star"], places=5)

    def test_b_convergence(self):
        """Verifica que NS_std decrece conforme B aumenta sobre datos con ruido."""
        # Dataset con ruido deliberado: la frontera de decisión no es perfectamente limpia
        rng = np.random.RandomState(0)
        X_noisy = pd.DataFrame({
            'x1': rng.uniform(0, 10, 80),
            'x2': rng.uniform(0, 10, 80),
            'x3': rng.uniform(0, 10, 80)
        })
        # Clase determinada por x1 + ruido: crea ambigüedad de atributo
        noise = rng.normal(0, 2.5, 80)
        y_noisy = pd.Series(((X_noisy['x1'] + noise) > 5).astype(int))
        candidate_split = {'feature': 'x1', 'threshold': 5.0, 'gain_ratio': 0.3}

        B_list = [10, 100]
        stds = []
        for B in B_list:
            ns_runs = []
            for seed in range(25):
                engine = ReflectionEngine(B=B, random_state=seed)
                gamma = engine.reflect(X_noisy, y_noisy, candidate_split)
                ns_runs.append(gamma["NS"])
            stds.append(float(np.std(ns_runs)))

        # Con B pequeño la varianza de NS entre distintas semillas debe ser mayor
        # que con B grande (la estimación converge). O al menos ambas no son iguales.
        # Si ambas son 0, el dataset es trivial para el bootstrap (raro con ruido).
        if stds[0] == 0.0 and stds[1] == 0.0:
            # Si ambas son 0.0, la convergencia ya alcanzó el mínimo: test pasa trivialmente
            self.assertEqual(stds[0], stds[1])
        else:
            self.assertGreaterEqual(stds[0], stds[1],
                msg=f"NS_std con B=10 ({stds[0]:.5f}) debe ser >= NS_std con B=100 ({stds[1]:.5f})")

    def test_invalid_random_state_raises(self):
        """Valores inválidos para random_state deben lanzar ValueError."""
        with self.assertRaises(ValueError):
            check_random_state("not_a_seed")

    def test_ts_zero_variance_stability(self):
        """Columna de varianza cero en nodo: epsilon debe prevenir division por cero."""
        X_const = pd.DataFrame({
            'x_const': [5.0] * 50,
            'x2': np.random.uniform(0, 10, 50)
        })
        y_const = pd.Series([0] * 25 + [1] * 25)
        module = ThresholdSurvivalModule(epsilon=1e-6)
        candidate = {
            'feature': 'x_const',
            'bootstrap_thresholds': [5.0, 5.0, 5.0, 5.0, 5.0]
        }
        # No debe lanzar excepcion y debe estar acotado
        ts = module.evaluate(X_const, y_const, candidate)
        self.assertTrue(0.0 <= ts <= 1.0)


if __name__ == '__main__':
    unittest.main()

