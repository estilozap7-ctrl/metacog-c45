"""
=========================================================
PyC45 Tests
Unit & Integration Testing Suite for Metacognitive Memory
=========================================================

Verifica la exactitud matemática, lógica y de integración del subsistema
de memoria metacognitiva (DecisionTrace, CompetitiveEstimator, MetaMemory,
MemoryIndex, ExperienceStatistics).
=========================================================
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.metacognition.decision_trace import DecisionTrace
from core.metacognition.competitive_estimator import CompetitiveEstimator
from core.metacognition.meta_memory import MetaMemory, MemoryIndex
from core.metacognition.experience_statistics import ExperienceStatistics
from core.metacognition.reflection_engine import ReflectionEngine


class TestDecisionTrace(unittest.TestCase):

    def test_trace_creation_and_dict_conversion(self):
        trace = DecisionTrace(
            node_id="node_0",
            depth=1,
            n_samples=100,
            class_distribution={0: 60, 1: 40},
            feature_selected="x1",
            threshold_selected=3.5,
            gain_ratio=0.75,
            ns=0.9,
            ts=0.85,
            p_star=0.9,
            p_vector={"x1": 0.9, "x2": 0.1},
            bootstrap_selections=["x1"]*9 + ["x2"],
            bootstrap_thresholds=[3.4, 3.5, 3.6]
        )
        self.assertEqual(trace.node_id, "node_0")
        self.assertEqual(trace.depth, 1)
        self.assertEqual(trace.n_samples, 100)
        self.assertEqual(trace.feature_selected, "x1")
        self.assertEqual(trace.threshold_selected, 3.5)
        self.assertEqual(trace.gain_ratio, 0.75)
        self.assertEqual(trace.ns, 0.9)
        self.assertEqual(trace.ts, 0.85)

        trace_dict = trace.to_dict()
        self.assertEqual(trace_dict["node_id"], "node_0")
        self.assertEqual(trace_dict["depth"], 1)
        self.assertEqual(trace_dict["ns"], 0.9)
        self.assertEqual(trace_dict["p_vector"]["x1"], 0.9)


class TestCompetitiveEstimator(unittest.TestCase):

    def setUp(self):
        self.estimator = CompetitiveEstimator(epsilon=1e-6, competitive_threshold=0.8)

    def test_absolute_dominance(self):
        # Escenario donde x1 domina completamente a x2
        gains = {"x1": 0.8, "x2": 0.1}
        res = self.estimator.estimate_competition(gains, "x1")
        
        # Delta = 0.8 - 0.1 = 0.7
        # CI = 1.0 - (0.7 / 0.800001) ≈ 1.0 - 0.875 = 0.125
        self.assertAlmostEqual(res["delta_gr"], 0.7, places=5)
        self.assertAlmostEqual(res["competitiveness_index"], 0.125, places=3)
        self.assertFalse(res["is_competitive"])
        self.assertEqual(res["alternate_feature"], "x2")

    def test_high_competition(self):
        # Escenario donde x1 y x2 son casi idénticos
        gains = {"x1": 0.8, "x2": 0.79}
        res = self.estimator.estimate_competition(gains, "x1")
        
        # Delta = 0.01
        # CI = 1.0 - (0.01 / 0.800001) ≈ 0.9875
        self.assertAlmostEqual(res["delta_gr"], 0.01, places=5)
        self.assertGreater(res["competitiveness_index"], 0.95)
        self.assertTrue(res["is_competitive"])

    def test_no_alternates(self):
        # Solo una variable
        gains = {"x1": 0.8}
        res = self.estimator.estimate_competition(gains, "x1")
        self.assertEqual(res["competitiveness_index"], 0.0)
        self.assertFalse(res["is_competitive"])

    def test_zero_gain_ratio_stability(self):
        # El mejor tiene Gain Ratio de 0
        gains = {"x1": 0.0, "x2": 0.0}
        res = self.estimator.estimate_competition(gains, "x1")
        self.assertEqual(res["competitiveness_index"], 0.0)
        self.assertFalse(res["is_competitive"])


class TestMetaMemoryAndIndex(unittest.TestCase):

    def setUp(self):
        self.memory = MetaMemory(w1=0.5, w2=0.5)
        self.trace1 = DecisionTrace(
            node_id="n1", depth=1, n_samples=100, class_distribution={0: 50, 1: 50},
            feature_selected="x1", threshold_selected=2.5, gain_ratio=0.5, ns=0.8, ts=0.8, p_star=0.8
        )
        self.trace2 = DecisionTrace(
            node_id="n2", depth=2, n_samples=50, class_distribution={0: 45, 1: 5},
            feature_selected="x2", threshold_selected=1.5, gain_ratio=0.6, ns=0.7, ts=0.9, p_star=0.7
        )
        self.trace3 = DecisionTrace(
            node_id="n3", depth=2, n_samples=60, class_distribution={0: 50, 1: 10},
            feature_selected="x1", threshold_selected=4.0, gain_ratio=0.4, ns=0.9, ts=0.7, p_star=0.9
        )
        self.memory.store(self.trace1)
        self.memory.store(self.trace2)
        self.memory.store(self.trace3)

    def test_memory_indexing(self):
        # Consultar por característica
        traces_x1 = self.memory.query_by_feature("x1")
        self.assertEqual(len(traces_x1), 2)
        self.assertIn(self.trace1, traces_x1)
        self.assertIn(self.trace3, traces_x1)

        # Consultar por profundidad
        traces_d2 = self.memory.query_by_depth(2)
        self.assertEqual(len(traces_d2), 2)
        self.assertIn(self.trace2, traces_d2)
        self.assertIn(self.trace3, traces_d2)

    def test_knn_similarity(self):
        # Nodo query: n_samples=55, clase={0: 48, 1: 7} -> Muy similar a n3 y n2, lejano a n1
        similar = self.memory.query_similar(n_samples=55, class_distribution={0: 48, 1: 7}, k=2)
        self.assertEqual(len(similar), 2)
        # El primero más cercano debería ser n3 o n2, no n1
        self.assertNotIn(self.trace1, similar)

    def test_memory_clear(self):
        self.memory.clear()
        self.assertEqual(self.memory.size(), 0)
        self.assertEqual(len(self.memory.query_by_feature("x1")), 0)


class TestExperienceStatistics(unittest.TestCase):

    def setUp(self):
        self.memory = MetaMemory()
        # n1: 100 muestras, ns=0.8, ts=0.9
        t1 = DecisionTrace(
            node_id="n1", depth=1, n_samples=100, class_distribution={0: 50, 1: 50},
            feature_selected="x1", threshold_selected=2.5, gain_ratio=0.5, ns=0.8, ts=0.9, p_star=0.8
        )
        # n2: 50 muestras, ns=0.6, ts=0.8
        t2 = DecisionTrace(
            node_id="n2", depth=2, n_samples=50, class_distribution={0: 40, 1: 10},
            feature_selected="x2", threshold_selected=1.5, gain_ratio=0.6, ns=0.6, ts=0.8, p_star=0.6
        )
        self.memory.store(t1)
        self.memory.store(t2)

    def test_averages_and_summary(self):
        summary = ExperienceStatistics.get_summary(self.memory)
        self.assertEqual(summary["total_decisions"], 2)
        self.assertAlmostEqual(summary["avg_ns"], 0.7, places=5)
        self.assertAlmostEqual(summary["avg_ts"], 0.85, places=5)

    def test_metacognitive_feature_importance(self):
        # MFI bruto x1 = 100 * 0.8 * 0.9 = 72
        # MFI bruto x2 = 50 * 0.6 * 0.8 = 24
        # Total = 96
        # Normalizado x1 = 72/96 = 0.75, x2 = 24/96 = 0.25
        mfi = ExperienceStatistics.calculate_metacognitive_feature_importance(
            self.memory.get_all_traces(), normalize=True
        )
        self.assertAlmostEqual(mfi["x1"], 0.75, places=5)
        self.assertAlmostEqual(mfi["x2"], 0.25, places=5)


class TestIntegrationMemorySubsystem(unittest.TestCase):

    def test_complete_metacognitive_flow(self):
        # 1. Simular un dataset
        rng = np.random.RandomState(42)
        data = {
            'x1': rng.uniform(0, 10, 100),
            'x2': rng.uniform(5, 15, 100)
        }
        df = pd.DataFrame(data)
        df['y'] = (df['x1'] > 5).astype(int)
        
        # 2. Inicializar ReflectionEngine y CompetitiveEstimator
        engine = ReflectionEngine(B=15, random_state=42)
        comp_est = CompetitiveEstimator()
        memory = MetaMemory()
        
        # 3. Simular la evaluación de un nodo
        # Gain ratios de variables candidatas
        gains = {"x1": 0.82, "x2": 0.12}
        best_feat = "x1"
        best_thresh = 5.0
        
        # Calcular competencia
        comp_res = comp_est.estimate_competition(gains, best_feat)
        
        # Ejecutar reflexión
        candidate = {"feature": best_feat, "threshold": best_thresh, "gain_ratio": gains[best_feat]}
        gamma = engine.reflect(df[['x1', 'x2']], df['y'], candidate)
        
        # 4. Registrar DecisionTrace en MetaMemory
        trace = DecisionTrace(
            node_id="root",
            depth=0,
            n_samples=len(df),
            class_distribution=df['y'].value_counts().to_dict(),
            feature_selected=best_feat,
            threshold_selected=best_thresh,
            gain_ratio=gains[best_feat],
            ns=gamma["NS"],
            ts=gamma["TS"],
            p_star=gamma["p_star"],
            p_vector=gamma["p_vector"],
            bootstrap_selections=gamma["bootstrap_selections"],
            bootstrap_thresholds=gamma["bootstrap_thresholds"]
        )
        
        memory.store(trace)
        
        # 5. Consultar estadísticas finales
        summary = ExperienceStatistics.get_summary(memory)
        
        self.assertEqual(memory.size(), 1)
        self.assertEqual(summary["total_decisions"], 1)
        self.assertGreater(summary["avg_ns"], 0.0)
        self.assertGreater(summary["avg_ts"], 0.0)


if __name__ == '__main__':
    unittest.main()
