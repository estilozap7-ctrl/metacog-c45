"""
=========================================================
MetaCog-C45 Tests
Unit & Integration Testing Suite for Decision Core
=========================================================

Verifica la exactitud lógica, matemática e integrativa del Decision Core
(SplitConfidenceScore, DecisionPolicy, DecisionValidator, DecisionAudit).
=========================================================
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.metacognition.decision_core import (
    SplitConfidenceScore,
    DecisionPolicy,
    DecisionOutcome,
    DecisionValidator,
    DecisionAudit
)
from core.metacognition import (
    ReflectionEngine,
    CompetitiveEstimator,
    MetaMemory
)


class TestSplitConfidenceScore(unittest.TestCase):

    def test_elasticities_calculation(self):
        calculator = SplitConfidenceScore(alpha_0=1.0, beta_0=1.0, lambda_1=1.0, lambda_2=1.0)
        
        # En la raíz (depth=0, n_samples=total_samples) -> alpha = 1.0, beta = 1.0
        alpha, beta = calculator.calculate_elasticities(depth=0, max_depth=10, n_samples=100, total_samples=100)
        self.assertAlmostEqual(alpha, 1.0)
        self.assertAlmostEqual(beta, 1.0)
        
        # En profundidad máxima con pocas muestras -> elasticidad incrementada
        alpha_deep, beta_deep = calculator.calculate_elasticities(depth=10, max_depth=10, n_samples=10, total_samples=100)
        # alpha = 1.0 * (1.0 + 1.0 * (10/10) + 1.0 * (1 - 10/100)) = 1.0 * (1.0 + 1.0 + 0.9) = 2.9
        self.assertAlmostEqual(alpha_deep, 2.9)
        self.assertAlmostEqual(beta_deep, 2.9)

    def test_scs_veto_effect(self):
        calculator = SplitConfidenceScore(gamma=0.5)
        
        # Caso ideal
        scs = calculator.calculate_scs(gr_normalized=1.0, ns=1.0, ts=1.0, ci=0.0, alpha=1.0, beta=1.0)
        self.assertEqual(scs, 1.0)
        
        # Veto de estabilidad de variable (ns = 0.0) -> SCS debe ser 0.0
        scs_ns_veto = calculator.calculate_scs(gr_normalized=1.0, ns=0.0, ts=1.0, ci=0.0, alpha=1.0, beta=1.0)
        self.assertEqual(scs_ns_veto, 0.0)
        
        # Veto de estabilidad de umbral (ts = 0.0) -> SCS debe ser 0.0
        scs_ts_veto = calculator.calculate_scs(gr_normalized=1.0, ns=1.0, ts=0.0, ci=0.0, alpha=1.0, beta=1.0)
        self.assertEqual(scs_ts_veto, 0.0)


class TestDecisionValidator(unittest.TestCase):

    def setUp(self):
        self.validator = DecisionValidator(theta_accept=0.5, theta_reject=0.2, theta_comp=0.75)
        self.comp_est = CompetitiveEstimator()
        self.engine = ReflectionEngine(B=10, random_state=42)
        
        # Dataset simple
        self.X = pd.DataFrame({'x1': np.random.uniform(0, 10, 100), 'x2': np.random.uniform(0, 10, 100)})
        self.y = pd.Series((self.X['x1'] > 5).astype(int))

    def test_verdict_accept(self):
        # Candidato dominante y estable
        top_k = [{"feature": "x1", "threshold": 5.0, "gain_ratio": 0.85}]
        outcome = self.validator.validate(
            X_u=self.X, y_u=self.y, top_k_candidates=top_k,
            depth=1, max_depth=10, total_samples=100,
            competitive_estimator=self.comp_est, reflection_engine=self.engine
        )
        self.assertEqual(outcome.verdict, DecisionPolicy.ACCEPT)
        self.assertEqual(outcome.feature_selected, "x1")
        self.assertTrue(outcome.scs > 0.5)

    def test_verdict_collapse_low_samples(self):
        # Menos muestras que el mínimo
        top_k = [{"feature": "x1", "threshold": 5.0, "gain_ratio": 0.85}]
        outcome = self.validator.validate(
            X_u=self.X.iloc[:3], y_u=self.y.iloc[:3], top_k_candidates=top_k,
            depth=1, max_depth=10, total_samples=100,
            competitive_estimator=self.comp_est, reflection_engine=self.engine
        )
        self.assertEqual(outcome.verdict, DecisionPolicy.COLLAPSE)
        self.assertIsNone(outcome.feature_selected)


class TestDecisionAudit(unittest.TestCase):

    def test_audit_generation(self):
        outcome = DecisionOutcome(
            feature_selected="x1",
            threshold_selected=5.0,
            gain_ratio=0.85,
            scs=0.9,
            verdict=DecisionPolicy.ACCEPT,
            justification="Consolidado exitosamente",
            ns=0.95,
            ts=0.95,
            p_star=0.95,
            competitiveness_index=0.1,
            is_competitive=False,
            alternate_feature="x2",
            uncertainty=0.1,
            top_k_candidates=[{"feature": "x1", "scs": 0.9, "p_vector": {"x1": 1.0}}]
        )
        
        y_dummy = pd.Series([0, 1, 0, 1, 0, 1])
        trace = DecisionAudit.audit_decision(
            node_id="n_root",
            X_u=None,
            y_u=y_dummy,
            depth=0,
            outcome=outcome
        )
        
        self.assertEqual(trace.node_id, "n_root")
        self.assertEqual(trace.scs, 0.9)
        self.assertEqual(trace.verdict, "ACCEPT")
        self.assertEqual(trace.feature_selected, "x1")
        self.assertEqual(trace.competitiveness_index, 0.1)


class TestIntegrationDecisionCore(unittest.TestCase):

    def test_integrated_flow(self):
        # 1. Preparar subsistemas
        comp_est = CompetitiveEstimator()
        engine = ReflectionEngine(B=15, random_state=42)
        validator = DecisionValidator()
        memory = MetaMemory()
        
        # 2. Dataset local
        X_u = pd.DataFrame({'x1': np.random.uniform(0, 10, 100), 'x2': np.random.uniform(0, 10, 100)})
        y_u = pd.Series((X_u['x1'] > 5).astype(int))
        
        # 3. Candidatos Top-K del greedy finder
        top_k = [
            {"feature": "x1", "threshold": 5.0, "gain_ratio": 0.85},
            {"feature": "x2", "threshold": 3.0, "gain_ratio": 0.12}
        ]
        
        # 4. Validar
        outcome = validator.validate(
            X_u=X_u, y_u=y_u, top_k_candidates=top_k,
            depth=0, max_depth=5, total_samples=100,
            competitive_estimator=comp_est, reflection_engine=engine
        )
        
        # 5. Auditar y guardar
        trace = DecisionAudit.audit_decision(
            node_id="node_0", X_u=X_u, y_u=y_u, depth=0, outcome=outcome, random_state=42
        )
        memory.store(trace)
        
        self.assertEqual(memory.size(), 1)
        self.assertEqual(memory.get_all_traces()[0].scs, outcome.scs)


if __name__ == '__main__':
    unittest.main()
