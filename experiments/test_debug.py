"""
Debug script to inspect difference in bootstrap generation between
original and optimized validate() implementations.
"""

from __future__ import annotations
import os
import sys
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.classifier import C45Classifier
from core.metacognition import DecisionValidator, ReflectionEngine, SplitConfidenceScore, CompetitiveEstimator

def debug_validate(self, X_u, y_u, top_k_candidates, depth, max_depth, total_samples,
                   competitive_estimator, reflection_engine):
    print(f"\n--- DEBUG NODE depth={depth} ---")
    print(f"X_u shape: {X_u.shape} | y_u shape: {y_u.shape}")
    print(f"reflection_engine.random_state: {reflection_engine.random_state}")
    
    # 1. Run Original
    from core.metacognition.reflection_engine import check_random_state
    rng_orig = check_random_state(reflection_engine.random_state)
    indices_orig = reflection_engine._get_bootstrap_indices(y_u, rng_orig)
    print(f"Original first 5 bootstrap indices: {indices_orig[:5].tolist()}")
    
    # 2. Run Optimized
    rng_opt = check_random_state(reflection_engine.random_state)
    indices_opt = reflection_engine._get_bootstrap_indices(y_u, rng_opt)
    print(f"Optimized first 5 bootstrap indices: {indices_opt[:5].tolist()}")
    
    # Run the standard validation
    from core.metacognition.decision_core import DecisionValidator
    # Call original method
    # Let's temporarily restore validate to avoid recursion
    DecisionValidator.validate = original_validate
    outcome = self.validate(X_u, y_u, top_k_candidates, depth, max_depth, total_samples,
                            competitive_estimator, reflection_engine)
    DecisionValidator.validate = debug_validate
    return outcome

# Save original
original_validate = DecisionValidator.validate
DecisionValidator.validate = debug_validate

def main():
    path = os.path.join(ROOT, 'experiments', 'data', 'diabetes.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    splits = data['cv_splits']
    train_idx, test_idx = splits[0]
    
    clf = C45Classifier(mode='metacog', random_state=42)
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])

if __name__ == '__main__':
    main()
