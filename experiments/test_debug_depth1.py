"""
Debug script to inspect difference in NS/TS/SCS calculation at depth 1
between original and optimized validate() implementations.
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
from core.metacognition.decision_core import DecisionOutcome

# We will collect detailed information at depth 1
records = []

original_validate = DecisionValidator.validate

def hooked_validate_debug(self, X_u, y_u, top_k_candidates, depth, max_depth, total_samples,
                          competitive_estimator, reflection_engine):
    
    # 1. Run Original
    outcome_orig = original_validate(
        self, X_u, y_u, top_k_candidates, depth, max_depth, total_samples,
        competitive_estimator, reflection_engine
    )
    
    # We will simulate the optimized path manually to capture details
    n_samples = len(y_u)
    
    # Recreate the optimized calculations
    from core.metacognition.reflection_engine import check_random_state
    rng = check_random_state(reflection_engine.random_state)
    
    bootstrap_selections = []
    bootstrap_runs = []

    for _ in range(reflection_engine.B):
        indices = reflection_engine._get_bootstrap_indices(y_u, rng)
        X_b = X_u.iloc[indices].reset_index(drop=True)
        y_b = y_u.iloc[indices].reset_index(drop=True)

        try:
            feat_b, thresh_b, _ = reflection_engine.finder.find_best_split(X_b, y_b)
            if feat_b is not None:
                bootstrap_selections.append(feat_b)
                bootstrap_runs.append((feat_b, thresh_b))
        except Exception:
            continue

    total_valid = len(bootstrap_selections)
    if total_valid > 0:
        unique_feats, counts = np.unique(bootstrap_selections, return_counts=True)
        p_vector = {feat: float(count / total_valid) for feat, count in zip(unique_feats, counts)}
    else:
        p_vector = {}

    max_gr = max(float(c.get("gain_ratio", 0.0)) for c in top_k_candidates)
    gain_ratios_map = {c["feature"]: float(c.get("gain_ratio", 0.0)) for c in top_k_candidates}
    alpha, beta = self.scs_calculator.calculate_elasticities(depth, max_depth, n_samples, total_samples)

    evaluated_candidates = []
    for cand in top_k_candidates:
        feat = cand["feature"]
        comp_meta = competitive_estimator.estimate_competition(gain_ratios_map, feat)
        ci = comp_meta["competitiveness_index"]
        
        c_gr_norm = float(cand.get("gain_ratio", 0.0)) / (max_gr + 1e-9)

        if total_valid == 0:
            ns, ts, p_star = 1.0, 1.0, 1.0
            c_bootstrap_thresholds = []
        else:
            p_star = float(p_vector.get(feat, 0.0))
            c_bootstrap_thresholds = [thresh_b for f_b, thresh_b in bootstrap_runs if f_b == feat]
            
            eval_payload = {
                "feature": feat,
                "p_star": p_star,
                "p_vector": p_vector,
                "bootstrap_thresholds": c_bootstrap_thresholds
            }
            ns = reflection_engine.ns_module.evaluate(X_u, y_u, eval_payload, None)
            ts = reflection_engine.ts_module.evaluate(X_u, y_u, eval_payload, None)

        scs = self.scs_calculator.calculate_scs(c_gr_norm, ns, ts, ci, alpha, beta)

        evaluated_candidates.append({
            "feature": feat,
            "ns": ns,
            "ts": ts,
            "p_star": p_star,
            "scs": scs,
            "p_vector": p_vector,
            "bootstrap_thresholds": c_bootstrap_thresholds
        })

    # Find matching original candidate details
    print(f"\n--- DEPTH {depth} COMPARISON ---")
    print(f"X_u shape: {X_u.shape} | y_u shape: {y_u.shape}")
    print(f"Original selected: {outcome_orig.feature_selected} | SCS: {outcome_orig.scs:.4f} | Verdict: {outcome_orig.verdict.value}")
    
    # Sort opt candidates by SCS
    evaluated_candidates.sort(key=lambda x: x["scs"], reverse=True)
    best_opt = evaluated_candidates[0]
    best_opt_verdict = "ACCEPT" if best_opt["scs"] >= self.theta_accept else ("COLLAPSE" if best_opt["scs"] < self.theta_reject else "DEFER")
    print(f"Optimized selected: {best_opt['feature']} | SCS: {best_opt['scs']:.4f} | Verdict: {best_opt_verdict}")
    
    print("\nCandidates details comparison:")
    for c_orig in outcome_orig.top_k_candidates:
        c_opt = next((c for c in evaluated_candidates if c["feature"] == c_orig["feature"]), None)
        print(f"  Feature: {c_orig['feature']}")
        print(f"    Original:  ns={c_orig['ns']:.4f} | ts={c_orig['ts']:.4f} | scs={c_orig['scs']:.4f} | p_star={c_orig.get('p_star', 0.0):.4f}")
        if c_opt:
            print(f"    Optimized: ns={c_opt['ns']:.4f} | ts={c_opt['ts']:.4f} | scs={c_opt['scs']:.4f} | p_star={c_opt['p_star']:.4f}")
            print(f"    p_vector Match: {c_orig.get('p_vector', {}) == c_opt['p_vector']}")
            print(f"    thresholds count match: {len(c_orig.get('bootstrap_thresholds', [])) == len(c_opt['bootstrap_thresholds'])}")
            if len(c_orig.get('bootstrap_thresholds', [])) != len(c_opt['bootstrap_thresholds']):
                print(f"      Orig thresholds: {c_orig.get('bootstrap_thresholds', [])[:5]}")
                print(f"      Opt thresholds:  {c_opt['bootstrap_thresholds'][:5]}")
    
    # Return the original outcome to keep tree building going
    return outcome_orig

DecisionValidator.validate = hooked_validate_debug

def main():
    path = os.path.join(ROOT, 'experiments', 'data', 'diabetes.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    splits = data['cv_splits']
    train_idx, test_idx = splits[0]
    
    # Use config that does not collapse to stump immediately (e.g. low theta_reject)
    cfg = {
        'theta_accept': 0.30,
        'theta_reject': 0.05,
        'gamma': 0.20,
        'B': 50,
        'random_state': 42
    }
    
    clf = C45Classifier(mode='metacog', **cfg)
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])

if __name__ == '__main__':
    main()
