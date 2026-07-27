"""
=========================================================
MetaCog-C45 — Profiling and Diagnostic Tool
experiments/diagnose_profile.py
=========================================================
Profiles the calibration pipeline execution times stage by stage
on exactly one fold and one configuration.

Breakdowns:
- Main calibration stages (Dataset load, Fit/Predict, export)
- Internal classifier functions (BestSplitFinder, ReflectionEngine, etc.)
- ReflectionEngine internal loop components (Bootstrap, NS, TS)
=========================================================
"""

from __future__ import annotations

import os
import sys
import time
import json
import pickle
import pandas as pd
import numpy as np

# Path setup
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.classifier import C45Classifier
from core.metacognition.reflection_engine import ReflectionEngine
from core.metacognition.decision_core import SplitConfidenceScore, CompetitiveEstimator, DecisionValidator
from core.best_split import BestSplitFinder

# Hook timers for internal profiling
timings = {
    'best_split_finder_total': 0.0,
    'best_split_finder_calls': 0,
    
    'reflection_engine_total': 0.0,
    'reflection_engine_calls': 0,
    
    'bootstrap_generation': 0.0,
    'bootstrap_finder_calls': 0,
    'bootstrap_finder_total': 0.0,
    
    'ns_evaluate': 0.0,
    'ts_evaluate': 0.0,
    
    'ci_evaluate': 0.0,
    'ci_calls': 0,
    
    'scs_evaluate': 0.0,
    'scs_calls': 0
}

# --- Monkey-Patch to intercept and time internal calls ---

# 1. BestSplitFinder
original_find_best_split = BestSplitFinder.find_best_split
def patched_find_best_split(self, X, y):
    t0 = time.time()
    res = original_find_best_split(self, X, y)
    timings['best_split_finder_total'] += (time.time() - t0) * 1000
    timings['best_split_finder_calls'] += 1
    return res
BestSplitFinder.find_best_split = patched_find_best_split

# 2. ReflectionEngine.reflect
original_reflect = ReflectionEngine.reflect
def patched_reflect(self, X_u, y_u, candidate, alternate=None):
    t0 = time.time()
    
    # We also want to measure internal parts of reflect, so we partially inline or measure within the call.
    # To do this cleanly, we can time the sub-calls inside patched_reflect by hooking their modules.
    res = original_reflect(self, X_u, y_u, candidate, alternate)
    
    timings['reflection_engine_total'] += (time.time() - t0) * 1000
    timings['reflection_engine_calls'] += 1
    return res
ReflectionEngine.reflect = patched_reflect

# 3. ReflectionEngine._get_bootstrap_indices
original_get_bootstrap = ReflectionEngine._get_bootstrap_indices
def patched_get_bootstrap(self, y_u, rng):
    t0 = time.time()
    res = original_get_bootstrap(self, y_u, rng)
    timings['bootstrap_generation'] += (time.time() - t0) * 1000
    return res
ReflectionEngine._get_bootstrap_indices = patched_get_bootstrap

# We also want to intercept BestSplitFinder calls THAT OCCUR INSIDE the bootstrap loop.
# We can do this by wrapping find_best_split inside reflect.
# Since we already patched find_best_split globally, we can track if it's called inside reflect.
in_reflect_loop = False
def patched_find_best_split_detailed(self, X, y):
    global in_reflect_loop
    t0 = time.time()
    res = original_find_best_split(self, X, y)
    dt = (time.time() - t0) * 1000
    if in_reflect_loop:
        timings['bootstrap_finder_total'] += dt
        timings['bootstrap_finder_calls'] += 1
    else:
        timings['best_split_finder_total'] += dt
        timings['best_split_finder_calls'] += 1
    return res
BestSplitFinder.find_best_split = patched_find_best_split_detailed

# Wrap reflect to set/unset in_reflect_loop
def patched_reflect_loop_wrapper(self, X_u, y_u, candidate, alternate=None):
    global in_reflect_loop
    t0 = time.time()
    
    # We will time NS and TS evaluations by timing their evaluate methods
    # But since we want to know what portion of reflect they take:
    in_reflect_loop = True
    res = original_reflect(self, X_u, y_u, candidate, alternate)
    in_reflect_loop = False
    
    timings['reflection_engine_total'] += (time.time() - t0) * 1000
    timings['reflection_engine_calls'] += 1
    return res
ReflectionEngine.reflect = patched_reflect_loop_wrapper

# 4. NS evaluate
from core.metacognition.reflection_engine import NodeStabilityModule
original_ns_eval = NodeStabilityModule.evaluate
def patched_ns_eval(self, X_u, y_u, candidate, alternate=None):
    t0 = time.time()
    res = original_ns_eval(self, X_u, y_u, candidate, alternate)
    timings['ns_evaluate'] += (time.time() - t0) * 1000
    return res
NodeStabilityModule.evaluate = patched_ns_eval

# 5. TS evaluate
from core.metacognition.reflection_engine import ThresholdSurvivalModule
original_ts_eval = ThresholdSurvivalModule.evaluate
def patched_ts_eval(self, X_u, y_u, candidate, alternate=None):
    t0 = time.time()
    res = original_ts_eval(self, X_u, y_u, candidate, alternate)
    timings['ts_evaluate'] += (time.time() - t0) * 1000
    return res
ThresholdSurvivalModule.evaluate = patched_ts_eval

# 6. CI evaluate (estimate_competition)
original_ci_eval = CompetitiveEstimator.estimate_competition
def patched_ci_eval(self, feature_gain_ratios, best_feature):
    t0 = time.time()
    res = original_ci_eval(self, feature_gain_ratios, best_feature)
    timings['ci_evaluate'] += (time.time() - t0) * 1000
    timings['ci_calls'] += 1
    return res
CompetitiveEstimator.estimate_competition = patched_ci_eval


# 7. SCS calculate
original_scs_calc = SplitConfidenceScore.calculate_scs
def patched_scs_calc(self, gr_normalized, ns, ts, ci, alpha, beta):
    t0 = time.time()
    res = original_scs_calc(self, gr_normalized, ns, ts, ci, alpha, beta)
    timings['scs_evaluate'] += (time.time() - t0) * 1000
    timings['scs_calls'] += 1
    return res
SplitConfidenceScore.calculate_scs = patched_scs_calc


def main():
    print("=" * 70)
    print("  MetaCog-C45 — PERFORMANCE PROFILING AND DIAGNOSTICS")
    print("=" * 70)

    # Config parameters to evaluate
    ds_name = "diabetes"
    cfg = {
        'theta_accept': 0.30,
        'theta_reject': 0.05,
        'gamma': 0.20
    }
    seed = 42

    stages_time = {}

    # 1. Dataset Loading
    t0 = time.time()
    ds_path = os.path.join(ROOT, 'experiments', 'data', f'{ds_name}.pkl')
    with open(ds_path, 'rb') as f:
        data = pickle.load(f)
    stages_time['Dataset loading'] = (time.time() - t0) * 1000

    # 2. Loading CV splits
    t0 = time.time()
    X = data['X']
    y = data['y']
    splits = data['cv_splits']
    train_idx, test_idx = splits[0]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_test  = y.iloc[test_idx]
    stages_time['Loading CV splits'] = (time.time() - t0) * 1000

    # 3. Classic C4.5 Training
    clf_classic = C45Classifier(mode='classic', random_state=seed)
    t0 = time.time()
    clf_classic.fit(X_train, y_train)
    stages_time['Classic training'] = (time.time() - t0) * 1000

    # 4. Classic C4.5 Prediction
    t0 = time.time()
    y_pred_c = clf_classic.predict(X_test)
    y_prob_c = clf_classic.predict_proba(X_test)
    stages_time['Classic prediction'] = (time.time() - t0) * 1000

    # Cache metrics for utility calculation
    from experiments.pipeline.metrics import calculate_all_metrics
    classic_metrics = calculate_all_metrics(y_test, y_pred_c, y_prob_c)
    classic_mcc = {'diabetes': classic_metrics['mcc']}
    from core.tree_utils import TreeUtils
    classic_nodes = {'diabetes': TreeUtils.count_nodes(clf_classic.root)}

    # Reset timings before MetaCog to isolate
    for k in timings:
        timings[k] = 0.0

    # 5. MetaCog C4.5 Training
    clf_metacog = C45Classifier(mode='metacog', random_state=seed, **cfg)
    t0 = time.time()
    clf_metacog.fit(X_train, y_train)
    stages_time['MetaCog training'] = (time.time() - t0) * 1000

    # 6. MetaCog C4.5 Prediction
    t0 = time.time()
    y_pred_m = clf_metacog.predict(X_test)
    y_prob_m = clf_metacog.predict_proba(X_test)
    stages_time['MetaCog prediction'] = (time.time() - t0) * 1000

    metacog_metrics = calculate_all_metrics(y_test, y_pred_m, y_prob_m)
    metacog_nodes = TreeUtils.count_nodes(clf_metacog.root)

    # Build dummy DataFrame with single fold result to time utility computation
    res_fold = {
        'dataset': ds_name,
        'mcc': metacog_metrics['mcc'],
        'nodes': metacog_nodes
    }
    df_fold = pd.DataFrame([res_fold])

    # 7. Utility computation
    t0 = time.time()
    # Mock calculate_utility from run_recalibration_grid
    w = 0.20
    target_cr = 0.40
    stump_val = 1.0
    
    cr = metacog_nodes / classic_nodes[ds_name] if classic_nodes[ds_name] > 0 else 1.0
    phi = abs(cr - target_cr)
    psi = stump_val if np.isclose(metacog_nodes, 1.0, atol=0.05) else 0.0
    util = metacog_metrics['mcc'] - w * phi - psi
    stages_time['Utility'] = (time.time() - t0) * 1000

    # 8. Checkpoint writing
    t0 = time.time()
    checkpoint_data = {
        'completed': ['ta_0.30_tr_0.05_g_0.20'],
        'folds': [res_fold]
    }
    cp_path = os.path.join(ROOT, 'experiments', 'results', 'recalibration_checkpoint_test.json')
    with open(cp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)
    stages_time['Checkpoint'] = (time.time() - t0) * 1000

    # 9. CSV writing (Export stage)
    t0 = time.time()
    csv_path = os.path.join(ROOT, 'experiments', 'results', 'recalibration_grid_results_test.csv')
    df_fold.to_csv(csv_path, index=False)
    stages_time['Export'] = (time.time() - t0) * 1000

    # Clean up test output files
    if os.path.exists(cp_path):
        os.remove(cp_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Print results
    print("\n" + "=" * 50)
    print(f"{'Stage':<25} {'Time (ms)':<15}")
    print("=" * 50)
    total_pipeline_time = 0.0
    for stage, t in stages_time.items():
        print(f"{stage:<25} {t:<15.2f}")
        total_pipeline_time += t
    print("-" * 50)
    print(f"{'Total Pipeline Time':<25} {total_pipeline_time:<15.2f}")
    print("=" * 50)

    # Identify bottleneck stage
    sorted_stages = sorted(stages_time.items(), key=lambda x: x[1], reverse=True)
    bottleneck_stage, bottleneck_t = sorted_stages[0]
    pct = (bottleneck_t / total_pipeline_time) * 100
    print(f"\n[DIAGNOSIS] Largest Bottleneck Stage: {bottleneck_stage} ({pct:.1f}% of runtime)")

    # Detail timings inside MetaCog Training
    metacog_train_t = stages_time['MetaCog training']
    print(f"\n" + "=" * 50)
    print("  DETAILED INTERNAL TIMINGS OF METACOG TRAINING")
    print("=" * 50)
    
    # Time spent in tree expansion vs evaluation
    # We know best_split_finder is called for candidates, and reflection_engine is called for meta-validation
    print(f"{'Internal Function / Operation':<35} {'Calls':<8} {'Total Time (ms)':<15} {'% of Training':<10}")
    print("-" * 70)
    
    def print_line(name, calls, ms):
        pct_train = (ms / metacog_train_t) * 100 if metacog_train_t > 0 else 0.0
        print(f"{name:<35} {int(calls):<8} {ms:<15.2f} {pct_train:<10.1f}%")

    print_line("BestSplitFinder.find_best_split", timings['best_split_finder_calls'], timings['best_split_finder_total'])
    print_line("ReflectionEngine.reflect", timings['reflection_engine_calls'], timings['reflection_engine_total'])
    print_line("CompetitiveEstimator.evaluate (CI)", timings['ci_calls'], timings['ci_evaluate'])
    print_line("SplitConfidenceScore (SCS)", timings['scs_calls'], timings['scs_evaluate'])
    
    # Rest of time is C4.5 standard tree recursion/structural work
    timed_meta_total = (timings['best_split_finder_total'] + 
                        timings['reflection_engine_total'] + 
                        timings['ci_evaluate'] + 
                        timings['scs_evaluate'])
    structural_t = max(0.0, metacog_train_t - timed_meta_total)
    print_line("Tree recursion / Structure expansion", 1, structural_t)
    print("=" * 70)

    # Breakdown inside ReflectionEngine
    re_total = timings['reflection_engine_total']
    print(f"\n" + "=" * 50)
    print("  REFLECTION ENGINE INTERNAL BREAKDOWN")
    print("=" * 50)
    print(f"{'Operation':<35} {'Calls':<8} {'Total Time (ms)':<15} {'% of Engine':<10}")
    print("-" * 70)
    
    def print_re_line(name, calls, ms):
        pct_re = (ms / re_total) * 100 if re_total > 0 else 0.0
        print(f"{name:<35} {int(calls):<8} {ms:<15.2f} {pct_re:<10.1f}%")

    print_re_line("Bootstrap index generation", timings['reflection_engine_calls'] * 50, timings['bootstrap_generation'])
    print_re_line("Bootstrap best split finder", timings['bootstrap_finder_calls'], timings['bootstrap_finder_total'])
    print_re_line("NodeStability evaluation (NS)", timings['reflection_engine_calls'], timings['ns_evaluate'])
    print_re_line("ThresholdSurvival evaluation (TS)", timings['reflection_engine_calls'], timings['ts_evaluate'])
    
    timed_re_subtotal = (timings['bootstrap_generation'] + 
                         timings['bootstrap_finder_total'] + 
                         timings['ns_evaluate'] + 
                         timings['ts_evaluate'])
    overhead_re = max(0.0, re_total - timed_re_subtotal)
    print_re_line("Bootstrap loop overhead", timings['reflection_engine_calls'], overhead_re)
    print("=" * 70)

if __name__ == '__main__':
    main()
