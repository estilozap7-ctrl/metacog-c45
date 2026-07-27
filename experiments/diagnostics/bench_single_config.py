"""
Benchmark: single evaluate_config_loky call with all 4 datasets, 5 folds.
Measures exactly how long one grid-search config takes end-to-end.
"""
import os
import sys
import time
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.run_recalibration_grid import evaluate_config_loky

ds_names = ['iris', 'wine', 'breast_cancer', 'adult']
data_dir  = os.path.join(ROOT, 'experiments', 'data')
cfg       = {'theta_accept': 0.7, 'theta_reject': 0.3, 'gamma': 0.5}

print(f'ROOT: {ROOT}')
print(f'Config: {cfg}')
print(f'Datasets: {ds_names}')
print(f'Folds: 5')
print()

# --- Per-dataset sequential timing ---
print('=== Per-dataset timing (1 fold each) ===')
for ds in ds_names:
    t0 = time.time()
    r = evaluate_config_loky(cfg, [ds], data_dir, 1, 42)
    elapsed = time.time() - t0
    print(f'  {ds}: {len(r)} records in {elapsed:.2f}s')

print()

# --- Full config timing (4 datasets, 5 folds) ---
print('=== Full config: 4 datasets x 5 folds ===')
t0 = time.time()
result = evaluate_config_loky(cfg, ds_names, data_dir, 5, 42)
elapsed = time.time() - t0
print(f'Done: {len(result)} records in {elapsed:.2f}s ({elapsed/len(result):.2f}s/fold avg)')
by_ds = {}
for r in result:
    by_ds.setdefault(r['dataset'], []).append(r)
for ds, rows in by_ds.items():
    print(f'  {ds}: {len(rows)} folds')
