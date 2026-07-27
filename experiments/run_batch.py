"""
=========================================================
MetaCog-C45 Experimental Framework
Batch Runner (Official Campaign)
=========================================================
"""

import os
import sys
import time
import pickle
import pandas as pd
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.classifier import C45Classifier
from experiments.metrics_utils import calculate_all_metrics
from core.tree_utils import TreeUtils

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def extract_structural_metrics(tree):
    return {
        'nodes': TreeUtils.count_nodes(tree),
        'leaves': TreeUtils.count_leaves(tree),
        'depth': TreeUtils.depth(tree)
    }

def run_batch(lote_num):
    print("=" * 60)
    print(f"EJECUTANDO CAMPAÑA OFICIAL - LOTE {lote_num}")
    print("=" * 60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lote_dir = os.path.join(RESULTS_DIR, f"lote{lote_num}")
    os.makedirs(lote_dir, exist_ok=True)
    
    # Obtener orden exacto
    dataset_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")])
    if not dataset_files:
        print("ERROR: Ejecute setup_datasets.py primero.")
        return
        
    start_idx = (lote_num - 1) * 10
    end_idx = lote_num * 10
    batch_files = dataset_files[start_idx:end_idx]
    
    if not batch_files:
        print(f"Lote {lote_num} vacío.")
        return

    all_results = []
    
    for df_file in batch_files:
        path = os.path.join(DATA_DIR, df_file)
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        name = data['name']
        X, y, splits = data['X'], data['y'], data['cv_splits']
        
        print(f"\n[{name}] Procesando 50-folds...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # CLASSIC
            t0 = time.time()
            clf_classic = C45Classifier(mode='classic')
            clf_classic.fit(X_train, y_train)
            t_train_c = (time.time() - t0) * 1000
            
            t0 = time.time()
            y_pred_c = clf_classic.predict(X_test)
            y_prob_c = clf_classic.predict_proba(X_test)
            t_inf_c = (time.time() - t0) * 1000
            
            mets_c = calculate_all_metrics(y_test, y_pred_c, y_prob_c)
            strc_c = extract_structural_metrics(clf_classic.root)
            
            # METACOG
            t0 = time.time()
            clf_meta = C45Classifier(mode='metacog')
            clf_meta.fit(X_train, y_train)
            t_train_m = (time.time() - t0) * 1000
            
            t0 = time.time()
            y_pred_m = clf_meta.predict(X_test)
            y_prob_m = clf_meta.predict_proba(X_test)
            t_inf_m = (time.time() - t0) * 1000
            
            mets_m = calculate_all_metrics(y_test, y_pred_m, y_prob_m)
            strc_m = extract_structural_metrics(clf_meta.root)
            
            all_results.append({'dataset': name, 'fold': fold_idx, 'mode': 'classic', **mets_c, **strc_c, 'train_ms': t_train_c, 'infer_ms': t_inf_c})
            all_results.append({'dataset': name, 'fold': fold_idx, 'mode': 'metacog', **mets_m, **strc_m, 'train_ms': t_train_m, 'infer_ms': t_inf_m})
                
        print(f"[{name}] Completado.")
        
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(lote_dir, "results.csv"), index=False)
    df_results.to_json(os.path.join(lote_dir, "results.json"), orient='records', indent=2)
    print(f"Lote {lote_num} guardado en {lote_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lote", type=int, required=True, help="Número de lote (1, 2, 3)")
    args = parser.parse_args()
    run_batch(args.lote)
