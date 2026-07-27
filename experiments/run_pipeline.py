"""
=========================================================
MetaCog-C45 Experimental Framework
Pipeline Runner (ERR Pilot Version)
=========================================================
Orquesta la inducción, validación cruzada y recolección
de métricas sobre los datasets piloto.
=========================================================
"""

import os
import sys
import time
import pickle
import pandas as pd
import json

# Asegurar que el path alcance core/
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

def run_pilot_experiment():
    print("=" * 60)
    print("INICIANDO EJECUCIÓN DEL PIPELINE EXPERIMENTAL (ERR)")
    print("=" * 60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Obtener datasets piloto
    dataset_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]
    if not dataset_files:
        print("ERROR: No hay datasets procesados. Ejecute setup_datasets.py primero.")
        return
        
    all_results = []
    
    for df_file in dataset_files:
        path = os.path.join(DATA_DIR, df_file)
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        name = data['name']
        X = data['X']
        y = data['y']
        splits = data['cv_splits']
        
        print(f"\n[{name}] Procesando 50-folds...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # 1. CLASSIC
            start_train = time.time()
            clf_classic = C45Classifier(mode='classic')
            clf_classic.fit(X_train, y_train)
            time_train_classic = (time.time() - start_train) * 1000
            
            start_infer = time.time()
            y_pred_c = clf_classic.predict(X_test)
            y_prob_c = clf_classic.predict_proba(X_test)
            time_infer_classic = (time.time() - start_infer) * 1000
            
            metrics_c = calculate_all_metrics(y_test, y_pred_c, y_prob_c)
            struct_c = extract_structural_metrics(clf_classic.root)
            
            # 2. METACOG
            start_train = time.time()
            clf_meta = C45Classifier(mode='metacog')
            clf_meta.fit(X_train, y_train)
            time_train_meta = (time.time() - start_train) * 1000
            
            start_infer = time.time()
            y_pred_m = clf_meta.predict(X_test)
            y_prob_m = clf_meta.predict_proba(X_test)
            time_infer_meta = (time.time() - start_infer) * 1000
            
            metrics_m = calculate_all_metrics(y_test, y_pred_m, y_prob_m)
            struct_m = extract_structural_metrics(clf_meta.root)
            
            # Registro en tabla
            for mode, mets, struc, t_tr, t_in in [
                ('classic', metrics_c, struct_c, time_train_classic, time_infer_classic),
                ('metacog', metrics_m, struct_m, time_train_meta, time_infer_meta)
            ]:
                row = {
                    'dataset': name,
                    'fold': fold_idx,
                    'mode': mode,
                    **mets,
                    **struc,
                    'train_ms': t_tr,
                    'infer_ms': t_in
                }
                all_results.append(row)
                
        print(f"[{name}] Completado.")
        
    df_results = pd.DataFrame(all_results)
    
    # Guardar resultados
    csv_path = os.path.join(RESULTS_DIR, "err_pilot_results.csv")
    json_path = os.path.join(RESULTS_DIR, "err_pilot_results.json")
    
    df_results.to_csv(csv_path, index=False)
    df_results.to_json(json_path, orient='records', indent=2)
    
    print("\n" + "=" * 60)
    print(f"RESULTADOS SERIALIZADOS EN: {RESULTS_DIR}")
    print("=" * 60)
    
    # Resumen rápido MD
    resumen = df_results.groupby(['dataset', 'mode']).agg({
        'mcc': 'mean',
        'f1': 'mean',
        'nodes': 'mean',
        'train_ms': 'mean'
    }).reset_index()
    
    print("\nResumen Rápido (Medias):")
    print(resumen.to_string(index=False))

if __name__ == "__main__":
    run_pilot_experiment()
