"""
=========================================================
MetaCog-C45 Experimental Framework
test_equivalence.py — Prueba de Equivalencia Científica
=========================================================
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.pipeline.run_dataset import run_dataset

def main():
    print("=" * 60)
    print("INICIANDO PRUEBA DE EQUIVALENCIA CIENTÍFICA (IRIS)")
    print("=" * 60)
    
    iris_path = os.path.join(ROOT, "experiments", "data", "iris.pkl")
    if not os.path.exists(iris_path):
        print(f"ERROR: {iris_path} no existe. Ejecuta setup_datasets.py primero.")
        sys.exit(1)
        
    print("1. Ejecutando pipeline en modo SERIAL (n_jobs=1)...")
    df_serial = run_dataset(iris_path, n_jobs=1, master_seed=42)
    
    print("2. Ejecutando pipeline en modo PARALELO (n_jobs=4)...")
    df_parallel = run_dataset(iris_path, n_jobs=4, master_seed=42)
    
    print("\n3. Validando equivalencia de resultados...")
    
    # Asegurar orden idéntico para la comparación directa
    df_serial = df_serial.sort_values(by=['fold', 'mode']).reset_index(drop=True)
    df_parallel = df_parallel.sort_values(by=['fold', 'mode']).reset_index(drop=True)
    
    # Verificar columnas comunes
    common_cols = [c for c in df_serial.columns if c not in ['train_ms', 'infer_ms']]
    
    mismatches = 0
    for col in common_cols:
        vals_s = df_serial[col].values
        vals_p = df_parallel[col].values
        
        # Comparar con tolerancia para punto flotante
        if df_serial[col].dtype in [float, int, np.float64, np.int64]:
            is_equal = np.allclose(vals_s, vals_p, equal_nan=True)
        else:
            # Para strings o categóricos
            is_equal = np.array_equal(vals_s, vals_p)
            
        if not is_equal:
            print(f"  [FAIL] Inconsistencia detectada en columna: {col}")
            mismatches += 1
        else:
            print(f"  [OK] Columna '{col}' es 100% idéntica.")
            
    print("\n" + "=" * 60)
    if mismatches == 0:
        print("  ESTADO DE LA PRUEBA: EQUIVALENCIA CIENTÍFICA VERIFICADA")
        print("  No hay diferencias en métricas predictivas, SCS ni topología del árbol.")
    else:
        print("  ESTADO DE LA PRUEBA: FALLO EN EQUIVALENCIA")
    print("=" * 60)

if __name__ == '__main__':
    main()
