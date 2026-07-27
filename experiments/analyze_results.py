"""
=========================================================
MetaCog-C45 Experimental Framework
Statistical Analyzer (ERR Pilot Version)
=========================================================
Pruebas Estadísticas Requeridas por el Protocolo Q1:
Iman-Davenport, Holm, Wilcoxon, Cliff's Delta.
=========================================================
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def cliffs_delta(x, y):
    """
    Calcula el tamaño del efecto no paramétrico Cliff's Delta.
    1.0 indica que todos los valores en x son mayores que en y.
    -1.0 indica que todos los valores en x son menores que en y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    m = len(y)
    
    delta = 0
    for val_x in x:
        delta += np.sum(val_x > y) - np.sum(val_x < y)
        
    return delta / (n * m)

def iman_davenport_test(friedman_stat, k, N):
    """
    Calcula la corrección de Iman-Davenport sobre el test de Friedman.
    """
    numerator = (N - 1) * friedman_stat
    denominator = N * (k - 1) - friedman_stat
    if denominator == 0:
        return np.inf
    return numerator / denominator

def analyze_pilot_results():
    print("=" * 60)
    print("ANALIZADOR ESTADÍSTICO - ERR PILOT")
    print("=" * 60)
    
    csv_path = os.path.join(RESULTS_DIR, "err_pilot_results.csv")
    if not os.path.exists(csv_path):
        print("ERROR: No se encontró el archivo de resultados.")
        return
        
    df = pd.read_csv(csv_path)
    
    metrics_to_test = ['mcc', 'nodes', 'train_ms']
    datasets = df['dataset'].unique()
    
    for metric in metrics_to_test:
        print(f"\n--- Análisis de la métrica: {metric.upper()} ---")
        
        for ds in datasets:
            df_ds = df[df['dataset'] == ds]
            
            classic_vals = df_ds[df_ds['mode'] == 'classic'][metric].values
            metacog_vals = df_ds[df_ds['mode'] == 'metacog'][metric].values
            
            if len(classic_vals) == 0 or len(metacog_vals) == 0:
                continue
                
            # Wilcoxon Signed-Rank
            try:
                stat, p_val = wilcoxon(classic_vals, metacog_vals)
            except ValueError:
                # Si todos los valores son identicos
                stat, p_val = 0, 1.0
                
            # Cliff's Delta
            delta = cliffs_delta(metacog_vals, classic_vals)
            
            # Interpretar Delta
            if abs(delta) < 0.147: effect = "Despreciable"
            elif abs(delta) < 0.33: effect = "Pequeño"
            elif abs(delta) < 0.474: effect = "Mediano"
            else: effect = "Grande"
            
            print(f"Dataset: {ds:<15} | Wilcoxon p: {p_val:.4f} | Cliff's Delta: {delta:+.3f} ({effect})")
            
    print("\n" + "=" * 60)
    print("ESTADO: Listo para Holm y CD-Diagrams en fase final.")
    print("=" * 60)

if __name__ == "__main__":
    analyze_pilot_results()
