"""
=========================================================
MetaCog-C45 Experimental Framework
Pipeline: statistics.py
=========================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan
    delta = sum(np.sum(val_x > y) - np.sum(val_x < y) for val_x in x)
    return delta / (n * m)

def iman_davenport_test(friedman_stat, k, N):
    numerator = (N - 1) * friedman_stat
    denominator = N * (k - 1) - friedman_stat
    if denominator == 0:
        return np.inf
    return numerator / denominator

def run_statistical_suite(df_results, metrics=['mcc', 'nodes', 'infer_ms']):
    """
    Ejecuta el protocolo estadístico completo sobre los resultados.
    """
    stats_out = []
    datasets = df_results['dataset'].unique()
    
    for metric in metrics:
        for ds in datasets:
            df_ds = df_results[df_results['dataset'] == ds]
            val_c = df_ds[df_ds['mode'] == 'classic'][metric].values
            val_m = df_ds[df_ds['mode'] == 'metacog'][metric].values
            
            if len(val_c) == 0 or len(val_m) == 0:
                continue
                
            try:
                stat_w, p_val_w = wilcoxon(val_c, val_m)
            except ValueError:
                stat_w, p_val_w = 0, 1.0
                
            cd = cliffs_delta(val_m, val_c)
            
            stats_out.append({
                'dataset': ds,
                'metric': metric,
                'wilcoxon_p': p_val_w,
                'cliffs_delta': cd
            })
            
    # La validación global Iman-Davenport requiere que se haga al final agrupando datasets
    return pd.DataFrame(stats_out)
