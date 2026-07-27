"""
=========================================================
MetaCog-C45 Experimental Framework
Pipeline: run_dataset.py (Optimizado en Paralelo)
=========================================================
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from .run_fold import run_fold

# -------------------------------------------------------
# Subclase TQDM para joblib.Parallel
# -------------------------------------------------------
class tqdm_parallel(Parallel):
    def __init__(self, use_tqdm=True, tqdm_kwargs=None, *args, **kwargs):
        self.use_tqdm = use_tqdm
        self.tqdm_kwargs = tqdm_kwargs or {}
        if use_tqdm and kwargs.get('verbose', 0) == 0:
            kwargs['verbose'] = 1
        super().__init__(*args, **kwargs)
        if use_tqdm:
            self._print = lambda msg, *args, **kwargs: None

    def __call__(self, iterable):
        if not self.use_tqdm:
            return super().__call__(iterable)
        with tqdm(**self.tqdm_kwargs) as self._pbar:
            return super().__call__(iterable)

    def print_progress(self):
        if self.use_tqdm:
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()
        else:
            super().print_progress()

# -------------------------------------------------------
# Función auxiliar a nivel de módulo (requerida por Loky)
# -------------------------------------------------------
def execute_single_fold_pair(X_train, y_train, X_test, y_test, name, fold_idx, seed, metacog_params=None):
    """
    Ejecuta sequentially classic y metacog para un fold específico.
    """
    # C4.5 Classic — siempre con parámetros por defecto
    res_classic = run_fold(X_train, y_train, X_test, y_test, mode='classic', random_state=seed)
    res_classic['dataset'] = name
    res_classic['fold'] = fold_idx

    # MetaCog-C45 — con hiperparámetros inyectados
    res_metacog = run_fold(
        X_train, y_train, X_test, y_test,
        mode='metacog',
        random_state=seed,
        metacog_params=metacog_params
    )
    res_metacog['dataset'] = name
    res_metacog['fold'] = fold_idx

    return res_classic, res_metacog

# -------------------------------------------------------
# Ejecución del dataset completo
# -------------------------------------------------------
def run_dataset(dataset_path, n_jobs=-1, master_seed=42, lote_num=1, total_lotes=3, dataset_idx=1, total_datasets=10, metacog_params=None):
    """
    Ejecuta el protocolo 5x10-CV para un dataset específico en paralelo.
    """
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        
    name = data['name']
    X = data['X']
    y = data['y']
    splits = data['cv_splits']
    
    # Derivación determinista de semillas locales por fold
    fold_seeds = [int(master_seed + fold_idx) for fold_idx in range(len(splits))]
    
    tqdm_kwargs = {
        'total': len(splits),
        'desc': f"Lote {lote_num}/{total_lotes} | Ds {dataset_idx}/{total_datasets} | {name:<15}",
        'unit': 'fold',
        'leave': False
    }
    
    # Ejecución paralela
    results_pairs = tqdm_parallel(n_jobs=n_jobs, tqdm_kwargs=tqdm_kwargs)(
        delayed(execute_single_fold_pair)(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[test_idx], y.iloc[test_idx],
            name, fold_idx, fold_seeds[fold_idx],
            metacog_params=metacog_params
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splits)
    )
    
    # Aplanar la lista de pares (classic, metacog)
    flat_results = []
    for classic_res, metacog_res in results_pairs:
        flat_results.append(classic_res)
        flat_results.append(metacog_res)
        
    return pd.DataFrame(flat_results)
