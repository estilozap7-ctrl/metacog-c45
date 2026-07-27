"""
=========================================================
MetaCog-C45 Experimental Framework
Pipeline: run_fold.py
=========================================================
"""

import time
import pandas as pd
from core.classifier import C45Classifier
from core.tree_utils import TreeUtils
from .metrics import calculate_all_metrics

def extract_structural_metrics(tree):
    return {
        'nodes': TreeUtils.count_nodes(tree),
        'leaves': TreeUtils.count_leaves(tree),
        'depth': TreeUtils.depth(tree)
    }

def run_fold(X_train, y_train, X_test, y_test, mode, random_state=None, metacog_params=None):
    """
    Ejecuta un solo fold para el modo indicado.
    """
    t0 = time.time()
    params = metacog_params if metacog_params else {}
    clf = C45Classifier(mode=mode, random_state=random_state, **params)
    clf.fit(X_train, y_train)
    t_train = (time.time() - t0) * 1000
    
    t0 = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    t_infer = (time.time() - t0) * 1000
    
    metrics = calculate_all_metrics(y_test, y_pred, y_prob)
    structural = extract_structural_metrics(clf.root)
    
    # Extraer métricas metacognitivas si existen
    if mode == 'metacog' and clf.experience_stats_:
        metrics['avg_ns'] = clf.experience_stats_.get('avg_ns', 0.0)
        metrics['avg_ts'] = clf.experience_stats_.get('avg_ts', 0.0)
    else:
        metrics['avg_ns'] = 0.0
        metrics['avg_ts'] = 0.0
        
    return {
        'mode': mode,
        **metrics,
        **structural,
        'train_ms': t_train,
        'infer_ms': t_infer
    }
