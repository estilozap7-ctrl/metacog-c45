"""
=========================================================
MetaCog-C45 Experimental Framework
Pipeline: metrics.py
=========================================================
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

def calculate_all_metrics(y_true, y_pred, y_prob):
    """
    Calcula las métricas exigidas por el Protocolo Q1.
    """
    is_multiclass = y_prob.shape[1] > 2
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if is_multiclass:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan 
    else:
        prob_pos = y_prob[:, 1]
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, prob_pos)
        except ValueError:
            metrics['roc_auc'] = np.nan
        try:
            metrics['pr_auc'] = average_precision_score(y_true, prob_pos)
        except ValueError:
            metrics['pr_auc'] = np.nan

    return metrics
