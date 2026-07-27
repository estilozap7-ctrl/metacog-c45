"""
=========================================================
MetaCog-C45 Experimental Framework
Metrics Utilities
=========================================================
Calcula el set completo de métricas Q1:
Accuracy, Balanced Accuracy, F1, MCC, ROC-AUC, PR-AUC.
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
    Asume clasificación binaria o multiclase.
    """
    
    # Determinar si es binario o multiclase basado en las probabilidades
    is_multiclass = y_prob.shape[1] > 2
    
    metrics = {}
    
    # Métricas hard
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    if is_multiclass:
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # ROC-AUC multiclase (OVR)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            metrics['roc_auc'] = np.nan
            
        # PR-AUC no está soportado nativamente en multiclase directo en sklearn por defecto
        # Calculamos macro-promedio manual o devolvemos NaN
        metrics['pr_auc'] = np.nan 
    else:
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Clase positiva es la clase 1 (índice 1 en probas)
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
