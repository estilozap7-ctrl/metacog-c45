"""
=========================================================
MetaCog-C45 Experimental Framework
Dataset Preparation (v2.0 Frozen)
=========================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
N_SPLITS = 10
N_REPEATS = 5
RANDOM_STATE = 42

OPENML_DATASETS = [
    "iris", "wine", "breast-w", "adult", "bank-marketing", 
    "MagicTelescope", "mushroom", "tic-tac-toe", "nursery", "vehicle", 
    "letter", "pendigits", "credit-g", "diabetes", "spambase", 
    "kr-vs-kp", "hypothyroid", "sick", "segment", "sonar", 
    "ionosphere", "glass", "ecoli", "yeast", "balance-scale", 
    "car", "haberman", "blood-transfusion-service-center", "kc1", "pc1"
]

def preprocess_dataset(X, y):
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name in ['category', 'object']:
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            
    cat_cols = X.select_dtypes(include=['category', 'object', 'bool']).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder()
        X[cat_cols] = encoder.fit_transform(X[cat_cols])
        
    X = X.astype(float)
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), name='class')
    return X, y_encoded, le.classes_

def setup_datasets():
    print("=" * 60)
    print("PREPARACIÓN DE DATASETS OFICIAL - 30 DATASETS (5x10-CV)")
    print("=" * 60)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    cv_generator = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    for name in OPENML_DATASETS:
        out_path = os.path.join(DATA_DIR, f"{name}.pkl")
        if os.path.exists(out_path):
            print(f"[CACHE] {name} listo.")
            continue
            
        print(f"Descargando: {name}...")
        try:
            data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
            X_raw, y_raw = data.data, data.target
            
            if len(y_raw) > 1000:
                print(f"  -> Reduciendo dimensionalidad masiva de {name} por restricciones computacionales...")
                idx = np.random.RandomState(RANDOM_STATE).choice(len(y_raw), min(1000, len(y_raw)), replace=False)
                X_raw = X_raw.iloc[idx].reset_index(drop=True)
                y_raw = y_raw.iloc[idx].reset_index(drop=True)
                
            X, y, classes = preprocess_dataset(X_raw, y_raw)
            splits = list(cv_generator.split(X, y))
                
            with open(out_path, "wb") as f:
                pickle.dump({'name': name, 'X': X, 'y': y, 'classes': classes, 'cv_splits': splits}, f)
            print(f"  [OK] {name} guardado.")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

if __name__ == "__main__":
    setup_datasets()
