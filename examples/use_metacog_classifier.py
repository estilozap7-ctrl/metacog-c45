"""
=========================================================
MetaCog-C45 Examples
Demonstration of MetaCog-C45 end-to-end (mode='metacog')
=========================================================

Muestra la inducción completa de un árbol de decisión
MetaCog-C45, usando la misma API de C45Classifier con
mode='metacog'. Al finalizar, expone la memoria de
experiencia y la Importancia Metacognitiva de Características.
=========================================================
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.classifier import C45Classifier


def run_metacog_classifier_demo():
    print("=" * 70)
    print("MetaCog-C45 DEMO  (mode='metacog')")
    print("=" * 70)

    # 1. Dataset sintético
    np.random.seed(77)
    N = 80
    X = pd.DataFrame({
        'score_credito': np.random.uniform(300, 850, N),
        'ingreso':       np.random.uniform(20000, 120000, N),
        'edad':          np.random.randint(18, 65, N)
    })
    y = pd.Series(((X['score_credito'] > 600) & (X['ingreso'] > 50000)).astype(int))

    split = int(N * 0.75)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # 2. Entrenar con MetaCog-C45
    print("\n[Paso 1] Entrenando árbol MetaCog-C45...")
    clf = C45Classifier(
        max_depth=5,
        min_samples_split=5,
        mode='metacog',
        B=20,
        random_state=77
    )
    clf.fit(X_train, y_train)

    print(f"\nÁrbol MetaCog-C45:")
    clf.print_tree()
    clf.summary()

    # 3. Predicciones
    acc_train = clf.score(X_train, y_train)
    acc_val   = clf.score(X_val, y_val)
    print(f"Exactitud Entrenamiento : {acc_train:.4f}")
    print(f"Exactitud Validación    : {acc_val:.4f}")

    # 4. Explorar la memoria metacognitiva
    print(f"\n[Paso 2] Memoria Metacognitiva:")
    print(f"  Decisiones registradas: {clf.memory_.size()}")

    # 5. Estadísticas de experiencia
    print(f"\n[Paso 3] Experience Statistics (MFI):")
    for feat, score in clf.experience_stats_['mfi_normalized'].items():
        print(f"  {feat:<15}: {score:.4f}")

    print(f"\n  avg_NS: {clf.experience_stats_['avg_ns']:.4f}")
    print(f"  avg_TS: {clf.experience_stats_['avg_ts']:.4f}")

    # 6. Inspeccionar una traza de decisión
    traces = clf.memory_.get_all_traces()
    if traces:
        trace0 = traces[0]
        print(f"\n[Paso 4] Primera traza de decisión:")
        print(f"  node_id         : {trace0.node_id}")
        print(f"  feature         : {trace0.feature_selected}")
        print(f"  threshold       : {trace0.threshold_selected:.4f}" if trace0.threshold_selected else "  threshold : LEAF")
        print(f"  scs             : {trace0.scs:.4f}")
        print(f"  verdict         : {trace0.verdict}")
        print(f"  justification   : {trace0.justification}")

    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN MetaCog-C45 FINALIZADA")
    print("=" * 70)


if __name__ == '__main__':
    run_metacog_classifier_demo()
