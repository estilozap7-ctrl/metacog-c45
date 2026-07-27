"""
=========================================================
MetaCog-C45 Framework
Main Demonstration Script (Script Principal)
=========================================================

Autor      : Luis Alberto Buelvas Cogollo
Proyecto   : MetaCog-C45

Descripción:
Punto de entrada principal para ejecutar una simulación del pipeline completo
del clasificador C4.5. Genera un dataset sintético, entrena, predice,
calcula métricas de evaluación, realiza poda (pruning) y muestra
la importancia de variables.
=========================================================
"""

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from core.classifier import C45Classifier
from core.pruning import DecisionTreePruner
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator
from visualization.feature_importance import FeatureImportance


def main():
    print("=" * 70)
    print("                 BIENVENIDO A MetaCog-C45 Framework")
    print("=" * 70)

    # 1. Generación de un conjunto de datos sintético realista
    # Queremos modelar si un estudiante aprueba un examen basado en:
    # - Horas de estudio semanal
    # - Porcentaje de asistencia a clases
    np.random.seed(42)
    n_samples = 40
    
    study_hours = np.random.uniform(2, 20, n_samples)
    attendance = np.random.uniform(50, 100, n_samples)
    
    # Criterio de aprobación: aprobar si (horas > 10 y asistencia > 70) o (horas > 15)
    # Agregamos algo de ruido aleatorio a la etiqueta para hacer el problema interesante
    pass_exam = ((study_hours > 10) & (attendance > 70)) | (study_hours > 15)
    noise = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
    pass_exam = np.where(noise, ~pass_exam, pass_exam).astype(int)

    df = pd.DataFrame({
        "Horas_Estudio": study_hours,
        "Asistencia": attendance,
        "Aprobado": pass_exam
    })

    print(f"Dataset sintético generado con {n_samples} registros.")
    print("\nPrimeras 5 observaciones:")
    print(df.head())

    # Dividir en entrenamiento (75%) y validación (25%)
    split = int(n_samples * 0.75)
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]

    X_train, y_train = train_df[["Horas_Estudio", "Asistencia"]], train_df["Aprobado"]
    X_val, y_val = val_df[["Horas_Estudio", "Asistencia"]], val_df["Aprobado"]

    # 2. Construcción y Entrenamiento del Clasificador C4.5
    print("\n--- Paso 1: Entrenando el Clasificador C4.5 ---")
    clf = C45Classifier(max_depth=4, min_samples_split=3, min_gain_ratio=0.01)
    clf.fit(X_train, y_train)

    print("\nÁrbol de Decisión Inicial (sin podar):")
    clf.print_tree()
    clf.summary()

    # Predicciones iniciales
    train_acc = clf.score(X_train, y_train)
    val_acc_before = clf.score(X_val, y_val)
    print(f"Exactitud en Entrenamiento : {train_acc:.4f}")
    print(f"Exactitud en Validación    : {val_acc_before:.4f}")

    # 3. Poda del Árbol (Pruning)
    print("\n--- Paso 2: Ejecutando la Poda por Error Reducido (REP) ---")
    DecisionTreePruner.prune(clf, X_val, y_val)

    print("\nÁrbol de Decisión Final (Podado):")
    clf.print_tree()
    clf.summary()

    val_acc_after = clf.score(X_val, y_val)
    print(f"Exactitud en Validación después de podar: {val_acc_after:.4f}")

    # 4. Evaluación de Métricas de Clasificación
    print("\n--- Paso 3: Matriz de Confusión y Métricas de Clasificación ---")
    preds = clf.predict(X_val)
    cm = ConfusionMatrix.from_predictions(y_val, preds)
    cm.summary()

    metrics = ClassificationMetrics(cm)
    metrics.summary()

    # 5. Evaluación de Probabilidades y Curva ROC-AUC
    print("\n--- Paso 4: Análisis de Probabilidades y ROC-AUC ---")
    probs = clf.predict_proba(X_val)[:, 1]
    
    roc_data = ROCAUCCalculator.calculate(y_val, probs)
    print(f"Área Bajo la Curva ROC (AUC): {roc_data['auc']:.4f}")

    # 6. Importancia de Variables (Feature Importance)
    print("\n--- Paso 5: Importancia de Variables en el Árbol ---")
    fi = FeatureImportance().fit(clf.root)
    print(fi.dataframe())

    print("\n" + "=" * 70)
    print("                 EJECUCIÓN DE SIMULACIÓN COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
