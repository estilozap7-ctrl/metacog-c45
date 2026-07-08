"""
=========================================================
PyC45 Examples
Customer Churn Prediction (Fuga de Clientes)
=========================================================

Este script muestra cómo cargar un dataset, entrenar el clasificador
C4.5, realizar la poda del árbol usando un conjunto de validación,
evaluar con métricas avanzadas (Matriz de Confusión, ROC-AUC)
y visualizar los resultados.
=========================================================
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.classifier import C45Classifier
from core.pruning import DecisionTreePruner
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator
from visualization.feature_importance import FeatureImportance
from visualization.tree_plot import TreePlotter


def main():
    print("=" * 60)
    # 1. Cargar el dataset
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/customer_churn.csv'))
    print(f"Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print("\nPrimeras filas del dataset:")
    print(df.head())
    print(f"Dimensiones: {df.shape}")
    
    # 2. Dividir en conjuntos de entrenamiento y validación (70% / 30%)
    # Usamos Numpy para mezclar y dividir manualmente para no requerir scikit-learn
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    split_idx = int(len(df) * 0.7)
    
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]
    
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    
    X_train = train_df.drop('churn', axis=1)
    y_train = train_df['churn']
    X_val = val_df.drop('churn', axis=1)
    y_val = val_df['churn']
    
    print(f"\nDatos Entrenamiento : {X_train.shape[0]} registros")
    print(f"Datos Validación    : {X_val.shape[0]} registros")
    
    # 3. Inicializar y entrenar el clasificador C4.5
    print("\n--- Entrenando el Clasificador C4.5 ---")
    clf = C45Classifier(max_depth=6, min_samples_split=2, min_gain_ratio=0.005)
    clf.fit(X_train, y_train)
    
    print("\nÁrbol de Decisión original (sin podar):")
    clf.print_tree()
    clf.summary()
    
    # Evaluar en validación antes de podar
    val_preds_before = clf.predict(X_val)
    acc_val_before = clf.score(X_val, y_val)
    print(f"Exactitud (Accuracy) en Validación antes de podar: {acc_val_before:.4f}")
    
    # 4. Realizar la poda del árbol (Pruning)
    print("\n--- Podando el Árbol usando el Conjunto de Validación ---")
    DecisionTreePruner.prune(clf, X_val, y_val)
    
    print("\nÁrbol de Decisión final (Podado):")
    clf.print_tree()
    clf.summary()
    
    # Evaluar en validación después de podar
    val_preds_after = clf.predict(X_val)
    acc_val_after = clf.score(X_val, y_val)
    print(f"Exactitud (Accuracy) en Validación después de podar: {acc_val_after:.4f}")
    
    # 5. Evaluar con Matriz de Confusión y Métricas
    print("\n--- Evaluación del Desempeño (Conjunto de Validación) ---")
    cm = ConfusionMatrix.from_predictions(y_val, val_preds_after)
    cm.summary()
    
    metrics = ClassificationMetrics(cm)
    metrics.summary()
    
    # 6. Calcular ROC AUC y graficar la curva
    print("\n--- Cálculo de ROC AUC ---")
    # Obtener probabilidades de predicción para la clase 1
    # predict_proba devuelve un array [[P(class 0), P(class 1)], ...]
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    roc_data = ROCAUCCalculator.calculate(y_val, val_probs)
    print(f"Área Bajo la Curva ROC (AUC): {roc_data['auc']:.4f}")
    
    # Guardar gráfico de la curva ROC
    roc_img_path = os.path.join(os.path.dirname(__file__), 'roc_curve.png')
    print(f"Guardando la curva ROC en: {roc_img_path}")
    plt.figure(figsize=(7, 5))
    plt.plot(roc_data["fpr"], roc_data["tpr"], color="#E67E22", lw=2, label=f"ROC (AUC = {roc_data['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="#2C3E50", lw=1.5, linestyle="--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Curva ROC - Predicción de Churn")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(roc_img_path, dpi=150)
    plt.close()
    
    # 7. Importancia de Variables (Feature Importance)
    print("\n--- Importancia de Variables ---")
    fi = FeatureImportance().fit(clf.root)
    print(fi.dataframe())
    
    # 8. Guardar la visualización del árbol como imagen
    tree_img_path = os.path.join(os.path.dirname(__file__), 'decision_tree.png')
    print(f"\nGenerando imagen del árbol en: {tree_img_path}")
    plotter = TreePlotter()
    
    # Evitar bloqueo en entornos sin interfaz gráfica guardando la imagen
    original_show = plt.show
    plt.show = lambda: None
    
    plotter.plot(clf.root, figsize=(12, 8))
    plt.savefig(tree_img_path, dpi=150)
    plt.close('all')
    
    # Restaurar plt.show original
    plt.show = original_show
    print(f"Imagen del árbol guardada con éxito en: {tree_img_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
