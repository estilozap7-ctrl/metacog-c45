"""
=========================================================
MetaCog-C45 Examples
Credit Card Default Prediction (Predicción de Impago)
=========================================================

Este script demuestra cómo cargar el dataset de clientes de tarjetas de crédito
(en formato Excel .xls), realizar un submuestreo debido al costo computacional
de C4.5 puro en Python, entrenar y evaluar el modelo.
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
    print("=" * 70)
    print("        PREDICCIÓN DE IMPAGO DE TARJETAS DE CRÉDITO CON PYC45")
    print("=" * 70)

    # 1. Cargar el dataset Excel
    excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/default_of_credit_card_clients.xls'))
    print(f"Cargando dataset desde: {excel_path}")
    
    # Leemos el archivo XLS (requiere la librería xlrd)
    df = pd.read_excel(excel_path)
    
    print(f"Dimensiones del dataset original: {df.shape}")
    print("\nPrimeras 5 observaciones:")
    print(df.head())

    # 2. Preprocesamiento de los datos
    # La columna 'ID' no aporta información para el árbol, la eliminamos.
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # La columna objetivo es 'default payment next month'
    target_col = 'default payment next month'
    
    # 3. Submuestreo (Subsampling)
    # NOTA: C4.5 evalúa todos los posibles umbrales de variables numéricas continuas
    # en cada nodo. Con 30,000 registros y 23 variables, construir un árbol completo en Python
    # puro puede tomar mucho tiempo. Por tanto, tomamos una muestra aleatoria representativa de
    # 1000 registros para entrenar y evaluar rápidamente.
    print("\nRealizando submuestreo de 1000 registros para rapidez computacional...")
    df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)
    print(f"Dimensiones del subconjunto de prueba: {df_sample.shape}")
    
    # 4. División de datos en Entrenamiento (70%) y Validación/Prueba (30%)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df_sample))
    split_idx = int(len(df_sample) * 0.70)

    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    train_df = df_sample.iloc[train_indices]
    val_df = df_sample.iloc[val_indices]

    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_val = val_df.drop(target_col, axis=1)
    y_val = val_df[target_col]

    # 5. Inicialización y Entrenamiento de C4.5
    # Limitamos max_depth a 4 para generar un árbol legible y evitar sobreajuste.
    print("\n--- Entrenando el Clasificador C4.5 ---")
    clf = C45Classifier(max_depth=4, min_samples_split=10, min_gain_ratio=0.01)
    clf.fit(X_train, y_train)

    print("\nÁrbol de Decisión Inicial (sin podar):")
    clf.print_tree()
    clf.summary()

    # Evaluar antes de la poda
    val_acc_before = clf.score(X_val, y_val)
    print(f"Exactitud (Accuracy) en Validación antes de podar: {val_acc_before:.4f}")

    # 6. Poda del Árbol (Pruning)
    print("\n--- Podando el Árbol usando el Conjunto de Validación ---")
    DecisionTreePruner.prune(clf, X_val, y_val)

    print("\nÁrbol de Decisión final (Podado):")
    clf.print_tree()
    clf.summary()

    # Evaluar después de la poda
    val_preds = clf.predict(X_val)
    val_acc_after = clf.score(X_val, y_val)
    print(f"Exactitud (Accuracy) en Validación después de podar: {val_acc_after:.4f}")

    # 7. Matriz de Confusión y Métricas de Clasificación
    print("\n--- Matriz de Confusión (Datos de Validación) ---")
    cm = ConfusionMatrix.from_predictions(y_val, val_preds)
    cm.summary()

    metrics = ClassificationMetrics(cm)
    metrics.summary()

    # 8. Cálculo de Curva ROC-AUC
    print("\n--- Análisis ROC-AUC ---")
    val_probs = clf.predict_proba(X_val)[:, 1]
    roc_data = ROCAUCCalculator.calculate(y_val, val_probs)
    print(f"Área Bajo la Curva ROC (AUC): {roc_data['auc']:.4f}")

    # Guardar Curva ROC
    roc_path = os.path.join(os.path.dirname(__file__), 'credit_default_roc.png')
    plt.figure(figsize=(7, 5))
    plt.plot(roc_data["fpr"], roc_data["tpr"], color="#E67E22", lw=2, label=f"ROC (AUC = {roc_data['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="#2C3E50", lw=1.5, linestyle="--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Curva ROC - Predicción de Impago")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Gráfico de curva ROC guardado en: {roc_path}")

    # 9. Importancia de Variables
    print("\n--- Importancia de Variables en el Modelo ---")
    fi = FeatureImportance().fit(clf.root)
    print(fi.dataframe())

    # 10. Guardar Imagen de Estructura de Árbol
    tree_path = os.path.join(os.path.dirname(__file__), 'credit_default_tree.png')
    plotter = TreePlotter()
    
    # Desactivamos visualización de interfaz gráfica local para guardar directo
    original_show = plt.show
    plt.show = lambda: None
    
    plotter.plot(clf.root, figsize=(14, 8))
    plt.savefig(tree_path, dpi=150)
    plt.close('all')
    plt.show = original_show
    print(f"\nGráfico de estructura del árbol guardado en: {tree_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
