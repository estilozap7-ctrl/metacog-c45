"""
=========================================================
PyC45 Framework
Web Report Generator (Generador de Informe Web)
=========================================================

Ejecuta el pipeline completo de análisis sobre el dataset de tarjetas de crédito,
genera gráficos de evaluación y exporta los resultados en formato JSON
para ser consumidos por la página web de informe técnico.
=========================================================
"""

import sys
import os
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.classifier import C45Classifier
from core.pruning import DecisionTreePruner
from core.entropy import EntropyCalculator
from core.tree_utils import TreeUtils
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator
from visualization.feature_importance import FeatureImportance


# =========================================================
# Utilidades
# =========================================================

def fig_to_base64(fig):
    """Convierte una figura de matplotlib a string base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return b64


def tree_to_dict(node):
    """Convierte un DecisionNode a estructura de diccionario recursiva."""
    if node is None:
        return None
    d = {
        "is_leaf": node.is_leaf,
        "samples": node.samples,
        "entropy": round(node.entropy, 4),
        "gain_ratio": round(node.gain_ratio, 4),
        "prediction": int(node.prediction) if node.prediction is not None else None,
        "probability": round(node.probability, 4) if node.probability else 0,
        "class_counts": {str(k): int(v) for k, v in node.class_counts.items()} if node.class_counts else {},
    }
    if not node.is_leaf:
        d["feature"] = node.feature
        d["threshold"] = round(float(node.threshold), 4)
        d["left"] = tree_to_dict(node.left)
        d["right"] = tree_to_dict(node.right)
    return d


def tree_to_text(node, indent=""):
    """Genera la representación en texto del árbol de decisión."""
    if node is None:
        return ""
    lines = []
    if node.is_leaf:
        lines.append(f"{indent}🍃 Hoja → Clase={node.prediction}  (n={node.samples})")
    else:
        lines.append(f"{indent}📊 {node.feature} ≤ {node.threshold:.4f}  [GainRatio={node.gain_ratio:.4f}]")
        lines.append(tree_to_text(node.left, indent + "│   "))
        lines.append(tree_to_text(node.right, indent + "│   "))
    return "\n".join(lines)


# =========================================================
# Estilo de gráficos
# =========================================================

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d27",
    "accent": "#6c63ff",
    "accent2": "#00d4aa",
    "danger": "#ff6b6b",
    "warning": "#ffd93d",
    "text": "#e4e4e7",
    "muted": "#71717a",
    "grid": "#27272a",
}

def setup_chart_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS["bg"],
        'axes.facecolor': COLORS["card"],
        'axes.edgecolor': COLORS["muted"],
        'axes.labelcolor': COLORS["text"],
        'text.color': COLORS["text"],
        'xtick.color': COLORS["text"],
        'ytick.color': COLORS["text"],
        'grid.color': COLORS["grid"],
        'grid.alpha': 0.4,
        'font.family': 'sans-serif',
        'font.size': 11,
    })

setup_chart_style()


# =========================================================
# Funciones de gráficos
# =========================================================

def generate_class_distribution_chart(y):
    """Gráfico de distribución de clases."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    labels = ['No Impago (0)', 'Impago (1)']
    colors = [COLORS["accent2"], COLORS["danger"]]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='none', width=0.5, alpha=0.9)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                f'{count:,}\n({count/len(y)*100:.1f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=COLORS["text"])
    ax.set_ylabel('Cantidad de Registros')
    ax.set_title('Distribución de Clases en la Muestra', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig_to_base64(fig)


def generate_confusion_matrix_chart(cm):
    """Gráfico de la matriz de confusión como heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    matrix = np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]])
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicho: 0\n(No Impago)', 'Predicho: 1\n(Impago)'], fontsize=10)
    ax.set_yticklabels(['Real: 0\n(No Impago)', 'Real: 1\n(Impago)'], fontsize=10)
    ax.set_title('Matriz de Confusión', fontsize=13, fontweight='bold', pad=12)
    labels_map = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            label = labels_map[i][j]
            color = '#000' if val > matrix.max()*0.5 else '#fff'
            ax.text(j, i, f'{label}\n{val}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig_to_base64(fig)


def generate_metrics_chart(metrics_dict):
    """Gráfico de barras horizontales de métricas."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors_bar = []
    for v in values:
        if v >= 0.8:
            colors_bar.append(COLORS["accent2"])
        elif v >= 0.5:
            colors_bar.append(COLORS["warning"])
        else:
            colors_bar.append(COLORS["danger"])
    bars = ax.barh(names[::-1], values[::-1], color=colors_bar[::-1], edgecolor='none', height=0.55, alpha=0.9)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold', color=COLORS["text"])
    ax.set_xlim(0, 1.15)
    ax.set_title('Métricas de Clasificación', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig_to_base64(fig)


def generate_roc_chart(roc_data):
    """Gráfico de la curva ROC."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.fill_between(roc_data["fpr"], roc_data["tpr"], alpha=0.15, color=COLORS["accent"])
    ax.plot(roc_data["fpr"], roc_data["tpr"], color=COLORS["accent"], lw=2.5,
            label=f'Curva ROC (AUC = {roc_data["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], color=COLORS["muted"], lw=1.5, linestyle='--', alpha=0.7, label='Clasificador Aleatorio')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    ax.set_title('Curva ROC', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig_to_base64(fig)


def generate_feature_importance_chart(fi_df):
    """Gráfico de importancia de variables."""
    fig, ax = plt.subplots(figsize=(7, max(3, len(fi_df)*0.55 + 1.5)))
    colors_bar = plt.cm.plasma(np.linspace(0.2, 0.85, len(fi_df)))
    bars = ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
                   color=colors_bar[::-1], edgecolor='none', height=0.5, alpha=0.9)
    for bar, val in zip(bars, fi_df["Importance"][::-1].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold', color=COLORS["text"])
    ax.set_xlim(0, max(fi_df["Importance"])*1.25)
    ax.set_title('Importancia de Variables (Gain Ratio Acumulado)', fontsize=13, fontweight='bold', pad=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig_to_base64(fig)


def generate_entropy_diagram():
    """Gráfico explicativo de la función de entropía."""
    fig, ax = plt.subplots(figsize=(6, 4))
    p = np.linspace(0.001, 0.999, 200)
    entropy = -p*np.log2(p) - (1-p)*np.log2(1-p)
    ax.plot(p, entropy, color=COLORS["accent"], lw=2.5)
    ax.fill_between(p, entropy, alpha=0.1, color=COLORS["accent"])
    ax.axvline(x=0.5, color=COLORS["danger"], lw=1.2, linestyle='--', alpha=0.6, label='Máxima incertidumbre (p=0.5)')
    ax.set_xlabel('Proporción de la clase positiva (p)', fontsize=11)
    ax.set_ylabel('Entropía H(p)', fontsize=11)
    ax.set_title('Función de Entropía de Shannon (Binaria)', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9, framealpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig_to_base64(fig)


# =========================================================
# Pipeline Principal
# =========================================================

def run_pipeline():
    """Ejecuta el pipeline C4.5 y retorna todos los datos para el informe."""
    print("=" * 60)
    print(" GENERANDO INFORME TÉCNICO WEB - PyC45")
    print("=" * 60)

    # --- 1. Carga de datos ---
    print("[1/8] Cargando dataset...")
    excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/default_of_credit_card_clients.xls'))
    df_full = pd.read_excel(excel_path)
    if 'ID' in df_full.columns:
        df_full = df_full.drop('ID', axis=1)
    target_col = 'default payment next month'
    
    dataset_info = {
        "total_rows": int(df_full.shape[0]),
        "total_cols": int(df_full.shape[1]),
        "columns": df_full.columns.tolist(),
        "target_column": target_col,
        "class_distribution_full": {
            "0": int((df_full[target_col] == 0).sum()),
            "1": int((df_full[target_col] == 1).sum()),
        }
    }

    # --- 2. Submuestreo ---
    print("[2/8] Realizando submuestreo...")
    np.random.seed(42)
    df_sample = df_full.sample(n=1000, random_state=42).reset_index(drop=True)
    
    # División 70/30
    shuffled = np.random.permutation(len(df_sample))
    split = int(len(df_sample) * 0.70)
    train_df = df_sample.iloc[shuffled[:split]]
    val_df = df_sample.iloc[shuffled[split:]]
    
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_val = val_df.drop(target_col, axis=1)
    y_val = val_df[target_col]
    
    dataset_info["sample_size"] = 1000
    dataset_info["train_size"] = len(X_train)
    dataset_info["val_size"] = len(X_val)
    dataset_info["class_distribution_sample"] = {
        "0": int((df_sample[target_col] == 0).sum()),
        "1": int((df_sample[target_col] == 1).sum()),
    }

    chart_class_dist = generate_class_distribution_chart(df_sample[target_col])

    # --- 3. Entrenamiento ---
    print("[3/8] Entrenando el clasificador C4.5...")
    clf = C45Classifier(max_depth=4, min_samples_split=10, min_gain_ratio=0.01)
    clf.fit(X_train, y_train)
    
    tree_before_text = tree_to_text(clf.root)
    tree_before_dict = tree_to_dict(clf.root)
    tree_before_stats = {
        "nodes": TreeUtils.count_nodes(clf.root),
        "leaves": TreeUtils.count_leaves(clf.root),
        "depth": TreeUtils.depth(clf.root),
    }
    acc_train = float(clf.score(X_train, y_train))
    acc_val_before = float(clf.score(X_val, y_val))

    # --- 4. Poda ---
    print("[4/8] Podando el árbol...")
    DecisionTreePruner.prune(clf, X_val, y_val)
    
    tree_after_text = tree_to_text(clf.root)
    tree_after_dict = tree_to_dict(clf.root)
    tree_after_stats = {
        "nodes": TreeUtils.count_nodes(clf.root),
        "leaves": TreeUtils.count_leaves(clf.root),
        "depth": TreeUtils.depth(clf.root),
    }
    acc_val_after = float(clf.score(X_val, y_val))

    # --- 5. Métricas ---
    print("[5/8] Calculando métricas de clasificación...")
    val_preds = clf.predict(X_val)
    cm = ConfusionMatrix.from_predictions(y_val, val_preds)
    metrics = ClassificationMetrics(cm)
    
    metrics_dict = {
        "Accuracy": float(metrics.accuracy),
        "Precision": float(metrics.precision),
        "Recall": float(metrics.recall),
        "Specificity": float(metrics.specificity),
        "F1-Score": float(metrics.f1_score),
        "Balanced Accuracy": float(metrics.balanced_accuracy),
        "Error Rate": float(metrics.error_rate),
        "MCC": float(metrics.mcc),
    }
    cm_dict = {"tp": int(cm.tp), "tn": int(cm.tn), "fp": int(cm.fp), "fn": int(cm.fn), "total": int(cm.total)}

    chart_cm = generate_confusion_matrix_chart(cm)
    chart_metrics = generate_metrics_chart(metrics_dict)

    # --- 6. ROC-AUC ---
    print("[6/8] Calculando curva ROC y AUC...")
    val_probs = clf.predict_proba(X_val)[:, 1]
    roc_data = ROCAUCCalculator.calculate(y_val, val_probs)
    auc_value = float(roc_data["auc"])

    chart_roc = generate_roc_chart(roc_data)

    # --- 7. Feature Importance ---
    print("[7/8] Calculando importancia de variables...")
    fi = FeatureImportance().fit(clf.root)
    fi_df = fi.dataframe()
    fi_list = [{"feature": row["Feature"], "importance": round(float(row["Importance"]), 6)}
               for _, row in fi_df.iterrows()]

    chart_fi = generate_feature_importance_chart(fi_df)
    chart_entropy = generate_entropy_diagram()

    # --- 8. Compilar resultados ---
    print("[8/8] Compilando resultados...")
    report_data = {
        "dataset": dataset_info,
        "hyperparameters": {
            "max_depth": 4,
            "min_samples_split": 10,
            "min_gain_ratio": 0.01,
        },
        "tree_before_pruning": {
            "text": tree_before_text,
            "stats": tree_before_stats,
            "accuracy_train": round(acc_train, 6),
            "accuracy_val": round(acc_val_before, 6),
        },
        "tree_after_pruning": {
            "text": tree_after_text,
            "stats": tree_after_stats,
            "accuracy_val": round(acc_val_after, 6),
        },
        "confusion_matrix": cm_dict,
        "metrics": metrics_dict,
        "roc_auc": auc_value,
        "feature_importance": fi_list,
        "charts": {
            "class_distribution": chart_class_dist,
            "confusion_matrix": chart_cm,
            "metrics": chart_metrics,
            "roc": chart_roc,
            "feature_importance": chart_fi,
            "entropy_diagram": chart_entropy,
        }
    }

    # Guardar JSON y JS
    output_path = os.path.join(os.path.dirname(__file__), 'report_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
        
    output_path_js = os.path.join(os.path.dirname(__file__), 'report_data.js')
    with open(output_path_js, 'w', encoding='utf-8') as f:
        f.write("window.reportData = ")
        json.dump(report_data, f, ensure_ascii=False, indent=2)
        f.write(";")
    
    print(f"\nDatos del informe exportados en: {output_path} y {output_path_js}")
    print("=" * 60)
    print(" INFORME GENERADO CON ÉXITO")
    print(" Abre web_verification/index.html en tu navegador.")
    print("=" * 60)

    return report_data


if __name__ == "__main__":
    run_pipeline()
