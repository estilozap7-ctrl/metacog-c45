import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Evitar popup GUI de Matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import io
import base64
import webbrowser

# Asegurar la importación del core del framework
sys.path.append(os.path.abspath('.'))

from core.classifier import C45Classifier
from core.pruning import DecisionTreePruner
from core.tree_utils import TreeUtils
from metrics.confusion_matrix import ConfusionMatrix
from metrics.classification_metrics import ClassificationMetrics
from metrics.roc_auc import ROCAUCCalculator
from visualization.feature_importance import FeatureImportance
from visualization.tree_plot import TreePlotter

# =========================================================
# ESTILOS Y HELPER FUNCTIONS PARA EL REPORTE WEB
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

def fig_to_base64(fig):
    """Convierte una figura de matplotlib a string base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return b64

def generate_class_distribution_chart(y):
    """Gráfico de distribución de clases."""
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    labels = [f"Clase {k}" for k in counts.index]
    colors = [COLORS["accent2"], COLORS["danger"]] if len(counts) == 2 else plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
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
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    matrix = np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]])
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicho: 0', 'Predicho: 1'], fontsize=10)
    ax.set_yticklabels(['Real: 0', 'Real: 1'], fontsize=10)
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
    setup_chart_style()
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
    setup_chart_style()
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
    setup_chart_style()
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
    setup_chart_style()
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

def tree_to_text(node, indent="", memory=None):
    """Genera la representación en texto del árbol de decisión (con Sistema 2 si está disponible)."""
    if node is None:
        return ""
    lines = []
    if node.is_leaf:
        lines.append(f"{indent}🍃 Hoja → Clase={node.prediction}  (n={node.samples}, prob={node.probability:.2%})")
    else:
        trace = None
        if memory is not None and getattr(node, "trace_id", None):
            trace = next((t for t in memory.get_all_traces() if t.node_id == node.trace_id), None)

        if trace is not None:
            lines.append(
                f"{indent}📊 {node.feature} ≤ {node.threshold:.4f}\n"
                f"{indent}   GainRatio = {node.gain_ratio:.4f}\n"
                f"{indent}   SCS = {trace.scs:.3f}\n"
                f"{indent}   NS = {trace.ns:.3f}\n"
                f"{indent}   TS = {trace.ts:.3f}\n"
                f"{indent}   CI = {trace.competitiveness_index:.3f}\n"
                f"{indent}   Decision = {trace.verdict}"
            )
        else:
            lines.append(f"{indent}📊 {node.feature} ≤ {node.threshold:.4f}  [GainRatio={node.gain_ratio:.4f}]")

        lines.append(tree_to_text(node.left, indent + "│   ", memory))
        lines.append(tree_to_text(node.right, indent + "│   ", memory))
    return "\n".join(lines)

def get_rule_lengths(node, current_len=0):
    """Calcula recursivamente la longitud de todas las reglas (caminos a las hojas)."""
    if node is None:
        return []
    if node.is_leaf:
        return [current_len]
    return get_rule_lengths(node.left, current_len + 1) + get_rule_lengths(node.right, current_len + 1)

def generate_sql_rules(node, path_conditions=[]):
    """Genera recursivamente las condiciones SQL para las hojas."""
    if node is None:
        return []
    if node.is_leaf:
        if not path_conditions:
            return [f"ELSE {node.prediction}"]
        cond_str = " AND ".join(path_conditions)
        return [f"WHEN {cond_str} THEN {node.prediction}"]
    
    left_conds = path_conditions + [f"{node.feature} <= {node.threshold:.4f}"]
    right_conds = path_conditions + [f"{node.feature} > {node.threshold:.4f}"]
    
    rules = []
    rules.extend(generate_sql_rules(node.left, left_conds))
    rules.extend(generate_sql_rules(node.right, right_conds))
    return rules

def get_sql_code(root, table_name="credit_data", target_col="prediction"):
    """Genera el código SQL CASE WHEN del árbol."""
    rules = generate_sql_rules(root)
    when_rules = [r for r in rules if r.startswith("WHEN")]
    else_rules = [r for r in rules if r.startswith("ELSE")]
    else_clause = else_rules[0] if else_rules else "ELSE 0"
    
    sql = ["SELECT", "  *,", "  CASE"]
    for rule in when_rules:
        sql.append(f"    {rule}")
    sql.append(f"    {else_clause}")
    sql.append(f"  END AS {target_col}")
    sql.append(f"FROM {table_name};")
    return "\n".join(sql)

def generate_python_rules(node, indent="    "):
    """Genera recursivamente el código de decisión en Python."""
    if node is None:
        return []
    if node.is_leaf:
        return [f"{indent}return {node.prediction}  # Hoja (n={node.samples}, prob={node.probability:.2%})"]
    
    lines = []
    lines.append(f"{indent}if client.get('{node.feature}', 0.0) <= {node.threshold:.4f}:")
    lines.extend(generate_python_rules(node.left, indent + "    "))
    lines.append(f"{indent}else:")
    lines.extend(generate_python_rules(node.right, indent + "    "))
    return lines

def get_python_code(root):
    """Genera el código de predicción en Python nativo."""
    lines = [
        "def predict_metacog(client):",
        '    """',
        '    Predicción generada automáticamente por MetaCog-C45.',
        '    client: diccionario con los valores de las variables.',
        '    """'
    ]
    lines.extend(generate_python_rules(root))
    return "\n".join(lines)

def get_fastapi_code(root, features):
    """Genera el código para servir el modelo con FastAPI."""
    python_func = get_python_code(root)
    schema_attrs = [f"    {feat}: float" for feat in features]
    schema_attrs_str = "\n".join(schema_attrs)
    
    code = [
        "from fastapi import FastAPI",
        "from pydantic import BaseModel",
        "import uvicorn",
        "",
        "app = FastAPI(title='MetaCog-C45 Model API', version='1.0')",
        "",
        "class ClientData(BaseModel):",
        schema_attrs_str,
        "",
        python_func,
        "",
        "@app.post('/predict')",
        "def predict(data: ClientData):",
        "    client_dict = data.dict()",
        "    pred = predict_metacog(client_dict)",
        "    return {'prediction': pred}",
        "",
        "if __name__ == '__main__':",
        "    uvicorn.run(app, host='0.0.0.0', port=8000)"
    ]
    return "\n".join(code)


def _build_metacog_report(clf) -> dict:
    """
    Extrae todos los datos del Sistema 2 (metacognitivo) del clasificador
    entrenado y los serializa para el reporte HTML/JSON.
    Retorna un dict vacío si el clasificador está en modo 'classic'.
    """
    if clf.mode != "metacog":
        return {"mode": "classic", "enabled": False}

    stats = clf.experience_stats_ or {}
    memory = clf.memory_

    # ── Trazas individuales de DecisionTrace ──────────────────────────────
    traces_list = []
    if memory is not None:
        for t in memory.get_all_traces():
            traces_list.append({
                "node_id":             t.node_id,
                "depth":               t.depth,
                "n_samples":           t.n_samples,
                "feature_selected":    t.feature_selected,
                "threshold_selected":  round(float(t.threshold_selected), 6) if t.threshold_selected is not None else None,
                "gain_ratio":          round(float(t.gain_ratio), 6),
                "ns":                  round(float(t.ns), 6),
                "ts":                  round(float(t.ts), 6),
                "p_star":              round(float(t.p_star), 6),
                "competitiveness_index": round(float(t.competitiveness_index), 6),
                "is_competitive":      bool(t.is_competitive),
                "alternate_feature":   t.alternate_feature,
                "scs":                 round(float(t.scs), 6),
                "verdict":             t.verdict,
                "justification":       t.justification,
                "uncertainty":         round(float(t.uncertainty), 6),
                "algorithm_version":   t.algorithm_version,
            })

    # ── Distribución de veredictos ────────────────────────────────────────
    verdict_counts: dict = {}
    scs_values: list = []
    ns_values:  list = []
    ts_values:  list = []
    ci_values:  list = []
    for t_dict in traces_list:
        v = t_dict["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        scs_values.append(t_dict["scs"])
        ns_values.append(t_dict["ns"])
        ts_values.append(t_dict["ts"])
        ci_values.append(t_dict["competitiveness_index"])

    def _safe_stats(vals):
        if not vals:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        import statistics as _st
        return {
            "min":  round(min(vals), 6),
            "max":  round(max(vals), 6),
            "mean": round(sum(vals) / len(vals), 6),
            "std":  round(_st.stdev(vals) if len(vals) > 1 else 0.0, 6),
        }

    return {
        "mode":            "metacog",
        "enabled":         True,
        "total_decisions": stats.get("total_decisions", len(traces_list)),
        "avg_ns":          round(float(stats.get("avg_ns", 0.0)), 6),
        "avg_ts":          round(float(stats.get("avg_ts", 0.0)), 6),
        "mfi_normalized":  {k: round(v, 6) for k, v in stats.get("mfi_normalized", {}).items()},
        "verdict_counts":  verdict_counts,
        "scs_stats":       _safe_stats(scs_values),
        "ns_stats":        _safe_stats(ns_values),
        "ts_stats":        _safe_stats(ts_values),
        "ci_stats":        _safe_stats(ci_values),
        "scs_values":      [round(v, 6) for v in scs_values],
        "ns_values":       [round(v, 6) for v in ns_values],
        "ts_values":       [round(v, 6) for v in ts_values],
        "ci_values":       [round(v, 6) for v in ci_values],
        "decision_traces": traces_list,
    }


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # ── Leer modo y parámetros del Sistema 2 desde la CLI (o usar defaults) ──
    _mode         = globals().get("_CLI_MODE",         "metacog")
    _B            = globals().get("_CLI_B",             50)
    _theta_accept = globals().get("_CLI_THETA_ACCEPT",  0.5)
    _theta_reject = globals().get("_CLI_THETA_REJECT",  0.05)
    _gamma        = globals().get("_CLI_GAMMA",         0.1)

    clear_screen()
    print("=" * 80)
    if _mode == "metacog":
        print("       MetaCog-C45 FRAMEWORK - EJECUTOR CON SISTEMA 2 (METACOGNITIVO)")
    else:
        print("       MetaCog-C45 FRAMEWORK - EJECUTOR EN MODO CLÁSICO C4.5")
    print("=" * 80)
    
    # 1. Escanear y listar datasets
    datasets_dir = "datasets"
    _cli_dataset = globals().get("_CLI_DATASET", None)
    _cli_target  = globals().get("_CLI_TARGET",  None)

    if _cli_dataset:
        file_path = _cli_dataset
        selected_file = os.path.basename(_cli_dataset)
        print(f"\nUsando dataset especificado por CLI: '{file_path}'")
    else:
        if not os.path.exists(datasets_dir):
            print(f"Error: La carpeta '{datasets_dir}' no existe.")
            return

        files = [f for f in os.listdir(datasets_dir) if f.endswith(('.csv', '.xls', '.xlsx'))]
        if not files:
            print("No se encontraron archivos de datos (.csv, .xls, .xlsx) en la carpeta 'datasets'.")
            return

        print("\nDatasets disponibles en la carpeta 'datasets':")
        for idx, file in enumerate(files, 1):
            print(f"  [{idx}] {file}")

        # Selección del dataset
        while True:
            try:
                choice = input(f"\nSelecciona el dataset con el que deseas trabajar (1-{len(files)}): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(files):
                    selected_file = files[choice_idx]
                    break
                else:
                    print(f"Por favor ingresa un número entre 1 y {len(files)}.")
            except ValueError:
                print("Entrada inválida. Debe ser un número.")

        file_path = os.path.join(datasets_dir, selected_file)

    print(f"\nCargando '{selected_file}'...")
    
    # 2. Cargar el dataset
    ext = os.path.splitext(selected_file)[1].lower()
    try:
        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return
        
    print(f"Dataset cargado con éxito. Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas.")
    
    # 3. Detectar o elegir columna objetivo (Target)
    target_col = None
    if _cli_target:
        if _cli_target in df.columns:
            target_col = _cli_target
            print(f"Columna objetivo especificada por CLI: '{target_col}'")
        else:
            print(f"[ADVERTENCIA] Columna '{_cli_target}' no encontrada en el dataset.")

    if not target_col:
        print("\nColumnas del dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        # Auto-detección de columnas objetivo comunes
        known_targets = ['churn', 'default payment next month', 'target', 'class', 'clase', 'aprobado', 'label']
        detected_target = None
        for col in df.columns:
            if col.lower() in known_targets:
                detected_target = col
                break

        if detected_target:
            if _cli_dataset:
                # Modo no interactivo: usar target detectado automáticamente
                target_col = detected_target
                print(f"Columna objetivo detectada automáticamente (modo CLI): '{target_col}'")
            else:
                use_detected = input(f"\n¿Deseas usar la columna objetivo detectada automáticamente '{detected_target}'? (S/n): ").strip().lower()
                if use_detected in ['', 's', 'si', 'yes', 'y']:
                    target_col = detected_target

        if not target_col:
            if _cli_dataset:
                target_col = df.columns[-1]
                print(f"Seleccionando última columna por defecto (modo CLI): '{target_col}'")
            else:
                while True:
                    try:
                        target_choice = input("\nIngresa el número o nombre exacto de la columna objetivo (target): ").strip()
                        if target_choice in df.columns:
                            target_col = target_choice
                            break
                        elif target_choice.isdigit():
                            col_idx = int(target_choice) - 1
                            if 0 <= col_idx < len(df.columns):
                                target_col = df.columns[col_idx]
                                break
                        print("Nombre de columna o número no válido. Intenta de nuevo.")
                    except ValueError:
                        print("Entrada inválida.")
                
    print(f"Columna objetivo seleccionada: '{target_col}'")
    
    # 4. Limpieza básica de datos e identificadores
    cols_to_drop = []
    for col in df.columns:
        if col.lower() in ['id', 'uuid', 'index', 'indice'] and col != target_col:
            cols_to_drop.append(col)
            
    if cols_to_drop:
        print(f"\nEliminando columnas identificadoras no predictivas automáticamente: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        
    # Asegurar que la columna objetivo no tiene nulos
    df = df.dropna(subset=[target_col])
    
    # Codificar variable objetivo si es categórica
    if df[target_col].dtype == object or str(df[target_col].dtype) == 'category':
        print(f"Codificando variable objetivo '{target_col}' de tipo categórico a numérico...")
        df[target_col] = pd.factorize(df[target_col])[0]
        
    total_rows_full = int(df.shape[0])
    total_cols_full = int(df.shape[1])
    class_dist_full = {
        "0": int((df[target_col] == 0).sum()),
        "1": int((df[target_col] == 1).sum()),
    }
        
    # Rellenar nulos básicos en las variables predictoras (imputación simple)
    null_cols = df.columns[df.isnull().any()]
    if not null_cols.empty:
        print(f"\nImputando valores nulos en las columnas: {list(null_cols)}")
        for col in null_cols:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Desconocido")
                
    # 5. Submuestreo si es un dataset grande
    max_safe_samples = 1500
    if len(df) > max_safe_samples:
        if _cli_dataset:
            df = df.sample(n=max_safe_samples, random_state=42).reset_index(drop=True)
            print(f"Dataset reducido a {df.shape[0]} registros para entrenamiento (modo CLI).")
        else:
            print(f"\n[NOTA] El dataset contiene {len(df)} registros.")
            print("Dado que C4.5 calcula los puntos de corte óptimos de variables continuas en Python puro,")
            print(f"se recomienda realizar un submuestreo de {max_safe_samples} registros para optimizar el tiempo de ejecución.")
            subsample = input(f"¿Deseas submuestrear a {max_safe_samples} registros? (S/n): ").strip().lower()
            if subsample in ['', 's', 'si', 'yes', 'y']:
                df = df.sample(n=max_safe_samples, random_state=42).reset_index(drop=True)
                print(f"Dataset reducido a {df.shape[0]} registros para entrenamiento.")
            
    sample_size = len(df)
    class_dist_sample = {
        "0": int((df[target_col] == 0).sum()),
        "1": int((df[target_col] == 1).sum()),
    }
            
    # Separar variables y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Codificar variables predictoras categóricas si quedan
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"Codificando variables predictoras categóricas a numéricas: {list(categorical_cols)}")
        for col in categorical_cols:
            X[col] = pd.factorize(X[col])[0]
            
    # 6. Partición de datos
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    split_idx = int(len(df) * 0.70)  # 70% entrenamiento, 30% validación
    
    X_train = X.iloc[shuffled_indices[:split_idx]]
    y_train = y.iloc[shuffled_indices[:split_idx]]
    X_val = X.iloc[shuffled_indices[split_idx:]]
    y_val = y.iloc[shuffled_indices[split_idx:]]
    
    print(f"\nDistribución de datos de validación cruzada:")
    print(f"  Entrenamiento: {X_train.shape[0]} registros")
    print(f"  Validación   : {X_val.shape[0]} registros")
    
    # 7. Ejecutar Pipeline del Framework
    import time
    print("\n" + "=" * 50)
    if _mode == "metacog":
        print("   ETAPA 1: ENTRENAMIENTO - MetaCog-C45 (Sistema 1 + Sistema 2)")
    else:
        print("   ETAPA 1: ENTRENAMIENTO DEL CLASIFICADOR C4.5 (modo clásico)")
    print("=" * 50)

    # ── Resolución de hiperparámetros (CLI > defaults) ──────────────────
    _max_depth_val   = globals().get("_CLI_MAX_DEPTH",  None) or 5
    _min_samples_val = globals().get("_CLI_MIN_SAMPLES", None) or 10

    clf = C45Classifier(
        mode=_mode,
        max_depth=_max_depth_val,
        min_samples_split=_min_samples_val,
        min_gain_ratio=0.01,
        B=_B,
        theta_accept=_theta_accept,
        theta_reject=_theta_reject,
        gamma=_gamma,
        random_state=42,
    )

    if _mode == "metacog":
        print("Iniciando MetaCog-C45 (Reflection Engine + Decision Core + MetaMemory)...")
        print(f"  B={_B} | theta_accept={_theta_accept} | theta_reject={_theta_reject} | gamma={_gamma}")
    else:
        print("Construyendo árbol C4.5 clásico (esto puede tomar unos segundos)...")

    t0 = time.time()
    clf.fit(X_train, y_train)
    training_time_s = time.time() - t0
    print(f"Tiempo de entrenamiento: {training_time_s:.4f} s")
    clf.summary()

    tree_before_text = tree_to_text(clf.root, memory=clf.memory_)
    tree_before_dict = tree_to_dict(clf.root)
    tree_before_stats = {
        "nodes": TreeUtils.count_nodes(clf.root),
        "leaves": TreeUtils.count_leaves(clf.root),
        "depth": TreeUtils.depth(clf.root),
    }
    acc_train = float(clf.score(X_train, y_train))
    
    print("\nÁrbol de Decisión Inicial (Sin Podar):")
    clf.print_tree()
    clf.summary()
    
    acc_val_before = clf.score(X_val, y_val)
    print(f"\nExactitud (Accuracy) en Validación antes de podar: {acc_val_before:.4f}")
    
    print("\n" + "=" * 50)
    print("   ETAPA 2: PODA DEL ÁRBOL (REDUCED ERROR PRUNING)")
    print("=" * 50)
    t1 = time.time()
    DecisionTreePruner.prune(clf, X_val, y_val)
    pruning_time_s = time.time() - t1
    print(f"Tiempo de poda: {pruning_time_s:.4f} s")

    tree_after_text = tree_to_text(clf.root, memory=clf.memory_)
    tree_after_dict = tree_to_dict(clf.root)
    nodes_after  = TreeUtils.count_nodes(clf.root)
    leaves_after = TreeUtils.count_leaves(clf.root)
    depth_after  = TreeUtils.depth(clf.root)
    tree_after_stats = {
        "nodes": nodes_after,
        "leaves": leaves_after,
        "depth": depth_after,
    }

    print("\nÁrbol de Decisión Final (Podado):")
    clf.print_tree()
    clf.summary()
    
    # Inferencia individual (para medir latencia por muestra)
    t2 = time.time()
    val_preds = clf.predict(X_val)
    inference_total_s = time.time() - t2
    n_val = len(X_val)
    inference_time_ms = (inference_total_s / n_val * 1000) if n_val > 0 else 0.0
    # Estimación de memoria ~ 200 bytes por nodo
    memory_nodes_kb = (nodes_after * 200) / 1024

    acc_val_after = clf.score(X_val, y_val)
    print(f"\nExactitud (Accuracy) en Validación después de podar: {acc_val_after:.4f}")
    print(f"Tiempo de inferencia promedio: {inference_time_ms:.4f} ms/muestra")
    
    print("\n" + "=" * 50)
    print("   ETAPA 3: EVALUACIÓN Y MÉTRICAS AVANZADAS")
    print("=" * 50)
    cm = ConfusionMatrix.from_predictions(y_val, val_preds)
    cm.summary()
    
    metrics = ClassificationMetrics(cm)
    metrics.summary()
    
    # 8. Cálculo de curva ROC-AUC
    print("\nCalculando curva ROC-AUC...")
    roc_data_dict = None
    auc_value = 0.5
    try:
        val_probs = clf.predict_proba(X_val)[:, 1]
        roc_data = ROCAUCCalculator.calculate(y_val, val_probs)
        print(f"Área Bajo la Curva ROC (AUC): {roc_data['auc']:.4f}")
        
        roc_data_dict = {
            "fpr": [float(x) for x in roc_data["fpr"]],
            "tpr": [float(x) for x in roc_data["tpr"]],
            "auc": float(roc_data["auc"])
        }
        auc_value = float(roc_data["auc"])
        
        # Guardar gráfico ROC
        roc_path = "visualizacion_roc.png"
        plt.figure(figsize=(7, 5))
        plt.plot(roc_data["fpr"], roc_data["tpr"], color="#16A085", lw=2, label=f"ROC (AUC = {roc_data['auc']:.4f})")
        plt.plot([0, 1], [0, 1], color="#7F8C8D", lw=1.5, linestyle="--")
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Tasa de Falsos Positivos (FPR)")
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
        plt.title(f"Curva ROC - Dataset: {selected_file}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"Gráfico de curva ROC guardado exitosamente en '{roc_path}'")
    except Exception as e:
        print(f"No se pudo calcular/graficar la curva ROC: {e}")
        
    if roc_data_dict is None:
        roc_data_dict = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}
        
    # 9. Importancia de variables
    print("\n" + "=" * 50)
    print("   ETAPA 4: IMPORTANCIA DE VARIABLES")
    print("=" * 50)
    fi_df = None
    fi_list = []
    try:
        fi = FeatureImportance().fit(clf.root)
        fi_df = fi.dataframe()
        print(fi_df)
        # Calcular la dirección de correlación de cada feature con el target
        correlations = {col: float(X[col].corr(y)) for col in X.columns}
        fi_list = []
        for _, row in fi_df.iterrows():
            feat = row["Feature"]
            corr = correlations.get(feat, 0.0)
            fi_list.append({
                "feature": feat,
                "importance": round(float(row["Importance"]), 6),
                "direction": "+" if corr >= 0 else "-",
                "correlation": round(corr, 4),
            })
    except Exception as e:
        print(f"No se pudo calcular la importancia de variables: {e}")

    # 9b. Comparación con scikit-learn (opcional)
    print("\n" + "=" * 50)
    print("   ETAPA 5: COMPARACIÓN CON SCIKIT-LEARN")
    print("=" * 50)
    sklearn_comparison = None
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        sk_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
        sk_clf.fit(X_train, y_train)
        sk_preds = sk_clf.predict(X_val)
        sk_probs = sk_clf.predict_proba(X_val)[:, 1]
        sklearn_comparison = {
            "pyc45_accuracy":  round(float(acc_val_after), 4),
            "sklearn_accuracy": round(float(accuracy_score(y_val, sk_preds)), 4),
            "pyc45_precision":  round(float(metrics.precision), 4),
            "sklearn_precision": round(float(precision_score(y_val, sk_preds, zero_division=0)), 4),
            "pyc45_recall":  round(float(metrics.recall), 4),
            "sklearn_recall": round(float(recall_score(y_val, sk_preds, zero_division=0)), 4),
            "pyc45_f1":  round(float(metrics.f1_score), 4),
            "sklearn_f1": round(float(f1_score(y_val, sk_preds, zero_division=0)), 4),
            "pyc45_auc":  round(float(auc_value), 4),
            "sklearn_auc": round(float(roc_auc_score(y_val, sk_probs)), 4),
            "pyc45_nodes":  nodes_after,
            "sklearn_nodes": int(sk_clf.tree_.node_count),
            "pyc45_depth":  depth_after,
            "sklearn_depth": int(sk_clf.get_depth()),
        }
        print("Comparación scikit-learn completada con éxito.")
        for k, v in sklearn_comparison.items():
            print(f"  {k}: {v}")
    except ImportError:
        print("scikit-learn no instalado. Se omite comparación.")
    except Exception as e:
        print(f"Error en comparación scikit-learn: {e}")

    # 9c. Explicabilidad de muestras individuales
    print("\n" + "=" * 50)
    print("   ETAPA 6: EXPLICABILIDAD DE MUESTRAS")
    print("=" * 50)
    explainability_samples = []
    try:
        from core.explainability import ExplainabilityEngine
        exp_engine = ExplainabilityEngine()
        n_explain = min(10, len(X_val))
        for i in range(n_explain):
            sample_row = X_val.iloc[i]
            actual = int(y_val.iloc[i])
            predicted = int(val_preds[i])
            try:
                path = exp_engine.decision_path(clf.root, sample_row)
                path_list = []
                for step in path:
                    step_dict = {}
                    if hasattr(step, 'is_leaf'):
                        step_dict['is_leaf'] = bool(step.is_leaf)
                    if hasattr(step, 'feature'):
                        step_dict['feature'] = str(step.feature) if step.feature is not None else None
                    if hasattr(step, 'threshold'):
                        step_dict['threshold'] = float(step.threshold) if step.threshold is not None else None
                    if hasattr(step, 'prediction'):
                        step_dict['prediction'] = int(step.prediction) if step.prediction is not None else None
                    if hasattr(step, 'probability'):
                        step_dict['node_probability'] = round(float(step.probability), 4) if step.probability is not None else None
                    if hasattr(step, 'samples'):
                        step_dict['samples'] = int(step.samples) if step.samples is not None else None
                    if hasattr(step, 'class_counts') and step.class_counts:
                        step_dict['class_counts'] = {str(k): int(v) for k, v in step.class_counts.items()}
                    # Determinar dirección (left/right) y valor de la feature
                    if not getattr(step, 'is_leaf', True) and hasattr(step, 'feature') and step.feature in sample_row.index:
                        fval = float(sample_row[step.feature])
                        step_dict['feature_value'] = round(fval, 4)
                        step_dict['direction'] = 'left' if fval <= step.threshold else 'right'
                    path_list.append(step_dict)
            except Exception as ep:
                path_list = [{"error": str(ep)}]
            explainability_samples.append({
                "sample_index": int(X_val.index[i]),
                "actual_label": actual,
                "predicted_label": predicted,
                "decision_path": path_list,
            })
        print(f"Explicabilidad capturada para {n_explain} muestras.")
    except Exception as e:
        print(f"No se pudo capturar explicabilidad: {e}")

    # 9d. Impacto de Negocio
    print("\n" + "=" * 50)
    print("   ETAPA 7: CÁLCULO DE IMPACTO FINANCIERO")
    print("=" * 50)
    business_impact = {}
    try:
        tp_val = int(cm.tp)
        fn_val = int(cm.fn)
        fp_val = int(cm.fp)
        # Estimación: cada verdadero positivo evita una pérdida promedio de $3,000
        # Cada falso negativo representa una pérdida de $3,000 no evitada
        # Cada falso positivo tiene un costo de revisión de $200
        avg_loss_per_default = 3000
        cost_false_positive = 200
        estimated_savings = tp_val * avg_loss_per_default
        fn_exposure = fn_val * avg_loss_per_default
        fp_cost = fp_val * cost_false_positive
        net_benefit = estimated_savings - fp_cost
        total_risk = (tp_val + fn_val) * avg_loss_per_default
        risk_coverage_pct = (estimated_savings / total_risk * 100) if total_risk > 0 else 0
        roi_pct = (net_benefit / fp_cost * 100) if fp_cost > 0 else float('inf')

        business_impact = {
            "estimated_savings_usd": estimated_savings,
            "fn_exposure_usd": fn_exposure,
            "fp_cost_usd": fp_cost,
            "net_benefit_usd": net_benefit,
            "risk_coverage_pct": round(risk_coverage_pct, 2),
            "model_roi_pct": round(min(roi_pct, 9999), 2),
            "summary_text": (
                f"El modelo intercepta {tp_val} clientes en riesgo, ahorrando ~${estimated_savings:,}. "
                f"Los {fn_val} falsos negativos representan una exposición de ~${fn_exposure:,}. "
                f"Los {fp_val} falsos positivos generan costos de revisión de ~${fp_cost:,}. "
                f"Beneficio neto estimado: ${net_benefit:,}."
            ),
            "scenario_table": [
                {"scenario": "✅ VP Interceptados", "description": f"{tp_val} clientes en riesgo identificados correctamente", "value": f"${estimated_savings:,} ahorro", "impact_class": "good", "impact_label": "Alto"},
                {"scenario": "❌ FN No Detectados", "description": f"{fn_val} clientes en riesgo no detectados", "value": f"${fn_exposure:,} exposición", "impact_class": "bad", "impact_label": "Crítico"},
                {"scenario": "⚠️ FP Revisión Manual", "description": f"{fp_val} clientes marcados erróneamente", "value": f"${fp_cost:,} costo", "impact_class": "ok", "impact_label": "Moderado"},
                {"scenario": "💰 Beneficio Neto", "description": "Ahorro menos costos operacionales", "value": f"${net_benefit:,}", "impact_class": "good" if net_benefit > 0 else "bad", "impact_label": "Positivo" if net_benefit > 0 else "Negativo"},
                {"scenario": "📊 Cobertura de Riesgo", "description": "% del riesgo total que el modelo identifica", "value": f"{risk_coverage_pct:.1f}%", "impact_class": "good" if risk_coverage_pct > 50 else "ok", "impact_label": "Buena" if risk_coverage_pct > 50 else "Moderada"},
            ]
        }
        print(business_impact['summary_text'])
    except Exception as e:
        print(f"No se pudo calcular el impacto de negocio: {e}")

    # 9e. Generación de código exportable
    print("\nGenerando código exportable del árbol...")
    code_export = {}
    try:
        dataset_base = os.path.splitext(selected_file)[0].replace(' ', '_').lower()
        code_export = {
            "sql": get_sql_code(clf.root, table_name=dataset_base, target_col="prediccion"),
            "python": get_python_code(clf.root),
            "fastapi": get_fastapi_code(clf.root, X.columns.tolist()),
        }
        print("Código SQL, Python y FastAPI generados correctamente.")
    except Exception as e:
        print(f"No se pudo generar el código exportable: {e}")
        
    # 10. Guardar Imagen del Árbol de Decisión
    print("\nGenerando imagen gráfica del árbol de decisión...")
    try:
        tree_path = "visualizacion_arbol.png"
        plotter = TreePlotter()
        
        # Evitar popup GUI de Matplotlib
        original_show = plt.show
        plt.show = lambda: None
        
        plotter.plot(clf.root, figsize=(14, 8))
        plt.savefig(tree_path, dpi=150)
        plt.close('all')
        plt.show = original_show
        
        print(f"Gráfico de estructura del árbol guardado exitosamente en '{tree_path}'")
    except Exception as e:
        print(f"No se pudo generar la imagen del árbol: {e}")
        
    # 11. Compilar y Exportar Resultados para Visualización Web
    print("\nExportando resultados para visualización web...")
    try:
        report_data = {
            "dataset_name": selected_file,
            "dataset": {
                "total_rows": total_rows_full,
                "total_cols": total_cols_full,
                "columns": df.columns.tolist(),
                "target_column": target_col,
                "class_distribution_full": class_dist_full,
                "sample_size": sample_size,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "class_distribution_sample": class_dist_sample,
            },
            "hyperparameters": {
                "mode": clf.mode,
                "max_depth": int(clf.max_depth),
                "min_samples_split": int(clf.min_samples_split),
                "min_gain_ratio": float(clf.min_gain_ratio),
                "B": int(clf.B),
                "theta_accept": float(clf.theta_accept),
                "theta_reject": float(clf.theta_reject),
                "gamma": float(clf.gamma),
            },
            "metacognitive_data": _build_metacog_report(clf),
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
                "total_nodes":    nodes_after,
                "leaf_nodes":     leaves_after,
                "max_depth_real": depth_after,
            },
            "confusion_matrix": {
                "tp": int(cm.tp),
                "tn": int(cm.tn),
                "fp": int(cm.fp),
                "fn": int(cm.fn),
                "total": int(cm.total)
            },
            "metrics": {
                "Accuracy": float(metrics.accuracy),
                "Precision": float(metrics.precision),
                "Recall": float(metrics.recall),
                "Specificity": float(metrics.specificity),
                "F1-Score": float(metrics.f1_score),
                "Balanced Accuracy": float(metrics.balanced_accuracy),
                "Error Rate": float(metrics.error_rate),
                "MCC": float(metrics.mcc),
            },
            "roc_auc": float(auc_value),
            "feature_importance": fi_list,
            "system_metrics": {
                "training_time_s":   round(training_time_s, 6),
                "pruning_time_s":    round(pruning_time_s, 6),
                "inference_time_ms": round(inference_time_ms, 6),
                "memory_nodes_kb":   round(memory_nodes_kb, 2),
            },
            "sklearn_comparison": sklearn_comparison,
            "explainability_samples": explainability_samples,
            "business_impact": business_impact,
            "code_export": code_export,
            "charts": {
                "class_distribution": generate_class_distribution_chart(df[target_col]),
                "confusion_matrix": generate_confusion_matrix_chart(cm),
                "metrics": generate_metrics_chart({
                    "Accuracy": float(metrics.accuracy),
                    "Precision": float(metrics.precision),
                    "Recall": float(metrics.recall),
                    "Specificity": float(metrics.specificity),
                    "F1-Score": float(metrics.f1_score),
                    "Balanced Accuracy": float(metrics.balanced_accuracy),
                    "Error Rate": float(metrics.error_rate),
                    "MCC": float(metrics.mcc),
                }),
                "roc": generate_roc_chart(roc_data_dict),
                "feature_importance": generate_feature_importance_chart(fi_df) if fi_df is not None and not fi_df.empty else "",
                "entropy_diagram": generate_entropy_diagram(),
            }
        }
        
        output_dir = "web_verification"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "report_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
            
        output_path_js = os.path.join(output_dir, "report_data.js")
        with open(output_path_js, "w", encoding="utf-8") as f:
            f.write("window.reportData = ")
            json.dump(report_data, f, ensure_ascii=False, indent=2)
            f.write(";")
            
        print(f"Datos de visualización guardados en: {output_path} y {output_path_js}")
        
        # Abrir reporte en el navegador automáticamente
        html_path = os.path.abspath(os.path.join(output_dir, "index.html"))
        print(f"Abriendo el reporte en el navegador: file:///{html_path.replace(os.sep, '/')}")
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
    except Exception as e:
        print(f"Error al generar/abrir el informe técnico web: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("                 PROCESAMIENTO Y ANÁLISIS COMPLETADO CON ÉXITO")
    print("=" * 80)

if __name__ == "__main__":
    # ── CLI arguments (Release 1.0) ────────────────────────────────────────
    # Cuando se invoca desde la CLI oficial (`metacog --dataset ...`)
    # o directamente con argumentos, el script opera en modo no-interactivo.
    # Sin argumentos, conserva el comportamiento interactivo original.
    import argparse as _argparse

    _cli_parser = _argparse.ArgumentParser(
        prog="ejecutar_con_dataset",
        description="MetaCog-C45 — Ejecutor de datasets con reporte HTML.",
        add_help=False,   # help ya lo gestiona el parser padre (metacog_c45/cli.py)
    )
    _cli_parser.add_argument("--dataset",        metavar="RUTA",    default=None)
    _cli_parser.add_argument("--target",         metavar="COLUMNA", default=None)
    _cli_parser.add_argument("--max_depth",      metavar="N",       type=int, default=None)
    _cli_parser.add_argument("--min_samples_split", metavar="N",    type=int, default=None)
    _cli_parser.add_argument("--no_browser",     action="store_true", default=False)
    _cli_parser.add_argument("--help", "-h",     action="store_true", default=False)
    # ── System 2 arguments ────────────────────────────────────────────────
    _cli_parser.add_argument(
        "--mode",
        choices=["classic", "metacog"],
        default="metacog",
        help="Modo de inducción: 'classic' (C4.5 estándar) o 'metacog' (MetaCog-C45 con Sistema 2). Por defecto: metacog.",
    )
    _cli_parser.add_argument("--B",             metavar="N",   type=int,   default=50,
        help="Réplicas bootstrap para el Reflection Engine (solo modo metacog, por defecto: 50).")
    _cli_parser.add_argument("--theta_accept",  metavar="F",   type=float, default=0.5,
        help="Umbral de aceptación del SCS (por defecto: 0.5).")
    _cli_parser.add_argument("--theta_reject",  metavar="F",   type=float, default=0.05,
        help="Umbral de colapso del SCS (por defecto: 0.05).")
    _cli_parser.add_argument("--gamma",         metavar="F",   type=float, default=0.1,
        help="Exponente de penalización de competencia CI (por defecto: 0.1).")

    _cli_args, _unknown = _cli_parser.parse_known_args()

    if _cli_args.help:
        _cli_parser.print_help()
    else:
        # Exponer los argumentos como variables de módulo para que main()
        # pueda leerlos en futuras versiones sin romper la interfaz actual.
        _CLI_DATASET        = _cli_args.dataset
        _CLI_TARGET         = _cli_args.target
        _CLI_MAX_DEPTH      = _cli_args.max_depth
        _CLI_MIN_SAMPLES    = _cli_args.min_samples_split
        _CLI_NO_BROWSER     = _cli_args.no_browser
        _CLI_MODE           = _cli_args.mode
        _CLI_B              = _cli_args.B
        _CLI_THETA_ACCEPT   = _cli_args.theta_accept
        _CLI_THETA_REJECT   = _cli_args.theta_reject
        _CLI_GAMMA          = _cli_args.gamma
        main()
